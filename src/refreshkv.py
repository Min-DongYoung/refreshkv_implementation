import inspect
import math
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .logging_utils import compact_indices, position_stats, token_type_summary


@dataclass
class RefreshKVConfig:
    model_name: str = "Qwen/Qwen2-7B-Instruct"
    device: str = "cuda"
    dtype: str = "bfloat16"
    seed: int = 42

    max_new_tokens: int = 128
    qc_stride: int = 10
    similarity_threshold: float = 0.95

    # Partial cache budget (K). If None, use ratio * prompt length.
    partial_cache_size: Optional[int] = None
    partial_cache_ratio: float = 0.125

    # Attention-score pooling (kernel size per Appendix A.1, PDF p.11).
    pool_kernel_size: int = 7
    pool_padding: str = "same"

    # Attention score aggregation across heads for GQA.
    head_aggregation: str = "max"  # max | mean | first

    # TopK ordering: "score_asc" matches reverse(arg top-k ...) then evict Cp[0].
    topk_order: str = "score_asc"  # score_asc | score_desc | pos_asc
    evict_policy: str = "front"  # front | lowest_score

    # Trigger modes: baseline (query similarity), entropy, hybrid (baseline OR entropy)
    trigger_mode: str = "baseline"

    # Entropy trigger settings
    entropy_threshold: float = 4.0
    entropy_use_zscore: bool = False
    entropy_zscore_k: float = 2.0
    entropy_warmup: int = 10

    # Behavior flags
    layerwise_qc: bool = False
    log_per_head_similarity: bool = False
    log_decoded_tokens: bool = False
    first_token_from_prefill: bool = True
    pending_merge_policy: str = "pending_kv"  # pending_kv | cp_tail

    # Optional attention implementation control (if set in loader)
    attn_implementation: str = "eager"

    # Flash/SDPA path with eager recompute for attention scores
    use_fast_attention: bool = False
    recompute_drop_self: bool = True

    # Logging / limits
    max_log_indices: int = 200


class EntropyStats:
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, x: float) -> None:
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self) -> float:
        if self.count < 2:
            return 0.0
        return self.m2 / (self.count - 1)

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    def zscore(self, x: float) -> float:
        std = self.std
        if std == 0.0:
            return 0.0
        return (x - self.mean) / std


class QueryCapture:
    def __init__(self, model, log_per_head: bool = False):
        self.model = model
        self.log_per_head = log_per_head
        self.handles = []
        self.q_mean: List[Optional[torch.Tensor]] = []
        self.q_per_head: List[Optional[torch.Tensor]] = []

    def _get_layers(self):
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        if hasattr(self.model, "base_model") and hasattr(self.model.base_model, "layers"):
            return self.model.base_model.layers
        raise RuntimeError("Unable to locate decoder layers on model.")

    def attach(self):
        layers = self._get_layers()
        self.q_mean = [None for _ in range(len(layers))]
        self.q_per_head = [None for _ in range(len(layers))]

        for idx, layer in enumerate(layers):
            attn = getattr(layer, "self_attn", None)
            if attn is None or not hasattr(attn, "q_proj"):
                raise RuntimeError("Attention module missing q_proj; cannot capture queries.")

            def _hook(module, inputs, kwargs, idx=idx):
                hidden_states = None
                if inputs:
                    hidden_states = inputs[0]
                else:
                    hidden_states = kwargs.get("hidden_states", None)
                if hidden_states is None:
                    raise RuntimeError("QueryCapture: hidden_states not found in hook inputs.")
                q = module.q_proj(hidden_states)
                bsz, seq_len, _ = q.shape
                num_heads = getattr(module, "num_heads", None) or getattr(module, "num_attention_heads", None)
                head_dim = getattr(module, "head_dim", None)
                out_features = q.shape[-1]
                if num_heads is None and head_dim is None and hasattr(module, "config"):
                    num_heads = getattr(module.config, "num_attention_heads", None)
                if num_heads is None and head_dim is not None:
                    num_heads = out_features // head_dim
                if head_dim is None and num_heads is not None:
                    head_dim = out_features // num_heads
                if num_heads is None or head_dim is None:
                    raise RuntimeError(
                        f"Attention module missing num_heads/head_dim (module={module.__class__.__name__})."
                    )
                q = q.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
                q_mean = q.mean(dim=1)  # [b, seq, head_dim]
                self.q_mean[idx] = q_mean[:, -1, :].detach()
                if self.log_per_head:
                    self.q_per_head[idx] = q[:, :, -1, :].detach()

            handle = attn.register_forward_pre_hook(_hook, with_kwargs=True)
            self.handles.append(handle)

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def _normalize_past_key_values(pkv):
    if pkv is None:
        return None
    if hasattr(pkv, "to_legacy_cache"):
        pkv = pkv.to_legacy_cache()
    if isinstance(pkv, tuple):
        pkv = list(pkv)
    return pkv


def _to_cache_if_needed(model, pkv):
    if pkv is None:
        return None
    if hasattr(pkv, "get_mask_sizes"):
        return pkv
    # Convert legacy list/tuple to a Cache object whenever possible.
    try:
        from transformers.cache_utils import DynamicCache

        return DynamicCache.from_legacy_cache(pkv)
    except Exception:
        return pkv


def _ensure_cache(model, pkv, cache_position, logger=None, step=None, tag=None):
    if pkv is None:
        return None
    if hasattr(pkv, "get_mask_sizes"):
        cache_obj = pkv
        converted = False
    else:
        try:
            from transformers.cache_utils import DynamicCache

            cache_obj = DynamicCache.from_legacy_cache(pkv)
            converted = True
        except Exception as exc:
            raise RuntimeError(
                "past_key_values must be a Cache object when cache_position is provided. "
                "Conversion to DynamicCache failed."
            ) from exc
    if not hasattr(cache_obj, "get_mask_sizes"):
        raise RuntimeError("past_key_values must implement Cache API (get_mask_sizes).")
    if logger is not None and step is not None:
        logger.log_event(
            {
                "step": step,
                "cache_call": {
                    "tag": tag,
                    "cache_type": cache_obj.__class__.__name__,
                    "from_legacy": converted,
                    "cache_position": int(cache_position) if cache_position is not None else None,
                },
            }
        )
    return cache_obj


def _model_forward(model, **kwargs):
    sig = inspect.signature(model.forward)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters and v is not None}
    if "past_key_values" in filtered:
        filtered["past_key_values"] = _to_cache_if_needed(model, filtered["past_key_values"])
    return model(**filtered)


def _force_eager_attention(model) -> None:
    # Best-effort switches to ensure attention weights are returned.
    if hasattr(model, "config"):
        model.config.use_cache = True
        model.config.output_attentions = True
        if hasattr(model.config, "attn_implementation"):
            model.config.attn_implementation = "eager"
        if hasattr(model.config, "_attn_implementation"):
            model.config._attn_implementation = "eager"
    if hasattr(model, "attn_implementation"):
        model.attn_implementation = "eager"
    if hasattr(model, "_attn_implementation"):
        model._attn_implementation = "eager"
    # Disable SDPA fast paths so attentions are materialized.
    if hasattr(torch, "backends") and hasattr(torch.backends, "cuda"):
        for fn_name, val in (
            ("enable_flash_sdp", False),
            ("enable_mem_efficient_sdp", False),
            ("enable_math_sdp", True),
        ):
            fn = getattr(torch.backends.cuda, fn_name, None)
            if callable(fn):
                fn(val)
    # Best-effort per-layer override
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        for layer in model.model.layers:
            attn = getattr(layer, "self_attn", None)
            if attn is not None and hasattr(attn, "attn_implementation"):
                attn.attn_implementation = "eager"


def _aggregate_attention(attn: torch.Tensor, mode: str, num_kv_heads: Optional[int] = None) -> torch.Tensor:
    # attn: [b, heads, tgt_len, src_len]
    attn = attn.float()
    last = attn[:, :, -1, :]  # [b, heads, src_len]
    num_heads = last.shape[1]
    if mode == "max":
        # For GQA, take max within each query-head group and keep group-wise scores.
        if num_kv_heads is not None and num_kv_heads > 0 and num_kv_heads < num_heads:
            group = num_heads // num_kv_heads
            grouped = last.view(last.shape[0], num_kv_heads, group, last.shape[-1])
            scores = grouped.max(dim=2).values  # [b, num_kv_heads, src_len]
        else:
            scores = last.max(dim=1).values
    elif mode == "mean":
        scores = last.mean(dim=1)
    elif mode == "first":
        scores = last[:, 0, :]
    else:
        raise ValueError(f"Unknown head_aggregation: {mode}")
    return scores  # [b, src_len]


def _pool_scores(scores: torch.Tensor, kernel_size: int, padding_mode: str) -> torch.Tensor:
    if kernel_size <= 1:
        return scores
    pad = kernel_size // 2 if padding_mode == "same" else 0
    if scores.dim() == 2:
        pooled = F.max_pool1d(scores.unsqueeze(1), kernel_size=kernel_size, stride=1, padding=pad)
        pooled = pooled.squeeze(1)
        return pooled
    if scores.dim() == 3:
        b, g, s = scores.shape
        flat = scores.reshape(b * g, 1, s)
        pooled = F.max_pool1d(flat, kernel_size=kernel_size, stride=1, padding=pad)
        pooled = pooled.reshape(b, g, s)
        return pooled
    raise ValueError(f"Unsupported scores shape for pooling: {scores.shape}")


def _select_topk(scores: torch.Tensor, k: int, order: str) -> Tuple[torch.Tensor, torch.Tensor]:
    # scores can be [src_len] or [group, src_len]
    if scores.dim() == 2:
        group, src_len = scores.shape
        flat = scores.reshape(-1)
        k = min(k, src_len)
        # take extra candidates to reduce duplicates
        cand_k = min(flat.numel(), max(k * 4, k))
        top_scores, top_idx = torch.topk(flat, k=cand_k, largest=True)
        selected = {}
        for sc, idx in zip(top_scores.tolist(), top_idx.tolist()):
            tok = idx % src_len
            if tok not in selected:
                selected[tok] = sc
            if len(selected) >= k:
                break
        if len(selected) < k:
            sorted_idx = torch.argsort(flat, descending=True)
            for idx in sorted_idx.tolist():
                tok = idx % src_len
                if tok not in selected:
                    selected[tok] = float(flat[idx].item())
                if len(selected) >= k:
                    break
        top_idx = torch.tensor(list(selected.keys()), device=scores.device)
        top_scores = torch.tensor(list(selected.values()), device=scores.device)
    else:
        k = min(k, scores.shape[-1])
        top_scores, top_idx = torch.topk(scores, k=k, largest=True)
    if order == "score_asc":
        order_idx = torch.argsort(top_scores, descending=False)
    elif order == "score_desc":
        order_idx = torch.argsort(top_scores, descending=True)
    elif order == "pos_asc":
        order_idx = torch.argsort(top_idx, descending=False)
    else:
        raise ValueError(f"Unknown topk_order: {order}")
    return top_idx[order_idx], top_scores[order_idx]


def _entropy_and_margin(logits: torch.Tensor) -> Tuple[float, float]:
    logits = logits.float()
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    top2 = torch.topk(logits, k=2, dim=-1).values
    margin = top2[..., 0] - top2[..., 1]
    ent = float(entropy.item())
    assert ent >= -1e-6, "Entropy should be non-negative"
    return ent, float(margin.item())


@dataclass
class CacheState:
    cf: List[Tuple[torch.Tensor, torch.Tensor]]
    cp: List[Tuple[torch.Tensor, torch.Tensor]]
    cp_idx_abs: List[List[int]]
    cp_scores: List[List[float]]
    pending_k: List[Optional[torch.Tensor]]
    pending_v: List[Optional[torch.Tensor]]
    pending_idx_abs: List[List[int]]
    q_last_full: List[torch.Tensor]
    q_last_full_per_head: Optional[List[torch.Tensor]]
    pos_full: List[int]
    k: int


class RefreshKVGenerator:
    def __init__(self, model, tokenizer, cfg: RefreshKVConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.entropy_stats = EntropyStats()
        self._layerwise_warned = False

    def _with_eager_attention(self):
        class _EagerCtx:
            def __init__(self, model):
                self.model = model
                self.prev_attn = getattr(model.config, "attn_implementation", None)
                self.prev_attn_priv = getattr(model.config, "_attn_implementation", None)
                self.prev_out_attn = getattr(model.config, "output_attentions", None)
                self.layer_prev = []
                self.prev_sdp = {}

            def __enter__(self):
                # Save current SDPA settings if available, then disable fast SDP paths.
                if hasattr(torch, "backends") and hasattr(torch.backends, "cuda"):
                    for name, getter, setter, value in (
                        ("flash", "flash_sdp_enabled", "enable_flash_sdp", False),
                        ("mem_efficient", "mem_efficient_sdp_enabled", "enable_mem_efficient_sdp", False),
                        ("math", "math_sdp_enabled", "enable_math_sdp", True),
                    ):
                        get_fn = getattr(torch.backends.cuda, getter, None)
                        set_fn = getattr(torch.backends.cuda, setter, None)
                        if callable(get_fn) and callable(set_fn):
                            try:
                                self.prev_sdp[name] = get_fn()
                                set_fn(value)
                            except Exception:
                                pass
                if hasattr(self.model, "config"):
                    self.model.config.attn_implementation = "eager"
                    self.model.config._attn_implementation = "eager"
                    self.model.config.output_attentions = True
                if hasattr(self.model, "model") and hasattr(self.model.model, "config"):
                    self.model.model.config.attn_implementation = "eager"
                    self.model.model.config._attn_implementation = "eager"
                    self.model.model.config.output_attentions = True
                if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
                    for layer in self.model.model.layers:
                        attn = getattr(layer, "self_attn", None)
                        if attn is not None and hasattr(attn, "attn_implementation"):
                            self.layer_prev.append(attn.attn_implementation)
                            attn.attn_implementation = "eager"
                            if hasattr(attn, "_attn_implementation"):
                                attn._attn_implementation = "eager"
                            if hasattr(attn, "config"):
                                attn.config.attn_implementation = "eager"
                                attn.config._attn_implementation = "eager"
                        else:
                            self.layer_prev.append(None)
                return self

            def __exit__(self, exc_type, exc, tb):
                if hasattr(self.model, "config"):
                    self.model.config.attn_implementation = self.prev_attn
                    self.model.config._attn_implementation = self.prev_attn_priv
                    if self.prev_out_attn is not None:
                        self.model.config.output_attentions = self.prev_out_attn
                if hasattr(self.model, "model") and hasattr(self.model.model, "config"):
                    self.model.model.config.attn_implementation = self.prev_attn
                    self.model.model.config._attn_implementation = self.prev_attn_priv
                    if self.prev_out_attn is not None:
                        self.model.model.config.output_attentions = self.prev_out_attn
                if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
                    for layer, prev in zip(self.model.model.layers, self.layer_prev):
                        attn = getattr(layer, "self_attn", None)
                        if attn is not None and hasattr(attn, "attn_implementation") and prev is not None:
                            attn.attn_implementation = prev
                            if hasattr(attn, "_attn_implementation"):
                                attn._attn_implementation = prev
                # Restore SDPA settings if we captured them.
                if hasattr(torch, "backends") and hasattr(torch.backends, "cuda"):
                    for name, setter in (
                        ("flash", "enable_flash_sdp"),
                        ("mem_efficient", "enable_mem_efficient_sdp"),
                        ("math", "enable_math_sdp"),
                    ):
                        if name in self.prev_sdp:
                            set_fn = getattr(torch.backends.cuda, setter, None)
                            if callable(set_fn):
                                try:
                                    set_fn(self.prev_sdp[name])
                                except Exception:
                                    pass
                return False

        return _EagerCtx(self.model)

    def _recompute_attn_for_token(
        self,
        input_ids: torch.Tensor,
        cache_legacy,
        abs_pos: int,
        logger=None,
        step: Optional[int] = None,
        tag: str = "recompute",
    ):
        cache = _ensure_cache(self.model, cache_legacy, cache_position=abs_pos, logger=logger, step=step, tag=tag)
        with self._with_eager_attention():
            # Diagnostics: log effective attention implementation and SDPA flags.
            if logger is not None and step is not None:
                layer0_attn = None
                layer0_attn_priv = None
                if hasattr(self.model, "model") and hasattr(self.model.model, "layers") and self.model.model.layers:
                    attn0 = getattr(self.model.model.layers[0], "self_attn", None)
                    if attn0 is not None:
                        layer0_attn = getattr(attn0, "attn_implementation", None)
                        layer0_attn_priv = getattr(attn0, "_attn_implementation", None)
                diag = {
                    "step": step,
                    "recompute_diag": {
                        "tag": tag,
                        "model_attn_impl": getattr(self.model.config, "attn_implementation", None),
                        "model_attn_impl_priv": getattr(self.model.config, "_attn_implementation", None),
                        "model_output_attn": getattr(self.model.config, "output_attentions", None),
                        "inner_attn_impl": getattr(getattr(self.model, "model", None), "config", None)
                        and getattr(self.model.model.config, "attn_implementation", None),
                        "inner_attn_impl_priv": getattr(getattr(self.model, "model", None), "config", None)
                        and getattr(self.model.model.config, "_attn_implementation", None),
                        "layer0_attn_impl": layer0_attn,
                        "layer0_attn_impl_priv": layer0_attn_priv,
                        "flash_sdp_enabled": (
                            torch.backends.cuda.flash_sdp_enabled()
                            if hasattr(torch.backends, "cuda")
                            and callable(getattr(torch.backends.cuda, "flash_sdp_enabled", None))
                            else None
                        ),
                        "mem_efficient_sdp_enabled": (
                            torch.backends.cuda.mem_efficient_sdp_enabled()
                            if hasattr(torch.backends, "cuda")
                            and callable(getattr(torch.backends.cuda, "mem_efficient_sdp_enabled", None))
                            else None
                        ),
                        "math_sdp_enabled": (
                            torch.backends.cuda.math_sdp_enabled()
                            if hasattr(torch.backends, "cuda")
                            and callable(getattr(torch.backends.cuda, "math_sdp_enabled", None))
                            else None
                        ),
                    },
                }
                logger.log_event(diag)
            with torch.inference_mode():
                out = _model_forward(
                    self.model,
                    input_ids=input_ids,
                    past_key_values=cache,
                    use_cache=False,  # do not mutate caches
                    output_attentions=True,
                    return_dict=True,
                    position_ids=torch.tensor([[abs_pos]], device=self._device()),
                    cache_position=torch.tensor([abs_pos], device=self._device()),
                )
        if out.attentions is None:
            # Include a hint about current attention implementation for debugging.
            attn_impl = getattr(self.model.config, "attn_implementation", None) or getattr(
                self.model.config, "_attn_implementation", None
            )
            raise RuntimeError(
                f"Recompute attention failed to return attentions (attn_implementation={attn_impl}). "
                "Ensure eager attention is enabled for recompute."
            )
        return out.attentions

    def _device(self):
        return next(self.model.parameters()).device

    def _prefill(self, input_ids: torch.Tensor, logger=None) -> Tuple[CacheState, torch.Tensor]:
        device = self._device()
        assert input_ids.shape[0] == 1, "Only batch size 1 is supported in this baseline."
        input_ids = input_ids.to(device)
        attention_mask = torch.ones_like(input_ids, device=device)
        position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
        cache_position = torch.arange(input_ids.shape[1], device=device)

        capture = QueryCapture(self.model, log_per_head=self.cfg.log_per_head_similarity)
        capture.attach()
        with torch.inference_mode():
            outputs = _model_forward(
                self.model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache_position=cache_position,
                use_cache=True,
                output_attentions=not self.cfg.use_fast_attention,
                return_dict=True,
            )
        capture.detach()

        past = _normalize_past_key_values(outputs.past_key_values)
        attentions = outputs.attentions
        logits = outputs.logits

        if past is None:
            raise RuntimeError("Model did not return past_key_values.")

        if attentions is None and not self.cfg.use_fast_attention:
            # Expect attentions in eager mode.
            raise RuntimeError(
                "Model did not return attentions in eager mode. Ensure attn_implementation='eager'."
            )

        num_layers = len(past)
        prompt_len = input_ids.shape[1]
        k = self.cfg.partial_cache_size or max(1, int(self.cfg.partial_cache_ratio * prompt_len))
        k = min(k, prompt_len)

        cp = []
        cp_idx_abs = []
        cp_scores = []
        q_last_full = []
        q_last_full_per_head = [] if self.cfg.log_per_head_similarity else None

        # If using fast attention, recompute attention for the last prompt token.
        if self.cfg.use_fast_attention:
            if prompt_len <= 1:
                raise RuntimeError("Prompt length must be > 1 for recompute with drop-self.")
            cf_excl_last = []
            for (k_full, v_full) in past:
                cf_excl_last.append((k_full[:, :, :-1, :], v_full[:, :, :-1, :]))
            attentions = self._recompute_attn_for_token(
                input_ids[:, -1:],
                cf_excl_last,
                abs_pos=prompt_len - 1,
                logger=logger,
                step=0,
                tag="prefill_recompute",
            )
            if attentions is None:
                raise RuntimeError("Recompute attention returned None at prefill.")

        num_kv_heads = getattr(self.model.config, "num_key_value_heads", None)
        for l in range(num_layers):
            attn = attentions[l]
            scores = _aggregate_attention(attn, self.cfg.head_aggregation, num_kv_heads=num_kv_heads)
            scores = _pool_scores(scores, self.cfg.pool_kernel_size, self.cfg.pool_padding)
            if self.cfg.recompute_drop_self and scores.shape[-1] > 1:
                scores = scores[..., :-1]
            if l == 0:
                k = min(k, scores.shape[-1])
            if k < 1:
                raise RuntimeError("No tokens available for topK selection after drop-self.")
            # PDF p.4 Algorithm 1: Cp initialized with reverse(arg top-k ...) then evict lowest.
            idx, sc = _select_topk(scores.squeeze(0), k=k, order=self.cfg.topk_order)

            cf_k, cf_v = past[l]
            assert idx.max().item() < cf_k.shape[2]
            cp_k = torch.index_select(cf_k, dim=2, index=idx)
            cp_v = torch.index_select(cf_v, dim=2, index=idx)
            assert cp_k.shape[2] == k
            cp.append((cp_k, cp_v))
            cp_idx_abs.append(idx.tolist())
            cp_scores.append(sc.tolist())

            q_last = capture.q_mean[l]
            if q_last is None:
                raise RuntimeError("Query capture failed during prefill.")
            q_last_full.append(q_last.squeeze(0))
            if q_last_full_per_head is not None:
                qh = capture.q_per_head[l]
                if qh is None:
                    raise RuntimeError("Per-head query capture failed during prefill.")
                q_last_full_per_head.append(qh.squeeze(0))

        pending_k = [None for _ in range(num_layers)]
        pending_v = [None for _ in range(num_layers)]
        pending_idx_abs = [[] for _ in range(num_layers)]
        pos_full = [1 for _ in range(num_layers)]

        state = CacheState(
            cf=past,
            cp=cp,
            cp_idx_abs=cp_idx_abs,
            cp_scores=cp_scores,
            pending_k=pending_k,
            pending_v=pending_v,
            pending_idx_abs=pending_idx_abs,
            q_last_full=q_last_full,
            q_last_full_per_head=q_last_full_per_head,
            pos_full=pos_full,
            k=k,
        )
        return state, logits

    def _merge_pending(self, state: CacheState, layer: int) -> None:
        if state.pending_k[layer] is None:
            return
        cf_k, cf_v = state.cf[layer]
        if self.cfg.pending_merge_policy == "cp_tail":
            pending_len = state.pending_k[layer].shape[2]
            cp_k, cp_v = state.cp[layer]
            if pending_len > cp_k.shape[2]:
                raise RuntimeError("pending_len exceeds Cp length; cannot merge from Cp tail safely.")
            tail_k = cp_k[:, :, -pending_len:, :]
            tail_v = cp_v[:, :, -pending_len:, :]
            state.cf[layer] = (torch.cat([cf_k, tail_k], dim=2), torch.cat([cf_v, tail_v], dim=2))
        else:
            pk = state.pending_k[layer]
            pv = state.pending_v[layer]
            state.cf[layer] = (torch.cat([cf_k, pk], dim=2), torch.cat([cf_v, pv], dim=2))
        state.pending_k[layer] = None
        state.pending_v[layer] = None
        state.pending_idx_abs[layer] = []

    def _append_pending(self, state: CacheState, layer: int, new_k: torch.Tensor, new_v: torch.Tensor, abs_idx: int) -> None:
        if state.pending_k[layer] is None:
            state.pending_k[layer] = new_k
            state.pending_v[layer] = new_v
        else:
            state.pending_k[layer] = torch.cat([state.pending_k[layer], new_k], dim=2)
            state.pending_v[layer] = torch.cat([state.pending_v[layer], new_v], dim=2)
        state.pending_idx_abs[layer].append(abs_idx)

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: Optional[int] = None,
        logger=None,
    ) -> Dict[str, Any]:
        cfg = self.cfg
        max_new_tokens = max_new_tokens or cfg.max_new_tokens

        if not cfg.first_token_from_prefill:
            raise NotImplementedError("first_token_from_prefill=False is not supported in this baseline.")

        state, prefill_logits = self._prefill(input_ids, logger=logger)
        full_token_ids = input_ids[0].tolist()
        prompt_len = len(full_token_ids)

        generated_ids: List[int] = []
        entropies: List[float] = []
        margins: List[float] = []
        refresh_latencies: List[float] = []
        non_refresh_latencies: List[float] = []
        overlap_ratios: List[float] = []

        # First token (optionally from prefill logits)
        if cfg.first_token_from_prefill:
            logits_last = prefill_logits[:, -1, :]
            entropy, margin = _entropy_and_margin(logits_last)
            entropies.append(entropy)
            margins.append(margin)
            next_token = int(torch.argmax(logits_last, dim=-1).item())
            generated_ids.append(next_token)
            full_token_ids.append(next_token)
            if logger is not None:
                logger.log_token(
                    {
                        "step": 1,
                        "token_id": next_token,
                        "token_str": self.tokenizer.decode([next_token], clean_up_tokenization_spaces=False)
                        if cfg.log_decoded_tokens
                        else None,
                        "entropy": entropy,
                        "logit_margin": margin,
                        "is_check_step": False,
                        "refresh_fired": False,
                    }
                )
        else:
            next_token = int(input_ids[0, -1].item())

        # Generation loop (starts from token already sampled above if first_token_from_prefill)
        step_start = 1 if cfg.first_token_from_prefill else 0
        current_token = torch.tensor([[next_token]], device=self._device())
        abs_pos = prompt_len  # absolute position of current_token

        for step in range(step_start + 1, max_new_tokens + 1):
            is_check = cfg.qc_stride > 0 and (step % cfg.qc_stride == 0)
            if cfg.layerwise_qc and (not self._layerwise_warned) and logger is not None:
                logger.log_event(
                    {
                        "step": step,
                        "warning": "layerwise_qc_ignored_for_execution",
                        "detail": "Uniform cache selection enforced per step for HF safety.",
                    }
                )
                self._layerwise_warned = True

            # Probe for queries (and entropy trigger) using partial caches
            q_probe = None
            q_probe_heads = None
            probe_entropy = None
            probe_margin = None
            probe_entropy_z = None
            entropy_fired = False
            per_layer_sim = None
            per_head_sim = None

            if is_check:
                # PDF p.3: query-comparison (QC) stride; compare query similarity per layer at check steps.
                capture = QueryCapture(self.model, log_per_head=cfg.log_per_head_similarity)
                capture.attach()
                with torch.inference_mode():
                    probe_cache = _ensure_cache(
                        self.model,
                        state.cp,
                        cache_position=abs_pos,
                        logger=logger,
                        step=step,
                        tag="qc_probe",
                    )
                    probe_out = _model_forward(
                        self.model,
                        input_ids=current_token,
                        past_key_values=probe_cache,
                        use_cache=False,
                        output_attentions=False,
                        return_dict=True,
                        position_ids=torch.tensor([[abs_pos]], device=self._device()),
                        cache_position=torch.tensor([abs_pos], device=self._device()),
                    )
                capture.detach()
                q_probe = [q.squeeze(0) if q is not None else None for q in capture.q_mean]
                if cfg.log_per_head_similarity and capture.q_per_head:
                    q_probe_heads = [q.squeeze(0) if q is not None else None for q in capture.q_per_head]

                # Entropy trigger uses probe logits
                if cfg.trigger_mode in ("entropy", "hybrid"):
                    probe_entropy, probe_margin = _entropy_and_margin(probe_out.logits[:, -1, :])
                    probe_entropy_z = None
                    if cfg.entropy_use_zscore:
                        if self.entropy_stats.count >= cfg.entropy_warmup:
                            probe_entropy_z = self.entropy_stats.zscore(probe_entropy)
                            entropy_fired = probe_entropy_z > cfg.entropy_zscore_k
                        self.entropy_stats.update(probe_entropy)
                    else:
                        entropy_fired = probe_entropy > cfg.entropy_threshold

                # Baseline per-layer similarity
                if cfg.trigger_mode in ("baseline", "hybrid"):
                    per_layer_sim = []
                    if cfg.log_per_head_similarity and state.q_last_full_per_head is not None:
                        per_head_sim = []
                    for l, q in enumerate(q_probe):
                        if q is None:
                            per_layer_sim.append(None)
                            if per_head_sim is not None:
                                per_head_sim.append(None)
                            continue
                        sim = F.cosine_similarity(q, state.q_last_full[l], dim=0).item()
                        assert -1.0001 <= sim <= 1.0001, "Cosine similarity out of range"
                        per_layer_sim.append(sim)
                        if per_head_sim is not None and q_probe_heads is not None:
                            qh = q_probe_heads[l]
                            qh_last = state.q_last_full_per_head[l]
                            if qh is None or qh_last is None:
                                per_head_sim.append(None)
                            else:
                                # cosine similarity per head
                                head_sims = F.cosine_similarity(qh, qh_last, dim=-1).tolist()
                                per_head_sim.append(head_sims)

            # Decide which cache to use per layer
            num_layers = len(state.cf)
            layer_use_full = [False for _ in range(num_layers)]
            uniform_enforced = False
            min_similarity = None
            use_full_decision = None
            decision_agg = None
            if is_check:
                if cfg.trigger_mode == "entropy":
                    use_full = bool(entropy_fired)
                    decision_agg = "entropy"
                elif cfg.trigger_mode == "baseline":
                    sims = [s for s in (per_layer_sim or []) if s is not None]
                    min_similarity = min(sims) if sims else 1.0
                    use_full = min_similarity <= cfg.similarity_threshold
                    decision_agg = "min"
                elif cfg.trigger_mode == "hybrid":
                    if entropy_fired:
                        use_full = True
                        decision_agg = "entropy"
                    else:
                        sims = [s for s in (per_layer_sim or []) if s is not None]
                        min_similarity = min(sims) if sims else 1.0
                        use_full = min_similarity <= cfg.similarity_threshold
                        decision_agg = "min"
                else:
                    use_full = False
                # HF safety: enforce uniform cache selection across layers at each step.
                layer_use_full = [use_full for _ in range(num_layers)]
                uniform_enforced = True
                use_full_decision = use_full

            # Safety invariant: no per-layer mixing within a step.
            if is_check and len(set(layer_use_full)) > 1:
                if logger is not None:
                    logger.log_event(
                        {
                            "step": step,
                            "error": "non_uniform_layer_use_full",
                            "layer_use_full": layer_use_full,
                        }
                    )
                raise RuntimeError("Non-uniform layer_use_full detected; per-layer mixing is unsafe under HF.")

            # Merge pending tokens into Cf for layers that will use full cache
            for l in range(num_layers):
                if is_check and layer_use_full[l]:
                    self._merge_pending(state, l)

            # Prepare past_key_values for actual forward
            pkv_in = []
            for l in range(num_layers):
                pkv_in.append(state.cf[l] if (is_check and layer_use_full[l]) else state.cp[l])
            pkv_cache = _ensure_cache(
                self.model,
                pkv_in,
                cache_position=abs_pos,
                logger=logger,
                step=step,
                tag="decode",
            )
            # Safety invariant: past lengths must match across layers for the chosen cache.
            past_lengths = [pkv_in[l][0].shape[2] for l in range(num_layers)]
            if len(set(past_lengths)) != 1:
                if logger is not None:
                    logger.log_event(
                        {
                            "step": step,
                            "error": "non_uniform_past_lengths",
                            "past_lengths": past_lengths,
                            "layer_use_full": layer_use_full,
                        }
                    )
                raise RuntimeError("Non-uniform past_key_values lengths across layers.")

            # Actual forward (with attention scores if needed)
            start_t = time.time()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            capture_actual = None
            if is_check and any(layer_use_full):
                capture_actual = QueryCapture(self.model, log_per_head=cfg.log_per_head_similarity)
                capture_actual.attach()
            with torch.inference_mode():
                out = _model_forward(
                    self.model,
                    input_ids=current_token,
                    past_key_values=pkv_cache,
                    use_cache=True,
                    output_attentions=not cfg.use_fast_attention,
                    return_dict=True,
                    position_ids=torch.tensor([[abs_pos]], device=self._device()),
                    cache_position=torch.tensor([abs_pos], device=self._device()),
                )
            if capture_actual is not None:
                capture_actual.detach()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latency_ms = (time.time() - start_t) * 1000.0

            past_out = _normalize_past_key_values(out.past_key_values)
            attn_out = out.attentions
            attn_recomp = None
            if cfg.use_fast_attention and is_check and any(layer_use_full):
                attn_recomp = self._recompute_attn_for_token(
                    current_token,
                    pkv_in,
                    abs_pos=abs_pos,
                    logger=logger,
                    step=step,
                    tag="refresh_recompute",
                )
            logits = out.logits[:, -1, :]

            # Pick next token
            entropy, margin = _entropy_and_margin(logits)
            entropies.append(entropy)
            margins.append(margin)
            next_token = int(torch.argmax(logits, dim=-1).item())

            refresh_any = is_check and any(layer_use_full)
            if refresh_any:
                refresh_latencies.append(latency_ms)
            else:
                non_refresh_latencies.append(latency_ms)

            # Update caches per layer
            refresh_layers = []
            refresh_diffs = {}
            num_kv_heads = getattr(self.model.config, "num_key_value_heads", None)
            for l in range(num_layers):
                present_k, present_v = past_out[l]

                if is_check and layer_use_full[l]:
                    # Full attention: update Cf, refresh Cp
                    state.cf[l] = (present_k, present_v)
                    refresh_layers.append(l)

                    # Update q_last_full from actual pass
                    if capture_actual is not None:
                        q_act = capture_actual.q_mean[l]
                        if q_act is not None:
                            state.q_last_full[l] = q_act.squeeze(0)
                        if cfg.log_per_head_similarity and state.q_last_full_per_head is not None:
                            qh_act = capture_actual.q_per_head[l]
                            if qh_act is not None:
                                state.q_last_full_per_head[l] = qh_act.squeeze(0)
                    elif q_probe is not None and q_probe[l] is not None:
                        state.q_last_full[l] = q_probe[l]
                        if cfg.log_per_head_similarity and state.q_last_full_per_head is not None:
                            if q_probe_heads is not None and q_probe_heads[l] is not None:
                                state.q_last_full_per_head[l] = q_probe_heads[l]

                    # Build new Cp from attention scores
                    if attn_out is None and attn_recomp is None:
                        raise RuntimeError(
                            "Attention scores missing; enable recompute_attn_scores or use eager attention."
                        )
                    # PDF p.4: refresh Cp using top-K tokens based on attention scores from full attention.
                    attn = attn_out[l] if attn_out is not None else attn_recomp[l]
                    scores = _aggregate_attention(attn, cfg.head_aggregation, num_kv_heads=num_kv_heads)
                    scores = _pool_scores(scores, cfg.pool_kernel_size, cfg.pool_padding)
                    if cfg.recompute_drop_self and scores.shape[-1] > 1:
                        scores = scores[..., :-1]
                    k = min(state.k, scores.shape[-1])
                    if k < 1:
                        raise RuntimeError("No tokens available for topK selection after drop-self.")
                    idx, sc = _select_topk(scores.squeeze(0), k=k, order=cfg.topk_order)
                    cf_k, cf_v = state.cf[l]
                    assert idx.max().item() < cf_k.shape[2]
                    cp_k = torch.index_select(cf_k, dim=2, index=idx)
                    cp_v = torch.index_select(cf_v, dim=2, index=idx)
                    assert cp_k.shape[2] == k

                    before_idx = state.cp_idx_abs[l]
                    after_idx = idx.tolist()
                    state.cp[l] = (cp_k, cp_v)
                    state.cp_idx_abs[l] = after_idx
                    state.cp_scores[l] = sc.tolist()
                    state.k = k
                    assert len(after_idx) == state.k
                    state.pos_full[l] = step + 1

                    # Refresh diff stats
                    before_set = set(before_idx)
                    after_set = set(after_idx)
                    added = sorted(list(after_set - before_set))
                    evicted = sorted(list(before_set - after_set))
                    overlap = len(before_set & after_set) / max(1, len(after_set))
                    overlap_ratios.append(overlap)

                    refresh_diffs[l] = {
                        "before_idx": compact_indices(before_idx, cfg.max_log_indices),
                        "after_idx": compact_indices(after_idx, cfg.max_log_indices),
                        "overlap_ratio": overlap,
                        "counts": {"evicted": len(evicted), "added": len(added)},
                        "evicted_pos_stats": position_stats(evicted),
                        "added_pos_stats": position_stats(added),
                    }

                    # Token-type summary if we can map positions
                    if logger is not None:
                        evicted_ids = [full_token_ids[i] for i in evicted if i < len(full_token_ids)]
                        added_ids = [full_token_ids[i] for i in added if i < len(full_token_ids)]
                        refresh_diffs[l]["evicted_token_types"] = token_type_summary(evicted_ids, self.tokenizer)
                        refresh_diffs[l]["added_token_types"] = token_type_summary(added_ids, self.tokenizer)

                else:
                    # Partial attention: update Cp in-place, keep Cf frozen
                    cp_k, cp_v = present_k, present_v
                    cp_len = cp_k.shape[2]
                    assert cp_len == len(state.cp_idx_abs[l]) + 1, "Cp length mismatch after append"

                    # Append new token to Cp index list
                    new_abs_idx = abs_pos
                    cp_idx = state.cp_idx_abs[l] + [new_abs_idx]
                    cp_scores = state.cp_scores[l] + [float("inf")]

                    if cfg.evict_policy == "front":
                        evict_idx = 0
                    else:
                        evict_idx = int(np.argmin(cp_scores))

                    cp_idx.pop(evict_idx)
                    cp_scores.pop(evict_idx)
                    cp_k = torch.cat([cp_k[:, :, :evict_idx, :], cp_k[:, :, evict_idx + 1 :, :]], dim=2)
                    cp_v = torch.cat([cp_v[:, :, :evict_idx, :], cp_v[:, :, evict_idx + 1 :, :]], dim=2)

                    assert cp_k.shape[2] == state.k, "Cp exceeds budget"
                    assert len(cp_idx) == state.k
                    state.cp[l] = (cp_k, cp_v)
                    state.cp_idx_abs[l] = cp_idx
                    state.cp_scores[l] = cp_scores

                    # Track pending tokens for Cf merge
                    new_k = present_k[:, :, -1:, :]
                    new_v = present_v[:, :, -1:, :]
                    self._append_pending(state, l, new_k, new_v, new_abs_idx)
                    if state.pending_k[l] is not None and state.pending_k[l].shape[2] > state.k:
                        # Warn via logger (documented ambiguity)
                        if logger is not None:
                            logger.log_event(
                                {
                                    "step": step,
                                    "warning": "pending_tokens_exceed_k",
                                    "layer": l,
                                    "pending_len": int(state.pending_k[l].shape[2]),
                                    "k": int(state.k),
                                }
                            )

            # Logging
            if logger is not None:
                logger.log_token(
                    {
                        "step": step,
                        "token_id": next_token,
                        "token_str": self.tokenizer.decode([next_token], clean_up_tokenization_spaces=False)
                        if cfg.log_decoded_tokens
                        else None,
                        "entropy": entropy,
                        "logit_margin": margin,
                        "is_check_step": bool(is_check),
                        "refresh_fired": bool(refresh_any),
                    }
                )

                if is_check or refresh_any:
                    overall_sim = None
                    if per_layer_sim is not None:
                        sims = [s for s in per_layer_sim if s is not None]
                        overall_sim = float(np.mean(sims)) if sims else None

                    logger.log_event(
                        {
                            "step": step,
                            "seq_len": prompt_len + len(generated_ids) + 1,
                            "is_check_step": bool(is_check),
                            "refresh_fired": bool(refresh_any),
                            "trigger": {
                                "mode": cfg.trigger_mode,
                                "qc_stride": cfg.qc_stride,
                                "similarity_threshold": cfg.similarity_threshold,
                                "overall_score": overall_sim,
                                "overall_agg": "mean",
                                "min_similarity": min_similarity,
                                "decision_agg": decision_agg,
                                "per_layer_similarity": per_layer_sim,
                                "per_head_similarity": per_head_sim,
                                "head_aggregation": cfg.head_aggregation,
                                "entropy": probe_entropy,
                                "entropy_z": probe_entropy_z if cfg.entropy_use_zscore else None,
                                "entropy_fired": entropy_fired,
                                "use_full": use_full_decision,
                                "uniform_enforced": uniform_enforced,
                            },
                            "refresh": {
                                "layers": refresh_layers,
                                "layer_diffs": refresh_diffs,
                                "layer_use_full": layer_use_full if is_check else None,
                            },
                            "timing_ms": {
                                "step_latency": latency_ms,
                                "is_refresh": bool(refresh_any),
                            },
                        }
                    )

            # Update generated sequence
            generated_ids.append(next_token)
            full_token_ids.append(next_token)
            current_token = torch.tensor([[next_token]], device=self._device())
            abs_pos += 1

        return {
            "generated_ids": generated_ids,
            "generated_text": self.tokenizer.decode(generated_ids, clean_up_tokenization_spaces=False),
            "entropies": entropies,
            "margins": margins,
            "refresh_latencies": refresh_latencies,
            "non_refresh_latencies": non_refresh_latencies,
            "overlap_ratios": overlap_ratios,
        }
