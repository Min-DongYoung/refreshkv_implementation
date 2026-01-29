import argparse
import os
import random
import re
from dataclasses import asdict

import numpy as np
import torch
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.logging_utils import RunLogger
from src.refreshkv import RefreshKVConfig, RefreshKVGenerator


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def force_eager_attention():
    if torch.cuda.is_available() and hasattr(torch.backends, "cuda"):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)


def assert_eager_attention(model):
    attn_impl = getattr(model.config, "attn_implementation", None)
    if attn_impl is None:
        attn_impl = getattr(model.config, "_attn_implementation", None)
    print(f"Attention implementation: {attn_impl}")
    if attn_impl != "eager":
        raise RuntimeError(
            f"Expected eager attention, got {attn_impl}. Disable SDPA/FlashAttention and set attn_implementation='eager'."
        )


def load_config(path: str) -> RefreshKVConfig:
    if not os.path.exists(path):
        return RefreshKVConfig()
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return RefreshKVConfig(**data)


def build_prompt(question: str, tokenizer) -> str:
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": question}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"Question: {question}\nAnswer:"


def extract_final_number(text: str) -> str:
    nums = re.findall(r"-?\\d+(?:\\.\\d+)?", text.replace(",", ""))
    return nums[-1] if nums else ""


def load_gsm8k(n: int, split: str = "test"):
    try:
        ds = load_dataset("gsm8k", "main", split=split)
        return [{"question": ex["question"], "answer": ex["answer"]} for ex in ds.select(range(n))]
    except Exception:
        fallback = [
            {"question": "If you have 3 apples and buy 4 more, how many apples do you have?", "answer": "#### 7"},
            {"question": "A car travels 10 miles, then 15 miles. Total distance?", "answer": "#### 25"},
            {"question": "What is 12 + 30?", "answer": "#### 42"},
            {"question": "If 9 candies are shared among 3 kids equally, how many each?", "answer": "#### 3"},
            {"question": "What is 7 * 8?", "answer": "#### 56"},
        ]
        return fallback[:n]


def parse_gsm8k_answer(answer: str) -> str:
    if "####" in answer:
        return answer.split("####")[-1].strip().replace(",", "")
    return extract_final_number(answer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--num_examples", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--mode", type=str, default="entropy", choices=["entropy", "hybrid"])
    parser.add_argument("--entropy_threshold", type=float, default=None)
    parser.add_argument("--entropy_use_zscore", action="store_true")
    parser.add_argument("--entropy_zscore_k", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.trigger_mode = args.mode
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.max_new_tokens is not None:
        cfg.max_new_tokens = args.max_new_tokens
    if args.entropy_threshold is not None:
        cfg.entropy_threshold = args.entropy_threshold
    if args.entropy_zscore_k is not None:
        cfg.entropy_zscore_k = args.entropy_zscore_k
    if args.entropy_use_zscore:
        cfg.entropy_use_zscore = True

    set_seed(cfg.seed)
    if cfg.use_fast_attention and cfg.attn_implementation not in ("sdpa", "flash_attention_2"):
        print(
            f"WARNING: use_fast_attention=True but attn_implementation='{cfg.attn_implementation}'. "
            "Switching to 'sdpa'."
        )
        cfg.attn_implementation = "sdpa"
    if cfg.attn_implementation == "eager" and not cfg.use_fast_attention:
        force_eager_attention()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=getattr(torch, cfg.dtype),
        attn_implementation=cfg.attn_implementation,
        device_map="auto" if cfg.device == "cuda" else None,
    )
    model.eval()
    if cfg.attn_implementation == "eager" and not cfg.use_fast_attention:
        assert_eager_attention(model)
    else:
        attn_impl = getattr(model.config, "attn_implementation", None) or getattr(
            model.config, "_attn_implementation", None
        )
        print(f"Attention implementation: {attn_impl}")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger = RunLogger(config=asdict(cfg))
    generator = RefreshKVGenerator(model, tokenizer, cfg)

    data = load_gsm8k(args.num_examples, split=args.split)

    total_tokens = 0
    refresh_steps = 0
    overlap_sum = 0.0
    overlap_count = 0
    refresh_lat = []
    non_refresh_lat = []
    entropy_vals = []
    margin_vals = []
    correct = 0

    for ex in data:
        prompt = build_prompt(ex["question"], tokenizer)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        out = generator.generate(input_ids, max_new_tokens=cfg.max_new_tokens, logger=logger)

        total_tokens += len(out["generated_ids"])
        refresh_steps += len(out["refresh_latencies"])
        refresh_lat.extend(out["refresh_latencies"])
        non_refresh_lat.extend(out["non_refresh_latencies"])
        entropy_vals.extend(out["entropies"])
        margin_vals.extend(out["margins"])
        if out["overlap_ratios"]:
            overlap_sum += sum(out["overlap_ratios"])
            overlap_count += len(out["overlap_ratios"])

        pred = extract_final_number(out["generated_text"])
        gold = parse_gsm8k_answer(ex["answer"])
        if pred and gold and pred == gold:
            correct += 1

    acc = correct / max(1, len(data))
    refresh_freq = refresh_steps / max(1, total_tokens)
    avg_overlap = overlap_sum / max(1, overlap_count)
    mean_refresh_lat = float(np.mean(refresh_lat)) if refresh_lat else 0.0
    mean_non_refresh_lat = float(np.mean(non_refresh_lat)) if non_refresh_lat else 0.0
    entropy_mean = float(np.mean(entropy_vals)) if entropy_vals else 0.0
    entropy_std = float(np.std(entropy_vals)) if entropy_vals else 0.0
    margin_mean = float(np.mean(margin_vals)) if margin_vals else 0.0
    margin_std = float(np.std(margin_vals)) if margin_vals else 0.0

    print(f"RefreshKV Ablation Summary ({cfg.trigger_mode})")
    print(f"- examples: {len(data)}")
    print(f"- refresh frequency: {refresh_steps}/{total_tokens} = {refresh_freq:.4f}")
    print(f"- avg overlap ratio: {avg_overlap:.4f}")
    print(f"- mean refresh latency (ms): {mean_refresh_lat:.2f}")
    print(f"- mean non-refresh latency (ms): {mean_non_refresh_lat:.2f}")
    print(f"- GSM8K accuracy: {acc:.3f}")
    print(f"- entropy mean/std: {entropy_mean:.4f} / {entropy_std:.4f}")
    print(f"- margin mean/std: {margin_mean:.4f} / {margin_std:.4f}")
    print(f"- logs: {logger.run_dir}")
    if refresh_steps == 0:
        print("WARNING: no refresh events fired; consider lowering entropy threshold or qc_stride.")

    logger.close()


if __name__ == "__main__":
    main()
