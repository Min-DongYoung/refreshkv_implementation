import argparse
import os
import random
from dataclasses import asdict

import numpy as np
import torch
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


def build_prompt(text: str, tokenizer) -> str:
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": text}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=str, default="cwe_4k,qa_2_4k,vt_4k,niah_multikey_1_4k")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--min_tokens", type=int, default=4096)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--dataset", type=str, default="rbiswasfc/ruler")
    args = parser.parse_args()

    cfg = RefreshKVConfig()
    cfg.trigger_mode = "baseline"
    cfg.qc_stride = 10
    cfg.similarity_threshold = 0.95
    cfg.partial_cache_ratio = 0.125
    cfg.pool_kernel_size = 7
    cfg.head_aggregation = "max"
    cfg.attn_implementation = "eager"
    cfg.max_new_tokens = args.max_new_tokens
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed(cfg.seed)
    force_eager_attention()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=getattr(torch, cfg.dtype),
        attn_implementation=cfg.attn_implementation,
        device_map="auto" if cfg.device == "cuda" else None,
    )
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    assert_eager_attention(model)

    logger = RunLogger(config=asdict(cfg))
    generator = RefreshKVGenerator(model, tokenizer, cfg)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    overall = {"tokens": 0, "refresh": 0, "overlap_sum": 0.0, "overlap_n": 0}

    for task in tasks:
        ds = load_dataset(args.dataset, task, split="validation")
        count = 0
        task_stats = {"tokens": 0, "refresh": 0, "overlap_sum": 0.0, "overlap_n": 0}
        for ex in ds:
            if count >= args.num_samples:
                break
            text = ex.get("input", "")
            if not text:
                continue
            prompt = build_prompt(text, tokenizer)
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
            tok_len = int(input_ids.shape[1])
            if tok_len < args.min_tokens or tok_len > args.max_tokens:
                continue
            out = generator.generate(input_ids, max_new_tokens=cfg.max_new_tokens, logger=logger)

            task_stats["tokens"] += len(out["generated_ids"])
            task_stats["refresh"] += len(out["refresh_latencies"])
            if out["overlap_ratios"]:
                task_stats["overlap_sum"] += sum(out["overlap_ratios"])
                task_stats["overlap_n"] += len(out["overlap_ratios"])
            count += 1

        if count == 0:
            print(f"[{task}] WARNING: no samples met token-length constraints.")
            continue

        overall["tokens"] += task_stats["tokens"]
        overall["refresh"] += task_stats["refresh"]
        overall["overlap_sum"] += task_stats["overlap_sum"]
        overall["overlap_n"] += task_stats["overlap_n"]

        freq = task_stats["refresh"] / max(1, task_stats["tokens"])
        overlap = task_stats["overlap_sum"] / max(1, task_stats["overlap_n"])
        print(f"[{task}] samples={count} refresh_freq={freq:.4f} overlap={overlap:.4f}")

    if overall["tokens"] > 0:
        freq = overall["refresh"] / overall["tokens"]
        overlap = overall["overlap_sum"] / max(1, overall["overlap_n"])
        print(f"[overall] refresh_freq={freq:.4f} overlap={overlap:.4f}")
        print(f"logs: {logger.run_dir}")
    logger.close()


if __name__ == "__main__":
    main()
