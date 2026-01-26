# RefreshKV Reproduction (Colab-friendly)

## Implementation Notes (from RefreshKV.pdf)

Below are paper-faithful definitions and rules extracted from `RefreshKV.pdf`. Page numbers refer to the PDF in this repo.

1) **Trigger definition (query-similarity trigger):**
   - At every **S-th decode step** (QC stride), for **each layer** `l`, compute **cosine similarity** between the **query vector of the current input token** (averaged across all query heads in layer `l`) and the **averaged query vector from the most recent full-attention step for that layer**. If similarity **> s**, decode with the **partial cache `C_p`** for that layer; otherwise decode with the **full cache `C_f`** for that layer. (PDF p.3)

2) **QC cadence / check schedule:**
   - The query-comparison (QC) similarity check is performed **only every S decode steps** (QC stride) to reduce overhead. Non-check steps decode with the partial cache. (PDF p.3)

3) **`C_p` construction & update rule (topK + local tail, budgets):**
   - **Initialization (prefill):** after full-attention prefill, compute **attention scores for the last input token** over past tokens **for all layers**, apply **max pooling over attention scores** to preserve local completeness, then **select top-K tokens** to initialize the partial cache `C_p` as `reverse(arg top-k_x in Cf (max_pool(a_L, kernel_size)))`. (PDF p.3-4)
   - **Partial-attention step:** generate using `C_p`, **append the new token's KV** to `C_p`, then **evict the token with the lowest attention score** from `C_p` to keep size fixed at **K**. (PDF p.4)
   - **Full-attention step:** **update `C_f` with the new tokens decoded since the last full step**, generate using full attention over `C_f` and obtain attention scores `a_L`, then **refresh `C_p` with top-K tokens from `C_f`** based on pooled attention scores. (PDF p.4)
   - The **partial cache size** is **K tokens**, described as **"local + topK tokens from a_L"** in the paper's figure/legend. (PDF p.4)
   - **Kernel size** for max pooling over attention scores: **7** (following SnapKV). (PDF p.11)
   - **Experimental budget choices:** default **K = 1/8 of input length**, except **NovelSumm** uses **K = 4096 (~1/25 L)**. (PDF p.5)

4) **Definition of "full attention":**
   - A **full-attention step** is one where the model **attends over the full KV cache `C_f`**, produces the next token, and yields attention scores `a_L` for updating `C_p`. (PDF p.4)

Notes on ambiguities / implementation choices:
- The pseudocode uses `reverse(arg top-k ...)` before eviction by `C_p = C_p[1:]`. We interpret this as **ordering the top-K by ascending attention score** so that the first element is the lowest-score token to evict; this is exposed as a config option. (PDF p.4)
- The paper does not explicitly specify behavior when **the number of locally appended tokens exceeds K** before a refresh. We default to **evicting lowest-score tokens first**, and expose fallback policies in config with a warning if `pending > K`. (PDF p.4)

(Other sections of this README will be filled in after implementation.)

## Quickstart (Colab)

```bash
pip install -r requirements.txt
python run_baseline.py --num_examples 10 --max_new_tokens 128
```

Entropy and hybrid ablations:

```bash
python run_ablation.py --mode entropy --num_examples 10 --max_new_tokens 128
python run_ablation.py --mode hybrid --num_examples 10 --max_new_tokens 128
```

Logs are written to `logs/<run_id>/` as JSONL files (`events.jsonl`, `tokens.jsonl`).

## Repo Layout

- `src/refreshkv.py` - core RefreshKV implementation (cache updates, triggers, logging hooks)
- `src/logging_utils.py` - JSONL writers + helper utilities
- `run_baseline.py` - baseline run on GSM8K subset + summary stats
- `run_ablation.py` - entropy / hybrid trigger ablations
- `configs/default.yaml` - default config (QC stride, thresholds, cache budget, etc.)
- `requirements.txt` - minimal dependencies

## Configuration Notes

Key defaults follow the paper's settings for Qwen2-7B:
- `model_name: Qwen/Qwen2-7B-Instruct`
- `qc_stride: 10`
- `similarity_threshold: 0.95`
- `partial_cache_ratio: 0.125`
- `pool_kernel_size: 7`

The model is loaded with `attn_implementation: eager` so attention scores are available for top-K selection (needed to match the PDF). If you switch to Flash Attention, you must ensure attention scores are still obtainable (the paper recomputes them; see Appendix A.1, PDF p.11).

## Logging

Two JSONL logs are produced per run:
- `events.jsonl`: one record per QC step (and refresh), including trigger stats, per-layer similarity scores, refresh diffs, overlap ratio, token-type summaries, and latency.
- `tokens.jsonl`: one record per generated token including entropy, logit margin, `is_check_step`, and `refresh_fired`.

## Assumptions / Implementation Choices (Documented)

These are exposed as config options and called out here explicitly:

1) **TopK ordering for eviction**  
   The pseudocode uses `reverse(arg top-k ...)` and then evicts `C_p[1:]` (PDF p.4). We interpret this as **ordering top-K by ascending attention score** so the lowest score is evicted from the front. Config: `topk_order`.

2) **Pending tokens vs. Cp tail for Cf merge**  
   The paper appends tokens from the Cp tail to Cf at a full step (PDF p.4). To avoid silent KV loss when pending > K, we track **pending KV explicitly** and merge from `pending_kv` by default (Invariant 2). Config: `pending_merge_policy` (`pending_kv` vs `cp_tail`).

3) **QC probe forward**  
   Query similarity is computed via a **probe forward** using the partial cache at QC steps, then the actual forward uses the selected caches. This keeps caches immutable during QC checks (Invariant 5) and matches the paper's "QC stride" scheduling (PDF p.3). Config: `qc_stride`, `layerwise_qc`.

4) **First generated token**  
   We use **prefill logits** to generate the first token (standard HF behavior). This means token #1 is produced with full attention; scheduling starts at token #2. This is configurable only by code changes (see `first_token_from_prefill` note in `src/refreshkv.py`).

5) **GQA attention aggregation**  
   For GQA models, attention scores are aggregated across heads by **max** (PDF footnote on p.3 and Appendix A.3, p.12-13). Config: `head_aggregation`.

## GSM8K Fallback

If GSM8K download fails, the runners use a built-in list of 5 arithmetic prompts to smoke-test the pipeline and logging.

## Reproducibility

The runs set a fixed seed and use greedy decoding (temperature=0) with `max_new_tokens` controlled via CLI.

## Environment

Tested for Google Colab, single A100, Python 3.10+ with PyTorch + Hugging Face Transformers.
