# 0008 — StreamingLLM attention-sink KV eviction on Pythia-160M

- **Upstream**: [mit-han-lab/streaming-llm](https://github.com/mit-han-lab/streaming-llm) (submodule `upstream/streaming-llm`, commit `2e50426`)
- **Component**: `streaming_llm/kv_cache.py` — `StartRecentKVCache` (keep the first `start_size` tokens + the last `recent_size` tokens), used unmodified from the submodule. Its pos-shift attention (`streaming_llm/pos_shift/modify_gpt_neox.py`) patches the transformers 4.33 GPT-NeoX attention API, so `run.py` re-implements it (keys cached un-rotated, RoPE applied by cache slot on every step) inside a small fp32 GPT-NeoX forward.
- **Model / data**: [EleutherAI/pythia-160m](https://huggingface.co/EleutherAI/pythia-160m) (375 MB, trained context 2048), first 4 books of [emozilla/pg19-test](https://huggingface.co/datasets/emozilla/pg19-test) (25 MB), 4096 scored tokens per book
- **Hardware**: RTX 4060 Laptop 8 GB, torch 2.14.1+cu126, fp32 with TF32 off, memory capped at 60% by `lab/gpu_guard.py`
- **Run**: `.venv\Scripts\python.exe experiments\0008-streamingllm-attention-sink\run.py` (about 12 min; peak GPU memory 3.1 GiB). The first run downloads the model and data through the Hugging Face hub (set `HTTPS_PROXY` if needed); later runs work with `HF_HUB_OFFLINE=1`.

## Question

When a small RoPE model streams past its training context with a fixed KV budget, how much do the first few "attention sink" tokens matter? Specifically: how many sinks are needed, how important StreamingLLM's position shift is, and how close sinks plus recent tokens get to re-encoding the window from scratch. Also, what does upstream's eviction cost per token?

## Setup

- Upstream's `eval_long_ppl.py` loop: feed one token, score the next token, then call `StartRecentKVCache` on the cache. 4 books are batched, so each step is batch 4. The cache holds C entries after eviction, so each query attends to at most C+1 tokens.
- Policies, for C ∈ {256, 512, 1024}:
  - `dense`: no eviction (the cache grows to 4096).
  - `window`: `start_size=0` (pure sliding window).
  - `sinkS shift`: `start_size=S` ∈ {1, 4, 8} with position shift (StreamingLLM).
  - `sink4 no-shift`: 4 sinks, but keys keep their original text positions.
  - `recompute`: re-encode the last C+1 tokens from scratch for every prediction (sliding window with re-computation, the quality reference for a C-token budget). It is too slow to run on every token, so it is scored on every 16th token. The "subsample" column gives every policy's perplexity on those same tokens.
- Checks: the forward matches `transformers`' `GPTNeoXForCausalLM` to 0.018 max abs logit diff, where logits reach 848. Token-by-token decoding with a cache matches the parallel forward to 0.008. Window attention with and without position shift agrees to 0.013, as expected because RoPE scores depend only on relative position.
- Attention probe: one dense 1024-token forward. Attention mass on the first tokens is averaged over queries t ≥ 256 and over heads.
- Eviction cost: CUDA-event time of the `StartRecentKVCache` call, on steps that actually evict.

## Findings

1. **Pythia-160M has strong attention sinks.** In layers 2–11, 23–69% of each query's attention goes to the first 4 tokens, and up to 18% goes to token 0 alone (a uniform distribution would put 0.18% on token 0). Layers 0–1 attend locally instead: 28% of their mass goes to the last 4 tokens.
2. **Pure window attention breaks as soon as eviction starts.** With C=512, perplexity in positions 512–1023 jumps from 37.4 (dense) to 62.8. It never recovers: overall perplexity is 40–50 across the three budgets. Quadrupling C from 256 to 1024 only brings it from 50.0 to 40.4.
3. **Keeping sinks with position shift fixes it at every budget.** Overall perplexity is 31.9 / 31.1 / 30.6 for C = 256 / 512 / 1024, and it stays flat at 27–34 across all buckets up to 4096 tokens. Dense attention is 29 within the 2048-token training context and blows up to 101, 2353 and 6229 in the last three buckets (117 overall).
4. **One sink gets most of the gain, and more than 4 adds nothing.** With 1 sink, perplexity is 1.2–1.4% worse than with 4. With 8 sinks it is the same as with 4 (within 0.06).
5. **Without position shift, sinks fail exactly like dense attention.** `sink4 no-shift` matches the shifted version until position 2560 (31.1 vs 30.8 in positions 2048–2559). It then reaches 583–697 and 2373–2869 in the last two buckets, the same failure point as dense. Once the query-to-sink distance exceeds the 2048-token training range, RoPE goes out of distribution. So dense attention also fails because of position, not because the cache is too large. The position shift, which keeps every distance below C, is what makes streaming work.
6. **StreamingLLM matches re-computation at roughly 1/5–1/12 of its cost.** On the every-16th-token subsample, `sink4 shift` scores 29.51 / 28.87 / 28.58 and `recompute` scores 29.61 / 28.78 / 28.64. Streaming costs 6–13 ms/token, while re-computation costs 34 / 69 / 151 ms/token (batch 4). Among the streaming policies, the ranking (sink with shift < window < sink without shift < dense) is the same on each of the 4 books.
7. **Upstream's eviction copies the whole cache on every token.** `StartRecentKVCache` uses `torch.cat` to rebuild every layer's K and V. On every evicting step this takes 0.75 / 1.44 / 2.84 ms for C = 256 / 512 / 1024, which is linear in C. At C=1024 this is about 600 MB of reads plus writes per step, around 210 GB/s, close to this GPU's memory bandwidth. That is 12–23% of the per-token time, while a ring buffer (or a fixed sink region plus an index-based write) would cost nothing. KV memory stays fixed at 18 / 36 / 72 MiB per sequence (fp32), compared with 288 MiB for dense at 4096 tokens.

Caveats: Pythia was trained on the Pile, which contains PG-19 books, so absolute perplexities may be optimistic. Only the comparisons between policies matter. The ms/token figures come from a launch-bound Python loop and vary by about ±25% between policies with the same C. Perplexities are deterministic: an earlier aborted run produced identical values.

Takeaway: for a RoPE model streaming past its context, keep at least 1 sink token (4 is the safe default), and always re-position the cache. Sinks without position shift extend nothing. With both, a 256-entry cache matches a 1024-token re-computation window to within 3% perplexity, and the remaining avoidable cost is upstream's full-cache copy on every eviction.

Full numbers: [results.md](results.md) / [results.json](results.json).
