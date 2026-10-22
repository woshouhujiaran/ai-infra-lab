# 0001 — nano-vllm prefix cache hit rate

- **Upstream**: [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) (pinned by the `upstream/nano-vllm` submodule)
- **Component**: `nanovllm/engine/block_manager.py` (paged KV cache + hash-based prefix caching), driven by the real `Scheduler`
- **Hardware**: CPU only — no model, no GPU; token ids are synthetic
- **Run**: `.venv\Scripts\python.exe experiments\0001-nanovllm-prefix-cache\run.py`

## Question

How much prefill compute does nano-vllm's prefix cache save for requests that share a system prompt, and what breaks it?

## Setup

64 requests = shared prefix (0 / 256 / 1024 / 2048 tokens) + 128 unique tokens, 32 output tokens each, `max_num_batched_tokens=16384`.

- `burst`: all requests arrive before the first step.
- `staggered`: one new request per engine step.
- `memory_pressure`: block_size 256, 1024-token prefix, 256 output tokens, only 24 / 48 / 96 KV blocks.

`prefix_hit_rate = 1 - prefill_tokens_computed / prompt_tokens`. After each run the script asserts every block is freed and every `ref_count` is 0.

## Findings

1. **Burst arrival wastes the cache.** Block hashes are registered in `Scheduler.postprocess -> hash_blocks`, i.e. only after a prefill step finishes. Every request packed into the first 16K-token prefill batch recomputes the shared prefix. With a 2048-token prefix the hit rate is 83.8% for burst vs 92.7% for staggered arrival.
2. **The last block is never reused.** `can_allocate` only probes `num_blocks - 1` blocks, so the hit rate stays below the theoretical `prefix / prompt` ratio.
3. **Block size does not change the hit rate here** (prefix lengths are multiples of 256), but sharing cuts peak KV usage: with staggered arrival and a 2048-token prefix, block_size=16 peaks at 768 blocks (12,288 tokens) vs 1,536 blocks (24,576 tokens) for burst.
4. **Memory pressure causes preemption and recompute.** With 24 / 48 / 96 blocks there were 42 / 26 / 20 preemptions, and the hit rate is non-monotonic (72.1% / 57.6% / 62.5%). A preempted sequence frees its blocks, and they can be reallocated before it is rescheduled.

Note: upstream `Config` requires `kvcache_block_size % 256 == 0`; block sizes 16 and 64 only exercise the BlockManager in isolation.

Full numbers: [results.md](results.md) / [results.json](results.json).
