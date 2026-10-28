# 0004 — nano-vllm RotaryEmbedding: cached cos/sin vs on-the-fly

- **Upstream**: [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) (pinned by the `upstream/nano-vllm` submodule)
- **Component**: `nanovllm/layers/rotary_embedding.py` — `RotaryEmbedding` (fp32 `cos_sin_cache` built at init, indexed by `positions`, applied via `apply_rotary_emb`)
- **Hardware**: RTX 4060 Laptop 8 GB, torch 2.14.1+cu126, triton-windows 3.8.0, memory capped at 60% by `lab/gpu_guard.py`
- **Run**: `.venv\Scripts\python.exe experiments\0004-nanovllm-rope\run.py` (about 90 s including a 41 s first compile)

## Question

nano-vllm precomputes a `max_position x head_dim` fp32 cos/sin table and gathers rows per token. How accurate is that table at long positions compared with an fp64 reference, does recomputing cos/sin on the fly change accuracy or speed, and does building the table in fp64 help?

## Setup

- Qwen3-0.6B attention shape: 16 q heads, 8 kv heads, `head_dim=128`, `rope_theta=1e6`, `max_position=40960`. bf16 q/k, fixed seed.
- Variants:
  - `cached_fp32`: upstream module as is (table built in fp32 on CPU at init).
  - `cached_fp64_built`: same module, table built in fp64 then stored as fp32 (same size, same forward).
  - `on_the_fly`: keeps only `inv_freq`; computes `positions * inv_freq`, `cos`, `sin` on the GPU every call.
  - Eager (`_torchdynamo_orig_callable`) and compiled (upstream's `@torch.compile` forward; `torch.compile` for on-the-fly) via Inductor + Triton.
- Reference: angles, cos/sin and the rotation all in fp64.
- Numerics: cos/sin table error over all 40960 positions; rotated q/k on 512 random tokens per position bucket. "Mismatch" = fraction of output elements that differ from the fp64 result rounded to bf16 (0 means the best possible bf16 answer).
- Throughput: 1 / 16 / 256 / 2048 / 8192 tokens at random positions, warmup 10, timed 100 calls with CUDA events.

## Findings

1. **The fp32 table drifts with position.** Max abs cos/sin error grows from 6.7e-5 (positions < 1k) to 3.2e-3 (positions 32k–40k): the fp32 angle `t * inv_freq` loses absolute precision as `t` grows. Building the table in fp64 keeps every position at 3e-8 (fp32 storage rounding only).
2. **On-the-fly is numerically identical to the cached table.** Same fp32 angle math, so the error per bucket matches `cached_fp32` exactly, eager or compiled. Recomputing does not buy accuracy.
3. **The drift shows up in the bf16 output.** Max / mean abs error is dominated by bf16 rounding (~0.0156 / ~0.0011 for every variant), but the share of elements that land on the wrong bf16 value rises from 0.12% (< 1k) to 0.72%, 2.4% and 3.6% (32k–40k) with the fp32 table. With the fp64-built table it stays at 0.002% in every bucket.
4. **Cached wins on speed once compiled.** Compiled cached is 3.3× faster than compiled on-the-fly for single-token decode (0.061 vs 0.199 ms) and 10–15% faster from 256 to 8192 tokens; at 16 tokens they tie. Compile gives about 2–5× over eager and cuts temp memory about 4.7× (8192 tokens: 48 vs 228 MiB).
5. **Eager timings below 256 tokens are launch-bound noise.** `cached_fp32_eager` and `cached_fp64_built_eager` run identical code yet differ by up to 1.9× at 1 / 16 tokens; at 2048+ tokens all eager variants are within 5%.
6. **The table is cheap.** 40960 x 128 fp32 = 20 MiB, shared by every layer through `get_rope`'s `lru_cache`.

Takeaway: keep the cache (it is faster than recomputing once compiled and only costs 20 MiB), but build it in fp64 — a one-line, init-time change that removes the long-context angle drift at zero runtime cost.

Full numbers: [results.md](results.md) / [results.json](results.json).
