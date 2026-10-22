# 0002 — nano-vllm Sampler: exponential race vs `torch.multinomial`

- **Upstream**: [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) (pinned by the `upstream/nano-vllm` submodule)
- **Component**: `nanovllm/layers/sampler.py` — samples `argmax(p / E)` with `E ~ Exp(1)` (equivalent to Gumbel-max) instead of `torch.multinomial`
- **Hardware**: RTX 4060 Laptop 8 GB, torch 2.14.1+cu126, memory capped at 60% by `lab/gpu_guard.py`
- **Run**: `.venv\Scripts\python.exe experiments\0002-nanovllm-sampler\run.py`

## Question

Is the exponential-race sampler exactly as correct as multinomial sampling, and is it actually faster at Qwen3 vocabulary size (151,936)?

## Setup

- Correctness: vocab 32, temperature 0.7, ~2M samples per method; compare empirical frequencies with `softmax(logits / T)`.
- Throughput: bf16 logits, batch 1 / 16 / 64 / 256, 10 warmup + 50 timed calls with CUDA events; "extra peak MiB" = peak allocation above the input logits.
- The upstream method is wrapped in `@torch.compile`; the eager body is called via `_torchdynamo_orig_callable` because Triton is not available on Windows.

## Findings

1. **Distributionally exact.** chi²/dof = 0.71 (nano-vllm) vs 0.82 (multinomial), total variation ≈ 0.001 for both — consistent with sampling noise.
2. **Faster only at small batch (eager).** Batch 1: 0.27 ms vs 0.42 ms (1.54×). Batch 16: 1.17×. Batch 64: 1.06×. Batch 256: 0.96× (slightly slower) — at large batch both are bandwidth-bound over a 151,936-wide row.
3. **About 1.5× more temporary memory.** `torch.empty_like(probs).exponential_()` allocates a second fp32 vocab-sized tensor: 445 MiB vs 297 MiB extra at batch 256. Under `torch.compile` this should fuse away; that is left to a follow-up experiment with `triton-windows`.

Full numbers: [results.md](results.md) / [results.json](results.json).
