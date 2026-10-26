# 0002 — nano-vllm Sampler: exponential race vs `torch.multinomial`, eager vs `torch.compile`

- **Upstream**: [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) (pinned by the `upstream/nano-vllm` submodule)
- **Component**: `nanovllm/layers/sampler.py`, which samples `argmax(p / E)` with `E ~ Exp(1)` (equivalent to Gumbel-max) instead of calling `torch.multinomial`. Upstream wraps `forward` in `@torch.compile`.
- **Hardware**: RTX 4060 Laptop 8 GB, torch 2.14.1+cu126, triton-windows 3.8.0, memory capped at 60% by `lab/gpu_guard.py`
- **Run**: `.venv\Scripts\python.exe experiments\0002-nanovllm-sampler\run.py` (about 30 s)

## Question

Is the exponential-race sampler as correct as multinomial sampling? Is it faster at the Qwen3 vocabulary size (151,936)? And how much of the benefit depends on the `@torch.compile` that upstream relies on?

## Setup

- Three variants: the upstream body run eagerly (called through `_torchdynamo_orig_callable`), the upstream compiled `Sampler` (Inductor + Triton via `triton-windows`), and a `softmax` + `torch.multinomial` baseline.
- Correctness: vocab 32, temperature 0.7, about 2M samples per method. Empirical frequencies are compared with `softmax(logits / T)`.
- Throughput: bf16 logits, batch 1 / 16 / 64 / 256, 10 warmup calls (which also absorb recompiles) and 50 timed calls measured with CUDA events. "Extra peak MiB" is the peak allocation above the input logits.

## Findings

1. **All three variants are distributionally exact.** chi²/dof is 0.68 (eager), 0.67 (compiled) and 0.84 (multinomial), and total variation is at most 0.001. Both are consistent with sampling noise.
2. **Eager mode is faster only at small batch sizes.** Eager vs multinomial: 2.0× at batch 1, 1.15× at 16, 1.08× at 64, and 0.95× (slightly slower) at 256.
3. **`torch.compile` is what makes the sampler fast at large batch sizes.** At batch 256 the compiled sampler takes 4.52 ms, against 9.39 ms eager (2.1×) and 8.96 ms for multinomial (2.0×). At batch 64 it is 1.46× faster than eager.
4. **Compiling cuts temporary memory to a single fp32 buffer.** At batch 256 the eager path allocates three vocab-sized fp32 tensors (`.float()`, `softmax`, `empty_like().exponential_()`), 445 MiB in total. Multinomial needs 297 MiB. The compiled kernel needs only 148 MiB, which is exactly one `[256, 151936]` fp32 buffer.
5. **Compiling costs time at batch 1.** Compiled takes 0.30 ms vs 0.22 ms eager. At this size the kernel is launch-bound, and Dynamo guard checks plus the generated kernels' launch overhead outweigh the fusion. The first compile took 2.9 s; later shapes trigger dynamic-shape recompiles that the warmup absorbs.

Takeaway: on Windows without Triton, `@torch.compile` cannot run, and the eager fallback loses the large-batch speedup while using 1.5× the memory of `torch.multinomial`. The sampler's design pays off only when it is compiled.

Full numbers: [results.md](results.md) / [results.json](results.json).
