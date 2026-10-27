# 0003 — nano-vllm RMSNorm: fused residual-add vs separate ops

- **Upstream**: [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) (pinned by the `upstream/nano-vllm` submodule)
- **Component**: `nanovllm/layers/layernorm.py` — `RMSNorm.add_rms_forward` (fused `x + residual` then RMS) vs add then `rms_forward`
- **Hardware**: RTX 4060 Laptop 8 GB, torch 2.14.1+cu126, triton-windows 3.8.0, memory capped at 60% by `lab/gpu_guard.py`
- **Run**: `.venv\Scripts\python.exe experiments\0003-nanovllm-rmsnorm\run.py` (about 50 s including compile)

## Question

nano-vllm fuses the residual add into RMSNorm and does the sum in fp32 before normalizing. Against a naive bf16 `x + residual` followed by the same RMS body: how much numerics do you lose, and does fusion actually win on latency (eager and `torch.compile`)?

## Setup

- Hidden size 4096 (Qwen3-8B class), bf16 activations, `eps=1e-6`, fixed seed.
- Reference: textbook `(x.float() + residual.float())` then RMS in fp32, cast back — same math as the fused body.
- Variants: fused / separate × eager (`_torchdynamo_orig_callable`) / compiled (Inductor + Triton via `triton-windows`).
- Shapes: token counts 1 / 16 / 256 / 2048 (decode through short prefill). Warmup 10, timed 50 calls with CUDA events.

## Findings

1. **Fused matches the fp32 reference bit-for-bit on the output.** Max abs / mean abs / max rel are all 0. Separate has max abs error 0.03125 and max relative ~0.014 — the sum is stored in bf16 before variance/`rsqrt`, so mantissa bits that fused keeps in fp32 are gone.
2. **The returned residual still matches.** On this CUDA build, bf16 `x + residual` round-trips the same as `(fp32 sum).to(bf16)`, so `residual_max_abs` is 0 for separate; only the *norm output* drifts.
3. **Eager fused is slower than separate at every size.** About 1.4–1.7×: the fused body pays for explicit `.float()` casts and a longer op chain without Inductor fusion.
4. **Compile flips the mid-size race.** At 16 / 256 tokens, fused compiled is 1.27× / 1.12× faster than separate compiled, and ~2.4× / 3.0× faster than fused eager. At 2048 tokens separate compiled edges ahead again (0.33 vs 0.38 ms) and uses less temp memory (32 vs 48 MiB).
5. **Compile cuts temporary memory.** Eager paths peak at ~80 MiB extra for 2048 tokens; fused compiled 48 MiB; separate compiled 32 MiB (one fewer live fp32 working buffer).

Takeaway: the fused path is there for **numerics** (exact fp32 residual-add before RMS), not for eager speed. Latency wins only show up under `torch.compile` at moderate token counts; at large prefill, separate compiled is slightly faster and leaner.

Full numbers: [results.md](results.md) / [results.json](results.json).
