# 0033 — Liger-Kernel Dynamic Tanh (`LigerDyTFunction`)

- **Upstream**: [linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel) (submodule `upstream/Liger-Kernel`), imported unmodified from `src/`.
- **Component**: `src/liger_kernel/ops/dyt.py` (`LigerDyTFunction` / `liger_dyt_fwd` / `liger_dyt_bwd`) and `transformers/dyt.py` (`LigerDyT`). Elementwise `y = γ · tanh(α · x) + β` (β optional) proposed as a LayerNorm replacement. Forward and backward are autotuned Triton kernels (`BLOCK_N ∈ {1024,2048,4096}`); backward partitions rows across SMs and reduces fp32 partial `dα` / `dγ` / `dβ`. Distinct from [0031](../0031-liger-layer-norm) (LayerNorm) and [0022](../0022-liger-rmsnorm) (RMSNorm).
- **Hardware**: RTX 4060 Laptop 8 GB, torch 2.14.1+cu126, triton-windows, memory capped at 60% by `lab/gpu_guard.py`. Peak allocated ~1.7 GB.
- **Run**: `.venv\Scripts\python.exe experiments\0033-liger-dyt\run.py` (~5 min; first call per new `H` pays autotune).

## Question

Is Liger's DyT a usable drop-in LayerNorm replacement on an Ada laptop GPU — and what do the autotune / `dα` buffer choices cost?

## Setup

- **Numerics**: T = 2048, H = 4096; fp32 / fp16 / bf16; α ∈ {0.5, 1, 2}; with and without β. Relative L2 of Y / dX / dα / dγ / dβ vs fp64 textbook DyT and vs eager / `torch.compile`.
- **Edges**: `|x|·1e3` / `1e6`, α = 0 / 1e-8 / 100 / 1e6, NaN / ±Inf cells, all-zero / all-one.
- **Autotune / DA**: cold vs warm latency for H ∈ {512…8192}; DA buffer shape `(SMs, cdiv(H,512))` vs winning `BLOCK_N`; 8× bitwise determinism of grads; `β=None` path.
- **Layouts**: row-strided / transposed / 3D; Module vs Function.
- **Memory / latency / bandwidth**: bf16 at GPT-2 / BERT / Qwen3-0.6B / Llama-3.2-1B / Llama-3-8B hidden sizes vs eager DyT / `F.layer_norm` / `torch.compile`.

## Findings

1. **Numerics sit on the dtype floor; Triton fp32 math beats half-precision eager.**
   - bf16 Y ≈ 1.66e-3 vs fp64 (same floor as other Liger ops); eager DyT is 1.9–2.3e-3 because it runs `tanh` in bf16. fp16 ≈ 2.1e-4; fp32 ≈ 3–6e-8. Without β, fp32 Y is bit-identical to eager. `dbeta` matches eager exactly in bf16; `dalpha` stays within ~1% of eager on normal inputs.

2. **Saturated / non-finite inputs match eager where the math is defined; `dα` is NaN on ±Inf (both paths).**
   - `|x|·1e6` and `α=1e6` forward outputs match eager (tanh saturated). Relative `dX`/`dα` comparisons explode when both sides are near-zero under saturation — not a silent divergence of Y. A single ±Inf cell leaves Y finite (tanh(±Inf)=±1) but makes `dα` NaN on **both** Liger and eager; NaN inputs propagate. `α=0` is exact (Y = β).

3. **Per-shape autotune is expensive; the DA buffer wastes most slots.**
   - First call at a new H costs **6–7 s** (~10⁴–10⁵× a warm forward); subsequent calls at that H are 0.04–0.58 ms. Winning `BLOCK_N` is 1024–2048, so `program_id(0)` uses only 25–50% of the `cdiv(H,512)` DA columns (50–75% unused) — harmless because of `zeros` + `reset_to_zero=["DA"]`, but the sizing is a leftover from a finer tile. Gradients are bit-deterministic across 8 repeats; `β=None` returns `dbeta=None` correctly.

4. **`@ensure_contiguous` is safe; Module ≡ Function.**
   - Row-strided x leaves the neighbouring canary untouched; transposed and 3D match contiguous eager within ~2e-3 / ~3e-3 on Y / dX (bf16 floor). `LigerDyT` Module matches Function bit-exactly.

5. **Forward runs at copy bandwidth and ties `F.layer_norm`; 3–4× faster than eager DyT.**
   - After warm-up, Liger hits 176–237 GB/s for H ≥ 512 (eager DyT 56–88 GB/s). At Llama-3-8B × 16k tokens: Liger 1.18 ms vs eager 4.64 ms vs `F.layer_norm` 1.18 ms. End-to-end fwd+bwd stays within ~1× of LayerNorm across the LLM shapes tested. Peak held ~1.7 GB.

Takeaway: Liger DyT is floor-accurate (strictly better than half-precision eager thanks to fp32 `tanh` in Triton), layout-safe, and as fast as cuDNN LayerNorm once warmed — a viable LN replacement on this GPU **if** you can absorb a multi-second autotune tax on every new hidden size (or cache configs across runs). Watch ±Inf → NaN `dα` (same as eager) and treat relative grad checks under deep tanh saturation as noise.

Full numbers: [results.md](results.md) / [results.json](results.json).
