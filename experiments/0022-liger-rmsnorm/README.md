# 0022 — Liger-Kernel RMSNorm (`LigerRMSNormFunction`)

- **Upstream**: [linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel) (submodule `upstream/Liger-Kernel`, commit `1ab779f`), imported unmodified from `src/`.
- **Component**: `src/liger_kernel/ops/rms_norm.py`. `LigerRMSNormFunction` runs a Triton RMSNorm with HF-matched casting modes (`llama` / `gemma` / `none`), an optional Gemma-style weight offset, a default **in-place backward that writes `dX` into the caller's `dY` storage**, and an automatic switch between a one-row-per-program kernel and a 16-row block kernel when `H ≤ 256` and `T ≥ 32768`.
- **Hardware**: RTX 4060 Laptop 8 GB, torch 2.14.1+cu126, triton-windows 3.8.0, memory capped at 60% by `lab/gpu_guard.py`. Peak allocated memory was 1.9 GB.
- **Run**: `.venv\Scripts\python.exe experiments\0022-liger-rmsnorm\run.py` (about 1 min).

## Question

Liger's RMSNorm is the drop-in behind most of its Llama / Gemma / Qwen patches. Do the three casting modes actually match HuggingFace and the fp64 floor? When does the default in-place `dY→dX` overwrite silently break training graphs (the reason Gemma2 patches set `in_place=False`)? Are the row and block kernels interchangeable? What do memory and bandwidth look like against eager HF, `F.rms_norm` and `torch.compile`?

## Setup

- **Aliasing** (bf16, T = 256, H = 1024): plain single RMSNorm; Gemma2-style `y2 = rms(rms(x) + residual)` with `residual.requires_grad`; an aux `0.5 * x.pow(2).sum()` built before the norm; `autograd.grad` twice with `retain_graph=True` using a **fresh clone** of `dY` each time, plus the same probes **reusing** the caller's `dY` tensor; leaf inputs. Each case runs with `in_place=True` and `False`. Gradients are compared with fp64 autograd on the same graph.
- **Numerics**: T = 2048, H = 4096; fp32 / fp16 / bf16; casting ∈ {llama, gemma, none}; offset ∈ {0, 1}. Relative L2 of Y / dX / dW against an fp64 textbook RMSNorm. Baselines: matching HF casting, `F.rms_norm` (offset folded into the weight), and `torch.compile` of the HF body (tensor-only closure; Python float args make Inductor try the CPU `cl` path on this machine).
- **Kernel path**: shapes that force the row kernel (`BLOCK_SIZE > 256` or `T < 32768` or `row_mode=True`) vs the block kernel (`H = 128`, `T = 32768`). Compare bit-identity and relative error against an explicit `row_mode=True` run.
- **Memory / latency**: bf16 at Qwen3-0.6B (H = 1024) and Llama-3.2-1B (H = 2048), T ∈ {2048, 8192, 16384}. Held memory after forward and extra peak through backward for Liger in-place / out-of-place, eager HF, `torch.compile`, `F.rms_norm`.
- **Bandwidth**: bf16, T = 4096, H from 512 to 11008 (incl. H = 2049, just past a power of two). Forward counted as 2·T·H·2 bytes, backward as 3·T·H·2; `do_bench` median of 3 rounds.

## Findings

1. **In-place backward is safe on the Gemma2 two-norm graph here, but reusing the caller's `dY` is a silent footgun.**
   - Plain, aux-before, leaf inputs and `rms(rms(x)+residual)` with a grad-requiring residual all match between `in_place=True` and `False` (bf16 dx / dw / dresidual within ~0.002–0.003 of fp64, i.e. at the dtype floor). Autograd delivers a private buffer into the first RMSNorm before the residual leaf accumulates, so the documented Gemma2 hazard did not show up in this graph.
   - With a **fresh `dY` clone** each time, a second `retain_graph` backward is bit-identical to the first for both in-place settings.
   - **Reusing the same `dY` tensor**: `in_place=True` overwrites it with `dX`, so the next call's upstream grad is wrong (rel error 2.0 vs a clean first call; 1.4 between the two reuse calls). `in_place=False` keeps the caller's tensor intact (0.0). This is why patches that may see the same grad buffer twice (or that hand `grad_outputs` to `autograd.grad` more than once) set `in_place=False`.

2. **`gemma` casting sits on the fp64 floor and matches HF; `llama` / `none` do not.**
   - In fp16 / bf16, `casting_mode="gemma"` (and offset 0 or 1) has Y / dX / dW at the floor and `liger_vs_hf` ≤ 1.5e-5 (bf16) / 5e-6 (fp16).
   - `llama` keeps dX on the floor but Y and dW sit at ~1.4–1.7× the floor; vs the HF llama body the gap is ~0.003–0.004 relative in bf16 (HF itself is ~1.7–2× the floor on Y, because it multiplies the weight in the input dtype after an fp32 rstd).
   - `none` is the worst of the three (bf16 Y / dX / dW ~1.2–1.7× floor).
   - `F.rms_norm` hits the floor when offset = 0 (it upcasts like a pure fp32 variance path), but with offset = 1 it is **not** Gemma: it scales in the input dtype, so Y / dX land ~1.5× the floor while Liger gemma stays on the floor.
   - In fp32 every variant is within a few ×1e-7 of the reference. `torch.compile` of the closed-over llama body works for the first shape; later Inductor compiles with a different offset / casting closure hit `InvalidCxxCompiler` (no `cl` on this Windows box) and were recorded as errors.

3. **The block kernel is only selected for narrow rows with many tokens, and it is not bit-identical to the row kernel.**
   - Rule in the wrapper: row if `BLOCK_SIZE > 256 or T < 32768 or row_mode`. So H = 4096 always takes the row path; H = 128 takes block only at T = 32768.
   - At that block-path cell, Y / dX / dW disagree with `row_mode=True` by ~2e-6–9e-6 relative (bf16) — enough to break bit-reproducibility across the threshold, small versus the dtype floor (~1.7e-3).

4. **In-place halves the backward activation peak; Liger matches `F.rms_norm` / compile on held memory and beats eager by 2×.**
   - Held after forward is one `[T, H]` for Liger / compile / `F.rms_norm`, two for eager HF (it materializes an fp32 working buffer).
   - Extra peak: in-place ≈ one `[T, H]`; out-of-place / compile / `F.rms_norm` ≈ two; eager ≈ six (Llama-3.2-1B at 16k tokens: 64 / 128 / 384 MiB).
   - Fwd+bwd latency: Liger is 2.3–4.6× faster than eager HF. At T = 2048, `F.rms_norm` can still win (0.14 vs 0.28 ms at H = 1024); from T = 8192 Liger pulls ahead of both `F.rms_norm` and compile on the larger hidden size.

5. **Forward runs at or above copy bandwidth; backward is 1.7–5.8× eager and beats `F.rms_norm` once H ≥ 1024.**
   - Forward Liger: 178–229 GB/s (copy 98–114; eager 64–100; compile / `F.rms_norm` comparable to Liger at power-of-two H). Padding H = 2049 → BLOCK 4096 is free for Liger (217 GB/s) but drops `F.rms_norm` to 128 GB/s.
   - Backward Liger out-of-place: 100–179 GB/s (5.1–5.8× eager for H ≥ 2048). In-place is 10–20% slower than out-of-place on the pure kernel bench (it still wins on end-to-end peak memory). No register spills up to H = 11008.

Takeaway: use `casting_mode="gemma"` when you want HF-Gemma numerics (floor-accurate); `llama` is closer to HF-Llama but not to the fp64 floor on Y / dW. Prefer `in_place=False` whenever the same `dY` storage might be read again (`autograd.grad` with a reused `grad_outputs`, or any graph that aliases the upstream grad) — the kernels themselves stay correct across a second `retain_graph` call if each call gets a fresh buffer. The row/block switch is a quiet bit-reproducibility cliff at H ≤ 256 and T ≥ 32768. On this GPU the kernel is bandwidth-competitive with `F.rms_norm` / `torch.compile` and far leaner than eager HF, with in-place buying another 2× off the backward peak.

Full numbers: [results.md](results.md) / [results.json](results.json).
