# 0023 — Liger-Kernel fused residual-add + RMSNorm (`LigerFusedAddRMSNormFunction`)

- **Upstream**: [linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel) (submodule `upstream/Liger-Kernel`, commit `1ab779f`), imported unmodified from `src/`.
- **Component**: `src/liger_kernel/ops/fused_add_rms_norm.py`. Implements the decoder-layer pattern `hidden = residual + hidden; residual = hidden; hidden = rmsnorm(hidden)` in one Triton forward (returns normalized `Y` and post-add residual `S`) and a fused backward that folds `dS_out` into `dX` (= `dR`). Casting modes match plain Liger RMSNorm (`llama` / `gemma` / `none`); `in_place` overwrites the caller's `dY` with `dX`.
- **Hardware**: RTX 4060 Laptop 8 GB, torch 2.14.1+cu126, triton-windows 3.8.0, memory capped at 60% by `lab/gpu_guard.py`. Peak allocated memory was ~3.0 GB.
- **Run**: `.venv\Scripts\python.exe experiments\0023-liger-fused-add-rmsnorm\run.py` (about 1–2 min).

## Question

Does fusing the residual add into RMSNorm buy memory / bandwidth over a separate `add` + `LigerRMSNormFunction`? Do the casting modes still match HF? When is `in_place` safe, given the backward returns `(dX, dX, dW)` (same tensor twice)? What happens on norm-only / residual-only graphs where one output is unused?

## Setup

- **API defaults**: inspect `Function.forward`, `liger_fused_add_rms_norm`, and `LigerFusedAddRMSNorm` for the `in_place` default vs the class docstring.
- **Output branches** (bf16, T = 256, H = 1024): loss on both `Y` and `S` / `Y` only / `S` only, with `in_place` True/False. Gradients vs fp64 fused reference.
- **Aliasing**: plain both-outputs backward (clone grad buffers so `in_place` cannot poison the reference); `retain_graph` twice with a fresh `dY` vs reusing the caller's `dY`; low-level `fused_add_rms_norm_backward` identity of the returned `dX` / `dR` pair.
- **Numerics**: T = 2048, H = 4096; fp32 / fp16 / bf16; casting ∈ {llama, gemma, none}; offset ∈ {0, 1}. Rel L2 of Y / S / dX / dR / dW vs fp64; baselines = matching HF fused eager and `torch.compile` of that body.
- **Memory / latency**: bf16 at Qwen3-0.6B (H = 1024) and Llama-3.2-1B (H = 2048), T ∈ {2048, 8192, 16384}. Fused in-place / out-of-place vs separate `add` + Liger RMSNorm, eager HF, `torch.compile`.
- **Bandwidth**: bf16, T = 4096, H from 512 to 11008 (incl. H = 2049). Forward counted as 3·T·H·2 bytes, backward as 4·T·H·2.

## Findings

1. **`in_place` defaults disagree with each other and with the docstring.**
   - `LigerFusedAddRMSNormFunction.forward` and the `nn.Module` default to `False`.
   - `liger_fused_add_rms_norm(...)` defaults to `True`.
   - The class docstring still says it "defaults to `True` to save memory." Call sites that go through the functional API silently opt into the overwrite; Module users do not.

2. **Norm-only and residual-only both work on this torch — unused outputs arrive as zeros, not `None`.**
   - All six (branch × in_place) cells match fp64 (bf16 dx/dr ~0.0017; residual-only is exact with dW = 0).
   - Leaf `x.grad` / `r.grad` are equal in value but **not** the same storage (autograd copies when assigning to two leaves).
   - The Triton's `has_dS_out = dS_out is not None` is therefore always true under current PyTorch multi-output Function grads; a genuine `None` would still crash on `.view` before that check. Norm-only pays an extra load of a zero `dS_out` buffer.

3. **`in_place=True` is correct on a clean graph but reusing the caller's `dY` is a silent footgun; the low-level API returns the same tensor for `dX` and `dR`.**
   - Plain both-outputs: in-place and out-of-place sit on the bf16 floor (~0.0017 / 0.0034); in-place does overwrite the caller's `dY` (`caller_dY_overwritten=True`).
   - Fresh `dY` clones: a second `retain_graph` call is bit-identical for both settings.
   - **Reusing the same `dY`**: in-place first call is already wrong vs a clean call (rel 1.60) and a second reuse drifts further (1.20 vs first); out-of-place stays at 0.
   - `fused_add_rms_norm_backward` returns `(dX, dX, dW)` — same `data_ptr` twice. With `in_place=True` that object is the input `dY`.

4. **`gemma` casting is closest to the fp64 floor and matches HF; `llama` / `none` leave Y / dW above the floor; S is a pure add.**
   - S matches HF bit-for-bit in every cell (rel 0) — it is just `X + R` written out.
   - bf16 `gemma`: Y / dW ~1.4× floor, dX on the floor; `liger_vs_hf` on Y / dW ≤ 1e-5.
   - bf16 `llama`: Y / dW ~1.7–2× floor; with offset 0, Y matches HF to ~2e-5, but dx/dr disagree with HF at ~0.002 (HF itself is above the floor on the backward).
   - `none` is the worst (bf16 Y ~1.6–1.9× floor; vs HF ~0.0037).
   - fp32 everything is within a few ×1e-7. `torch.compile` of the closed-over HF body hits `InvalidCxxCompiler` (no `cl`) on this Windows box after the first shape.

5. **Fused in-place cuts the backward peak by 25% vs out-of-place / separate Liger; both beat eager by ~4.7×.**
   - Held after forward is two `[T, H]` (Y + S) for fused / separate / compile, five for eager (fp32 working buffers).
   - Extra peak at Llama-3.2-1B × 16k: in-place 192 MiB, out-of-place / separate / compile 256 MiB, eager 896 MiB.
   - Fwd+bwd: fused is 3.5–5.3× faster than eager HF and within ~0–20% of separate `add` + Liger RMSNorm (fusion's win is memory, not a large latency gap over already-Triton RMSNorm). `torch.compile` is competitive at small T and loses at large H×T.

6. **Fused forward sits at or above copy×1.5 bandwidth; backward is 2–12× eager and ~1.0–1.2× a separate Liger RMSNorm graph.**
   - Forward fused: 145–171 GB/s (copy 98–114; eager ~32–42; separate 112–162). Padding H = 2049 → BLOCK 4096 costs ~20% vs H = 2048.
   - Backward fused out-of-place: 59–232 GB/s (in-place 10–30% slower on the pure kernel bench). No register spills up to H = 11008.

Takeaway: prefer `casting_mode="gemma"` when you want HF-Gemma numerics; treat the functional API's default `in_place=True` as a footgun if the same `dY` buffer might be reused. Fusion's real prize vs separate Liger RMSNorm is the in-place backward peak (one fewer `[T, H]`), not a large throughput jump. Norm-only / residual-only are fine on torch 2.14 because unused outputs are zeros — but the Triton `None` path is still unguarded.

Full numbers: [results.md](results.md) / [results.json](results.json).
