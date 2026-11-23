# 0030 — Liger-Kernel Total Variation Distance (`LigerTVDLossFunction`)

- **Upstream**: [linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel) (submodule `upstream/Liger-Kernel`, commit `1ab779f`), imported unmodified from `src/`.
- **Component**: `src/liger_kernel/ops/tvd.py` (`LigerTVDLossFunction` / `tv_distance_forward_triton` / `tvd_backward_triton`) and the `LigerTVDLoss` module wrapper. Forward: one Triton program per row, `BLOCK_SIZE = min(16384, next_power_of_2(V))`, TVD = `0.5 * |p - q|` accumulated in an always-fp32 loss buffer, with the reduction scale fused into both the per-row loss and the stored `d/dp` (subgradient `0.5 * sgn(p - q)`, zero on ties). Backward only returns a gradient for `p`; `q` / labels / reduction / `ignore_index` get `None`. When `grad_output` equals scalar `1.0` (`torch.equal`), the saved grads buffer is returned by reference.
- **Hardware**: RTX 4060 Laptop 8 GB, torch 2.14.1+cu126, triton-windows 3.8.0, memory capped at 60% (4.8 GiB) by `lab/gpu_guard.py`. Peak allocated ~4.5 GiB.
- **Run**: `.venv\Scripts\python.exe experiments\0030-liger-tvd\run.py` (~70 s once kernels are warm; `run.py numerics ignore ties edges api` skips the big sweeps).

## Question

How does Liger's Triton TVD match an eager `0.5*|p-q|` reference on reductions, `ignore_index`, ties, half precision, and non-probability inputs? What does fusing the gradient into the forward (and always returning fp32 loss) buy vs eager torch and `torch.compile` at 32k–152k vocabularies?

## Setup

- **Numerics**: BT 64 × V 32000; probability pairs and upstream-style raw-`randn` `p`; reductions `batchmean` / `sum` / `mean` / `none`; fp32 / fp16 / bf16. Loss and `d/dp` vs fp64 analytic and vs eager torch.
- **ignore_index**: ~40% ignored labels; all-ignored batch (division by `n_non_ignore`).
- **Ties**: self-distillation; half-row exact ties (padded-vocab style); fp32 Δ=1e-5 cast to fp16/bf16 near-ties; equal zero pads.
- **Edges**: negative mass, NaN / `±inf`, unnormalized `p`, V = 1, V = 32768 (multi-block loop past `MAX_FUSED_SIZE = 16384`).
- **API**: last-layer vs scaled `grad_output`, no `q` gradient (vs eager), `retain_graph` / same-storage return of saved grads, non-contiguous `q`, module wrapper, `torch.compile`, always-fp32 loss dtype, fp16 `mean` vs `batchmean` grad magnitude, bad label shape.
- **Perf**: bf16 `batchmean`, V ∈ {32000, 128256, 151936}, BT 256–4096; Liger vs eager torch vs `torch.compile` of the eager formula on selected cells. Separate `reduction="none"` sweep for the fp32 `(BT, V)` tile.

## Findings

1. **Always-fp32 loss beats half-precision torch; gradients are exact `±0.5` (scaled) and match the analytic subgradient.**
   - On every cell Liger returns `float32` (including bf16 / fp16 inputs); eager torch returns bf16 for a bf16 `batchmean`. Probability pairs: Liger bf16 loss sits at ~1.6e-4 vs the dtype floor (~1.3e-3 for torch); fp16 Liger is ~1e-5 vs torch's ~9e-5. Gradients for `batchmean` / `sum` / `none` are bit-exact vs the fp64 free-variable subgradient in all three dtypes (the stored values are exactly `±0.5` or 0 before / after the reduction scale).
   - Upstream unit tests skip fp16; with raw `randn` as `p` (the test convention), eager fp16 `batchmean` / `sum` / `mean` overflow to `inf` while Liger's fp32 accumulate stays finite (~2.5e-5 vs ref). Prefer Liger (or fp32) whenever `p` is not already a probability simplex.

2. **`ignore_index` is correct when some rows survive; an all-ignored batch divides by zero.**
   - Partial ignore (~40%): loss and grads match the reference at ~1e-8; ignored rows have grad abs max 0. `sum` with every label ignored returns 0 with zero grads. `batchmean` / `mean` raise `ZeroDivisionError` (`scale = 1.0 / n_non_ignore` with `n_non_ignore = 0`). The eager reference returns a disconnected `0.0` tensor instead — both paths are unsafe for a fully-masked microbatch; guard before calling.

3. **Ties use exact equality; `q` never gets a gradient.**
   - Self-distillation and trailing exact ties zero the tied slots' grads in fp32 / fp16 / bf16. A fp32 Δ=1e-5 becomes 249 exact bf16 ties (~0.19% of entries) and 0 fp16 ties — the subgradient follows the cast values, not the pre-cast ones. Equal zero pads contribute no grad.
   - Liger's backward returns `None` for `q` (teacher-style API). The same eager formula would give `q` a nonzero grad (`abs max 0.0156` on the probe). `grad_output == 1.0` returns the saved grads buffer by `data_ptr` (scaled outputs allocate a new tensor); `torch.compile(LigerTVDLoss)` matches eager bit-for-bit; non-contiguous `q` is fine via `ensure_contiguous`.

4. **Forward already materializes the full `(BT, V)` grad buffer — cheap backward, ~3× lower peak than eager, and 1.5–3× faster at large shapes.**
   - bf16 `batchmean`: Liger forward peak equals held (the bf16 grads tile; ~126 MiB at 2048×32k) and backward adds ~0 MiB. Eager torch peaks at ~3× the input (378 MiB) from `abs` / reduction temps. Sustained Liger forward is ~110–150 GB/s vs eager's ~40–55 GB/s; fwd+bwd at 2048×32k is 3.4 vs 10.3 ms (3.0×), at 1024×128k 6.4 vs 20.1 ms (3.1×). `torch.compile` of the eager formula can match or beat Liger's forward alone (160–232 GB/s) but loses on fwd+bwd (6.97 / 10.6 / 12.7 ms on the three compile cells) because it still builds grads in backward.
   - Eager OOMs under the 4.8 GiB cap at 2048×128k while Liger still runs (502 MiB held). `reduction="none"`: Liger always holds an fp32 `(BT, V)` tile (~1× expected); torch's none path stays in the input dtype (half the tile for bf16 inputs) and is slightly cheaper when the per-element map is required.

Takeaway: use Liger TVD for an always-fp32 loss, fused forward gradients, and ~3× lower peak / faster fwd+bwd than eager `0.5*|p-q|` at LLM vocab sizes; keep at least one non-ignored label per batch, treat `q` as a no-grad teacher, and remember upstream's fp16 skip is about eager overflow on raw inputs — Liger itself is fine.

Full numbers: [results.md](results.md) / [results.json](results.json).
