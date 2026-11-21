# 0028 — Liger-Kernel KL divergence (`LigerKLDivLossFunction`)

- **Upstream**: [linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel) (submodule `upstream/Liger-Kernel`, commit `1ab779f`), imported unmodified from `src/`.
- **Component**: `src/liger_kernel/ops/kl_div.py` (`LigerKLDivLossFunction` / `kldiv_forward_triton` / `kldiv_backward_triton`) and the `LigerKLDIVLoss` module wrapper. Forward: one Triton program per row, `BLOCK_SIZE = min(16384, next_power_of_2(V))`, fp32 accumulation into a `(BT,)` or `(BT, V)` buffer, then wrapper reduction (`batchmean` / `sum` / `mean` / `none`) matching PyTorch. Backward: writes `-target` (or `-exp(target)` when `log_target`) into a fresh buffer, optionally scales by `grad_output` (skips the mul when `torch.equal(grad_output, tensor(1.0))`), applies the reduction divisor; target gets no gradient.
- **Hardware**: RTX 4060 Laptop 8 GB, torch 2.14.1+cu126, triton-windows 3.8.0, memory capped at 60% (4.8 GiB) by `lab/gpu_guard.py`. Peak allocated ~4.5 GiB.
- **Run**: `.venv\Scripts\python.exe experiments\0028-liger-kl-div\run.py` (about 20–60 s once kernels are warm; `run.py numerics eps edges api` skips the big sweeps).

## Question

How does Liger's Triton KL match `F.kl_div` on reductions, `log_target`, half precision, and non-finite / out-of-support targets? What does its always-fp32 loss path and per-row tiling buy (or cost) vs eager torch and `torch.compile` at 32k–152k vocabularies?

## Setup

- **Numerics**: BT 64 × V 32000; `log_target` True/False; reductions `batchmean` / `sum` / `mean` / `none`; fp32 / fp16 / bf16. Loss and d(input) vs an fp64 analytic reference and vs `F.kl_div`. Soft / log-softmax teacher pairs.
- **Eps**: 90% zeros per row (renormalized), `eps` ∈ {1e-10 … 1e-2}; Liger clamps `target` with `eps` before `log`, torch uses the `0·log(0)=0` convention.
- **Edges**: padded zeros, one-hot, negative / >1 targets, `±inf` / NaN in input or target, `log_target` with `-inf` padding / `+inf`, V = 1.
- **API**: last-layer vs scaled `grad_output`, `reduction="none"`, no target grad, `retain_graph`, non-contiguous target (via `ensure_contiguous`), module wrapper, `torch.compile` of `LigerKLDIVLoss`, bf16 loss dtype, fp16 `mean` vs `batchmean` gradient underflow.
- **Perf**: bf16 `batchmean`, V ∈ {32000, 128256, 151936}, BT 256–4096; Liger vs `F.kl_div` vs `torch.compile(F.kl_div)` on selected cells. Separate `reduction="none"` sweep for the fp32 `(BT, V)` tile.

## Findings

1. **Loss is always fp32 and more accurate than half-precision torch; gradients match torch (including its half-precision limits).**
   - On every cell Liger returns `float32` (including bf16 inputs); `F.kl_div` returns bf16 for a bf16 `batchmean`. Liger's fp32 accumulate sits on the analytic floor (5e-8–2e-7) while torch bf16 / fp16 loss error equals the dtype floor (0.15–0.21%).
   - Gradients: exact in fp32; in bf16 at the floor (~0–0.2%); in fp16 `batchmean` / `sum` / `none` both Liger and torch sit at ~1.7% / 0 vs the fp64 free-variable grad because the stored grad is the input dtype. `reduction="mean"` in fp16 zeros the entire gradient (`nonzero_frac = 0`, abs max 0): values are `-target / (BT·V)` and underflow fp16; `batchmean` keeps 99.7% nonzero. Prefer `batchmean` (and bf16/fp32) for half-precision training.

2. **`eps` clamp is silent and only bites when raised; negative targets become finite wrong losses.**
   - Default `eps=1e-10` matches torch on sparse (90% zero) rows (grad identical always, since `d/d input = -target`). At `eps=1e-2` Liger loss is 30.7% higher (4.29 vs 3.29) — it treats every zero as mass `eps`.
   - A negative target entry: torch → `nan`; Liger → finite `2.37` because `log(max(target, eps))` replaces the negative with `eps`. Out-of-range mass (`target > 1`) matches torch. Keep targets non-negative probabilities (or use `log_target=True`).

3. **Non-finite inputs mostly match torch; `log_target` with `-inf` is NaN on both.**
   - Padded zero columns and one-hot targets match. Student `-inf` → `inf` loss; student `+inf` → `-inf` loss; NaN input → NaN loss with finite grads; NaN target → NaN loss and NaN grads.
   - `log_target=True` with `-inf` padded teacher columns: both Liger and torch return `nan` (`exp(-inf)·(-inf - input)`). A `tl.where` / `xlogy` guard would fix it the same way as experiment 0025's JSD rewrite; Liger does not.

4. **API is clean: reductions, compile, contiguous copies, no target grad.**
   - Last-layer and `×2.5` `grad_output` match the analytic grad; `reduction="none"` with ones `grad_output` matches `-target`; a second `retain_graph` backward is bit-identical; target `.grad is None`; non-contiguous targets work (`ensure_contiguous`); `torch.compile(LigerKLDIVLoss)` matches eager bit-for-bit on the tested cell. Module wrapper with custom `eps` works.

5. **Forward is near copy bandwidth with ~0 extra peak; 2.8–4.6× faster than eager `F.kl_div`, tied with `torch.compile`.**
   - bf16 `batchmean`: Liger forward ~167–247 GB/s (0.20 → 4.3 ms from 256×32k to 2048×128k) with ~0 MiB forward peak above the live tensors (only a `(BT,)` fp32 buffer). Eager torch is stuck at ~50–60 GB/s with a forward peak of ~1.5–3× the input (48 → 1506 MiB). `torch.compile(F.kl_div)` matches Liger's forward time and also drops the forward peak to ~0; on fwd+bwd compile is often a bit ahead of Liger (3.3 vs 4.6 ms at 2048×32k; 6.6 vs 9.1 ms at 1024×128k) because Liger still materializes a full bf16 grad buffer.
   - Backward peak is the same for Liger and torch (~1× input: the grad). Setup OOMs under the 4.8 GiB cap past ~2048×128k / 1024×152k when allocating the bf16 pair (plus a temporary fp32 softmax).
   - `reduction="none"`: both hold the expected `(BT, V)` fp32 tile; Liger's peak equals held, torch peaks at 3× (extra temps). Prefer `batchmean` unless the per-element map is required.

Takeaway: use Liger KL for fp32-accurate loss under half-precision inputs and a cheap forward vs eager torch; prefer `reduction="batchmean"` (never fp16 `mean`), keep targets valid probabilities or use `log_target` without `-inf` padding, and know that `torch.compile(F.kl_div)` already closes most of the latency gap while Liger still wins on always-fp32 loss dtype.

Full numbers: [results.md](results.md) / [results.json](results.json).
