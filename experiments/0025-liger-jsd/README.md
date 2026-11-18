# 0025 — Liger-Kernel JSD (`LigerJSDFunction`)

- **Upstream**: [linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel) (submodule `upstream/Liger-Kernel`, commit `1ab779f`), imported unmodified from `src/`.
- **Component**: `src/liger_kernel/ops/jsd.py` + `ops/backends/_triton/jsd.py` (generalized Jensen-Shannon divergence on student / teacher log-probs, used for distillation). Forward clones the student, runs `_jsd_kernel` (one program per row, 4096-column blocks, dX written in place), writes a per-element fp32 loss tile of shape `(BT, V)`, sums it and casts to the input dtype. Backward returns the saved dX.
- **Hardware**: RTX 4060 Laptop 8 GB, torch 2.14.1+cu126, triton-windows 3.8.0, memory capped at 60% (4.8 GiB) by `lab/gpu_guard.py`. Peak allocated 3.5 GiB. The CuTe DSL / cuTile backends are not installed, so the dispatcher uses the Triton impl.
- **Run**: `.venv\Scripts\python.exe experiments\0025-liger-jsd\run.py` (about 1 minute).

## Question

Does the Triton JSD kernel handle the log-probs real distillation code produces (padded or top-k truncated vocabularies with `-inf`)? Is its numerics / API behavior what the docstring says? What does the `(BT, V)` loss tile cost in memory and time at 128k–152k vocabularies compared with eager (TRL-style), `torch.compile` and a row-loss rewrite?

## Setup

- **Numerics**: BT=64, V=32000, beta ∈ {0, 0.1, 0.5, 0.9, 1}, fp32 / fp16 / bf16, "far" (teacher ≠ student, loss ≈ 0.25–1.3) and "close" (noise 0.1) pairs; rel L2 error of loss and dX vs an fp64 analytic reference (checked against fp64 autograd to 1e-15). Compared with the eager TRL-style `F.kl_div` + `logsumexp` in the input dtype.
- **Non-finite**: 16 × 16384; `-inf` on 384 / a whole 4096-column block in both models (padded vocab), teacher top-50 only, student top-50 only; finite sentinels (`-1e9`, fp16 min, fp32 min); end to end through `log_softmax` with a `-inf`-padded vocabulary.
- **API**: `ignore_index` rows, all rows ignored, `grad_output ≠ 1`, `retain_graph` with an in-place edit of the returned gradient, gradient w.r.t. the target, non-contiguous inputs.
- **Rewrite (`RowJSD`, in `run.py`)**: same math, `tl.where` guards for zero probabilities and for blocks that are `-inf` in both models, per-row loss (no `(BT, V)` tile), forward is loss-only and saves nothing but X / Y, backward recomputes dX from X / Y with `grad_output` read on the device (no `torch.equal` + `.item()` sync).
- **Memory / latency**: bf16 log-probs, beta 0.5, V ∈ {32000, 128256, 151936}, BT 256–6144; upstream vs `RowJSD` vs eager bf16 vs `torch.compile` (fp32 upcast, 4 cells). Held / peak memory above the live X, Y; `do_bench` latency of fwd+bwd and of `torch.no_grad()` forward.

## Findings

1. **`-inf` log-probs make the loss NaN, and through `log_softmax` the whole gradient row NaN.**
   - Liger loss is `nan` whenever either model has a `-inf` entry that the formula multiplies by a zero probability: padded vocab (both `-inf`, 384 columns), teacher top-k at beta 0 / 0.5, student top-k at beta 0.5 / 1. Only the legitimately infinite cases (`KL(Q||P)` with a truncated teacher, `KL(P||Q)` with a truncated student) return `inf`, as they should.
   - dX is NaN in the padded lanes (2.3% for 384 of 16384), 25% when a whole 4096-block is `-inf` (the per-block `max` is `-inf` and `-inf - -inf` = NaN; this also hits beta 0, where 384 padded lanes alone leave dX finite), and 99.7% of the row for a student top-50 at beta 0.5 / 1.
   - End to end (`-inf`-padded logits → `log_softmax` → JSD → backward) **100% of d logits are NaN**: one NaN in the incoming gradient poisons the `log_softmax` backward row sum.
   - The eager TRL-style loss behaves the same. Finite sentinels (`-1e9`, fp16 min, fp32 min) are exact (loss and dX within 1e-7 of fp64), and `X.clamp(min=finfo.min)` before the loss gives d logits within 4e-7 of the fp64 valid-columns reference.
   - A `tl.where` guard (`P > 0`, `Q > 0`, `max == -inf → 0`) fixes all of it at no cost: `RowJSD` matches the fp64 reference to 3e-7 (fp32) in every scenario, including `inf` where the divergence really is infinite, and gives a finite end-to-end gradient with `-inf` padding.

2. **Output loss is cast to the input dtype; the kernel itself computes in fp32.**
   - bf16 / fp16 loss error equals the dtype rounding floor (0.005–0.2%), fp32 sums over the `(BT, V)` tile are at 3e-8–3e-7; `RowJSD` returns the fp32 sum (1e-7).
   - dX is at the bf16 floor (1.6–1.9e-3) for both Liger and `RowJSD`; in fp16 close pairs reach 9–16% dX error at beta 0.1–0.9 (0.5% at beta 0 / 1; identical for both, so it is the fp16 storage of tiny `1/BT`-scaled values, not the kernel), so prefer bf16 or fp32 there.
   - Eager bf16 is unusable for close pairs: 3.8–6.1× relative loss error (and 17–109% dX error) at beta ∈ (0, 1), because it forms `log M` and the KL terms in bf16.
   - In fp32, beta ∈ (0, 1) on a close pair has 4e-5–7e-5 loss error for Liger and `RowJSD` alike (cancellation between `β·P·(Y − log M)` and `(1 − β)·Q·(X − log M)`).

3. **beta = 0 / 1 are rescaled limits, not the limits of the general formula.**
   - The general loss goes to 0 at both ends (beta 1e-4: 1.3e-4; beta 0.999: 1.3e-3), but beta = 0 returns `KL(P||Q)` (1.31) and beta = 1 returns `KL(Q||P)` (1.30), i.e. `lim loss/beta` and `lim loss/(1 − beta)`.
   - Gradient norm jumps the same way (beta 1e-4: 1e-4 of beta 0's; beta 1 is 2.5× beta 0's). Moving beta from 0.01 to 0 multiplies the loss and gradient by ~100 — retune the learning rate or loss weight when sweeping beta across the endpoints.

4. **API behavior.**
   - `ignore_index` rows get exactly zero gradient and the loss is a mean over non-ignored rows (1.4e-7 vs fp64). With every row ignored the loss is 0 and the gradient is 0 (no 0/0 NaN).
   - `grad_output ≠ 1` and non-contiguous inputs are correct (2.6e-7); the caller's tensors are not modified.
   - With `grad_output == 1` backward returns the saved dX itself (same data pointer). An in-place edit of that gradient (e.g. `g.mul_(0)`) makes a second `retain_graph` backward raise autograd's version-counter error — loud, not silent. Leaf `.grad` accumulates correctly (2.0× after two backwards). `RowJSD` saves nothing it returns, so this does not arise.
   - The target gets no gradient (`grad is None`) while the gradient w.r.t. the log-target is as large as the student's (norm 2.8e-3 vs 2.3e-3). Pass a detached teacher or use another loss if the teacher is trained jointly.

5. **The `(BT, V)` fp32 loss tile makes Liger JSD ≈ 3× the input in memory and ~100 GB/s; a row-loss kernel is 2.2–2.4× faster in fwd+bwd, ~5× in eval, and holds nothing.**
   - Liger: held after forward = 1× the input (the saved dX), peak = 3× (clone 2 B/elem + fp32 tile 4 B/elem). 1.5 GiB for BT = 2048 at 128k vocab, 1.8 GiB at 152k; BT = 3072 OOMs at both (4.8 GiB cap), where `RowJSD` still runs 6144 × 128k (peak 1.5 GiB, which is just the returned dX).
   - Forward under `torch.no_grad()` costs the same as fwd+bwd (6.1 vs 6.4 ms at 32k × 2048), because dX is always computed and cloned. Latency is ~100–107 GB/s on 5 accesses of the bf16 input; `RowJSD` is 230–245 GB/s: fwd+bwd 2.2–2.4× faster (24.5 → 10.8 ms at 128k × 2048; 29.0 → 12.9 ms at 152k × 2048) and no-grad forward 4.8–5.6× faster (4.3 ms at 128k × 2048), with 0 MiB held and 0 MiB peak.
   - Eager TRL-style (bf16) is ~18× slower than `RowJSD` (8× slower than Liger; 13 GB/s) and OOMs at 128k / 152k × 2048 (holds 6× the input after forward).
   - `torch.compile` of an fp32-upcast version is *faster and leaner than Liger*: 1.05–1.14× faster fwd+bwd (21.8 vs 24.5 ms at 128k × 2048) with 0.66× the peak (1.0 vs 1.5 GiB), but still half of `RowJSD`'s throughput (120 vs 242 GB/s).

Takeaway: Liger's JSD is fine for finite log-probs, but guard or clamp `-inf` log-probs before calling it (padded vocabularies, top-k / top-p teachers), expect to pay a fp32 `(BT, V)` tile and an unconditional dX in eval too, and mind the beta endpoints' scale. A ~60-line row-loss kernel removes the NaNs, the tile and the eval cost.

Full numbers: [results.md](results.md) / [results.json](results.json).
