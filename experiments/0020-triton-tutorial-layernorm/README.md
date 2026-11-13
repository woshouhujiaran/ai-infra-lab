# 0020 — Triton tutorial layer norm (fused forward + lock-based dW/db reduction)

- **Upstream**: [triton-lang/triton](https://github.com/triton-lang/triton) (submodule `upstream/triton`, tag v3.8.0, commit `c01b677`). `python/tutorials/05-layer-norm.py` is imported unmodified; only its top-level `test_layer_norm(...)` call and benchmark are skipped.
- **Component**: `_layer_norm_fwd_fused`, `_layer_norm_bwd_dx_fused`, `_layer_norm_bwd_dwdb` and the `LayerNorm` autograd wrapper. Both passes run one program per row with the whole row in one block (≤ 64 KB). In the backward pass, each row adds its `dy * xhat` and `dy` into one of `GROUP_SIZE_M` partial buffers (`row % GROUP_SIZE_M`), guarded by a `tl.atomic_cas` spin lock. The buffers have the dtype of `x`. A second kernel sums the buffers in fp32.
- **Hardware**: RTX 4060 Laptop 8 GB (24 SMs), torch 2.14.1+cu126, triton-windows 3.8.0, memory capped at 60% by `lab/gpu_guard.py`. Peak allocated memory was 2.6 GB.
- **Run**: `.venv\Scripts\python.exe experiments\0020-triton-tutorial-layernorm\run.py` (about 4.5 min).

## Question

The tutorial claims to be faster than PyTorch's layer norm, and its backward pass is a common template for Triton norm kernels (spin-locked partial buffers, then a reduction kernel). How does the wrapper handle the layouts autograd actually produces? How accurate are dW and db when the partial sums are kept in fp16 or bf16 over realistic token counts? Is the lock-ordered reduction deterministic, and how does `GROUP_SIZE_M` trade contention against accuracy? How fast is it against torch native, `torch.compile` and a lock-free rewrite?

## Setup

- **Layouts** (1024 × 2048 rows, fp16): contiguous 2D and 3D x; a row-strided x (`big[:, :N]`, as produced by slicing a fused projection); a permuted x (`(T, B, N).transpose(0, 1)`); and losses that give the backward pass a non-contiguous `dy`: `y.sum()`, which gives a stride-0 expanded `dy` with one element of storage; `y.mean()`, run in fp32 because fp16 cannot hold 1/numel; and `(y.transpose(0, 1) * g).sum()`, which gives a permuted `dy`. Each case runs in a child process and is compared with fp64 autograd on the same logical tensors. For the row-strided case, a 4 MiB block is freed with a canary tensor placed right after it, so that `y` lands in the freed block and any write past `y` shows up in the canary.
- **Fixed wrapper**: contiguous copies of x and `dy`, a contiguous `y`, and the lock-free backward pass described next.
- **Lock-free backward** (`_bwd_rows_fp32` in `run.py`): 4 × 24 programs. Each one walks a contiguous run of rows and keeps fp32 dW/db partials in registers. The upstream `_layer_norm_bwd_dwdb` kernel then sums the partials. There are no locks and the summation order is fixed. It uses `max(tutorial warps, BLOCK / 512)` warps (up to 32), because its three extra fp32 row vectors spill at 8 warps for N ≥ 16k.
- **Numerics**: N ∈ {1024, 4096}; M from 1k to 64k rows (32k for N = 4096); fp32, fp16 and bf16. The inputs follow the tutorial: x = -2.3 + 0.5·N(0,1), w and b ~ U(0,1), dy = 0.1·N(0,1). The "biased" runs add 0.05 to dy. The error metric is relative L2 error against an fp64 reference computed in row chunks. `floor` is that reference rounded once to the dtype.
- **Determinism**: 20 backward calls on one graph. The metric is the number of distinct (dW, db) bit patterns.
- **GROUP_SIZE_M sweep**: G ∈ {1, 4, 16, 64, 128, 256, 1024} at M = 16384, measuring backward time, determinism over 10 calls, and dW/db error with biased dy.
- **Bandwidth**: M = 4096, N from 512 to 32768, fp16 and bf16. Forward is counted as 2·M·N bytes and backward as 3·M·N bytes, divided by the `do_bench` median, which is taken over 3 interleaved rounds. Backward means `autograd.grad` for x, w and b, including the wrapper's buffer allocations. `torch.compile` runs with `donated_buffer=False` so that `retain_graph=True` works.

## Findings

1. **The wrapper assumes contiguous rows everywhere and silently corrupts memory when they aren't.** It allocates `y = empty_like(x)` and `dx = empty_like(dy)`, then indexes x, y, `dy` and dx with the stride of `x.reshape(-1, N)`.
   - **Row-strided x** (`big[:, :N]`): `y` comes out contiguous, but the kernel writes row r at r·2N. Half the rows of `y` are garbage (511 of 1024 wrong, NaN), and rows past M/2 are written into whatever follows `y`. Here that was the canary, and exactly half of it was overwritten (1,048,576 of 2,097,152 elements). The backward pass reads `dy` with the same wrong stride, so dW and db are wrong too (relative error 1.44 and 1.05).
   - **Permuted x**: `y` inherits the permuted strides but is filled in row-major order, so 1022 of 1024 rows of `y` are wrong (relative error 1.0). dW is also wrong (1.44), because the backward pass reads the saved non-contiguous x with the contiguous copy's stride.
   - **`y.sum()` loss**: autograd passes a stride-0 `dy` with 4 bytes of storage, and the kernel reads M·N elements from it. dx, dW and db all came back NaN. No CUDA error was raised.
   - **Permuted `dy`**: dW is wrong (1.41) and dx is off (0.039), while db is correct, because a sum doesn't care about order.
   - **`y.mean()` is fine**: `MeanBackward` divides, which materializes a contiguous `dy`.
   - Contiguous 2D and 3D inputs are correct. The fixed wrapper (contiguous x, `y` and `dy`) is correct in every case.
2. **The row limit is 64 KB.** N = 16384 works in fp32 and N = 32768 works in fp16/bf16. One more element raises `This layer norm doesn't support feature dim >= 64KB.`
3. **In fp16/bf16, dW and db accumulate in x.dtype and lose accuracy as the token count grows.** Each buffer accumulates M / G rows before the fp32 stage.
   - The table shows tutorial dW error as a multiple of the dtype floor (zero-mean dy):

     | rows per buffer | 4 | 8 | 32 | 64 | 256 |
     |---|---|---|---|---|---|
     | fp16 | 1.8× | 2.3× | 4.1× | 5.6× | 11–12× |
     | bf16 | 1.8× | 2.4× | 4.1× | 5.5× | 11–12× |

   - In absolute terms, bf16 dW at 64k tokens × 1024 or 32k × 4096 has a relative error of 1.9% against a floor of 0.16%. The error grows roughly as √(rows per buffer). Zero-mean db behaves the same way.
   - With a biased dy, db stays within 1.2–1.4× of the floor at the default G, because the large common term rounds well.
   - In every fp16/bf16 cell, allocating the same buffers in fp32 (a one-line change), the lock-free rewrite and torch native all land exactly on the floor. With fp32 inputs, every variant is limited by fp32 accumulation; there the tutorial's dW error (3.7e-7 at 64k rows) is below the lock-free rewrite's (5.2e-7) and torch native's (8.6e-7).
4. **The default `GROUP_SIZE_M` is just large enough to be mostly deterministic.** Rows that share a lock race only when they are co-resident on the GPU.
   - At N = 1024 (4 warps, 40 registers), 12 programs fit per SM, so 288 are co-resident. That is just above the default G = 256. The final run gave one bit pattern in every 20-call cell. An earlier run of the same script found 2–3 distinct dW patterns in 3 of the 4 fp16/bf16 N = 1024 cells (spread up to 3.2e-3 relative in bf16), plus 2 patterns in one fp32 cell that used fp32 buffers.
   - At N = 4096 (8 warps, 72 registers), only 3 programs fit per SM (72 co-resident), and the default G = 128 was deterministic.
   - Below those counts the result changes with every call. G ≤ 128 at N = 1024 and G ≤ 16 at N = 4096 gave 10 distinct patterns in 10 calls, and G = 64 at N = 4096 gave 1–3.
   - Torch native and the lock-free rewrite were bit-identical everywhere.
5. **Small G is a cliff for both speed and accuracy.**
   - At M = 16384, G = 1 serializes every row through one lock. Backward takes 31–43 ms instead of 0.5–2.0 ms, and the bf16 db is 84% wrong: once the running sum reaches a few hundred, adding 0.05 no longer changes a bf16 value (fp16: 30% wrong). G = 4 is still 5–15× slower than the default, with a bf16 db error of 54%.
   - From G = 64 up, time is flat. At N = 4096, G = 1024 is 2× slower again, from zeroing and re-reading 2 × 8 MiB of buffers.
   - The lock-free rewrite is 1.01–1.10× faster than the default G, deterministic, and at the floor.
   - Contended cells (G ≤ 16) varied by up to 2× between runs; the rest varied by 10–15%.
6. **Speed: the tutorial is faster than torch native in the backward pass and matches copy bandwidth in the forward pass, but it degrades at wide rows.**
   - **Forward**: the tutorial runs at 182–230 GB/s for N ≤ 16k, close to copy speed (195–226). Torch native is 2× slower at N = 512 (91 GB/s) and equal from N = 3072. At N = 32768 the tutorial drops to 167–183 GB/s, while `torch.compile` holds 230.
   - **Backward**: the tutorial is 1.24–1.71× faster than torch native for N = 1k–4k, 1.31–1.37× faster at 8k–16k and 1.13–1.17× faster at 32k. At N = 512 they are even (0.97×).
   - Beyond 4k the tutorial loses occupancy: 128 registers per thread at 8k (2 programs per SM), 255 at 16k, and 302 bytes of spills at 32k. It falls to 99–104 GB/s at 32k, where `torch.compile` (143 GB/s, 1.6× torch native) is the fastest.
   - The lock-free rewrite is 1.01–1.71× faster than torch native and beats the tutorial at 8k–16k (208 vs 162–165 GB/s at 8k, 174–176 vs 142–149 at 16k). It also spills at 32k, even with 32 warps (88–97 GB/s).
   - `torch.compile`'s backward pass is slower than torch native at N ≤ 1024 and only 1.05–1.45× faster at 2k–16k.

Takeaway: as a kernel, the tutorial layer norm is fast: it matches copy speed in the forward pass and is 1.2–1.7× faster than torch native in the backward pass up to 16k columns. As written, though, it is not a drop-in `F.layer_norm`. The wrapper silently produces wrong results or writes past `y` for row-strided or permuted x. It also returns NaN gradients for the simplest `y.sum()` loss, because it never makes x or `dy` contiguous. Keeping dW/db partials in fp16/bf16 costs up to 11–12× the dtype's rounding error at 32k–64k tokens. A small `GROUP_SIZE_M` turns the spin-lock reduction into a serial, nondeterministic and badly rounded sum. Calling `.contiguous()` on x and `dy`, allocating the partials in fp32 and keeping G at or above the number of co-resident programs fixes the correctness issues. A lock-free fixed-order reduction gets floor accuracy and bit-reproducibility at the same or better speed for N ≤ 16k.

Full numbers: [results.md](results.md) / [results.json](results.json).
