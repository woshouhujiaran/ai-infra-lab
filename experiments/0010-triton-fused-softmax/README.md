# 0010 — Triton tutorial fused softmax vs `torch.softmax`

- **Upstream**: [triton-lang/triton](https://github.com/triton-lang/triton) (shallow submodule `upstream/triton` pinned at tag `v3.8.0`, matching the installed `triton-windows` 3.8.0). The tutorial file is imported unmodified; `run.py` executes it only up to its unit test, so the tutorial's own benchmark and plot are skipped.
- **Component**: `python/tutorials/02-fused-softmax.py`. This covers `softmax_kernel` (one row per block of `next_power_of_2(N)` elements, persistent grid looping over rows) and the `softmax()` launcher, which calls `warmup()` on every call and sizes the grid from the compiled kernel's registers and shared memory.
- **Hardware**: RTX 4060 Laptop 8 GB (24 SMs, 99 KiB shared memory per block, 32 MB L2, 256 GB/s theoretical DRAM), torch 2.14.1+cu126, triton-windows 3.8.0, memory capped at 60% by `lab/gpu_guard.py`.
- **Run**: `.venv\Scripts\python.exe experiments\0010-triton-fused-softmax\run.py` (about 1 min, mostly `torch.compile`).

## Question

The tutorial claims its fused kernel is "significantly faster" than `torch.softmax` for rows that fit on-chip. Does that hold on a laptop Ada GPU with current PyTorch? Where does the one-row-per-block design break down: padding to a power of two, register spills, or shared memory? How much does the launcher's per-call `warmup()` cost when the input is small?

## Setup

- Input: `M = 4096` rows of fp32 `N(0, 1)` logits, with row length N ∈ {256, 512, 1024, 2048, 4096, 4097, 8192, 8193, 12288, 16384, 32768, 65536}. 4097 and 8193 are one past a power of two, so the block doubles and almost half of it is masked.
- Variants:
  - `tutorial`: the tutorial's `softmax()` as written (8 warps).
  - `cached_w8`: the same kernel and grid, but the compiled kernel and grid are cached per shape instead of recomputed on every call.
  - `best warps`: the cached launcher with the fastest `num_warps` in {4, 8, 16, 32}, using the tutorial's occupancy formula for each.
  - `torch.softmax`; `torch.compile(torch.softmax, dynamic=False)`, which uses an Inductor-generated Triton kernel; the tutorial's `naive_softmax` (five eager ops).
  - `copy`: `y.copy_(x)`, which moves the same bytes with no compute.
- Bandwidth = 2·M·N·4 bytes divided by the median time from `triton.testing.do_bench`, which flushes L2 before every repetition (the tutorial's metric). All `torch.compile` shapes are compiled before any timing starts.
- Per-call cost: 2000 back-to-back calls on 64×1024 and 4096×1024. "Issue" is the CPU time to enqueue the calls; "wall" runs until the GPU finishes.
- Numerics: 512 rows per N, compared with fp64 `torch.softmax`.
- Repeatability: four full or partial runs agreed within about 3% for N ≥ 512 and within about 12% at N = 256. In the first runs, before shapes were precompiled, `torch.compile` at N = 256 measured 8.5 GB/s twice. That was compilation work overlapping the measurement, and precompiling removed it.

## Findings

1. **The tutorial's speedup over `torch.softmax` does not reproduce here.** `torch.softmax` already runs at copy bandwidth for every N (210–231 GB/s, 82–90% of theoretical). The tutorial kernel as written is never faster: 0.73× at N = 256, 0.83× at 512, 0.91× at 1024, 0.93–0.99× from 2048 to 8192, and 0.98–1.0× from 8193 to 16384. `torch.compile` matches `torch.softmax` (0.99–1.05×) up to N = 32768.
2. **Fixed 8 warps is the wrong choice at both ends.** Small rows want 4 warps: at N = 256, 8 warps put one element in each thread, and 16 or 32 warps fall to 105 and 45 GB/s. That still leaves the best Triton variant at 0.79× of torch for N = 256 and 0.87× for N = 512. From N = 2048 to 8193, 8–32 warps reach 232–236 GB/s (up to 92% of theoretical). That is 1.02–1.04× of `torch.softmax` and slightly above `copy_`. This is the only range where a Triton variant wins, and the win is small.
3. **Padding to a power of two costs registers and occupancy, but not bandwidth.** Going from N = 4096 to 4097 doubles the block to 8192. At 8 warps, registers rise from 38 to 128 per thread, shared memory from 16 to 32 KiB, and occupancy drops from 6 to 2 programs per SM. Bandwidth stays the same (222 vs 220 GB/s; best 234 vs 234), because masked lanes generate no DRAM traffic. At N = 8193 the 4- and 8-warp kernels hit the 255-register cap. Only the 4-warp kernel suffers (214 GB/s), and spills appear only at 4 warps for N ≥ 12288 (8 bytes, about −3%).
4. **Hard limit: the kernel cannot run fp32 rows of 32768 or more on this GPU.** Its shared-memory use is about 4·BLOCK_SIZE bytes (65,568 B at N = 16384), so N = 32768 needs 131,104 B against a 101,376 B per-block limit. Triton raises `OutOfResources` for every `num_warps` setting. `torch.softmax` keeps 229–231 GB/s at 32768 and 65536. `torch.compile` gets 229 and 214 GB/s there, using its looped reduction for long rows.
5. **The launcher's per-call `warmup()` dominates small calls.** At 64×1024 the kernel itself takes about 9 µs, but the tutorial launcher costs 36.4 µs per call. Caching the compiled kernel and grid cuts that to 17.8 µs. `torch.softmax` takes 7.9 µs, `torch.compile` 47.5 µs (Dynamo guards plus the wrapper), and `naive_softmax` 77 µs. Issue time equals wall time in every case, so all of these are CPU-bound. At 4096×1024 the GPU is the bottleneck and the tutorial, cached and torch variants all take about 35–38 µs per call. These back-to-back calls run from the warm 32 MB L2, so they are faster than the cold-cache `do_bench` figure of 150–170 µs.
6. **Fusion matters only against the naive version.** `naive_softmax` runs at 59–66 GB/s for N ≥ 2048, which is 3.5–3.9× slower than any fused kernel. That is close to the tutorial's ~4× DRAM-traffic estimate. At N = 65536 it runs out of memory under the 4.8 GB cap, because it allocates four full-size temporaries.
7. **Numerics are fine everywhere.** All variants pass `torch.allclose` against fp32 `torch.softmax`. The Triton kernel's approximate `tl.exp` gives a maximum relative error of 7.0–9.7e-7 against fp64, compared with 4.3–6.7e-7 for `torch.softmax`. Row sums stay within 1.8e-7 of 1.

Takeaway: on an RTX 4060 Laptop with torch 2.14, `torch.softmax` is already bandwidth-bound and is the right default. The tutorial kernel is useful as a teaching example but not as a replacement. It is up to 27% slower on short rows and adds 4.6× the per-call CPU cost on small inputs. Because each row must fit in shared memory, it cannot run fp32 rows of 32k or more on consumer Ada GPUs. If you want a custom row-wise kernel, cache the compiled launch, choose `num_warps` by row length (4 for N ≤ 1024, 8–16 for 2k–16k), and use a looped or online reduction for long rows.

Full numbers: [results.md](results.md) / [results.json](results.json).
