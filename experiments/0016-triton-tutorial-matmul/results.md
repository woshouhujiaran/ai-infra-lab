triton c01b6774b1 `03-matrix-multiplication.py`, triton-windows 3.8.0, torch 2.14.1+cu126, NVIDIA GeForce RTX 4060 Laptop GPU (memory cap 0.6). Times are `triton.testing.do_bench` medians (L2 flushed before each rep), best of 4 interleaved rounds whose order alternates (medians of rounds are in results.json). Each timed block starts after 2 s of GPU warm-up.

Clock-state reference, cuBLAS fp16 4096^3 TFLOPS at the start of each section: square 25.96, group 23.78, llm 22.33, epilogue 27.41, odd 27.06

## 1. Tutorial CUDA autotune configs at 4096^3 (99 KiB = 101,376 B shared memory per block)

| config (BMxBNxBK group warps stages) | dtype | shared B | fits | regs | spills | 4096^3 fp16 autotune ms |
|---|---|---|---|---|---|---|
| 128x256x64 g8 w8 s3 | fp16 | 98304 | True | 236 | 0 | 4.924 |
| 64x256x32 g8 w4 s4 | fp16 | 61440 | True | 252 | 0 | 5.908 |
| 128x128x32 g8 w4 s4 | fp16 | 49152 | True | 250 | 0 | 5.516 |
| 128x64x32 g8 w4 s4 | fp16 | 36864 | True | 166 | 0 | 5.299 |
| 64x128x32 g8 w4 s4 | fp16 | 36864 | True | 166 | 0 | 5.290 |
| 128x32x32 g8 w4 s4 | fp16 | 30720 | True | 96 | 0 | 5.318 |
| 64x32x32 g8 w2 s5 | fp16 | 24576 | True | 122 | 0 | 5.504 |
| 32x64x32 g8 w2 s5 | fp16 | 24576 | True | 121 | 0 | 5.701 |
| 128x256x128 g8 w8 s3 | fp16 | 196608 | False |  |  |  |
| 256x128x128 g8 w8 s3 | fp16 | 196608 | False |  |  |  |
| 256x64x128 g8 w4 s4 | fp16 | 245760 | False |  |  |  |
| 64x256x128 g8 w4 s4 | fp16 | 245760 | False |  |  |  |
| 128x128x128 g8 w4 s4 | fp16 | 196608 | False |  |  |  |
| 128x64x64 g8 w4 s4 | fp16 | 73728 | True | 168 | 0 | 5.385 |
| 64x128x64 g8 w4 s4 | fp16 | 73728 | True | 165 | 0 | 5.458 |
| 128x32x64 g8 w4 s4 | fp16 | 61440 | True | 127 | 0 | 5.678 |
| 128x256x64 g8 w8 s3 | fp8e4m3 | 49152 | True | 255 | 138 |  |
| 64x256x32 g8 w4 s4 | fp8e4m3 | 30720 | True | 255 | 72 |  |
| 128x128x32 g8 w4 s4 | fp8e4m3 | 24576 | True | 255 | 62 |  |
| 128x64x32 g8 w4 s4 | fp8e4m3 | 18432 | True | 183 | 0 |  |
| 64x128x32 g8 w4 s4 | fp8e4m3 | 18432 | True | 198 | 0 |  |
| 128x32x32 g8 w4 s4 | fp8e4m3 | 15360 | True | 120 | 0 |  |
| 64x32x32 g8 w2 s5 | fp8e4m3 | 12288 | True | 122 | 0 |  |
| 32x64x32 g8 w2 s5 | fp8e4m3 | 12288 | True | 120 | 0 |  |
| 128x256x128 g8 w8 s3 | fp8e4m3 | 98304 | True | 255 | 162 |  |
| 256x128x128 g8 w8 s3 | fp8e4m3 | 98304 | True | 255 | 148 |  |
| 256x64x128 g8 w4 s4 | fp8e4m3 | 122880 | False |  |  |  |
| 64x256x128 g8 w4 s4 | fp8e4m3 | 122880 | False |  |  |  |
| 128x128x128 g8 w4 s4 | fp8e4m3 | 98304 | True | 255 | 150 |  |
| 128x64x64 g8 w4 s4 | fp8e4m3 | 36864 | True | 244 | 0 |  |
| 64x128x64 g8 w4 s4 | fp8e4m3 | 36864 | True | 238 | 0 |  |
| 128x32x64 g8 w4 s4 | fp8e4m3 | 30720 | True | 149 | 0 |  |

## 2a. Square M = N = K, TFLOPS

| S | cublas | tutorial | acc16 | int64 | tutorial_e5m2 | tutorial_e4m3 | promote_e4m3 | scaled_mm_e4m3 | scaled_mm_e4m3_fast | tutorial / cuBLAS | acc16 / cuBLAS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 256 | 2.52 | 2.97 | 3.65 | 3.29 | 3.29 | 4.09 | 3.29 | 3.65 | 3.65 | 1.18 | 1.45 |
| 512 | 11.93 | 13.77 | 15.43 | 13.09 | 17.43 | 17.43 | 16.37 | 20.18 | 20.18 | 1.15 | 1.29 |
| 1024 | 21.63 | 24.68 | 40.37 | 24.97 | 38.83 | 38.83 | 38.14 | 46.58 | 40.37 | 1.14 | 1.87 |
| 2048 | 29.08 | 26.46 | 56.29 | 29.38 | 49.06 | 49.06 | 50.23 | 55.92 | 57.06 | 0.91 | 1.94 |
| 3072 | 29.88 | 29.57 | 56.34 | 30.52 | 51.38 | 51.29 | 49.54 | 58.43 | 59.60 | 0.99 | 1.89 |
| 4096 | 27.18 | 27.74 | 48.01 | 28.33 | 46.93 | 49.37 | 46.09 | 54.96 | 58.84 | 1.02 | 1.77 |
| 6144 | 26.58 | 26.44 | 47.90 | 27.53 | 45.58 | 44.57 | 41.97 | 52.19 | 53.23 | 0.99 | 1.80 |
| 8192 | 25.37 | 25.10 | 46.59 | 26.32 | 43.90 | 42.35 | 42.27 | 50.27 | 53.42 | 0.99 | 1.84 |

## 2b. Autotune: chosen config and first-call wall time (s, includes compilation on a cold cache)

| S | tutorial | acc16 | int64 | tutorial_e5m2 | tutorial_e4m3 | promote_e4m3 |
|---|---|---|---|---|---|---|
| 256 | 128x32x32 g8 w4 s4 (1.5 s) | 32x64x32 g8 w2 s5 (14.0 s) | 128x32x64 g8 w4 s4 (15.8 s) | 32x64x32 g8 w2 s5 (15.2 s) | 128x32x64 g8 w4 s4 (1.8 s) | 32x64x32 g8 w2 s5 (14.9 s) |
| 512 | 32x64x32 g8 w2 s5 (1.2 s) | 128x64x32 g8 w4 s4 (1.2 s) | 128x32x32 g8 w4 s4 (1.2 s) | 128x32x64 g8 w4 s4 (1.5 s) | 128x32x64 g8 w4 s4 (1.5 s) | 128x32x64 g8 w4 s4 (1.5 s) |
| 1024 | 32x64x32 g8 w2 s5 (1.2 s) | 128x128x32 g8 w4 s4 (1.2 s) | 128x32x32 g8 w4 s4 (1.2 s) | 128x32x64 g8 w4 s4 (1.5 s) | 128x32x64 g8 w4 s4 (1.5 s) | 128x32x64 g8 w4 s4 (1.5 s) |
| 2048 | 128x256x64 g8 w8 s3 (1.3 s) | 128x128x32 g8 w4 s4 (1.3 s) | 128x128x32 g8 w4 s4 (1.3 s) | 128x64x64 g8 w4 s4 (1.7 s) | 128x64x64 g8 w4 s4 (1.7 s) | 128x32x64 g8 w4 s4 (1.6 s) |
| 3072 | 128x256x64 g8 w8 s3 (1.5 s) | 128x128x32 g8 w4 s4 (1.4 s) | 128x128x32 g8 w4 s4 (1.5 s) | 128x64x64 g8 w4 s4 (1.9 s) | 128x64x64 g8 w4 s4 (1.9 s) | 128x32x64 g8 w4 s4 (1.8 s) |
| 4096 | 128x256x64 g8 w8 s3 (1.7 s) | 128x128x32 g8 w4 s4 (1.6 s) | 128x128x32 g8 w4 s4 (1.7 s) | 128x64x64 g8 w4 s4 (2.2 s) | 128x64x64 g8 w4 s4 (2.2 s) | 64x128x64 g8 w4 s4 (2.1 s) |
| 6144 | 64x128x32 g8 w4 s4 (2.5 s) | 128x256x64 g8 w8 s3 (2.1 s) | 128x128x32 g8 w4 s4 (2.5 s) | 128x64x64 g8 w4 s4 (3.3 s) | 128x64x64 g8 w4 s4 (3.4 s) | 128x32x64 g8 w4 s4 (2.9 s) |
| 8192 | 64x128x32 g8 w4 s4 (4.8 s) | 128x128x32 g8 w4 s4 (3.2 s) | 128x128x32 g8 w4 s4 (4.5 s) | 128x64x64 g8 w4 s4 (5.8 s) | 128x64x64 g8 w4 s4 (5.8 s) | 128x32x64 g8 w4 s4 (4.7 s) |

## 3. GROUP_SIZE_M sweep, TFLOPS (tile fixed to the autotuned winner; outputs bit-identical)

| M x N x K | tile | tiles (m, n) | g=1 | g=2 | g=4 | g=8 | g=16 | g=32 | g=64 | best / g=1 |
|---|---|---|---|---|---|---|---|---|---|---|
| 4096x4096x4096 | 128x256x64 g8 w8 s3 | (32, 16) | 28.60 | 28.80 | 28.63 | 29.46 | 29.44 | 29.42 | 29.41 | 1.03 |
| 8192x8192x8192 | 64x128x32 g8 w4 s4 | (128, 64) | 15.77 | 25.93 | 26.80 | 27.14 | 27.46 | 27.13 | 26.68 | 1.74 |
| 4096x14336x4096 | 64x128x32 g8 w4 s4 | (64, 112) | 15.52 | 25.08 | 25.03 | 26.00 | 25.64 | 25.64 | 25.60 | 1.68 |

## 4a. LLM linear y = x @ W.T, W [N, K] fp16: time in us (weight GB/s for M <= 16, else TFLOPS)

`tutorial_wT_view` passes the column-major `W.T` view as B; `tutorial_rowmajor_b` uses a pre-transposed contiguous copy (separate autotuner).

| N x K | M | F.linear | tutorial_wT_view | tutorial_rowmajor_b | config (view / row-major) |
|---|---|---|---|---|---|
| 4096x4096 | 1 | 177.2 (189.4 GB/s) | 196.6 (170.7 GB/s) | 196.6 (170.7 GB/s) | 32x64x32 g8 w2 s5 / 32x64x32 g8 w2 s5 |
| 4096x4096 | 16 | 205.8 (163.0 GB/s) | 194.6 (172.4 GB/s) | 193.5 (173.4 GB/s) | 32x64x32 g8 w2 s5 / 32x64x32 g8 w2 s5 |
| 4096x4096 | 128 | 227.3 (18.9 TF) | 208.9 (20.6 TF) | 206.8 (20.8 TF) | 128x64x64 g8 w4 s4 / 128x64x64 g8 w4 s4 |
| 4096x4096 | 1024 | 1169.4 (29.4 TF) | 1156.1 (29.7 TF) | 1156.1 (29.7 TF) | 128x128x32 g8 w4 s4 / 128x128x32 g8 w4 s4 |
| 4096x4096 | 4096 | 4892.7 (28.1 TF) | 4713.5 (29.2 TF) | 4635.6 (29.6 TF) | 128x256x64 g8 w8 s3 / 128x128x32 g8 w4 s4 |
| 14336x4096 | 1 | 507.9 (231.2 GB/s) | 574.5 (204.4 GB/s) | 570.4 (205.9 GB/s) | 64x128x64 g8 w4 s4 / 64x128x32 g8 w4 s4 |
| 14336x4096 | 16 | 573.4 (204.8 GB/s) | 576.5 (203.7 GB/s) | 572.4 (205.2 GB/s) | 32x64x32 g8 w2 s5 / 64x128x32 g8 w4 s4 |
| 14336x4096 | 128 | 643.1 (23.4 TF) | 612.4 (24.6 TF) | 617.5 (24.3 TF) | 32x64x32 g8 w2 s5 / 128x32x32 g8 w4 s4 |
| 14336x4096 | 1024 | 4332.0 (27.8 TF) | 4364.3 (27.6 TF) | 4339.2 (27.7 TF) | 128x256x64 g8 w8 s3 / 128x128x32 g8 w4 s4 |
| 14336x4096 | 4096 | 17904.6 (26.9 TF) | 19125.2 (25.1 TF) | 17652.7 (27.2 TF) | 64x128x32 g8 w4 s4 / 128x256x64 g8 w8 s3 |

## 4b. Re-autotuning per new batch size (fresh autotuner, x [M, 4096] @ W.T [4096, 4096])

Total first-call time for M = 1..32: 54.5 s; median 1.20 s per new M; second call median 0.249 ms.

| M | first call s | bench s | second call ms | config |
|---|---|---|---|---|
| 1 | 1.27 | 1.27 | 0.321 | 32x64x32 g8 w2 s5 |
| 2 | 17.03 | 17.03 | 0.388 | 32x64x32 g8 w2 s5 |
| 3 | 1.25 | 1.25 | 0.281 | 32x64x32 g8 w2 s5 |
| 4 | 1.23 | 1.23 | 0.242 | 32x64x32 g8 w2 s5 |
| 5 | 1.22 | 1.22 | 0.292 | 32x64x32 g8 w2 s5 |
| 6 | 1.21 | 1.21 | 0.289 | 32x64x32 g8 w2 s5 |
| 7 | 1.20 | 1.20 | 0.215 | 32x64x32 g8 w2 s5 |
| 8 | 1.21 | 1.20 | 0.248 | 32x64x32 g8 w2 s5 |
| 9 | 1.20 | 1.20 | 0.353 | 32x64x32 g8 w2 s5 |
| 10 | 1.20 | 1.20 | 0.331 | 32x64x32 g8 w2 s5 |
| 11 | 1.20 | 1.20 | 0.221 | 32x64x32 g8 w2 s5 |
| 12 | 1.20 | 1.20 | 0.257 | 32x64x32 g8 w2 s5 |
| 13 | 1.20 | 1.20 | 0.197 | 32x64x32 g8 w2 s5 |
| 14 | 1.20 | 1.20 | 0.256 | 32x64x32 g8 w2 s5 |
| 15 | 1.20 | 1.20 | 0.206 | 32x64x32 g8 w2 s5 |
| 16 | 1.20 | 1.20 | 0.306 | 32x64x32 g8 w2 s5 |
| 17 | 1.20 | 1.20 | 0.207 | 32x64x32 g8 w2 s5 |
| 18 | 1.20 | 1.20 | 0.221 | 32x64x32 g8 w2 s5 |
| 19 | 1.20 | 1.20 | 0.250 | 32x64x32 g8 w2 s5 |
| 20 | 1.20 | 1.20 | 0.197 | 32x64x32 g8 w2 s5 |
| 21 | 1.20 | 1.20 | 0.233 | 32x64x32 g8 w2 s5 |
| 22 | 1.20 | 1.20 | 0.302 | 32x64x32 g8 w2 s5 |
| 23 | 1.20 | 1.20 | 0.219 | 32x64x32 g8 w2 s5 |
| 24 | 1.20 | 1.20 | 0.212 | 32x64x32 g8 w2 s5 |
| 25 | 1.20 | 1.20 | 0.309 | 32x64x32 g8 w2 s5 |
| 26 | 1.21 | 1.21 | 0.251 | 32x64x32 g8 w2 s5 |
| 27 | 1.20 | 1.20 | 0.231 | 32x64x32 g8 w2 s5 |
| 28 | 1.20 | 1.20 | 0.224 | 32x64x32 g8 w2 s5 |
| 29 | 1.20 | 1.20 | 0.216 | 32x64x32 g8 w2 s5 |
| 30 | 1.20 | 1.20 | 0.224 | 32x64x32 g8 w2 s5 |
| 31 | 1.20 | 1.20 | 0.267 | 32x64x32 g8 w2 s5 |
| 32 | 1.20 | 1.20 | 0.340 | 32x64x32 g8 w2 s5 |

## 5. Fused leaky_relu epilogue, ms

| M x N x K | fused | tutorial + F.leaky_relu | cuBLAS + F.leaky_relu | tutorial, no activation | fused vs separate: differing elems / max abs diff | rel err vs fp64 (fused / separate) |
|---|---|---|---|---|---|---|
| 4096x4096x4096 | 4.985 | 5.427 | 5.499 | 5.752 | 2192897 / 3e-05 | 0.000208 / 0.000208 |
| 1024x14336x4096 | 4.152 | 4.346 | 4.554 | 4.102 | 1923405 / 3e-05 | 0.000208 / 0.000208 |
| 4096x14336x4096 | 17.860 | 18.685 | 18.406 | 18.169 | 7682774 / 3e-05 | 0.000208 / 0.000208 |

## 6a. fp16 inputs: relative RMS error vs fp64, 512x512 output, N(0, 1) inputs

| K | fp16_output_rounding | tutorial | acc16 | cublas | cublas_no_reduced_reduction |
|---|---|---|---|---|---|
| 256 | 0.000207 | 0.000207 | 0.000842 | 0.000207 | 0.000207 |
| 1024 | 0.000208 | 0.000208 | 0.00166 | 0.000302 | 0.000208 |
| 4096 | 0.000207 | 0.000207 | 0.00332 | 0.000392 | 0.000207 |
| 16384 | 0.000209 | 0.00021 | 0.00661 | 0.000514 | 0.000209 |
| 65536 | 0.000207 | 0.000231 | 0.0133 | 0.000809 | 0.000213 |

## 6b. fp8 inputs: relative RMS error vs fp64 of the quantized fp8 inputs (accumulation + fp16 output rounding only)

fp8 inputs are plain casts of the same fp16 inputs (no scaling, as in the tutorial). Total error vs the original fp16 inputs is dominated by quantization and given in the last two columns.

| K | tutorial_e5m2 | split_acc_e5m2 | imprecise_limit_e5m2 | promote_e5m2 | tutorial_e4m3 | split_acc_e4m3 | imprecise_limit_e4m3 | promote_e4m3 | scaled_mm_e4m3 | scaled_mm_e4m3_fast | total err tutorial e5m2 | total err tutorial e4m3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 256 | 0.000253 | 0.000253 | 0.000253 | 0.000216 | 0.000305 | 0.000305 | 0.000305 | 0.000226 | 0.000251 | 0.000305 | 0.0742 | 0.0376 |
| 1024 | 0.000404 | 0.000404 | 0.000404 | 0.000216 | 0.000587 | 0.000587 | 0.000587 | 0.000224 | 0.000249 | 0.000587 | 0.0748 | 0.0375 |
| 4096 | 0.000888 | 0.000888 | 0.000888 | 0.000216 | 0.00136 | 0.00136 | 0.00136 | 0.000225 | 0.000249 | 0.00136 | 0.0747 | 0.0374 |
| 16384 | 0.00214 | 0.00214 | 0.00214 | 0.000215 | 0.00313 | 0.00313 | 0.00313 | 0.000225 | 0.000249 | 0.00313 | 0.0746 | 0.0376 |
| 65536 | 0.00518 | 0.00518 | 0.00518 | 0.000216 | 0.00707 | 0.00707 | 0.00707 | 0.000225 | 0.000249 | 0.00707 | 0.0749 | 0.0381 |

## 7. Shapes not divisible by 16, TFLOPS

| M x N x K | cuBLAS | tutorial | tutorial / cuBLAS | tutorial config | rel err (tutorial / cuBLAS) |
|---|---|---|---|---|---|
| 4096x4096x4096 | 27.94 | 27.92 | 1.00 | 128x256x64 g8 w8 s3 | 0.000208 / 0.000208 |
| 4096x4096x4095 | 18.41 | 19.74 | 1.07 | 128x128x32 g8 w4 s4 | 0.000208 / 0.000208 |
| 4096x4096x4097 | 24.48 | 22.50 | 0.92 | 128x64x64 g8 w4 s4 | 0.000208 / 0.000208 |
| 4096x4096x4104 | 28.45 | 26.16 | 0.92 | 128x128x32 g8 w4 s4 | 0.000208 / 0.000208 |
| 4096x4097x4096 | 18.31 | 19.23 | 1.05 | 128x128x32 g8 w4 s4 | 0.000208 / 0.000208 |
| 4097x4096x4096 | 22.18 | 21.23 | 0.96 | 64x128x32 g8 w4 s4 | 0.000208 / 0.000208 |
| 4095x4095x4095 | 18.84 | 15.59 | 0.83 | 64x256x32 g8 w4 s4 | 0.000208 / 0.000208 |

## 8. int32 offsets: A [M, 4096] fp16 with M*K around 2^31, B [4096, 64] (fixed tile 128x64x32; each case in a fresh process; rows compared with cuBLAS)

| case | M | max A offset | A GiB | wrong rows | first wrong row | result |
|---|---|---|---|---|---|---|
| tutorial:below | 524288 | 2147483647 | 4.0 | 0 |  | correct |
| tutorial:above | | | | | | exit 1: Search for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information. |
| int64:above | 524544 | 2148532223 | 4.002 | 0 |  | correct |

Runtime: 801 s; section timings (s): {'overflow_s': 15.4, 'configs_s': 28.2, 'warm_s': 3.0, 'square_s': 215.6, 'group_s': 43.3, 'llm_s': 108.5, 'retune_s': 54.5, 'epilogue_s': 23.5, 'numerics_s': 147.4, 'odd_s': 148.0}
