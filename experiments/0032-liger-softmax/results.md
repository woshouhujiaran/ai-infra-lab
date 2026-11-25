# 0032 — Liger Softmax results

GPU: {'name': 'NVIDIA GeForce RTX 4060 Laptop GPU', 'memory_total_mib': 8188, 'memory_used_mib': 3, 'utilization_percent': 0, 'temperature_c': 46, 'memory_fraction_cap': 0.6, 'torch': '2.14.1+cu126', 'cuda': '12.6'}
Peak allocated (tracker): 3328.0 MiB
Wall time: 30.7s

## 1. Numerics

| dtype | scale | y_rel | dx_rel | y_floor | dx_floor | y_vs_F | dx_vs_F | multi | BLOCK | sum_err |
|---|---|---|---|---|---|---|---|---|---|---|
| fp32 | 1 | 9.325e-08 | 1.003e-07 | 9.665e-08 | 1.035e-07 | 9.288e-08 | 1.063e-07 | False | 4096 | 1.573e-07 |
| fp32 | 10 | 8.608e-08 | 2.782e-07 | 8.856e-08 | 2.790e-07 | 1.149e-07 | 3.319e-07 | False | 4096 | 3.038e-07 |
| fp16 | 1 | 2.080e-04 | 5.434e-04 | 5.929e-06 | 5.007e-04 | 7.804e-06 | 8.148e-04 | False | 4096 | 1.746e-05 |
| fp16 | 10 | 1.677e-04 | 0.001126 | 4.186e-07 | 0.001145 | 1.334e-05 | 0.001108 | False | 4096 | 3.516e-04 |
| bf16 | 1 | 0.001658 | 0.002865 | 1.101e-05 | 0.003298 | 1.547e-05 | 0.003055 | False | 4096 | 2.372e-04 |
| bf16 | 10 | 0.00137 | 0.007895 | 2.484e-09 | 0.008085 | 2.484e-09 | 0.007793 | False | 4096 | 0.002761 |

## 2. Edges

| case | liger/fp64 | F/fp64 | liger/F | nan_l | nan_F | dx_vs_F | dx_mask | sum_err | note |
|---|---|---|---|---|---|---|---|---|---|
| normal | 0.001632 | 0.001632 | 0 | False | False | 0.003223 | - | 1.771e-04 | - |
| half_row_neg_inf | 0.001656 | 0.001656 | 1.046e-05 | False | False | 0.003212 | 0 | 2.275e-04 | - |
| all_neg_inf | nan | nan | nan | True | True | - | - | nan | skipped_nan_y |
| nan_inf_cells | nan | nan | nan | True | True | - | - | nan | skipped_nan_y |
| large_logits_1e3 | 2.526e-08 | 2.526e-08 | 0 | False | False | 0.002548 | - | 1.127e-07 | - |
| all_equal_zero | 0 | 0 | 0 | False | False | 3.609e-04 | - | 0 | - |
| one_hotish | 1.683e-42 | 1.683e-42 | 0 | False | False | 0 | - | 0 | - |

## 3. Public path (multi_block_launch)

| V | BLOCK | warps | multi | pow2>=V | sum_err | err |
|---|---|---|---|---|---|---|
| 1 | 1 | 4 | False | True | 0 | - |
| 2 | 2 | 4 | False | True | 0.001953 | - |
| 3 | 4 | 4 | False | True | 0.002197 | - |
| 63 | 64 | 4 | False | True | 5.035e-04 | - |
| 64 | 64 | 4 | False | True | 6.638e-04 | - |
| 127 | 128 | 4 | False | True | 4.015e-04 | - |
| 128 | 128 | 4 | False | True | 4.368e-04 | - |
| 1023 | 1024 | 4 | False | True | 2.128e-04 | - |
| 1024 | 1024 | 4 | False | True | 2.449e-04 | - |
| 2049 | 4096 | 8 | False | True | 1.352e-04 | - |
| 4096 | 4096 | 8 | False | True | 1.667e-04 | - |
| 8191 | 8192 | 16 | False | True | 1.624e-04 | - |
| 8192 | 8192 | 16 | False | True | 9.108e-05 | - |
| 16384 | 16384 | 16 | False | True | 3.749e-05 | - |
| 32768 | 32768 | 32 | False | True | 7.922e-05 | - |
| 65535 | 65536 | 32 | False | True | 4.026e-05 | - |
| 65536 | 65536 | 32 | False | True | 7.602e-05 | - |

### Over-limit

| V | calculate_settings | liger |
|---|---|---|
| 65537 | RuntimeError: Cannot launch Triton kernel since n = 65537 exceeds the recommended Triton blocksize = | RuntimeError: Cannot launch Triton kernel since n = 65537 exceeds the recommended Triton blocksize = |
| 65537 | RuntimeError: Cannot launch Triton kernel since n = 65537 exceeds the recommended Triton blocksize = | RuntimeError: Cannot launch Triton kernel since n = 65537 exceeds the recommended Triton blocksize = |
| 128256 | RuntimeError: Cannot launch Triton kernel since n = 128256 exceeds the recommended Triton blocksize  | RuntimeError: Cannot launch Triton kernel since n = 128256 exceeds the recommended Triton blocksize  |

### Forced multi-block vs public single-block

| V | pub_B | force_B | y_vs_pub | y_vs_F | dx_vs_pub | dx_vs_F | ms_single | ms_multi | multi/single | err |
|---|---|---|---|---|---|---|---|---|---|---|
| 2048 | 2048 | 1024 | - | - | - | - | - | - | - | CompilationError: at 12:8: |
| 4096 | 4096 | 1024 | - | - | - | - | - | - | - | CompilationError: at 12:8: |
| 8192 | 8192 | 1024 | - | - | - | - | - | - | - | CompilationError: at 12:8: |
| 16384 | 16384 | 1024 | - | - | - | - | - | - | - | CompilationError: at 12:8: |

## 4. Layouts

| case | contig | y_vs_F | dx_vs_F | canary_ok | mod_y | err |
|---|---|---|---|---|---|---|
| contiguous | True | 2.372e-05 | 0.003126 | - | 0 | - |
| row_strided_x | False | 2.372e-05 | 0.003126 | True | - | - |
| transposed_x | False | 1.431e-05 | 0.003098 | - | - | - |
| input_3d | True | 2.228e-05 | 0.003139 | - | - | - |

## 5. Memory / latency

| label | T | V | liger_fwd | F_fwd | cmp_fwd | liger_fb | F_fb | held_MiB | peak_MiB | multi |
|---|---|---|---|---|---|---|---|---|---|---|
| attn S=512 | 512 | 512 | 0.1659 | 0.01099 | 0.009811 | 0.0215 | 0.02788 | 2 | 2.5 | False |
| attn S=2048 | 256 | 2048 | 0.01382 | 0.02708 | 0.04238 | 0.03368 | 0.04722 | 4.5 | 5.5 | False |
| attn S=8192 | 128 | 8192 | 0.03371 | 0.08011 | 0.07034 | 0.07799 | 0.1449 | 9 | 11 | False |
| attn S=16384 | 64 | 16384 | 0.0371 | 0.06664 | 0.1018 | 0.09028 | 0.1382 | 10 | 12 | False |
| vocab 32k | 256 | 32000 | 0.2628 | 0.5676 | 0.6615 | 0.7117 | 1.156 | 66 | 82 | False |
| vocab 64k-1 | 64 | 65535 | 0.4558 | 0.3387 | 0.586 | 0.98 | 0.6816 | 48 | 56 | False |
| vocab 64k | 64 | 65536 | 0.2906 | 0.3593 | 0.7151 | 0.6161 | 0.7465 | 40 | 48 | False |

## 6. Bandwidth (bf16, T=4096)

| V | BLOCK | util | liger_GB/s | F_GB/s | cmp_GB/s | liger_ms | F_ms | multi |
|---|---|---|---|---|---|---|---|---|
| 512 | 512 | 1 | 62.87 | 60.32 | 164 | 0.1334 | 0.1391 | False |
| 1024 | 1024 | 1 | 164.2 | 75.1 | 40.82 | 0.1022 | 0.2234 | False |
| 2048 | 2048 | 1 | 130.4 | 73.22 | 47.04 | 0.2573 | 0.4583 | False |
| 2049 | 4096 | 0.5002 | 182 | 95.86 | 178.9 | 0.1844 | 0.3502 | False |
| 4096 | 4096 | 1 | 225.7 | 181 | 214.8 | 0.2974 | 0.3707 | False |
| 8192 | 8192 | 1 | 231.3 | 218.5 | 220.5 | 0.5803 | 0.6142 | False |
| 16384 | 16384 | 1 | 226.4 | 229.5 | 233.7 | 1.186 | 1.17 | False |
| 32768 | 32768 | 1 | 225.8 | 234.3 | 185 | 2.377 | 2.291 | False |
| 65536 | 65536 | 1 | 234.4 | 234.8 | 130.9 | 4.581 | 4.573 | False |
