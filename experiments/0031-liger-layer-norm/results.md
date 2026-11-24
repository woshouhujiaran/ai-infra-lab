# 0031 — Liger-Kernel LayerNorm results

GPU: NVIDIA GeForce RTX 4060 Laptop GPU | torch 2.14.1+cu126 | seed 0

## Numerics (vs fp64 + F.layer_norm)

| dtype | bias | floor_y | liger_y | F_y | liger_dx | F_dx | liger_dw | F_dw | vs_F_y | compile_y |
|---|---|---|---|---|---|---|---|---|---|---|
| fp32 | True | 2.527e-08 | 4.654e-08 | 4.847e-08 | 5.586e-08 | 6.422e-08 | 1.845e-07 | 1.637e-07 | 4.121e-08 | 4.895e-08 |
| fp32 | False | 2.529e-08 | 6.118e-08 | 6.416e-08 | 5.583e-08 | 6.421e-08 | 1.883e-07 | 1.725e-07 | 5.545e-08 | 6.492e-08 |
| fp16 | True | 2.072e-04 | 2.072e-04 | 2.072e-04 | 2.074e-04 | 2.074e-04 | 2.067e-04 | 2.067e-04 | 2.844e-06 | 2.072e-04 |
| fp16 | False | 2.076e-04 | 2.076e-04 | 2.076e-04 | 2.074e-04 | 2.074e-04 | 2.131e-04 | 2.131e-04 | 3.807e-06 | 2.076e-04 |
| bf16 | True | 0.001656 | 0.001656 | 0.001656 | 0.001659 | 0.001659 | 0.001633 | 0.001633 | 6.074e-06 | 0.001656 |
| bf16 | False | 0.001659 | 0.001659 | 0.001659 | 0.001658 | 0.001658 | 0.001704 | 0.001704 | 8.557e-06 | 0.001659 |

## Layouts (strided / transposed / stride-0 dy)

| case | contig | y_rel | dx_rel | dw_rel | vs_F_dx | extra |
|---|---|---|---|---|---|---|
| contiguous | True | 0.001663 | 0.001657 | 0.001702 | 1.078e-05 | {} |
| row_strided_x | False | 0.001663 | 0.001657 | 0.001702 | 1.078e-05 | {'canary_unchanged': True} |
| transposed_x | False | 0.001661 | 0.001658 | 0.001657 | 9.425e-06 | {} |
| strided_weight_bias | True | 0.001662 | 0.001654 | 0.001702 | 6.727e-06 | {} |
| input_3d | True | 0.001657 | 0.001656 | 0.001657 | 1.301e-05 | {} |
| stride0_dy_from_ysum | - | - | True | True | 1.536e-05 | {'dx_abs_mean': 0.8099376793134319} |

## Edges / limits

| case | detail |
|---|---|
| H_65536 | {'H': 65536, 'BLOCK_SIZE': 65536, 'num_warps': 32, 'fwd_ok': True, 'y_finite': True, 'bwd_ok': True, 'dx_finite': True, 'dw_finite': True} |
| H_65537 | {'H': 65537, 'calculate_settings': 'RuntimeError: Cannot launch Triton kernel since n = 65537 exceeds the recommended Triton blocksize = 65536.'} |
| H_65535 | {'H': 65535, 'BLOCK_SIZE': 65536, 'num_warps': 32, 'fwd_ok': True, 'y_finite': True, 'bwd_ok': True, 'dx_finite': True, 'dw_finite': True} |
| grad_w_and_b | {'dx_finite': True, 'w_grad_is_none': False, 'b_grad_is_none': False, 'dw_abs_mean': 1.2186482548713684, 'db_abs_mean': 1.260059878230095} |
| freeze_w | {'dx_finite': True, 'w_grad_is_none': True, 'b_grad_is_none': False, 'db_abs_mean': 1.260059878230095} |
| freeze_b | {'dx_finite': True, 'w_grad_is_none': False, 'b_grad_is_none': True, 'dw_abs_mean': 1.2186482548713684} |
| freeze_both | {'dx_finite': True, 'w_grad_is_none': True, 'b_grad_is_none': True} |
| nan_row | {'y_nan': True, 'y_inf': False, 'F_y_nan': True, 'F_y_inf': False} |
| inf_row | {'y_nan': True, 'y_inf': False, 'F_y_nan': True, 'F_y_inf': False} |
| all_nan | {'y_nan': True, 'y_inf': False, 'F_y_nan': True, 'F_y_inf': False} |
| module_bias_false | {'has_bias_param': True, 'bias_abs_max': 0.0, 'bias_requires_grad': True, 'fwd_bwd_ok': True, 'dx_finite': True} |
| functional_bias_none | {'y_rel_vs_zero_bias': 0.0, 'dx_rel': 0.0, 'dw_rel': 0.0} |

## SM-partitioned dW/db determinism (sm=24, repeats=8)

| T | rows/prog | bit_dx | bit_dw | bit_db | dx_vs_F | dw_vs_F | db_vs_F |
|---|---|---|---|---|---|---|---|
| 1 | 1 | True | True | True | 0 | 0 | 0 |
| 3 | 1 | True | True | True | 0 | 0 | 0 |
| 23 | 1 | True | True | True | 4.391e-08 | 0 | 0 |
| 24 | 1 | True | True | True | 2.654e-09 | 0 | 0 |
| 25 | 2 | True | True | True | 0 | 0 | 0 |
| 256 | 11 | True | True | True | 1.251e-05 | 8.526e-07 | 0 |
| 4096 | 171 | True | True | True | 1.916e-05 | 6.697e-06 | 0 |

## Memory / latency (bf16)

| shape | T | H | liger_held | liger_peak | liger_ms | F_ms | compile_ms |
|---|---|---|---|---|---|---|---|
| GPT-2-124M | 2048 | 768 | 6 | 10.2 | 2.719 | 0.186 | 0.384 |
| GPT-2-124M | 8192 | 768 | 24.1 | 36.2 | 1.446 | 0.911 | 0.429 |
| GPT-2-124M | 16384 | 768 | 48.1 | 72.3 | 1.468 | 1.355 | 0.842 |
| BERT-base | 2048 | 768 | 6 | 9.2 | 0.232 | 0.186 | 0.186 |
| BERT-base | 8192 | 768 | 24.1 | 36.2 | 0.751 | 0.725 | 0.43 |
| BERT-base | 16384 | 768 | 48.1 | 72.3 | 1.467 | 1.569 | 0.848 |
| Llama-3.2-1B | 2048 | 2048 | 16 | 24.4 | 0.389 | 0.361 | 0.723 |
| Llama-3.2-1B | 8192 | 2048 | 64.1 | 96.5 | 1.327 | 1.549 | 2.966 |
| Llama-3.2-1B | 16384 | 2048 | 128.1 | 192.5 | 2.588 | 2.999 | 6.214 |
| Qwen3-0.6B | 2048 | 1024 | 8 | 12.2 | 0.25 | 0.237 | 0.496 |
| Qwen3-0.6B | 8192 | 1024 | 32.1 | 48.3 | 0.821 | 1.031 | 1.916 |
| Qwen3-0.6B | 16384 | 1024 | 64.1 | 96.3 | 1.672 | 2.075 | 3.98 |

## Bandwidth vs H (bf16, T=4096)

| H | block | copy | fwd_L | fwd_F | bwd_L | bwd_F | L/F_fwd | L/F_bwd |
|---|---|---|---|---|---|---|---|---|
| 512 | 512 | 93.1 | 146.3 | 85.3 | 47.3 | 100.7 | 1.72 | 0.47 |
| 768 | 1024 | 97.5 | 189 | 112.7 | 72 | 124.5 | 1.68 | 0.58 |
| 1024 | 1024 | 103.7 | 202.3 | 143.7 | 100.3 | 117 | 1.41 | 0.86 |
| 2048 | 2048 | 105.7 | 218.5 | 187.2 | 154.1 | 115.9 | 1.17 | 1.33 |
| 2049 | 4096 | 105.1 | 92.6 | 60.4 | 98.7 | 117.4 | 1.53 | 0.84 |
| 4096 | 4096 | 110 | 220.7 | 208.7 | 196.4 | 115 | 1.06 | 1.71 |
| 8192 | 8192 | 110.5 | 222.9 | 222.2 | 209.2 | 122.1 | 1 | 1.71 |
| 11008 | 16384 | 111.1 | 196.5 | 225.2 | 127.9 | 114 | 0.87 | 1.12 |
| 16384 | 16384 | 113.9 | 227.2 | 226.6 | 182.2 | 109.3 | 1 | 1.67 |

