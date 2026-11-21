# 0028 — Liger-Kernel KL divergence results

GPU: NVIDIA GeForce RTX 4060 Laptop GPU | torch 2.14.1+cu126 | seed 0

## Numerics (vs fp64 analytic + torch)

| log_t | dtype | red | floor | liger_loss_rel | torch_loss_rel | liger_grad_rel | torch_grad_rel | liger_dtype |
|---|---|---|---|---|---|---|---|---|
| False | fp32 | batchmean | 4.253e-08 | 6.704e-08 | 4.253e-08 | 0 | 0 | float32 |
| False | fp32 | sum | 4.253e-08 | 6.704e-08 | 4.253e-08 | 0 | 0 | float32 |
| False | fp32 | mean | 3.155e-09 | 3.155e-09 | 1.102e-07 | 5.390e-08 | 5.390e-08 | float32 |
| False | fp32 | none | 2.528e-08 | 1.352e-07 | 2.306e-07 | 0 | 0 | float32 |
| False | fp16 | batchmean | 1.515e-04 | 9.079e-08 | 1.515e-04 | 0.01687 | 0.01687 | float32 |
| False | fp16 | sum | 1.515e-04 | 9.079e-08 | 1.515e-04 | 0 | 0 | float32 |
| False | fp16 | mean | 0.001719 | 5.500e-08 | 0.001719 | 1 | 1 | float32 |
| False | fp16 | none | 2.483e-04 | 1.331e-07 | 0.001505 | 0 | 0 | float32 |
| False | bf16 | batchmean | 0.001736 | 5.633e-08 | 0.001736 | 0 | 0 | float32 |
| False | bf16 | sum | 0.001736 | 5.633e-08 | 0.001736 | 0 | 0 | float32 |
| False | bf16 | mean | 7.799e-04 | 4.235e-09 | 7.799e-04 | 0.001702 | 0.001816 | float32 |
| False | bf16 | none | 0.001636 | 1.392e-07 | 0.01245 | 0 | 0 | float32 |
| True | fp32 | batchmean | 1.194e-08 | 1.194e-08 | 1.194e-08 | 2.037e-07 | 4.009e-08 | float32 |
| True | fp32 | sum | 1.194e-08 | 1.194e-08 | 1.194e-08 | 2.037e-07 | 4.009e-08 | float32 |
| True | fp32 | mean | 5.120e-08 | 5.548e-08 | 5.548e-08 | 2.227e-07 | 7.018e-08 | float32 |
| True | fp32 | none | 2.556e-08 | 1.975e-07 | 4.903e-08 | 2.037e-07 | 4.009e-08 | float32 |
| True | fp16 | batchmean | 2.038e-04 | 4.089e-08 | 2.038e-04 | 0.01688 | 0.01688 | float32 |
| True | fp16 | sum | 2.038e-04 | 4.089e-08 | 2.038e-04 | 3.207e-04 | 3.207e-04 | float32 |
| True | fp16 | mean | 0.001338 | 1.205e-07 | 0.001338 | 1 | 1 | float32 |
| True | fp16 | none | 2.525e-04 | 2.001e-07 | 3.709e-04 | 3.207e-04 | 3.207e-04 | float32 |
| True | bf16 | batchmean | 0.002097 | 1.064e-07 | 0.002097 | 0.001649 | 0.001649 | float32 |
| True | bf16 | sum | 0.002097 | 1.064e-07 | 0.002097 | 0.001649 | 0.001649 | float32 |
| True | bf16 | mean | 4.197e-04 | 1.858e-07 | 4.197e-04 | 0.002248 | 0.002361 | float32 |
| True | bf16 | none | 0.001606 | 1.999e-07 | 0.002271 | 0.001649 | 0.001649 | float32 |

## Eps clamp vs torch xlogy (90% zeros per row)

| eps | liger | torch | ref_eps | xlogy | liger_vs_torch | grad_rel |
|---|---|---|---|---|---|---|
| 1.000e-10 | 3.285 | 3.285 | 3.285 | 3.285 | 0 | 0 |
| 1.000e-08 | 3.285 | 3.285 | 3.285 | 3.285 | 0 | 0 |
| 1.000e-06 | 3.285 | 3.285 | 3.285 | 3.285 | 0 | 0 |
| 1.000e-04 | 3.285 | 3.285 | 3.285 | 3.285 | 6.387e-06 | 0 |
| 0.01 | 4.294 | 3.285 | 4.294 | 3.285 | 0.307 | 0 |

## Edge cases

| case | liger_loss | torch_loss | liger_finite_g | torch_finite_g | liger_g_nan | torch_g_nan |
|---|---|---|---|---|---|---|
| pad_zeros_512 | 1.072 | 1.072 | True | True | False | False |
| one_hot | 9.599 | 9.599 | True | True | False | False |
| neg_target | 2.366 | nan | True | True | False | False |
| gt1_target | 15.78 | 15.78 | True | True | False | False |
| neg_inf_input | inf | inf | True | True | False | False |
| pos_inf_input | -inf | -inf | True | True | False | False |
| nan_input | nan | nan | True | True | False | False |
| nan_target | nan | nan | False | False | True | True |
| log_target_neg_inf_pad | nan | nan | True | True | False | False |
| log_target_pos_inf | inf | inf | False | False | False | False |
| v1 | 0 | 0 | True | True | False | False |

## API / autograd

- `last_layer_grad_rel_vs_ref`: 0
- `scaled_2p5_grad_rel_vs_2p5_ref`: 2.602e-08
- `scaled_over_last`: 2.5
- `none_ones_grad_rel_vs_ref`: 0
- `none_vs_batchmean_scaled`: 0
- `target_grad_is_none`: True
- `input_grad_is_not_none`: True
- `retain_graph_second_rel`: 0
- `noncontig_target_ok`: True
- `noncontig_target_grad_rel`: 0
- `module_wrapper_grad_rel`: 0
- `module_wrapper_loss`: 0.5417
- `compile_loss_rel_vs_eager`: 0
- `compile_grad_rel_vs_eager`: 0
- `compile_loss_dtype`: float32
- `none_loss_dtype`: float32
- `none_loss_shape`: [32, 4096]
- `batchmean_loss_dtype`: float32
- `torch_none_dtype`: float32
- `torch_batchmean_dtype`: float32
- `liger_bf16_batchmean_dtype`: float32
- `torch_bf16_batchmean_dtype`: bfloat16
- `fp16_mean_grad_nonzero_frac`: 0
- `fp16_batchmean_grad_nonzero_frac`: 0.9969
- `fp16_mean_grad_abs_max`: 0
- `fp16_batchmean_grad_abs_max`: 2.623e-04

## Memory / latency (bf16, batchmean)

| impl | vocab | BT | held | peak_f | peak_b | ms_fwd | ms_fb | GB/s |
|---|---|---|---|---|---|---|---|---|
| liger | 32k | 256 | 0 | 0 | 32 | 0.1956 | 0.6219 | 167.5 |
| torch | 32k | 256 | 0 | 48 | 32 | 0.5484 | 0.8521 | 59.8 |
| liger | 32k | 1024 | 0 | 0 | 125 | 0.6304 | 2.5 | 207.9 |
| torch | 32k | 1024 | 0 | 187.5 | 125 | 2.598 | 4.231 | 50.5 |
| liger | 32k | 2048 | 0 | 0.01 | 252 | 1.146 | 4.645 | 228.8 |
| torch | 32k | 2048 | 0 | 378 | 252 | 5.103 | 8.395 | 51.4 |
| compile_torch | 32k | 2048 | 0 | 0 | 126 | 1.143 | 3.301 | 229.3 |
| liger | 32k | 3072 | 0 | 0.01 | 376 | 1.663 | 6.86 | 236.5 |
| torch | 32k | 3072 | 0 | 564 | 376 | 7.621 | 12.58 | 51.6 |
| liger | 32k | 4096 | 0 | 0.02 | 500 | 2.179 | 9.322 | 240.6 |
| torch | 32k | 4096 | 0 | 750 | 500 | 10.09 | 16.76 | 52 |
| liger | Llama-3 128k | 256 | 0 | 0 | 125.2 | 0.6307 | 2.421 | 208.2 |
| torch | Llama-3 128k | 256 | 0 | 187.9 | 125.2 | 2.602 | 4.237 | 50.5 |
| liger | Llama-3 128k | 1024 | 0 | 0 | 501 | 2.189 | 9.129 | 240 |
| torch | Llama-3 128k | 1024 | 0 | 751.5 | 501 | 10.12 | 16.78 | 51.9 |
| compile_torch | Llama-3 128k | 1024 | 0 | 0 | 250.5 | 2.18 | 6.635 | 241 |
| liger | Llama-3 128k | 2048 | 0 | 0.01 | 1004 | 4.255 | 18.18 | 246.9 |
| torch | Llama-3 128k | 2048 | 0 | 1506 | 1004 | 20.19 | 33.43 | 52 |
| - | Llama-3 128k | 3072 | OOM | - | - | - | - | - |
| - | Llama-3 128k | 4096 | OOM | - | - | - | - | - |
| liger | Qwen3 152k | 256 | 0 | 0 | 148.4 | 0.7294 | 2.833 | 213.3 |
| torch | Qwen3 152k | 256 | 0 | 222.6 | 148.4 | 3.072 | 5.023 | 50.6 |
| liger | Qwen3 152k | 1024 | 0 | 0 | 593.5 | 2.575 | 10.81 | 241.7 |
| torch | Qwen3 152k | 1024 | 0 | 890.2 | 593.5 | 11.99 | 19.88 | 51.9 |
| compile_torch | Qwen3 152k | 1024 | 0 | 0.01 | 296.8 | 2.568 | 7.863 | 242.3 |
| - | Qwen3 152k | 2048 | OOM | - | - | - | - | - |
| - | Qwen3 152k | 3072 | OOM | - | - | - | - | - |
| - | Qwen3 152k | 4096 | est_peak_over_budget | - | - | - | - | - |

## reduction=none fp32 loss tile

| impl | BT | V | held | peak_f | ms_fwd | expected_tile |
|---|---|---|---|---|---|---|
| liger | 64 | 32000 | 7.81 | 7.81 | 0.1332 | 7.81 |
| torch | 64 | 32000 | 7.81 | 23.44 | 0.2008 | 7.81 |
| liger | 256 | 32000 | 32 | 32 | 0.569 | 31.25 |
| torch | 256 | 32000 | 32 | 96 | 1.139 | 31.25 |
| liger | 64 | 128256 | 32 | 32 | 0.5571 | 31.31 |
| torch | 64 | 128256 | 32 | 96 | 1.145 | 31.31 |
| liger | 128 | 128256 | 62.62 | 62.62 | 1.129 | 62.62 |
| torch | 128 | 128256 | 62.62 | 187.9 | 2.273 | 62.62 |

