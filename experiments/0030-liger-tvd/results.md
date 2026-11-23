# 0030 — Liger-Kernel Total Variation Distance results

GPU: NVIDIA GeForce RTX 4060 Laptop GPU | torch 2.14.1+cu126 | seed 0

## Numerics (vs fp64 analytic + eager torch)

| kind | dtype | red | floor | liger_loss_rel | torch_loss_rel | liger_grad_rel | torch_grad_rel | liger_dtype |
|---|---|---|---|---|---|---|---|---|
| probs | fp32 | batchmean | 1.444e-08 | 1.444e-08 | 6.045e-08 | 0 | 0 | float32 |
| probs | fp32 | sum | 1.444e-08 | 1.444e-08 | 6.045e-08 | 0 | 0 | float32 |
| probs | fp32 | mean | 1.364e-08 | 1.364e-08 | 1.364e-08 | 4.750e-08 | 4.750e-08 | float32 |
| probs | fp32 | none | 2.608e-08 | 2.608e-08 | 2.608e-08 | 0 | 0 | float32 |
| probs | fp16 | batchmean | 8.660e-05 | 1.009e-05 | 8.660e-05 | 0 | 0 | float32 |
| probs | fp16 | sum | 8.660e-05 | 1.009e-05 | 8.660e-05 | 0 | 0 | float32 |
| probs | fp16 | mean | 6.402e-04 | 1.010e-05 | 6.402e-04 | 0.02344 | 0.02344 | float32 |
| probs | fp16 | none | 7.707e-04 | 7.707e-04 | 7.707e-04 | 0 | 0 | float32 |
| probs | bf16 | batchmean | 0.001264 | 1.551e-04 | 0.001264 | 0 | 0 | float32 |
| probs | bf16 | sum | 0.001264 | 1.551e-04 | 0.001264 | 0 | 0 | float32 |
| probs | bf16 | mean | 7.670e-04 | 1.550e-04 | 7.670e-04 | 5.493e-04 | 5.493e-04 | float32 |
| probs | bf16 | none | 0.001687 | 0.001687 | 0.001687 | 0 | 0 | float32 |
| raw | fp32 | batchmean | 1.424e-08 | 6.230e-08 | 1.424e-08 | 0 | 0 | float32 |
| raw | fp32 | sum | 1.424e-08 | 6.230e-08 | 1.424e-08 | 0 | 0 | float32 |
| raw | fp32 | mean | 3.217e-08 | 3.217e-08 | 3.217e-08 | 4.750e-08 | 4.750e-08 | float32 |
| raw | fp32 | none | 2.535e-08 | 2.535e-08 | 2.535e-08 | 0 | 0 | float32 |
| raw | fp16 | batchmean | 8.256e-05 | 2.514e-05 | inf | 0 | 0 | float32 |
| raw | fp16 | sum | inf | 2.514e-05 | inf | 0 | 0 | float32 |
| raw | fp16 | mean | 1.035e-04 | 2.511e-05 | inf | 0.02344 | 0.02344 | float32 |
| raw | fp16 | none | 4.186e-05 | 4.186e-05 | 4.186e-05 | 0 | 0 | float32 |
| raw | bf16 | batchmean | 0.002094 | 2.182e-05 | 0.002094 | 0 | 0 | float32 |
| raw | bf16 | sum | 0.002094 | 2.182e-05 | 0.002094 | 0 | 0 | float32 |
| raw | bf16 | mean | 9.968e-04 | 2.186e-05 | 9.968e-04 | 5.493e-04 | 5.493e-04 | float32 |
| raw | bf16 | none | 5.011e-05 | 5.011e-05 | 5.011e-05 | 0 | 0 | float32 |

## ignore_index

| red | n_ign | liger_loss_rel | torch_loss_rel | liger_g_rel | liger_ign_g_max | torch_ign_g_max |
|---|---|---|---|---|---|---|
| batchmean | 25 | 2.877e-08 | 2.877e-08 | 1.304e-08 | 0 | 0 |
| sum | 25 | 5.711e-09 | 5.711e-09 | 0 | 0 | 0 |
| mean | 25 | 2.877e-08 | 2.877e-08 | 1.304e-08 | 0 | 0 |
| none | 25 | 2.591e-08 | 2.591e-08 | 0 | 0 | 0 |

### All labels ignored

| red | liger_loss | torch_loss | liger_g_max | torch_g_max |
|---|---|---|---|---|
| batchmean | ERR:ZeroDivisionError | ERR:RuntimeError | - | - |
| sum | 0 | ERR:RuntimeError | 0 | - |
| mean | ERR:ZeroDivisionError | ERR:RuntimeError | - | - |

## Ties / near-ties

| case | dtype | red | liger_loss | extra |
|---|---|---|---|---|
| self_distill | fp32 | batchmean | 0 | {'torch_loss': 0.0, 'liger_grad_abs_max': 0.0, 'torch_grad_abs_max': 0.0, 'liger_grad_nonzero_frac': 0.0} |
| self_distill | fp32 | none | 0 | {'torch_loss': 0.0, 'liger_grad_abs_max': 0.0, 'torch_grad_abs_max': 0.0, 'liger_grad_nonzero_frac': 0.0} |
| self_distill | fp16 | batchmean | 0 | {'torch_loss': 0.0, 'liger_grad_abs_max': 0.0, 'torch_grad_abs_max': 0.0, 'liger_grad_nonzero_frac': 0.0} |
| self_distill | fp16 | none | 0 | {'torch_loss': 0.0, 'liger_grad_abs_max': 0.0, 'torch_grad_abs_max': 0.0, 'liger_grad_nonzero_frac': 0.0} |
| self_distill | bf16 | batchmean | 0 | {'torch_loss': 0.0, 'liger_grad_abs_max': 0.0, 'torch_grad_abs_max': 0.0, 'liger_grad_nonzero_frac': 0.0} |
| self_distill | bf16 | none | 0 | {'torch_loss': 0.0, 'liger_grad_abs_max': 0.0, 'torch_grad_abs_max': 0.0, 'liger_grad_nonzero_frac': 0.0} |
| half_tied_pad | fp32 | batchmean | 0.2 | {'tied_grad_nonzero_frac': 0.0, 'untied_grad_rel': 0.0, 'liger_grad_rel_vs_ref': 0.0} |
| half_tied_pad | fp16 | batchmean | 0.1972 | {'tied_grad_nonzero_frac': 0.0, 'untied_grad_rel': 0.0, 'liger_grad_rel_vs_ref': 0.0} |
| half_tied_pad | bf16 | batchmean | 0.1997 | {'tied_grad_nonzero_frac': 0.0, 'untied_grad_rel': 0.0, 'liger_grad_rel_vs_ref': 0.0} |
| near_tie_1e-5 | fp16 | batchmean | 0.0205 | {'exact_ties_after_cast': 0, 'tie_frac': 0.0, 'grad_zero_frac': 0.0, 'liger_grad_rel_vs_ref': 0.0} |
| near_tie_1e-5 | bf16 | batchmean | 0.0205 | {'exact_ties_after_cast': 249, 'tie_frac': 0.00189971923828125, 'grad_zero_frac': 0.00189971923828125, 'liger_grad_rel_vs_ref': 0.0} |
| equal_zero_pad_512 | bf16 | batchmean | 0.35 | {'pad_grad_nonzero_frac': 0.0, 'live_grad_nonzero_frac': 0.9979161024093628} |

## Edge cases

| case | liger_loss | torch_loss | liger_finite_g | torch_finite_g | liger_g_nan | loss_rel |
|---|---|---|---|---|---|---|
| probs_ok | 0.3966 | 0.3966 | True | True | False | 7.515e-08 |
| both_zero_pad | 0.3713 | 0.3713 | True | True | False | 8.026e-08 |
| neg_p | 0.4966 | 0.4966 | True | True | False | 1.200e-07 |
| neg_q | 0.4966 | 0.4966 | True | True | False | 0 |
| nan_p | nan | nan | True | True | False | - |
| nan_q | nan | nan | True | True | False | - |
| inf_p | inf | inf | True | True | False | - |
| neg_inf_q | inf | inf | True | True | False | - |
| unnormalized_p | 0.7068 | 0.7068 | True | True | False | 8.433e-08 |
| v1 | 0 | 0 | True | True | False | 0 |
| v32768_multiblock | 0.397 | 0.397 | True | True | False | 7.506e-08 |

## API / autograd

- `last_layer_grad_rel_vs_ref`: 0
- `scaled_2p5_grad_rel_vs_2p5_ref`: 0
- `scaled_over_last`: 2.5
- `none_ones_grad_rel_vs_ref`: 0
- `q_grad_is_none`: True
- `p_grad_is_not_none`: True
- `torch_q_grad_is_none`: False
- `torch_q_grad_abs_max`: 0.01562
- `retain_graph_second_rel`: 0
- `retain_graph_second_abs_max`: 0.01562
- `retain_graph_second_zero`: False
- `grad_output_1_returns_same_storage`: True
- `grad_output_2_returns_same_storage`: False
- `noncontig_q_ok`: True
- `noncontig_q_grad_rel`: 0
- `module_wrapper_grad_rel`: 0
- `module_wrapper_loss`: 0.3972
- `compile_loss_equal`: True
- `compile_grad_equal`: True
- `compile_loss_rel_vs_eager`: 0
- `compile_grad_rel_vs_eager`: 0
- `liger_none_dtype`: float32
- `liger_none_shape`: [32, 4096]
- `liger_batchmean_dtype`: float32
- `torch_none_dtype`: float32
- `torch_batchmean_dtype`: float32
- `liger_bf16_batchmean_dtype`: float32
- `torch_bf16_batchmean_dtype`: bfloat16
- `fp16_mean_grad_nonzero_frac`: 0.998
- `fp16_batchmean_grad_nonzero_frac`: 0.998
- `fp16_mean_grad_abs_max`: 9.537e-07
- `fp16_batchmean_grad_abs_max`: 0.03125
- `fp16_mean_expected_abs`: 9.537e-07
- `bad_label_shape_raises`: True

## Memory / latency (bf16, batchmean)

| impl | vocab | BT | held | peak_f | peak_b | ms_fwd | ms_fb | GB/s |
|---|---|---|---|---|---|---|---|---|
| liger | 32k | 256 | 16 | 16 | 0 | 6.663 | 0.5739 | 4.9 |
| torch | 32k | 256 | 16 | 48 | 48 | 0.5779 | 1.157 | 56.7 |
| liger | 32k | 1024 | 62.5 | 62.5 | 0 | 1.241 | 2.19 | 105.6 |
| torch | 32k | 1024 | 62.5 | 187.5 | 187.5 | 3.257 | 6.544 | 40.2 |
| liger | 32k | 2048 | 126 | 126 | 0 | 1.854 | 3.402 | 141.4 |
| torch | 32k | 2048 | 126 | 378 | 378 | 4.924 | 10.29 | 53.2 |
| compile_torch | 32k | 2048 | 126 | 158 | 158 | 1.634 | 6.969 | 160.4 |
| liger | 32k | 3072 | 188 | 188 | 0 | 3.549 | 6.27 | 110.8 |
| torch | 32k | 3072 | 188 | 564 | 564 | 7.308 | 15.18 | 53.8 |
| liger | 32k | 4096 | 250 | 250 | 0 | 3.629 | 6.386 | 144.5 |
| torch | 32k | 4096 | 250 | 750 | 750 | 9.416 | 19.99 | 55.7 |
| liger | Llama-3 128k | 256 | 62.63 | 62.63 | 0 | 0.9728 | 1.716 | 135 |
| torch | Llama-3 128k | 256 | 62.63 | 187.9 | 187.9 | 2.51 | 5.153 | 52.3 |
| liger | Llama-3 128k | 1024 | 250.5 | 250.5 | 0 | 3.625 | 6.39 | 144.9 |
| torch | Llama-3 128k | 1024 | 250.5 | 751.5 | 751.5 | 9.857 | 20.13 | 53.3 |
| compile_torch | Llama-3 128k | 1024 | 250.5 | 282.5 | 282.5 | 3.088 | 10.63 | 170.1 |
| liger | Llama-3 128k | 2048 | 502 | 502 | 0 | 6.938 | 12.64 | 151.4 |
| torch | Llama-3 128k | 2048 | OOM | - | - | - | - | - |
| - | Llama-3 128k | 3072 | OOM | - | - | - | - | - |
| - | Llama-3 128k | 4096 | OOM | - | - | - | - | - |
| liger | Qwen3 152k | 256 | 74.19 | 74.19 | 0 | 1.088 | 2.129 | 143 |
| torch | Qwen3 152k | 256 | 74.19 | 222.6 | 222.6 | 3.719 | 7.714 | 41.8 |
| liger | Qwen3 152k | 1024 | 296.8 | 296.8 | 0 | 4.376 | 7.803 | 142.2 |
| torch | Qwen3 152k | 1024 | 296.8 | 890.2 | 890.2 | 11.58 | 23.6 | 53.7 |
| compile_torch | Qwen3 152k | 1024 | 296.8 | 328.8 | 328.8 | 2.677 | 12.66 | 232.5 |
| - | Qwen3 152k | 2048 | OOM | - | - | - | - | - |
| - | Qwen3 152k | 3072 | OOM | - | - | - | - | - |
| - | Qwen3 152k | 4096 | est_peak_over_budget | - | - | - | - | - |

## reduction=none fp32 loss tile

| impl | BT | V | held | peak_f | ms_fwd | expected_tile |
|---|---|---|---|---|---|---|
| liger | 64 | 32000 | 8.28 | 12.19 | 0.1192 | 7.81 |
| torch | 64 | 32000 | 3.91 | 7.81 | 0.1106 | 7.81 |
| liger | 256 | 32000 | 32 | 48 | 0.5648 | 31.25 |
| torch | 256 | 32000 | 16 | 32 | 0.3829 | 31.25 |
| liger | 64 | 128256 | 32 | 48 | 0.5278 | 31.31 |
| torch | 64 | 128256 | 16 | 32 | 0.3846 | 31.31 |
| liger | 128 | 128256 | 62.62 | 94.63 | 1.379 | 62.62 |
| torch | 128 | 128256 | 32 | 64 | 1.446 | 62.62 |

