bitsandbytes 0.50.2, torch 2.14.1+cu126, NVIDIA GeForce RTX 4060 Laptop GPU (memory cap 0.6), runtime 654.5 s

## Code books (create_dynamic_map)

| map | min | max | smallest_nonzero_abs | codes_below_1e-3 | codes_above_0.1 |
|---|---|---|---|---|---|
| signed (m) | -0.993 | 1.0 | 5.5e-07 | 30 | 129 |
| unsigned (v) | 0.0 | 1.0 | 3.25e-07 | 30 | 129 |

## torch re-implementation vs CUDA kernel (1048676 elements, 50 steps, grad range 10^3, teacher-forced per step)

| optimizer | weight_decay | m_code_mismatch_frac | v_code_mismatch_frac | max_code_diff | absmax_max_rel_diff | param_max_diff_over_max_update | free_running_drift_rel |
|---|---|---|---|---|---|---|---|
| Adam8bit | 0.0 | 1.335e-06 | 1.467e-05 | 1 | 1.31e-05 | 4.47e-05 | 6.77e-05 |
| AdamW8bit | 0.01 | 1.335e-06 | 1.467e-05 | 1 | 1.31e-05 | 4.449e-05 | 6.77e-05 |

## weight_decay=0.1 semantics (lr 0.01, 20 steps): rel. diff of final params

| optimizer | rel_diff_vs_torch_Adam_L2 | rel_diff_vs_torch_AdamW |
|---|---|---|
| bnb.optim.Adam 32-bit | 0.1135 | 0.0009593 |
| bnb.optim.AdamW 32-bit | 0.1135 | 0.0009593 |
| bnb.optim.Adam8bit | 0.1146 | 0.01523 |

## Optimizer state on nanoGPT GPTConfig() (GPT-2 124M shapes, vocab 50304, tied wte / lm_head): 124475904 params in 148 tensors (98 tensors / 121344 elements below min_8bit_size); no-sync step bitwise equal: True

| optimizer | state_mib | paged_state_mib | bytes_per_param | torch_allocated_delta_mib | nvidia_smi_delta_mib | nvidia_smi_after_delete_mib | page_manager_tensors_after_delete | params_with_fp32_state | checkpoint_mib | step_ms_wall | step_ms_wall_no_per_param_sync |
|---|---|---|---|---|---|---|---|---|---|---|---|
| torch AdamW fused | 949.7 | 0.0 | 8.0 | 951.0 | 1016 | 0 | 0 | 124475904 | 949.8 | 20.101 |  |
| bnb AdamW 32-bit | 949.7 | 0.0 | 8.0 | 950.9 | 1016 | 4 | 0 | 124475904 | 949.8 | 38.015 | 20.161 |
| bnb AdamW8bit | 241.8 | 0.0 | 2.037 | 249.1 | 216 | -1 | 0 | 121344 | 241.9 | 25.341 | 14.564 |
| bnb PagedAdamW8bit | 241.8 | 237.2 | 2.037 | 4.6 | 338 | 334 | 100 | 121344 | 241.9 | 34.553 |  |

## step() latency, one tensor (20 steps after 5 warmup, wall clock)

| dtype | n | optimizer | us_per_step | us_gpu_events | effective_GBps |
|---|---|---|---|---|---|
| float32 | 262144 | torch Adam fused | 73.7 | 71.1 | 99.5 |
| float32 | 262144 | bnb Adam 32-bit | 68.7 | 66.8 | 106.8 |
| float32 | 262144 | bnb Adam8bit | 88.7 | 86.8 | 47.3 |
| float32 | 262144 | bnb PagedAdam8bit | 95.1 | 93.2 | 44.1 |
| float32 | 1048576 | torch Adam fused | 55.0 | 52.5 | 533.8 |
| float32 | 1048576 | bnb Adam 32-bit | 149.6 | 147.0 | 196.3 |
| float32 | 1048576 | bnb Adam8bit | 178.0 | 175.8 | 94.2 |
| float32 | 1048576 | bnb PagedAdam8bit | 204.5 | 202.5 | 82.1 |
| float32 | 4194304 | torch Adam fused | 517.9 | 512.3 | 226.8 |
| float32 | 4194304 | bnb Adam 32-bit | 711.5 | 708.9 | 165.1 |
| float32 | 4194304 | bnb Adam8bit | 613.9 | 611.0 | 109.3 |
| float32 | 4194304 | bnb PagedAdam8bit | 595.9 | 593.8 | 112.6 |
| float32 | 16777216 | torch Adam fused | 2325.1 | 2317.7 | 202.0 |
| float32 | 16777216 | bnb Adam 32-bit | 2481.2 | 2478.9 | 189.3 |
| float32 | 16777216 | bnb Adam8bit | 3440.6 | 3436.8 | 78.0 |
| float32 | 16777216 | bnb PagedAdam8bit | 3148.1 | 3144.4 | 85.3 |
| float32 | 67108864 | torch Adam fused | 8000.7 | 7990.9 | 234.9 |
| float32 | 67108864 | bnb Adam 32-bit | 8567.6 | 8560.6 | 219.3 |
| float32 | 67108864 | bnb Adam8bit | 7643.5 | 7637.0 | 140.5 |
| float32 | 67108864 | bnb PagedAdam8bit | 7748.1 | 7744.2 | 138.6 |
| bfloat16 | 262144 | torch Adam fused | 123.5 | 119.9 | 29.7 |
| bfloat16 | 262144 | bnb Adam 32-bit | 76.4 | 73.3 | 75.5 |
| bfloat16 | 262144 | bnb Adam8bit | 104.5 | 101.6 | 25.1 |
| bfloat16 | 262144 | bnb PagedAdam8bit | 127.7 | 125.4 | 20.5 |
| bfloat16 | 1048576 | torch Adam fused | 71.0 | 68.5 | 206.8 |
| bfloat16 | 1048576 | bnb Adam 32-bit | 198.2 | 194.8 | 116.4 |
| bfloat16 | 1048576 | bnb Adam8bit | 220.2 | 217.7 | 47.6 |
| bfloat16 | 1048576 | bnb PagedAdam8bit | 203.1 | 201.0 | 51.6 |
| bfloat16 | 4194304 | torch Adam fused | 186.2 | 183.1 | 315.3 |
| bfloat16 | 4194304 | bnb Adam 32-bit | 487.4 | 484.4 | 189.3 |
| bfloat16 | 4194304 | bnb Adam8bit | 475.2 | 472.3 | 88.3 |
| bfloat16 | 4194304 | bnb PagedAdam8bit | 497.3 | 494.8 | 84.3 |
| bfloat16 | 16777216 | torch Adam fused | 1062.3 | 1056.0 | 221.1 |
| bfloat16 | 16777216 | bnb Adam 32-bit | 1771.0 | 1767.3 | 208.4 |
| bfloat16 | 16777216 | bnb Adam8bit | 1938.2 | 1934.4 | 86.6 |
| bfloat16 | 16777216 | bnb PagedAdam8bit | 2447.5 | 2443.3 | 68.5 |
| bfloat16 | 67108864 | torch Adam fused | 4081.0 | 4073.0 | 230.2 |
| bfloat16 | 67108864 | bnb Adam 32-bit | 6429.7 | 6426.9 | 229.6 |
| bfloat16 | 67108864 | bnb Adam8bit | 7459.2 | 7455.6 | 90.0 |
| bfloat16 | 67108864 | bnb PagedAdam8bit | 7420.8 | 7418.1 | 90.4 |

## Update error vs in-block gradient dynamic range (1048576 elements, 300 steps, last 50 scored vs fp32 Adam)

| range_decades | variant | update_rel_err | update_cos | v_zero_frac | m_zero_frac | smallest_decile_rel_err | smallest_decile_mag_ratio | largest_decile_rel_err | max_abs_update_over_lr | ref_max_abs_update_over_lr |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | bnb Adam8bit | 0.0389 | 0.9994 | 0.0 | 0.0 | 0.03886 | 0.9789 | 0.03889 | 1.471 | 1.496 |
| 0 | dynamic / tensor-wise | 0.113 | 0.9984 | 0.0 | 1.907e-06 | 0.1131 | 0.8908 | 0.1131 | 1.474 | 1.496 |
| 0 | linear / block 256 | 0.02182 | 0.9998 | 0.0 | 0.009461 | 0.02173 | 0.9938 | 0.022 | 1.512 | 1.496 |
| 0 | linear / tensor-wise | 0.06871 | 0.9992 | 0.0 | 0.0149 | 0.06869 | 0.9369 | 0.06876 | 1.49 | 1.496 |
| 0 | dynamic / block 256, sqrt(v) (fix) | 0.04495 | 0.9993 | 0.0 | 0.0 | 0.04483 | 0.971 | 0.04471 | 1.503 | 1.496 |
| 2 | bnb Adam8bit | 0.1075 | 0.9943 | 0.0 | 1.907e-06 | 0.1795 | 1.029 | 0.03079 | 2.21 | 1.496 |
| 2 | dynamic / tensor-wise | 0.1377 | 0.9912 | 0.0 | 8.583e-06 | 0.2002 | 1.013 | 0.102 | 2.494 | 1.496 |
| 2 | linear / block 256 | 3893.0 | 0.01444 | 0.417 | 0.1221 | 3630.0 | 114.4 | 0.01739 | 3363000.0 | 1.496 |
| 2 | linear / tensor-wise | 5948.0 | 0.01138 | 0.5487 | 0.2334 | 4325.0 | 88.27 | 0.0601 | 5830000.0 | 1.496 |
| 2 | dynamic / block 256, sqrt(v) (fix) | 0.09577 | 0.9954 | 0.0 | 1.907e-06 | 0.1455 | 0.9914 | 0.03421 | 1.972 | 1.496 |
| 3 | bnb Adam8bit | 21.79 | 0.1315 | 0.0008507 | 2.575e-05 | 68.95 | 1.785 | 0.03386 | 36710.0 | 1.496 |
| 3 | dynamic / tensor-wise | 159.1 | 0.01905 | 0.02497 | 8.011e-05 | 503.5 | 21.37 | 0.08872 | 118600.0 | 1.496 |
| 3 | linear / block 256 | 3102.0 | 0.01387 | 0.603 | 0.3687 | 60.81 | 4.421 | 0.01997 | 2237000.0 | 1.496 |
| 3 | linear / tensor-wise | 4637.0 | 0.01076 | 0.6938 | 0.4816 | 3.13 | 4.129 | 0.07496 | 6304000.0 | 1.496 |
| 3 | dynamic / block 256, sqrt(v) (fix) | 0.1349 | 0.9909 | 0.0 | 2.575e-05 | 0.2366 | 0.9986 | 0.03594 | 2.49 | 1.496 |
| 4 | bnb Adam8bit | 226.4 | 0.0583 | 0.1875 | 0.0002232 | 394.3 | 68.84 | 0.03758 | 47380.0 | 1.496 |
| 4 | dynamic / tensor-wise | 320.3 | 0.05753 | 0.2586 | 0.0005922 | 401.5 | 70.57 | 0.07887 | 97500.0 | 1.496 |
| 4 | linear / block 256 | 2750.0 | 0.01282 | 0.6971 | 0.5201 | 3.115 | 4.112 | 0.0236 | 2778000.0 | 1.496 |
| 4 | linear / tensor-wise | 4130.0 | 0.01039 | 0.7685 | 0.6097 | 3.115 | 4.112 | 0.07498 | 5823000.0 | 1.496 |
| 4 | dynamic / block 256, sqrt(v) (fix) | 0.1971 | 0.9809 | 0.0 | 0.0002232 | 0.3824 | 0.9887 | 0.0384 | 3.387 | 1.496 |
| 5 | bnb Adam8bit | 216.3 | 0.09064 | 0.3468 | 0.002184 | 131.1 | 43.68 | 0.04104 | 54730.0 | 1.496 |
| 5 | dynamic / tensor-wise | 300.6 | 0.07777 | 0.4061 | 0.005807 | 132.1 | 45.7 | 0.08164 | 88680.0 | 1.496 |
| 5 | linear / block 256 | 2474.0 | 0.01255 | 0.7543 | 0.612 | 3.028 | 4.004 | 0.02842 | 3842000.0 | 1.496 |
| 5 | linear / tensor-wise | 3036.0 | 0.01043 | 0.8139 | 0.6861 | 3.028 | 4.004 | 0.08267 | 3732000.0 | 1.496 |
| 5 | dynamic / block 256, sqrt(v) (fix) | 0.2902 | 0.9594 | 0.0 | 0.002184 | 0.5932 | 0.9657 | 0.04155 | 5.519 | 1.496 |
| 6 | bnb Adam8bit | 196.2 | 0.101 | 0.4539 | 0.009012 | 45.84 | 27.19 | 0.04308 | 54860.0 | 1.492 |
| 6 | dynamic / tensor-wise | 270.6 | 0.08131 | 0.5049 | 0.008921 | 55.64 | 43.63 | 0.08101 | 80300.0 | 1.492 |
| 6 | linear / block 256 | 2022.0 | 0.01193 | 0.793 | 0.6736 | 2.634 | 3.487 | 0.03483 | 2446000.0 | 1.492 |
| 6 | linear / tensor-wise | 3266.0 | 0.009549 | 0.844 | 0.7373 | 2.634 | 3.487 | 0.5814 | 6387000.0 | 1.492 |
| 6 | dynamic / block 256, sqrt(v) (fix) | 0.4619 | 0.9001 | 0.0 | 0.009012 | 1.078 | 1.092 | 0.04527 | 7.775 | 1.492 |

## Training: nanoGPT GPT 6 layers x 384, 6 heads, block 256, vocab 256, no bias, 1200 steps x 32 x 256 tokens

| run | status | final_val_loss | final_train_loss_avg50 | loss_spikes | ms_per_step | v_zero_frac | block_grad_span_decades_median | block_grad_span_decades_p99 | block_grad_span_decades_max |
|---|---|---|---|---|---|---|---|---|---|
| torch AdamW fused | ok | 1.0125 | 0.7522 | 2 | 38.22 | 5.549e-05 | 0.6787 | 1.453 | 5.399 |
| torch AdamW fused (seed 1) | ok | 1.0204 | 0.7578 | 15 | 39.37 | 5.354e-05 | 0.6757 | 1.484 | 5.556 |
| bnb AdamW 32-bit | ok | 1.0109 | 0.75 | 3 | 47.29 |  |  |  |  |
| bnb AdamW8bit | ok | 1.0194 | 0.7544 | 2 | 43.72 | 4.67e-05 |  |  |  |
| linear int8 / block 256 (ref) | ok | 11.7713 | 10.0301 | 555 | 55.9 |  |  |  |  |
| torch AdamW fused, lr 3e-3 | ok | 1.0609 | 0.7987 | 1 | 41.2 | 0.0009037 | 1.119 | 4.56 | 6.411 |
| bnb AdamW8bit, lr 3e-3 | ok | 1.0509 | 0.7908 | 0 | 44.45 | 0.0001963 |  |  |  |
| linear int8 / block 256 (ref), lr 3e-3 | ok | 8.8984 | 6.0105 | 592 | 50.62 |  |  |  |  |
| dynamic / block 256, sqrt(v) (ref) | ok | 1.0134 | 0.7537 | 3 | 75.92 |  |  |  |  |
| dynamic / block 256, sqrt(v) (ref), lr 3e-3 | ok | 1.0628 | 0.7978 | 1 | 74.48 |  |  |  |  |

v_zero_frac: fraction of v in 8-bit-eligible tensors at the end that is code 0 (bnb) or below 1.63e-07 x block absmax, i.e. would round to code 0 (torch fp32). Block grad span: log10(max / min sqrt(v)) per 256-element block.


Val loss curves:

| run | 200 | 400 | 600 | 800 | 1000 | 1200 |
|---|---|---|---|---|---|---|
| torch AdamW fused | 2.1331 | 1.5215 | 1.2748 | 1.1369 | 1.0486 | 1.0125 |
| torch AdamW fused (seed 1) | 2.108 | 1.5341 | 1.285 | 1.1503 | 1.0572 | 1.0204 |
| bnb AdamW 32-bit | 2.1329 | 1.519 | 1.2613 | 1.1214 | 1.0463 | 1.0109 |
| bnb AdamW8bit | 2.1297 | 1.5317 | 1.272 | 1.1389 | 1.0539 | 1.0194 |
| linear int8 / block 256 (ref) | 87.8069 | 98.1955 | 45.5059 | 89.1413 | 21.7511 | 11.7713 |
| torch AdamW fused, lr 3e-3 | 2.0288 | 1.5463 | 1.3453 | 1.2005 | 1.1064 | 1.0609 |
| bnb AdamW8bit, lr 3e-3 | 1.9928 | 1.5389 | 1.3323 | 1.1799 | 1.0935 | 1.0509 |
| linear int8 / block 256 (ref), lr 3e-3 | 145.7845 | 74.5163 | 58.1834 | 26.0382 | 18.7851 | 8.8984 |
| dynamic / block 256, sqrt(v) (ref) | 2.1335 | 1.5362 | 1.2703 | 1.1334 | 1.0502 | 1.0134 |
| dynamic / block 256, sqrt(v) (ref), lr 3e-3 | 1.9872 | 1.549 | 1.3289 | 1.1954 | 1.1032 | 1.0628 |
