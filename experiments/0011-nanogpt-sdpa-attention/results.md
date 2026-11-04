nanoGPT 3adf61e154, torch 2.14.1+cu126, cuDNN 91002, FlashAttention compiled: False, NVIDIA GeForce RTX 4060 Laptop GPU (memory cap 0.6), runtime 99.2 s

## SDPA dispatch (fwd+bwd at T=1024, B=16; top CUDA kernels by device time)

| mode | variant | backend | top kernels |
|---|---|---|---|
| bf16_autocast | linears_only | no attention kernels | ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_nt<br>ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_nn<br>ampere_bf16_s1688gemm_bf16_128x128_ldg8_relu_f2f_tn |
| bf16_autocast | sdpa_default | efficient (CUTLASS mem-efficient) | _Z52fmha_cutlassB_bf16_aligned_64x64_k64_seqaligned_sm80N22PyTorchMemEffAttention23Attenti<br>ampere_bf16_s1688gemm_bf16_128x128_ldg8_relu_f2f_tn<br>ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_nt |
| bf16_autocast | sdpa_cudnn | cudnn | cudnn_generated_fort_native_sdpa_sm80_flash_bprop_wmma_f16_knob_28_64x128x64_1x4x1_kernel0<br>ampere_bf16_s1688gemm_bf16_128x128_ldg8_relu_f2f_tn<br>ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_nt |
| bf16_autocast | sdpa_math | unfused (bmm + softmax) | _ZN2at6native29vectorized_elementwise_kernelILi4ENS0_13BinaryFunctorIfffNS0_15binary_inter<br>_ZN43_GLOBAL__N__bcfaaeff_10_SoftMax_cu_0af7be2121softmax_warp_backwardIfffLi10ELb0ELb0ELi<br>_ZN43_GLOBAL__N__bcfaaeff_10_SoftMax_cu_0af7be2120softmax_warp_forwardIfffLi10ELb0ELb0ELi3 |
| bf16_autocast | manual | unfused (bmm + softmax) | _ZN2at6native29vectorized_elementwise_kernelILi4EZNS0_25bfloat16_copy_kernel_cudaERNS_18Te<br>_ZN2at6native27unrolled_elementwise_kernelIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIt<br>_ZN2at6native29vectorized_elementwise_kernelILi4ENS0_13BinaryFunctorIfffNS0_15binary_inter |
| bf16_autocast | manual_compile | inductor triton (unfused attention) | triton_per_fused__softmax__softmax_backward_data__to_490d8951_4<br>ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_nn<br>_ZN7cutlass7Kernel2I66cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_nt_align8EE |
| fp32_tf32 | linears_only | no attention kernels | _ZN7cutlass7Kernel2I52cutlass_80_tensorop_s1688gemm_128x128_32x3_nt_align4EEvNT_6ParamsE<br>_ZN7cutlass7Kernel2I52cutlass_80_tensorop_s1688gemm_128x128_32x3_nn_align4EEvNT_6ParamsE<br>_ZN7cutlass7Kernel2I52cutlass_80_tensorop_s1688gemm_128x256_16x3_tn_align4EEvNT_6ParamsE |
| fp32_tf32 | sdpa_default | efficient (CUTLASS mem-efficient) | _Z40fmha_cutlassB_f32_aligned_64x64_k64_sm80N22PyTorchMemEffAttention23AttentionBackwardKe<br>_Z39fmha_cutlassF_f32_aligned_64x64_rf_sm80N22PyTorchMemEffAttention15AttentionKernelIfN7c<br>_ZN7cutlass7Kernel2I52cutlass_80_tensorop_s1688gemm_128x128_32x3_nt_align4EEvNT_6ParamsE |
| fp32_tf32 | sdpa_cudnn | unsupported | No available kernel. Aborting execution. |
| fp32_tf32 | sdpa_math | unfused (bmm + softmax) | _ZN2at6native29vectorized_elementwise_kernelILi4ENS0_13BinaryFunctorIfffNS0_15binary_inter<br>_ZN43_GLOBAL__N__bcfaaeff_10_SoftMax_cu_0af7be2121softmax_warp_backwardIfffLi10ELb0ELb0ELi<br>_ZN43_GLOBAL__N__bcfaaeff_10_SoftMax_cu_0af7be2120softmax_warp_forwardIfffLi10ELb0ELb0ELi3 |
| fp32_tf32 | manual | unfused (bmm + softmax) | _ZN2at6native29vectorized_elementwise_kernelILi4ENS0_13AUnaryFunctorIfffNS0_15binary_inter<br>_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_44_GLOBAL<br>Memcpy DtoD (Device -> Device) |

## bf16_autocast: latency (ms, `do_bench` median) and extra peak memory (MiB) at 16384 tokens per batch

| T | B | variant | fwd ms | fwd MiB | fwd+bwd ms | fwd+bwd MiB | fwd x manual | fwd+bwd x manual | mask buffer MiB |
|---|---|---|---|---|---|---|---|---|---|
| 256 | 64 | linears_only | 3.204 | 124.6 | 9.945 | 197.8 | 3.82 | 3.25 |  |
| 256 | 64 | sdpa_default | 3.437 | 148.6 | 11.605 | 295.3 | 3.56 | 2.78 |  |
| 256 | 64 | sdpa_cudnn | 3.434 | 148.6 | 12.36 | 343.3 | 3.56 | 2.61 |  |
| 256 | 64 | sdpa_math | 16.979 | 723.9 | 36.589 | 965.8 | 0.72 | 0.88 |  |
| 256 | 64 | manual | 12.242 | 579.5 | 32.279 | 893.8 | 1.0 | 1.0 | 0.2 |
| 256 | 64 | manual_compile | 5.935 | 169.1 | 16.104 | 367.3 | 2.06 | 2.0 | 0.2 |
| 512 | 32 | linears_only | 3.204 | 124.6 | 9.945 | 197.8 | 6.21 | 5.2 |  |
| 512 | 32 | sdpa_default | 3.639 | 148.6 | 12.866 | 295.3 | 5.47 | 4.02 |  |
| 512 | 32 | sdpa_cudnn | 3.587 | 148.6 | 13.667 | 343.3 | 5.55 | 3.79 |  |
| 512 | 32 | sdpa_math | 26.384 | 1156.7 | 54.398 | 1733.8 | 0.75 | 0.95 |  |
| 512 | 32 | manual | 19.898 | 1059.5 | 51.744 | 1662.0 | 1.0 | 1.0 | 1.0 |
| 512 | 32 | manual_compile | 7.629 | 265.1 | 20.748 | 559.3 | 2.61 | 2.49 | 1.0 |
| 1024 | 16 | linears_only | 3.207 | 124.6 | 9.946 | 197.8 | 11.02 | 9.12 |  |
| 1024 | 16 | sdpa_default | 4.186 | 148.6 | 15.612 | 295.3 | 8.45 | 5.81 |  |
| 1024 | 16 | sdpa_cudnn | 4.038 | 148.6 | 15.045 | 343.3 | 8.76 | 6.03 |  |
| 1024 | 16 | sdpa_math | 45.202 | 2023.7 | 89.92 | 3269.8 | 0.78 | 1.01 |  |
| 1024 | 16 | manual | 35.353 | 2019.5 | 90.733 | 3198.8 | 1.0 | 1.0 | 4.0 |
| 1024 | 16 | manual_compile | 11.027 | 457.1 | 30.1 | 943.3 | 3.21 | 3.01 | 4.0 |
| 2048 | 8 | linears_only | 3.215 | 124.6 | 9.975 | 197.8 | 20.59 |  |  |
| 2048 | 8 | sdpa_default | 5.289 | 148.6 | 21.352 | 295.3 | 12.51 |  |  |
| 2048 | 8 | sdpa_cudnn | 4.95 | 148.6 | 18.647 | 343.3 | 13.37 |  |  |
| 2048 | 8 | sdpa_math | 90.078 | 3763.7 | OOM | OOM | 0.73 |  |  |
| 2048 | 8 | manual | 66.189 | 3939.5 | OOM | OOM | 1.0 |  | 16.0 |
| 2048 | 8 | manual_compile | 22.811 | 841.1 | 67.539 | 3199.3 | 2.9 |  | 16.0 |
| 4096 | 4 | linears_only | 3.21 | 124.6 | 9.976 | 197.8 |  |  |  |
| 4096 | 4 | sdpa_default | 7.479 | 148.6 | 33.094 | 295.3 |  |  |  |
| 4096 | 4 | sdpa_cudnn | 6.777 | 148.6 | 25.708 | 343.3 |  |  |  |
| 4096 | 4 | sdpa_math | OOM | OOM | OOM | OOM |  |  |  |
| 4096 | 4 | manual | OOM | OOM | OOM | OOM |  |  | 64.0 |
| 4096 | 4 | manual_compile | 43.047 | 1609.1 | OOM | OOM |  |  | 64.0 |
| 8192 | 2 | linears_only | 3.21 | 124.6 | 9.976 | 197.8 |  |  |  |
| 8192 | 2 | sdpa_default | 11.843 | 148.6 | 55.774 | 295.3 |  |  |  |
| 8192 | 2 | sdpa_cudnn | 10.465 | 148.6 | 39.833 | 343.3 |  |  |  |
| 8192 | 2 | sdpa_math | OOM | OOM | OOM | OOM |  |  |  |
| 8192 | 2 | manual | OOM | OOM | OOM | OOM |  |  | 256.0 |
| 8192 | 2 | manual_compile | OOM | OOM | OOM | OOM |  |  | 256.0 |
| 16384 | 1 | linears_only | 3.209 | 124.6 | 9.978 | 197.8 |  |  |  |
| 16384 | 1 | sdpa_default | 20.519 | 148.6 | 103.327 | 295.3 |  |  |  |
| 16384 | 1 | sdpa_cudnn | 17.703 | 148.6 | 67.948 | 343.3 |  |  |  |
| 16384 | 1 | sdpa_math | OOM | OOM | OOM | OOM |  |  |  |
| 16384 | 1 | manual | OOM | OOM | OOM | OOM |  |  | 1024.0 |
| 16384 | 1 | manual_compile | OOM | OOM | OOM | OOM |  |  | 1024.0 |

## fp32_tf32: latency (ms, `do_bench` median) and extra peak memory (MiB) at 16384 tokens per batch

| T | B | variant | fwd ms | fwd MiB | fwd+bwd ms | fwd+bwd MiB | fwd x manual | fwd+bwd x manual | mask buffer MiB |
|---|---|---|---|---|---|---|---|---|---|
| 256 | 64 | linears_only | 7.82 | 192.0 | 18.929 | 338.3 | 2.12 | 2.43 |  |
| 256 | 64 | sdpa_default | 7.411 | 240.0 | 27.385 | 484.5 | 2.24 | 1.68 |  |
| 256 | 64 | sdpa_cudnn | unsupported | unsupported | unsupported | unsupported |  |  |  |
| 256 | 64 | sdpa_math | 16.935 | 624.4 | 41.995 | 962.3 | 0.98 | 1.1 |  |
| 256 | 64 | manual | 16.596 | 528.1 | 46.077 | 962.3 | 1.0 | 1.0 | 0.2 |
| 512 | 32 | linears_only | 5.736 | 192.0 | 18.94 | 338.3 | 4.41 | 3.61 |  |
| 512 | 32 | sdpa_default | 9.002 | 240.0 | 33.864 | 484.5 | 2.81 | 2.02 |  |
| 512 | 32 | sdpa_cudnn | unsupported | unsupported | unsupported | unsupported |  |  |  |
| 512 | 32 | sdpa_math | 25.055 | 1057.2 | 58.506 | 1730.3 | 1.01 | 1.17 |  |
| 512 | 32 | manual | 25.282 | 912.2 | 68.374 | 1730.5 | 1.0 | 1.0 | 1.0 |
| 1024 | 16 | linears_only | 5.725 | 192.0 | 18.926 | 338.3 | 7.46 | 5.97 |  |
| 1024 | 16 | sdpa_default | 12.176 | 240.0 | 46.884 | 484.5 | 3.51 | 2.41 |  |
| 1024 | 16 | sdpa_cudnn | unsupported | unsupported | unsupported | unsupported |  |  |  |
| 1024 | 16 | sdpa_math | 41.287 | 1924.2 | 91.608 | 3266.3 | 1.03 | 1.23 |  |
| 1024 | 16 | manual | 42.703 | 1681.0 | 113.008 | 3267.3 | 1.0 | 1.0 | 4.0 |
| 2048 | 8 | linears_only | 5.734 | 192.0 | 18.924 | 338.3 |  |  |  |
| 2048 | 8 | sdpa_default | 18.514 | 240.0 | 73.082 | 484.5 |  |  |  |
| 2048 | 8 | sdpa_cudnn | unsupported | unsupported | unsupported | unsupported |  |  |  |
| 2048 | 8 | sdpa_math | 80.962 | 3664.2 | OOM | OOM |  |  |  |
| 2048 | 8 | manual | OOM | OOM | OOM | OOM |  |  | 16.0 |
| 4096 | 4 | linears_only | 5.743 | 192.0 | 18.931 | 338.3 |  |  |  |
| 4096 | 4 | sdpa_default | 31.206 | 240.0 | 125.381 | 484.5 |  |  |  |
| 4096 | 4 | sdpa_cudnn | unsupported | unsupported | unsupported | unsupported |  |  |  |
| 4096 | 4 | sdpa_math | OOM | OOM | OOM | OOM |  |  |  |
| 4096 | 4 | manual | OOM | OOM | OOM | OOM |  |  | 64.0 |
| 8192 | 2 | linears_only | 5.745 | 192.0 | 18.942 | 338.3 |  |  |  |
| 8192 | 2 | sdpa_default | 56.423 | 240.0 | 229.994 | 484.5 |  |  |  |
| 8192 | 2 | sdpa_cudnn | unsupported | unsupported | unsupported | unsupported |  |  |  |
| 8192 | 2 | sdpa_math | OOM | OOM | OOM | OOM |  |  |  |
| 8192 | 2 | manual | OOM | OOM | OOM | OOM |  |  | 256.0 |
| 16384 | 1 | linears_only | 5.746 | 192.0 | 18.952 | 338.3 |  |  |  |
| 16384 | 1 | sdpa_default | 106.631 | 240.0 | 439.117 | 484.5 |  |  |  |
| 16384 | 1 | sdpa_cudnn | unsupported | unsupported | unsupported | unsupported |  |  |  |
| 16384 | 1 | sdpa_math | OOM | OOM | OOM | OOM |  |  |  |
| 16384 | 1 | manual | OOM | OOM | OOM | OOM |  |  | 1024.0 |

## Training with attention dropout 0.1 (bf16 autocast, fwd+bwd)

| T | B | variant | backend | fwd+bwd ms | fwd+bwd MiB |
|---|---|---|---|---|---|
| 1024 | 16 | sdpa_default | efficient (CUTLASS mem-efficient) | 21.421 | 295.3 |
| 1024 | 16 | sdpa_cudnn | unsupported | unsupported | unsupported |
| 1024 | 16 | manual | unfused (bmm + softmax) | 107.06 | 3198.8 |
| 1024 | 16 | manual_compile | inductor triton (unfused attention) | 32.075 | 1158.0 |
| 4096 | 4 | sdpa_default | efficient (CUTLASS mem-efficient) | 52.091 | 295.3 |
| 4096 | 4 | sdpa_cudnn | unsupported | unsupported | unsupported |
| 4096 | 4 | manual | OOM | OOM | OOM |
| 4096 | 4 | manual_compile | OOM | OOM | OOM |

## Error vs fp64 manual attention (T=1024, B=2; relative L2 error)

| mode | variant | y rel | y max abs | dx rel | dW_qkv rel |
|---|---|---|---|---|---|
| bf16_autocast | sdpa_default | 0.00387 | 0.00522 | 0.00481 | 0.00493 |
| bf16_autocast | sdpa_cudnn | 0.00387 | 0.00522 | 0.00476 | 0.0049 |
| bf16_autocast | sdpa_math | 0.00374 | 0.00522 | 0.0045 | 0.00465 |
| bf16_autocast | manual | 0.00394 | 0.00522 | 0.00482 | 0.00496 |
| bf16_autocast | manual_compile | 0.00394 | 0.00522 | 0.00482 | 0.00496 |
| fp32_tf32 | sdpa_default | 0.000359 | 0.000516 | 0.000465 | 0.000465 |
| fp32_tf32 | sdpa_cudnn | unsupported |  |  |  |
| fp32_tf32 | sdpa_math | 0.000424 | 0.000516 | 0.000564 | 0.000564 |
| fp32_tf32 | manual | 0.000424 | 0.000516 | 0.000564 | 0.000564 |
