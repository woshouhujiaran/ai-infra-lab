Liger-Kernel 1ab779ff09, torch 2.14.1+cu126, triton 3.8.0, NVIDIA GeForce RTX 4060 Laptop GPU (memory cap 0.6)

## Numerics (1024 tokens, bf16 inputs, targets sampled from the model's softmax, logit std 3; reference = fp32 math on the same bf16 inputs)

"strict" = rerun with `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False` (PyTorch's default is True, which lets cuBLAS do split-K reductions in bf16).

| shape | variant | loss | loss abs err | dX rel err | dW rel err | dX rel err strict | dW rel err strict |
|---|---|---|---|---|---|---|---|
| Qwen3-0.6B head | fp32 ref rounded to bf16 | 7.529891 | 0.0 | 0.001653 | 0.001439 | 0.001653 | 0.001439 |
| Qwen3-0.6B head | torch_bf16 | 7.53125 | 0.00136 | 0.003567 | 0.002526 | 0.002693 | 0.002526 |
| Qwen3-0.6B head | torch_fp32 | 7.529796 | 9.54e-05 | 0.003508 | 0.002475 | 0.002617 | 0.002475 |
| Qwen3-0.6B head | torch_compile | 7.529796 | 9.58e-05 | 0.003508 | 0.002475 | 0.002617 | 0.002475 |
| Qwen3-0.6B head | liger_ce | 7.53125 | 0.00136 | 0.003508 | 0.002475 | 0.002617 | 0.002475 |
| Qwen3-0.6B head | liger_flce | 7.529796 | 9.58e-05 | 0.002615 | 0.002666 | 0.002615 | 0.002666 |
| Qwen3-0.6B head | liger_flce_c8 | 7.5298 | 9.16e-05 | 0.003098 | 0.002631 | 0.002614 | 0.002631 |
| Qwen3-0.6B head | liger_flce_fp32acc | 7.529796 | 9.58e-05 | 0.002615 | 0.002475 | 0.002615 | 0.002475 |
| Llama-3.2-1B head | fp32 ref rounded to bf16 | 7.43311 | 0.0 | 0.001662 | 0.001448 | 0.001662 | 0.001448 |
| Llama-3.2-1B head | torch_bf16 | 7.4375 | 0.00439 | 0.003566 | 0.002493 | 0.00265 | 0.002493 |
| Llama-3.2-1B head | torch_fp32 | 7.433333 | 0.000223 | 0.003515 | 0.002434 | 0.00258 | 0.002434 |
| Llama-3.2-1B head | torch_compile | 7.433333 | 0.000223 | 0.003515 | 0.002434 | 0.00258 | 0.002434 |
| Llama-3.2-1B head | liger_ce | 7.4375 | 0.00439 | 0.003515 | 0.002434 | 0.00258 | 0.002434 |
| Llama-3.2-1B head | liger_flce | 7.433333 | 0.000223 | 0.00258 | 0.002621 | 0.00258 | 0.002621 |
| Llama-3.2-1B head | liger_flce_c8 | 7.433333 | 0.000223 | 0.01178 | 0.002582 | 0.002579 | 0.002582 |
| Llama-3.2-1B head | liger_flce_fp32acc | 7.433333 | 0.000223 | 0.00258 | 0.002434 | 0.00258 | 0.002434 |
| h1024 / 32k vocab | fp32 ref rounded to bf16 | 6.251103 | 0.0 | 0.001653 | 0.001648 | 0.001653 | 0.001648 |
| h1024 / 32k vocab | torch_bf16 | 6.25 | 0.0011 | 0.003973 | 0.003187 | 0.00321 | 0.003187 |
| h1024 / 32k vocab | torch_fp32 | 6.251323 | 0.00022 | 0.003893 | 0.00311 | 0.003109 | 0.00311 |
| h1024 / 32k vocab | torch_compile | 6.251323 | 0.00022 | 0.003893 | 0.00311 | 0.003109 | 0.00311 |
| h1024 / 32k vocab | liger_ce | 6.25 | 0.0011 | 0.003893 | 0.00311 | 0.003109 | 0.00311 |
| h1024 / 32k vocab | liger_flce | 6.251333 | 0.00023 | 0.003539 | 0.003843 | 0.003122 | 0.003843 |
| h1024 / 32k vocab | liger_flce_c8 | 6.251323 | 0.00022 | 0.005268 | 0.003464 | 0.00311 | 0.003464 |
| h1024 / 32k vocab | liger_flce_fp32acc | 6.251333 | 0.00023 | 0.003539 | 0.003123 | 0.003122 | 0.003123 |

## One training step of LM head + CE (forward + backward; 2 warmup, 10 timed, CUDA events)

Extra peak = max allocated during the step minus allocated before it (x, W, targets); it includes the dX and dW that the step returns.

| shape | tokens | bf16 logits MiB | dX+dW MiB | variant | chunks | extra peak MiB | ms/step | speedup vs torch_fp32 |
|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B head | 1024 | 296.8 | 298.8 | torch_bf16 |  | 890.3 | 41.76 | 1.408 |
| Qwen3-0.6B head | 1024 | 296.8 | 298.8 | torch_fp32 |  | 1782.0 | 58.805 | 1.0 |
| Qwen3-0.6B head | 1024 | 296.8 | 298.8 | torch_compile |  | 595.5 | 35.76 | 1.644 |
| Qwen3-0.6B head | 1024 | 296.8 | 298.8 | liger_ce |  | 595.5 | 35.125 | 1.674 |
| Qwen3-0.6B head | 1024 | 296.8 | 298.8 | liger_flce | 128 | 301.1 | 674.682 | 0.087 |
| Qwen3-0.6B head | 1024 | 296.8 | 298.8 | liger_flce_c8 | 16 | 317.3 | 91.847 | 0.64 |
| Qwen3-0.6B head | 1024 | 296.8 | 298.8 | liger_flce_fp32acc | 128 | 892.8 | 1036.502 | 0.057 |
| Qwen3-0.6B head | 4096 | 1187.0 | 304.8 | torch_bf16 |  | 3564.0 | 159.93 |  |
| Qwen3-0.6B head | 4096 | 1187.0 | 304.8 | torch_fp32 |  | OOM | OOM |  |
| Qwen3-0.6B head | 4096 | 1187.0 | 304.8 | torch_compile |  | 1492.8 | 140.749 |  |
| Qwen3-0.6B head | 4096 | 1187.0 | 304.8 | liger_ce |  | 1492.8 | 137.093 |  |
| Qwen3-0.6B head | 4096 | 1187.0 | 304.8 | liger_flce | 128 | 314.0 | 699.44 |  |
| Qwen3-0.6B head | 4096 | 1187.0 | 304.8 | liger_flce_c8 | 16 | 379.0 | 151.576 |  |
| Qwen3-0.6B head | 4096 | 1187.0 | 304.8 | liger_flce_fp32acc | 128 | 898.8 | 1051.05 |  |
| Qwen3-0.6B head | 16384 | 4748.0 | 328.8 | torch_bf16 |  | OOM | OOM |  |
| Qwen3-0.6B head | 16384 | 4748.0 | 328.8 | torch_fp32 |  | OOM | OOM |  |
| Qwen3-0.6B head | 16384 | 4748.0 | 328.8 | torch_compile |  | OOM | OOM |  |
| Qwen3-0.6B head | 16384 | 4748.0 | 328.8 | liger_ce |  | OOM | OOM |  |
| Qwen3-0.6B head | 16384 | 4748.0 | 328.8 | liger_flce | 128 | 365.9 | 775.749 |  |
| Qwen3-0.6B head | 16384 | 4748.0 | 328.8 | liger_flce_c8 | 16 | 625.6 | 585.124 |  |
| Qwen3-0.6B head | 16384 | 4748.0 | 328.8 | liger_flce_fp32acc | 128 | 922.8 | 1143.975 |  |
| Llama-3.2-1B head | 1024 | 250.5 | 505.0 | torch_bf16 |  | 756.5 | 61.632 | 1.221 |
| Llama-3.2-1B head | 1024 | 250.5 | 505.0 | torch_fp32 |  | 1506.0 | 75.245 | 1.0 |
| Llama-3.2-1B head | 1024 | 250.5 | 505.0 | torch_compile |  | 756.5 | 60.316 | 1.248 |
| Llama-3.2-1B head | 1024 | 250.5 | 505.0 | liger_ce |  | 756.5 | 58.079 | 1.296 |
| Llama-3.2-1B head | 1024 | 250.5 | 505.0 | liger_flce | 32 | 513.8 | 286.994 | 0.262 |
| Llama-3.2-1B head | 1024 | 250.5 | 505.0 | liger_flce_c8 | 8 | 537.3 | 77.786 | 0.967 |
| Llama-3.2-1B head | 1024 | 250.5 | 505.0 | liger_flce_fp32acc | 32 | 1508.0 | 437.333 | 0.172 |
| Llama-3.2-1B head | 4096 | 1002.0 | 517.0 | torch_bf16 |  | 3006.0 | 247.538 |  |
| Llama-3.2-1B head | 4096 | 1002.0 | 517.0 | torch_fp32 |  | OOM | OOM |  |
| Llama-3.2-1B head | 4096 | 1002.0 | 517.0 | torch_compile |  | 1520.0 | 238.127 |  |
| Llama-3.2-1B head | 4096 | 1002.0 | 517.0 | liger_ce |  | 1520.0 | 236.367 |  |
| Llama-3.2-1B head | 4096 | 1002.0 | 517.0 | liger_flce | 32 | 549.3 | 308.568 |  |
| Llama-3.2-1B head | 4096 | 1002.0 | 517.0 | liger_flce_c8 | 8 | 644.0 | 243.991 |  |
| Llama-3.2-1B head | 4096 | 1002.0 | 517.0 | liger_flce_fp32acc | 32 | 1520.0 | 461.87 |  |
| Llama-3.2-1B head | 16384 | 4008.0 | 565.0 | torch_bf16 |  | OOM | OOM |  |
| Llama-3.2-1B head | 16384 | 4008.0 | 565.0 | torch_fp32 |  | OOM | OOM |  |
| Llama-3.2-1B head | 16384 | 4008.0 | 565.0 | torch_compile |  | OOM | OOM |  |
| Llama-3.2-1B head | 16384 | 4008.0 | 565.0 | liger_ce |  | OOM | OOM |  |
| Llama-3.2-1B head | 16384 | 4008.0 | 565.0 | liger_flce | 32 | 692.1 | 973.172 |  |
| Llama-3.2-1B head | 16384 | 4008.0 | 565.0 | liger_flce_c8 | 8 | 1068.1 | 977.918 |  |
| Llama-3.2-1B head | 16384 | 4008.0 | 565.0 | liger_flce_fp32acc | 32 | 1568.1 | 1008.465 |  |
| h1024 / 32k vocab | 1024 | 62.5 | 64.5 | torch_bf16 |  | 187.5 | 8.52 | 1.376 |
| h1024 / 32k vocab | 1024 | 62.5 | 64.5 | torch_fp32 |  | 375.0 | 11.723 | 1.0 |
| h1024 / 32k vocab | 1024 | 62.5 | 64.5 | torch_compile |  | 127.0 | 7.528 | 1.557 |
| h1024 / 32k vocab | 1024 | 62.5 | 64.5 | liger_ce |  | 127.0 | 7.896 | 1.485 |
| h1024 / 32k vocab | 1024 | 62.5 | 64.5 | liger_flce | 32 | 66.5 | 36.655 | 0.32 |
| h1024 / 32k vocab | 1024 | 62.5 | 64.5 | liger_flce_c8 | 4 | 80.1 | 8.456 | 1.386 |
| h1024 / 32k vocab | 1024 | 62.5 | 64.5 | liger_flce_fp32acc | 32 | 189.5 | 55.92 | 0.21 |
| h1024 / 32k vocab | 4096 | 250.0 | 70.5 | torch_bf16 |  | 750.0 | 33.902 | 1.391 |
| h1024 / 32k vocab | 4096 | 250.0 | 70.5 | torch_fp32 |  | 1500.0 | 47.146 | 1.0 |
| h1024 / 32k vocab | 4096 | 250.0 | 70.5 | torch_compile |  | 320.5 | 30.444 | 1.549 |
| h1024 / 32k vocab | 4096 | 250.0 | 70.5 | liger_ce |  | 320.5 | 30.788 | 1.531 |
| h1024 / 32k vocab | 4096 | 250.0 | 70.5 | liger_flce | 32 | 78.3 | 41.725 | 1.13 |
| h1024 / 32k vocab | 4096 | 250.0 | 70.5 | liger_flce_c8 | 4 | 133.0 | 31.072 | 1.517 |
| h1024 / 32k vocab | 4096 | 250.0 | 70.5 | liger_flce_fp32acc | 32 | 195.5 | 61.23 | 0.77 |
| h1024 / 32k vocab | 16384 | 1000.0 | 94.5 | torch_bf16 |  | 3000.0 | 141.974 |  |
| h1024 / 32k vocab | 16384 | 1000.0 | 94.5 | torch_fp32 |  | OOM | OOM |  |
| h1024 / 32k vocab | 16384 | 1000.0 | 94.5 | torch_compile |  | 1094.5 | 128.456 |  |
| h1024 / 32k vocab | 16384 | 1000.0 | 94.5 | liger_ce |  | 1094.5 | 125.204 |  |
| h1024 / 32k vocab | 16384 | 1000.0 | 94.5 | liger_flce | 32 | 125.8 | 128.852 |  |
| h1024 / 32k vocab | 16384 | 1000.0 | 94.5 | liger_flce_c8 | 4 | 344.6 | 127.316 |  |
| h1024 / 32k vocab | 16384 | 1000.0 | 94.5 | liger_flce_fp32acc | 32 | 219.6 | 135.786 |  |
