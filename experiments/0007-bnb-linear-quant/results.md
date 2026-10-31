bitsandbytes 0.50.2, torch 2.14.1+cu126, NVIDIA GeForce RTX 4060 Laptop GPU

## Resident weight memory (no bias)

| shape | variant | MiB | vs fp16 | bits/weight |
|---|---|---|---|---|
| qwen3-0.6b up_proj | fp16 | 6.0 | 1.0 | 16.0 |
| qwen3-0.6b up_proj | int8 | 3.012 | 0.502 | 8.031 |
| qwen3-0.6b up_proj | int8_t6 | 3.012 | 0.502 | 8.031 |
| qwen3-0.6b up_proj | fp4 | 1.688 | 0.2813 | 4.501 |
| qwen3-0.6b up_proj | nf4 | 1.688 | 0.2813 | 4.501 |
| qwen3-0.6b up_proj | nf4_dq | 1.551 | 0.2585 | 4.135 |
| llama-7b q_proj | fp16 | 32.0 | 1.0 | 16.0 |
| llama-7b q_proj | int8 | 16.016 | 0.5005 | 8.008 |
| llama-7b q_proj | int8_t6 | 16.016 | 0.5005 | 8.008 |
| llama-7b q_proj | fp4 | 9.0 | 0.2813 | 4.5 |
| llama-7b q_proj | nf4 | 9.0 | 0.2813 | 4.5 |
| llama-7b q_proj | nf4_dq | 8.256 | 0.258 | 4.128 |
| llama-7b up_proj | fp16 | 86.0 | 1.0 | 16.0 |
| llama-7b up_proj | int8 | 44.042 | 0.5121 | 8.194 |
| llama-7b up_proj | int8_t6 | 44.042 | 0.5121 | 8.194 |
| llama-7b up_proj | fp4 | 24.688 | 0.2871 | 4.593 |
| llama-7b up_proj | nf4 | 24.688 | 0.2871 | 4.593 |
| llama-7b up_proj | nf4_dq | 22.685 | 0.2638 | 4.22 |

## Relative error vs fp64 reference (256 tokens; outlier = 25x on in_features/512 channels)

| shape | weights | variant | weight rel err | output rel err (normal x) | output rel err (outlier x) |
|---|---|---|---|---|---|
| qwen3-0.6b up_proj | gaussian | fp16 | 0.0 | 0.0002538 | 0.0002534 |
| qwen3-0.6b up_proj | gaussian | int8 | 0.007882 | 0.01116 | 0.04855 |
| qwen3-0.6b up_proj | gaussian | int8_t6 | 0.007882 | 0.01116 | 0.009902 |
| qwen3-0.6b up_proj | gaussian | fp4 | 0.1219 | 0.1222 | 0.1228 |
| qwen3-0.6b up_proj | gaussian | nf4 | 0.09201 | 0.09224 | 0.09199 |
| qwen3-0.6b up_proj | gaussian | nf4_dq | 0.09203 | 0.09226 | 0.09202 |
| llama-7b q_proj | gaussian | fp16 | 0.0 | 0.0002954 | 0.0002907 |
| llama-7b q_proj | gaussian | int8 | 0.008673 | 0.01226 | 0.07222 |
| llama-7b q_proj | gaussian | int8_t6 | 0.008673 | 0.01226 | 0.01089 |
| llama-7b q_proj | gaussian | fp4 | 0.1219 | 0.122 | 0.1216 |
| llama-7b q_proj | gaussian | nf4 | 0.09197 | 0.09196 | 0.09219 |
| llama-7b q_proj | gaussian | nf4_dq | 0.092 | 0.09198 | 0.09221 |
| llama-7b q_proj | student_t3 | fp16 | 0.0 | 0.0002957 | 0.00029 |
| llama-7b q_proj | student_t3 | int8 | 0.04429 | 0.04507 | 0.08389 |
| llama-7b q_proj | student_t3 | int8_t6 | 0.04429 | 0.04507 | 0.04443 |
| llama-7b q_proj | student_t3 | fp4 | 0.187 | 0.187 | 0.1863 |
| llama-7b q_proj | student_t3 | nf4 | 0.1222 | 0.1222 | 0.1222 |
| llama-7b q_proj | student_t3 | nf4_dq | 0.1226 | 0.1226 | 0.1226 |
| llama-7b up_proj | gaussian | fp16 | 0.0 | 0.000254 | 0.0002355 |
| llama-7b up_proj | gaussian | int8 | 0.008669 | 0.01226 | 0.07247 |
| llama-7b up_proj | gaussian | int8_t6 | 0.008669 | 0.01226 | 0.01094 |
| llama-7b up_proj | gaussian | fp4 | 0.122 | 0.1219 | 0.1221 |
| llama-7b up_proj | gaussian | nf4 | 0.092 | 0.09201 | 0.09204 |
| llama-7b up_proj | gaussian | nf4_dq | 0.09202 | 0.09203 | 0.09207 |

## Latency (fp16 activations, 100 calls after 10 warmup, CUDA events)

| shape | variant | input | tokens | ms/call | speedup vs fp16 | extra peak MiB |
|---|---|---|---|---|---|---|
| qwen3-0.6b up_proj | fp16 | normal | 1 | 0.0338 | 1.0 | 0.01 |
| qwen3-0.6b up_proj | fp16 | normal | 16 | 0.0292 | 1.0 | 0.09 |
| qwen3-0.6b up_proj | fp16 | normal | 256 | 0.1947 | 1.0 | 1.5 |
| qwen3-0.6b up_proj | fp16 | normal | 2048 | 1.4208 | 1.0 | 12.0 |
| qwen3-0.6b up_proj | int8 | normal | 1 | 0.1853 | 0.182 | 0.02 |
| qwen3-0.6b up_proj | int8 | outlier | 1 | 0.1369 | 0.247 | 0.02 |
| qwen3-0.6b up_proj | int8 | normal | 16 | 0.1194 | 0.245 | 0.3 |
| qwen3-0.6b up_proj | int8 | outlier | 16 | 0.1249 | 0.234 | 0.3 |
| qwen3-0.6b up_proj | int8 | normal | 256 | 0.1365 | 1.426 | 4.75 |
| qwen3-0.6b up_proj | int8 | outlier | 256 | 0.2562 | 0.76 | 4.75 |
| qwen3-0.6b up_proj | int8 | normal | 2048 | 0.9352 | 1.519 | 38.01 |
| qwen3-0.6b up_proj | int8 | outlier | 2048 | 0.938 | 1.515 | 38.01 |
| qwen3-0.6b up_proj | int8_t6 | normal | 1 | 0.2152 | 0.157 | 0.02 |
| qwen3-0.6b up_proj | int8_t6 | outlier | 1 | 0.5322 | 0.064 | 0.05 |
| qwen3-0.6b up_proj | int8_t6 | normal | 16 | 0.3105 | 0.094 | 0.3 |
| qwen3-0.6b up_proj | int8_t6 | outlier | 16 | 0.5836 | 0.05 | 0.31 |
| qwen3-0.6b up_proj | int8_t6 | normal | 256 | 0.4055 | 0.48 | 4.75 |
| qwen3-0.6b up_proj | int8_t6 | outlier | 256 | 0.598 | 0.326 | 4.76 |
| qwen3-0.6b up_proj | int8_t6 | normal | 2048 | 1.1372 | 1.249 | 38.01 |
| qwen3-0.6b up_proj | int8_t6 | outlier | 2048 | 1.6532 | 0.859 | 38.03 |
| qwen3-0.6b up_proj | fp4 | normal | 1 | 0.0512 | 0.66 | 0.01 |
| qwen3-0.6b up_proj | fp4 | normal | 16 | 0.0724 | 0.403 | 0.09 |
| qwen3-0.6b up_proj | fp4 | normal | 256 | 0.2875 | 0.677 | 1.5 |
| qwen3-0.6b up_proj | fp4 | normal | 2048 | 1.4973 | 0.949 | 18.0 |
| qwen3-0.6b up_proj | nf4 | normal | 1 | 0.1221 | 0.277 | 0.01 |
| qwen3-0.6b up_proj | nf4 | normal | 16 | 0.1193 | 0.245 | 0.09 |
| qwen3-0.6b up_proj | nf4 | normal | 256 | 0.2867 | 0.679 | 1.5 |
| qwen3-0.6b up_proj | nf4 | normal | 2048 | 1.4937 | 0.951 | 18.0 |
| qwen3-0.6b up_proj | nf4_dq | normal | 1 | 0.1522 | 0.222 | 0.01 |
| qwen3-0.6b up_proj | nf4_dq | normal | 16 | 0.1432 | 0.204 | 0.09 |
| qwen3-0.6b up_proj | nf4_dq | normal | 256 | 0.2888 | 0.674 | 1.5 |
| qwen3-0.6b up_proj | nf4_dq | normal | 2048 | 1.5056 | 0.944 | 18.38 |
| llama-7b q_proj | fp16 | normal | 1 | 0.151 | 1.0 | 0.01 |
| llama-7b q_proj | fp16 | normal | 16 | 0.1533 | 1.0 | 0.12 |
| llama-7b q_proj | fp16 | normal | 256 | 0.9124 | 1.0 | 2.0 |
| llama-7b q_proj | fp16 | normal | 2048 | 6.2659 | 1.0 | 16.0 |
| llama-7b q_proj | int8 | normal | 1 | 0.1797 | 0.84 | 0.03 |
| llama-7b q_proj | int8 | outlier | 1 | 0.1355 | 1.114 | 0.03 |
| llama-7b q_proj | int8 | normal | 16 | 0.4487 | 0.342 | 0.44 |
| llama-7b q_proj | int8 | outlier | 16 | 0.4408 | 0.348 | 0.44 |
| llama-7b q_proj | int8 | normal | 256 | 0.3712 | 2.458 | 7.0 |
| llama-7b q_proj | int8 | outlier | 256 | 0.3703 | 2.464 | 7.0 |
| llama-7b q_proj | int8 | normal | 2048 | 2.5732 | 2.435 | 56.01 |
| llama-7b q_proj | int8 | outlier | 2048 | 2.3987 | 2.612 | 56.01 |
| llama-7b q_proj | int8_t6 | normal | 1 | 0.2404 | 0.628 | 0.03 |
| llama-7b q_proj | int8_t6 | outlier | 1 | 0.6295 | 0.24 | 0.25 |
| llama-7b q_proj | int8_t6 | normal | 16 | 0.6571 | 0.233 | 0.44 |
| llama-7b q_proj | int8_t6 | outlier | 16 | 1.1186 | 0.137 | 0.5 |
| llama-7b q_proj | int8_t6 | normal | 256 | 0.6811 | 1.34 | 7.0 |
| llama-7b q_proj | int8_t6 | outlier | 256 | 0.9338 | 0.977 | 7.07 |
| llama-7b q_proj | int8_t6 | normal | 2048 | 3.2021 | 1.957 | 56.01 |
| llama-7b q_proj | int8_t6 | outlier | 2048 | 3.7927 | 1.652 | 56.1 |
| llama-7b q_proj | fp4 | normal | 1 | 0.1237 | 1.221 | 0.01 |
| llama-7b q_proj | fp4 | normal | 16 | 0.548 | 0.28 | 0.12 |
| llama-7b q_proj | fp4 | normal | 256 | 1.2506 | 0.73 | 34.0 |
| llama-7b q_proj | fp4 | normal | 2048 | 6.5735 | 0.953 | 48.0 |
| llama-7b q_proj | nf4 | normal | 1 | 0.1171 | 1.289 | 0.01 |
| llama-7b q_proj | nf4 | normal | 16 | 0.5513 | 0.278 | 0.12 |
| llama-7b q_proj | nf4 | normal | 256 | 1.2255 | 0.745 | 34.0 |
| llama-7b q_proj | nf4 | normal | 2048 | 6.5321 | 0.959 | 48.0 |
| llama-7b q_proj | nf4_dq | normal | 1 | 0.1169 | 1.292 | 0.01 |
| llama-7b q_proj | nf4_dq | normal | 16 | 0.5289 | 0.29 | 0.12 |
| llama-7b q_proj | nf4_dq | normal | 256 | 1.2423 | 0.734 | 36.0 |
| llama-7b q_proj | nf4_dq | normal | 2048 | 6.5411 | 0.958 | 50.0 |
| llama-7b up_proj | fp16 | normal | 1 | 0.3813 | 1.0 | 0.02 |
| llama-7b up_proj | fp16 | normal | 16 | 0.4047 | 1.0 | 0.34 |
| llama-7b up_proj | fp16 | normal | 256 | 2.1514 | 1.0 | 6.0 |
| llama-7b up_proj | fp16 | normal | 2048 | 13.2969 | 1.0 | 43.0 |
| llama-7b up_proj | int8 | normal | 1 | 0.3626 | 1.052 | 0.07 |
| llama-7b up_proj | int8 | outlier | 1 | 0.3669 | 1.039 | 0.07 |
| llama-7b up_proj | int8 | normal | 16 | 1.0067 | 0.402 | 1.07 |
| llama-7b up_proj | int8 | outlier | 16 | 0.9803 | 0.413 | 1.07 |
| llama-7b up_proj | int8 | normal | 256 | 0.7393 | 2.91 | 17.75 |
| llama-7b up_proj | int8 | outlier | 256 | 0.7241 | 2.971 | 17.75 |
| llama-7b up_proj | int8 | normal | 2048 | 4.9355 | 2.694 | 137.01 |
| llama-7b up_proj | int8 | outlier | 2048 | 4.3679 | 3.044 | 137.01 |
| llama-7b up_proj | int8_t6 | normal | 1 | 0.492 | 0.775 | 0.07 |
| llama-7b up_proj | int8_t6 | outlier | 1 | 0.7144 | 0.534 | 0.67 |
| llama-7b up_proj | int8_t6 | normal | 16 | 1.3677 | 0.296 | 1.07 |
| llama-7b up_proj | int8_t6 | outlier | 16 | 1.8165 | 0.223 | 1.24 |
| llama-7b up_proj | int8_t6 | normal | 256 | 1.1378 | 1.891 | 17.75 |
| llama-7b up_proj | int8_t6 | outlier | 256 | 1.6014 | 1.343 | 17.92 |
| llama-7b up_proj | int8_t6 | normal | 2048 | 6.8741 | 1.934 | 137.01 |
| llama-7b up_proj | int8_t6 | outlier | 2048 | 7.6785 | 1.732 | 137.21 |
| llama-7b up_proj | fp4 | normal | 1 | 0.2649 | 1.439 | 0.02 |
| llama-7b up_proj | fp4 | normal | 16 | 0.9467 | 0.427 | 0.34 |
| llama-7b up_proj | fp4 | normal | 256 | 3.2933 | 0.653 | 5.38 |
| llama-7b up_proj | fp4 | normal | 2048 | 12.8448 | 1.035 | 129.0 |
| llama-7b up_proj | nf4 | normal | 1 | 0.2604 | 1.464 | 0.02 |
| llama-7b up_proj | nf4 | normal | 16 | 0.96 | 0.422 | 0.34 |
| llama-7b up_proj | nf4 | normal | 256 | 3.3566 | 0.641 | 5.38 |
| llama-7b up_proj | nf4 | normal | 2048 | 13.0128 | 1.022 | 129.0 |
| llama-7b up_proj | nf4_dq | normal | 1 | 0.2762 | 1.381 | 0.02 |
| llama-7b up_proj | nf4_dq | normal | 16 | 0.9478 | 0.427 | 0.34 |
| llama-7b up_proj | nf4_dq | normal | 256 | 3.3737 | 0.638 | 6.0 |
| llama-7b up_proj | nf4_dq | normal | 2048 | 13.0138 | 1.022 | 135.0 |
