Liger-Kernel 1ab779ff09, triton-windows 3.8.0, torch 2.14.1+cu126, NVIDIA GeForce RTX 4060 Laptop GPU (memory cap 0.6), runtime 64.1 s, peak allocated 1890.2 MiB

## 1. Aliasing (bf16, T = 256, H = 1024; rel L2 vs fp64; `in_place` overwrites dY with dX)

| case | in_place | dx rel | dw rel | dresidual | 2nd clean vs 1st | reuse dY vs clean | reuse 2nd vs 1st | stash | error |
|---|---|---|---|---|---|---|---|---|---|
| plain | True | 0.0017 | 0.0029 | None | None | None | None | None |  |
| plain | False | 0.0017 | 0.0029 | None | None | None | None | None |  |
| gemma2_two_norm | True | 0.0023 | 0.0028 | 0.0017 | None | None | None | None |  |
| gemma2_two_norm | False | 0.0023 | 0.0028 | 0.0017 | None | None | None | None |  |
| aux_pow_before | True | 0.0021 | 0.0029 | None | None | None | None | None |  |
| aux_pow_before | False | 0.0021 | 0.0029 | None | None | None | None | None |  |
| retain_graph_twice | True | 0.0016 | 0.0028 | None | 0.0 | 1.9987 | 1.4178 | None |  |
| retain_graph_twice | False | 0.0016 | 0.0028 | None | 0.0 | 0.0 | 0.0 | None |  |
| leaf_input | True | None | None | None | None | None | None | {'x': 0.0} |  |
| leaf_input | False | None | None | None | None | None | None | {'x': 0.0} |  |

## 2. Error vs fp64 (T = 2048, H = 4096; rel L2; `floor` = fp64 result rounded to the dtype; `liger_vs_hf` = Liger vs matching HF casting)

| dtype | mode | offset | out | floor | liger | eager_hf | F.rms_norm | compile | liger vs hf |
|---|---|---|---|---|---|---|---|---|---|
| fp32 | llama | 0.0 | y | 0.0 | 6.053e-08 | 6.105e-08 | 6.088e-08 | 6.070e-08 | 4.273e-08 |
| fp32 | llama | 0.0 | dx | 0.0 | 6.060e-08 | 6.621e-08 | 6.097e-08 | 6.078e-08 | 5.709e-08 |
| fp32 | llama | 0.0 | dw | 0.0 | 1.839e-07 | 1.211e-07 | 1.622e-07 | 4.073e-07 | 2.099e-07 |
| fp32 | llama | 1.0 | y | 0.0 | 6.708e-08 | 6.753e-08 | 6.735e-08 | InductorError: InvalidCxxCompiler: Compiler: cl is not found | 4.263e-08 |
| fp32 | llama | 1.0 | dx | 0.0 | 6.715e-08 | 7.221e-08 | 6.744e-08 | InductorError: InvalidCxxCompiler: Compiler: cl is not found | 5.694e-08 |
| fp32 | llama | 1.0 | dw | 0.0 | 1.839e-07 | 1.211e-07 | 1.622e-07 | InductorError: InvalidCxxCompiler: Compiler: cl is not found | 2.099e-07 |
| fp32 | gemma | 0.0 | y | 0.0 | 6.053e-08 | 6.105e-08 | 6.088e-08 | InductorError: InvalidCxxCompiler: Compiler: cl is not found | 4.273e-08 |
| fp32 | gemma | 0.0 | dx | 0.0 | 6.060e-08 | 6.621e-08 | 6.097e-08 | InductorError: InvalidCxxCompiler: Compiler: cl is not found | 5.709e-08 |
| fp32 | gemma | 0.0 | dw | 0.0 | 1.839e-07 | 1.211e-07 | 1.622e-07 | InductorError: InvalidCxxCompiler: Compiler: cl is not found | 2.099e-07 |
| fp32 | gemma | 1.0 | y | 0.0 | 6.708e-08 | 6.753e-08 | 6.735e-08 | InductorError: InvalidCxxCompiler: Compiler: cl is not found | 4.263e-08 |
| fp32 | gemma | 1.0 | dx | 0.0 | 6.715e-08 | 7.221e-08 | 6.744e-08 | InductorError: InvalidCxxCompiler: Compiler: cl is not found | 5.694e-08 |
| fp32 | gemma | 1.0 | dw | 0.0 | 1.839e-07 | 1.211e-07 | 1.622e-07 | InductorError: InvalidCxxCompiler: Compiler: cl is not found | 2.099e-07 |
| fp32 | none | 0.0 | y | 0.0 | 6.053e-08 | 6.105e-08 | 6.088e-08 | InductorError: InvalidCxxCompiler: Compiler: cl is not found | 4.273e-08 |
| fp32 | none | 0.0 | dx | 0.0 | 6.060e-08 | 6.621e-08 | 6.097e-08 | InductorError: InvalidCxxCompiler: Compiler: cl is not found | 5.709e-08 |
| fp32 | none | 0.0 | dw | 0.0 | 1.839e-07 | 1.211e-07 | 1.622e-07 | InductorError: InvalidCxxCompiler: Compiler: cl is not found | 2.099e-07 |
| fp32 | none | 1.0 | y | 0.0 | 6.708e-08 | 6.753e-08 | 6.735e-08 | InductorError: InvalidCxxCompiler: Compiler: cl is not found | 4.263e-08 |
| fp32 | none | 1.0 | dx | 0.0 | 6.715e-08 | 7.221e-08 | 6.744e-08 | InductorError: InvalidCxxCompiler: Compiler: cl is not found | 5.694e-08 |
| fp32 | none | 1.0 | dw | 0.0 | 1.839e-07 | 1.211e-07 | 1.622e-07 | InductorError: InvalidCxxCompiler: Compiler: cl is not found | 2.099e-07 |
| fp16 | llama | 0.0 | y | 2.075e-04 | 2.937e-04 | 3.783e-04 | 2.075e-04 | InductorError: InvalidCxxCompiler: Compiler: cl is not found | 4.003e-04 |
| fp16 | llama | 0.0 | dx | 2.074e-04 | 2.074e-04 | 4.308e-04 | 2.074e-04 | InductorError: InvalidCxxCompiler: Compiler: cl is not found | 4.779e-04 |
| fp16 | llama | 0.0 | dw | 2.136e-04 | 3.515e-04 | 4.424e-04 | 2.136e-04 | InductorError: InvalidCxxCompiler: Compiler: cl is not found | 4.981e-04 |
| fp16 | llama | 1.0 | y | 2.072e-04 | 2.938e-04 | 4.458e-04 | 3.141e-04 | InductorError: InvalidCxxCompiler: Compiler: cl is not found | 5.058e-04 |
| fp16 | llama | 1.0 | dx | 2.075e-04 | 2.075e-04 | 4.915e-04 | 3.143e-04 | InductorError: InvalidCxxCompiler: Compiler: cl is not found | 5.333e-04 |
| fp16 | llama | 1.0 | dw | 2.136e-04 | 3.515e-04 | 4.424e-04 | 2.136e-04 | InductorError: InvalidCxxCompiler: Compiler: cl is not found | 4.981e-04 |
| fp16 | gemma | 0.0 | y | 2.075e-04 | 2.075e-04 | 2.075e-04 | 2.075e-04 | InductorError: InvalidCxxCompiler: Compiler: cl is not found | 3.905e-06 |
| fp16 | gemma | 0.0 | dx | 2.074e-04 | 2.074e-04 | 2.074e-04 | 2.074e-04 | InductorError: InvalidCxxCompiler: Compiler: cl is not found | 5.229e-06 |
| fp16 | gemma | 0.0 | dw | 2.136e-04 | 2.136e-04 | 2.136e-04 | 2.136e-04 | InductorError: InvalidCxxCompiler: Compiler: cl is not found | 1.689e-06 |
| fp16 | gemma | 1.0 | y | 2.072e-04 | 2.072e-04 | 2.072e-04 | 3.141e-04 | InductorError: InvalidCxxCompiler: Compiler: cl is not found | 3.900e-06 |
| fp16 | gemma | 1.0 | dx | 2.075e-04 | 2.075e-04 | 2.075e-04 | 3.143e-04 | InductorError: InvalidCxxCompiler: Compiler: cl is not found | 4.430e-06 |
| fp16 | gemma | 1.0 | dw | 2.136e-04 | 2.136e-04 | 2.136e-04 | 2.136e-04 | InductorError: InvalidCxxCompiler: Compiler: cl is not found | 1.689e-06 |
| fp16 | none | 0.0 | y | 2.075e-04 | 2.693e-04 | 3.783e-04 | 2.075e-04 | InductorError: InvalidCxxCompiler: Compiler: cl is not found | 4.594e-04 |
| fp16 | none | 0.0 | dx | 2.074e-04 | 3.457e-04 | 4.308e-04 | 2.074e-04 | InductorError: InvalidCxxCompiler: Compiler: cl is not found | 4.963e-04 |
| fp16 | none | 0.0 | dw | 2.136e-04 | 3.480e-04 | 4.424e-04 | 2.136e-04 | InductorError: InvalidCxxCompiler: Compiler: cl is not found | 5.024e-04 |
| fp16 | none | 1.0 | y | 2.072e-04 | 3.577e-04 | 4.458e-04 | 3.141e-04 | InductorError: InvalidCxxCompiler: Compiler: cl is not found | 4.593e-04 |
| fp16 | none | 1.0 | dx | 2.075e-04 | 3.461e-04 | 4.915e-04 | 3.143e-04 | InductorError: InvalidCxxCompiler: Compiler: cl is not found | 5.503e-04 |
| fp16 | none | 1.0 | dw | 2.136e-04 | 3.480e-04 | 4.424e-04 | 2.136e-04 | InductorError: InvalidCxxCompiler: Compiler: cl is not found | 5.024e-04 |
| bf16 | llama | 0.0 | y | 0.0017 | 0.0023 | 0.0029 | 0.0017 | InductorError: InvalidCxxCompiler: Compiler: cl is not found | 0.0029 |
| bf16 | llama | 0.0 | dx | 0.0017 | 0.0017 | 0.0033 | 0.0017 | InductorError: InvalidCxxCompiler: Compiler: cl is not found | 0.0037 |
| bf16 | llama | 0.0 | dw | 0.0017 | 0.0029 | 0.0034 | 0.0017 | InductorError: InvalidCxxCompiler: Compiler: cl is not found | 0.0037 |
| bf16 | llama | 1.0 | y | 0.0017 | 0.0023 | 0.0034 | 0.0025 | InductorError: InvalidCxxCompiler: Compiler: cl is not found | 0.0038 |
| bf16 | llama | 1.0 | dx | 0.0017 | 0.0017 | 0.0037 | 0.0025 | InductorError: InvalidCxxCompiler: Compiler: cl is not found | 0.0041 |
| bf16 | llama | 1.0 | dw | 0.0017 | 0.0029 | 0.0034 | 0.0017 | InductorError: InvalidCxxCompiler: Compiler: cl is not found | 0.0037 |
| bf16 | gemma | 0.0 | y | 0.0017 | 0.0017 | 0.0017 | 0.0017 | InductorError: InvalidCxxCompiler: Compiler: cl is not found | 1.337e-05 |
| bf16 | gemma | 0.0 | dx | 0.0017 | 0.0017 | 0.0017 | 0.0017 | InductorError: InvalidCxxCompiler: Compiler: cl is not found | 1.083e-05 |
| bf16 | gemma | 0.0 | dw | 0.0017 | 0.0017 | 0.0017 | 0.0017 | InductorError: InvalidCxxCompiler: Compiler: cl is not found | 1.047e-07 |
| bf16 | gemma | 1.0 | y | 0.0017 | 0.0017 | 0.0017 | 0.0025 | InductorError: InvalidCxxCompiler: Compiler: cl is not found | 8.678e-06 |
| bf16 | gemma | 1.0 | dx | 0.0017 | 0.0017 | 0.0017 | 0.0025 | InductorError: InvalidCxxCompiler: Compiler: cl is not found | 1.488e-05 |
| bf16 | gemma | 1.0 | dw | 0.0017 | 0.0017 | 0.0017 | 0.0017 | InductorError: InvalidCxxCompiler: Compiler: cl is not found | 1.047e-07 |
| bf16 | none | 0.0 | y | 0.0017 | 0.0021 | 0.0029 | 0.0017 | InductorError: InvalidCxxCompiler: Compiler: cl is not found | 0.0035 |
| bf16 | none | 0.0 | dx | 0.0017 | 0.0028 | 0.0033 | 0.0017 | InductorError: InvalidCxxCompiler: Compiler: cl is not found | 0.0039 |
| bf16 | none | 0.0 | dw | 0.0017 | 0.0028 | 0.0034 | 0.0017 | InductorError: InvalidCxxCompiler: Compiler: cl is not found | 0.004 |
| bf16 | none | 1.0 | y | 0.0017 | 0.0028 | 0.0034 | 0.0025 | InductorError: InvalidCxxCompiler: Compiler: cl is not found | 0.0035 |
| bf16 | none | 1.0 | dx | 0.0017 | 0.0028 | 0.0037 | 0.0025 | InductorError: InvalidCxxCompiler: Compiler: cl is not found | 0.0043 |
| bf16 | none | 1.0 | dw | 0.0017 | 0.0028 | 0.0034 | 0.0017 | InductorError: InvalidCxxCompiler: Compiler: cl is not found | 0.004 |

## 3. Kernel path selection (bf16; wrapper picks row if `BLOCK_SIZE > 256 or T < 32768 or row_mode`)

| label | T | H | row_mode | BLOCK | warps | path | y == row | y rel | dx rel | dw rel |
|---|---|---|---|---|---|---|---|---|---|---|
| row_forced_wide | 8192 | 4096 | True | 4096 | 8 | row | True | 0.0 | 0.0 | 0.0 |
| auto_wide_few_rows | 1024 | 4096 | None | 4096 | 8 | row | True | 0.0 | 0.0 | 0.0 |
| auto_narrow_many_rows | 32768 | 128 | None | 128 | 4 | block | False | 9.207e-06 | 7.152e-06 | 2.097e-06 |
| row_forced_narrow_many | 32768 | 128 | True | 128 | 4 | row | True | 0.0 | 0.0 | 0.0 |
| auto_narrow_few_rows | 1024 | 128 | None | 128 | 4 | row | True | 0.0 | 0.0 | 0.0 |

## 4. Memory + latency (bf16; held = after forward incl. output; extra peak = max during fwd+bwd above inputs; do_bench median)

### held after forward (MiB)

| shape | T | H | liger_inplace | liger_oop | eager_hf | torch_compile | F_rms_norm |
|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | 2048 | 1024 | 4.0 | 4.0 | 8.0 | 4.0 | 4.0 |
| Qwen3-0.6B | 8192 | 1024 | 16.0 | 16.0 | 32.0 | 16.0 | 16.0 |
| Qwen3-0.6B | 16384 | 1024 | 32.1 | 32.1 | 64.0 | 32.0 | 32.1 |
| Llama-3.2-1B | 2048 | 2048 | 8.0 | 8.0 | 16.0 | 8.0 | 8.0 |
| Llama-3.2-1B | 8192 | 2048 | 32.0 | 32.0 | 64.0 | 32.0 | 32.0 |
| Llama-3.2-1B | 16384 | 2048 | 64.1 | 64.1 | 128.0 | 64.0 | 64.1 |

### extra peak (MiB)

| shape | T | H | liger_inplace | liger_oop | eager_hf | torch_compile | F_rms_norm |
|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | 2048 | 1024 | 4.1 | 8.1 | 32.0 | 8.0 | 8.0 |
| Qwen3-0.6B | 8192 | 1024 | 16.1 | 32.1 | 96.0 | 33.0 | 32.0 |
| Qwen3-0.6B | 16384 | 1024 | 32.2 | 64.2 | 192.0 | 66.0 | 64.1 |
| Llama-3.2-1B | 2048 | 2048 | 8.2 | 16.2 | 50.0 | 16.0 | 16.0 |
| Llama-3.2-1B | 8192 | 2048 | 32.2 | 64.2 | 192.0 | 64.0 | 64.0 |
| Llama-3.2-1B | 16384 | 2048 | 64.3 | 128.3 | 384.0 | 128.0 | 128.1 |

### fwd + bwd (ms)

| shape | T | H | liger_inplace | liger_oop | eager_hf | torch_compile | F_rms_norm |
|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | 2048 | 1024 | 0.279 | 0.269 | 0.637 | 0.198 | 0.138 |
| Qwen3-0.6B | 8192 | 1024 | 0.497 | 0.537 | 1.825 | 0.538 | 0.706 |
| Qwen3-0.6B | 16384 | 1024 | 1.077 | 1.212 | 4.576 | 1.061 | 1.393 |
| Llama-3.2-1B | 2048 | 2048 | 0.226 | 0.234 | 0.61 | 0.26 | 0.196 |
| Llama-3.2-1B | 8192 | 2048 | 1.027 | 1.159 | 4.578 | 1.291 | 1.369 |
| Llama-3.2-1B | 16384 | 2048 | 2.023 | 2.295 | 9.228 | 2.631 | 2.643 |

## 5. Bandwidth (bf16, T = 4096; fwd 2·T·H·2 bytes, bwd 3·T·H·2 bytes; do_bench median of 3 rounds; `xE` = speedup vs eager HF)

| H | BLOCK | lanes | copy | fwd liger | fwd eager | fwd compile | fwd F | bwd liger oop | bwd liger in-place | bwd eager | bwd compile | bwd F | fwd liger xE | bwd liger oop xE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 512 | 512 | 1.0 | 97.5 | 178.1 | 86.2 | 199.8 | 174.3 | 99.9 | 89.7 | 57.7 | 110.7 | 151.7 | 2.07 | 1.73 |
| 1024 | 1024 | 1.0 | 109.2 | 218.5 | 100.5 | 210.1 | 204.8 | 152.6 | 130.7 | 50.6 | 141.2 | 130.0 | 2.17 | 3.02 |
| 2048 | 2048 | 1.0 | 110.0 | 222.9 | 94.7 | 211.4 | 214.9 | 171.9 | 160.6 | 33.9 | 190.5 | 119.3 | 2.35 | 5.07 |
| 2049 | 4096 | 0.5 | 110.8 | 217.1 | 94.5 | 206.2 | 128.1 | 160.7 | 141.7 | 34.0 | 110.0 | 121.4 | 2.3 | 4.73 |
| 3072 | 4096 | 0.75 | 110.7 | 222.4 | 71.2 | 215.6 | 219.4 | 179.0 | 152.6 | 30.9 | 138.3 | 118.5 | 3.12 | 5.79 |
| 4096 | 4096 | 1.0 | 111.8 | 226.8 | 68.7 | 218.5 | 224.4 | 176.2 | 145.4 | 30.4 | 140.4 | 123.5 | 3.3 | 5.8 |
| 8192 | 8192 | 1.0 | 114.2 | 228.9 | 64.3 | 224.8 | 231.2 | 175.7 | 139.6 | 30.4 | 143.7 | 129.3 | 3.56 | 5.77 |
| 11008 | 16384 | 0.672 | 113.8 | 228.1 | 65.7 | 233.6 | 231.1 | 174.0 | 137.8 | 30.5 | 144.8 | 115.7 | 3.47 | 5.7 |

### Kernel resources (row forward)

| H | kernel | BLOCK | warps | regs/thread | spills (bytes) | shared |
|---|---|---|---|---|---|---|
| 512 | fwd_row | 512 | 4 | 19 | 0 | 16 |
| 1024 | fwd_row | 1024 | 4 | 26 | 0 | 16 |
| 2048 | fwd_row | 2048 | 8 | 26 | 0 | 32 |
| 2049 | fwd_row | 4096 | 8 | 48 | 0 | 8192 |
| 3072 | fwd_row | 4096 | 8 | 39 | 0 | 32 |
| 4096 | fwd_row | 4096 | 8 | 39 | 0 | 32 |
| 8192 | fwd_row | 8192 | 16 | 39 | 0 | 64 |
| 11008 | fwd_row | 16384 | 16 | 63 | 0 | 64 |
