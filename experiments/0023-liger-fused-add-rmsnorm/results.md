Liger-Kernel 1ab779ff09, triton-windows 3.8.0, torch 2.14.1+cu126, NVIDIA GeForce RTX 4060 Laptop GPU (memory cap 0.6), runtime 68.5 s, peak allocated 3050.2 MiB

## 0. `in_place` defaults

- Function.forward default: `False`
- `liger_fused_add_rms_norm` functional default: `True`
- `LigerFusedAddRMSNorm` module default: `False`
- Docstring claims default True: `True`

## 1. Output branches (bf16, T = 256, H = 1024; on this torch, unused Function outputs get **zero tensors**, not None)

| branch | in_place | ok | dx rel | dr rel | dw rel | dx is dr (same ptr) | dx==dr | error |
|---|---|---|---|---|---|---|---|---|
| both | False | True | 0.0017 | 0.0017 | 0.0034 | False | True |  |
| both | True | True | 0.0017 | 0.0017 | 0.0034 | False | True |  |
| norm_only | False | True | 0.0017 | 0.0017 | 0.0034 | False | True |  |
| norm_only | True | True | 0.0017 | 0.0017 | 0.0034 | False | True |  |
| residual_only | False | True | 0.0 | 0.0 | 0.0 | False | True |  |
| residual_only | True | True | 0.0 | 0.0 | 0.0 | False | True |  |

## 2. Aliasing (bf16, T = 256, H = 1024)

| case | in_place | dx rel | dr rel | dw rel | 2nd vs 1st | reuse dY vs clean | reuse 2nd vs 1st | dx is dr | error |
|---|---|---|---|---|---|---|---|---|---|
| plain_both | True | 0.0017 | 0.0017 | 0.0034 | None | None | None | False |  |
| plain_both | False | 0.0017 | 0.0017 | 0.0034 | None | None | None | False |  |
| retain_reuse_dY | True | None | None | None | 0.0 | 1.6005 | 1.2013 | True |  |
| retain_reuse_dY | False | None | None | None | 0.0 | 0.0 | 0.0 | True |  |
| bwd_returns_dX_twice | True | None | None | None | None | None | None | True |  |
| bwd_returns_dX_twice | False | None | None | None | None | None | None | True |  |

## 3. Error vs fp64 (T = 2048, H = 4096; rel L2)

| dtype | mode | offset | out | floor | liger | eager_hf | compile | liger vs hf |
|---|---|---|---|---|---|---|---|---|
| fp32 | llama | 0.0 | y | 0.0 | 6.516e-08 | 6.522e-08 | 6.466e-08 | 4.204e-08 |
| fp32 | llama | 0.0 | s | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| fp32 | llama | 0.0 | dx | 0.0 | 4.341e-08 | 5.441e-08 | 5.143e-08 | 5.329e-08 |
| fp32 | llama | 0.0 | dr | 0.0 | 4.341e-08 | 5.441e-08 | 5.143e-08 | 5.329e-08 |
| fp32 | llama | 0.0 | dw | 0.0 | 1.872e-07 | 1.222e-07 | 4.100e-07 | 2.112e-07 |
| fp32 | llama | 1.0 | y | 0.0 | 7.109e-08 | 7.120e-08 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 4.217e-08 |
| fp32 | llama | 1.0 | s | 0.0 | 0.0 | 0.0 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 0.0 |
| fp32 | llama | 1.0 | dx | 0.0 | 5.440e-08 | 6.216e-08 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 5.658e-08 |
| fp32 | llama | 1.0 | dr | 0.0 | 5.440e-08 | 6.216e-08 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 5.658e-08 |
| fp32 | llama | 1.0 | dw | 0.0 | 1.872e-07 | 1.222e-07 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 2.112e-07 |
| fp32 | gemma | 0.0 | y | 0.0 | 6.516e-08 | 6.522e-08 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 4.204e-08 |
| fp32 | gemma | 0.0 | s | 0.0 | 0.0 | 0.0 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 0.0 |
| fp32 | gemma | 0.0 | dx | 0.0 | 4.341e-08 | 5.441e-08 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 5.329e-08 |
| fp32 | gemma | 0.0 | dr | 0.0 | 4.341e-08 | 5.441e-08 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 5.329e-08 |
| fp32 | gemma | 0.0 | dw | 0.0 | 1.872e-07 | 1.222e-07 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 2.112e-07 |
| fp32 | gemma | 1.0 | y | 0.0 | 7.109e-08 | 7.120e-08 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 4.217e-08 |
| fp32 | gemma | 1.0 | s | 0.0 | 0.0 | 0.0 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 0.0 |
| fp32 | gemma | 1.0 | dx | 0.0 | 5.440e-08 | 6.216e-08 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 5.658e-08 |
| fp32 | gemma | 1.0 | dr | 0.0 | 5.440e-08 | 6.216e-08 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 5.658e-08 |
| fp32 | gemma | 1.0 | dw | 0.0 | 1.872e-07 | 1.222e-07 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 2.112e-07 |
| fp32 | none | 0.0 | y | 0.0 | 6.516e-08 | 6.522e-08 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 4.204e-08 |
| fp32 | none | 0.0 | s | 0.0 | 0.0 | 0.0 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 0.0 |
| fp32 | none | 0.0 | dx | 0.0 | 4.341e-08 | 5.441e-08 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 5.329e-08 |
| fp32 | none | 0.0 | dr | 0.0 | 4.341e-08 | 5.441e-08 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 5.329e-08 |
| fp32 | none | 0.0 | dw | 0.0 | 1.872e-07 | 1.222e-07 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 2.112e-07 |
| fp32 | none | 1.0 | y | 0.0 | 7.109e-08 | 7.120e-08 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 4.217e-08 |
| fp32 | none | 1.0 | s | 0.0 | 0.0 | 0.0 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 0.0 |
| fp32 | none | 1.0 | dx | 0.0 | 5.440e-08 | 6.216e-08 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 5.658e-08 |
| fp32 | none | 1.0 | dr | 0.0 | 5.440e-08 | 6.216e-08 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 5.658e-08 |
| fp32 | none | 1.0 | dw | 0.0 | 1.872e-07 | 1.222e-07 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 2.112e-07 |
| fp16 | llama | 0.0 | y | 2.072e-04 | 3.697e-04 | 3.697e-04 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 4.281e-06 |
| fp16 | llama | 0.0 | s | 2.250e-04 | 2.250e-04 | 2.250e-04 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 0.0 |
| fp16 | llama | 0.0 | dx | 2.073e-04 | 2.073e-04 | 2.789e-04 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 2.967e-04 |
| fp16 | llama | 0.0 | dr | 2.073e-04 | 2.073e-04 | 2.789e-04 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 2.967e-04 |
| fp16 | llama | 0.0 | dw | 2.088e-04 | 4.208e-04 | 4.202e-04 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 3.220e-05 |
| fp16 | llama | 1.0 | y | 2.072e-04 | 3.693e-04 | 4.321e-04 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 3.189e-04 |
| fp16 | llama | 1.0 | s | 2.250e-04 | 2.250e-04 | 2.250e-04 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 0.0 |
| fp16 | llama | 1.0 | dx | 2.074e-04 | 2.074e-04 | 3.448e-04 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 3.770e-04 |
| fp16 | llama | 1.0 | dr | 2.074e-04 | 2.074e-04 | 3.448e-04 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 3.770e-04 |
| fp16 | llama | 1.0 | dw | 2.088e-04 | 4.208e-04 | 4.202e-04 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 3.220e-05 |
| fp16 | gemma | 0.0 | y | 2.072e-04 | 3.058e-04 | 3.058e-04 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 4.034e-06 |
| fp16 | gemma | 0.0 | s | 2.250e-04 | 2.250e-04 | 2.250e-04 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 0.0 |
| fp16 | gemma | 0.0 | dx | 2.073e-04 | 2.073e-04 | 3.013e-04 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 3.547e-04 |
| fp16 | gemma | 0.0 | dr | 2.073e-04 | 2.073e-04 | 3.013e-04 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 3.547e-04 |
| fp16 | gemma | 0.0 | dw | 2.088e-04 | 3.077e-04 | 3.077e-04 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 1.875e-06 |
| fp16 | gemma | 1.0 | y | 2.072e-04 | 3.056e-04 | 3.056e-04 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 4.003e-06 |
| fp16 | gemma | 1.0 | s | 2.250e-04 | 2.250e-04 | 2.250e-04 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 0.0 |
| fp16 | gemma | 1.0 | dx | 2.074e-04 | 2.074e-04 | 3.032e-04 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 3.564e-04 |
| fp16 | gemma | 1.0 | dr | 2.074e-04 | 2.074e-04 | 3.032e-04 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 3.564e-04 |
| fp16 | gemma | 1.0 | dw | 2.088e-04 | 3.077e-04 | 3.077e-04 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 1.875e-06 |
| fp16 | none | 0.0 | y | 2.072e-04 | 3.507e-04 | 4.353e-04 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 4.579e-04 |
| fp16 | none | 0.0 | s | 2.250e-04 | 2.250e-04 | 2.250e-04 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 0.0 |
| fp16 | none | 0.0 | dx | 2.073e-04 | 2.574e-04 | 3.707e-04 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 4.331e-04 |
| fp16 | none | 0.0 | dr | 2.073e-04 | 2.574e-04 | 3.707e-04 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 4.331e-04 |
| fp16 | none | 0.0 | dw | 2.088e-04 | 4.025e-04 | 4.740e-04 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 4.879e-04 |
| fp16 | none | 1.0 | y | 2.072e-04 | 4.163e-04 | 4.901e-04 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 4.581e-04 |
| fp16 | none | 1.0 | s | 2.250e-04 | 2.250e-04 | 2.250e-04 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 0.0 |
| fp16 | none | 1.0 | dx | 2.074e-04 | 2.794e-04 | 4.328e-04 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 4.926e-04 |
| fp16 | none | 1.0 | dr | 2.074e-04 | 2.794e-04 | 4.328e-04 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 4.926e-04 |
| fp16 | none | 1.0 | dw | 2.088e-04 | 4.025e-04 | 4.740e-04 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 4.879e-04 |
| bf16 | llama | 0.0 | y | 0.0017 | 0.003 | 0.003 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 1.750e-05 |
| bf16 | llama | 0.0 | s | 0.0018 | 0.0018 | 0.0018 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 0.0 |
| bf16 | llama | 0.0 | dx | 0.0017 | 0.0017 | 0.0022 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 0.0024 |
| bf16 | llama | 0.0 | dr | 0.0017 | 0.0017 | 0.0022 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 0.0024 |
| bf16 | llama | 0.0 | dw | 0.0017 | 0.0034 | 0.0034 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 6.663e-06 |
| bf16 | llama | 1.0 | y | 0.0017 | 0.003 | 0.0035 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 0.0026 |
| bf16 | llama | 1.0 | s | 0.0018 | 0.0018 | 0.0018 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 0.0 |
| bf16 | llama | 1.0 | dx | 0.0017 | 0.0017 | 0.0028 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 0.003 |
| bf16 | llama | 1.0 | dr | 0.0017 | 0.0017 | 0.0028 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 0.003 |
| bf16 | llama | 1.0 | dw | 0.0017 | 0.0034 | 0.0034 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 6.663e-06 |
| bf16 | gemma | 0.0 | y | 0.0017 | 0.0024 | 0.0024 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 9.736e-06 |
| bf16 | gemma | 0.0 | s | 0.0018 | 0.0018 | 0.0018 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 0.0 |
| bf16 | gemma | 0.0 | dx | 0.0017 | 0.0017 | 0.0024 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 0.0028 |
| bf16 | gemma | 0.0 | dr | 0.0017 | 0.0017 | 0.0024 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 0.0028 |
| bf16 | gemma | 0.0 | dw | 0.0017 | 0.0024 | 0.0024 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 0.0 |
| bf16 | gemma | 1.0 | y | 0.0017 | 0.0024 | 0.0024 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 7.912e-06 |
| bf16 | gemma | 1.0 | s | 0.0018 | 0.0018 | 0.0018 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 0.0 |
| bf16 | gemma | 1.0 | dx | 0.0017 | 0.0017 | 0.0024 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 0.0028 |
| bf16 | gemma | 1.0 | dr | 0.0017 | 0.0017 | 0.0024 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 0.0028 |
| bf16 | gemma | 1.0 | dw | 0.0017 | 0.0024 | 0.0024 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 0.0 |
| bf16 | none | 0.0 | y | 0.0017 | 0.0028 | 0.0035 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 0.0037 |
| bf16 | none | 0.0 | s | 0.0018 | 0.0018 | 0.0018 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 0.0 |
| bf16 | none | 0.0 | dx | 0.0017 | 0.002 | 0.0029 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 0.0034 |
| bf16 | none | 0.0 | dr | 0.0017 | 0.002 | 0.0029 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 0.0034 |
| bf16 | none | 0.0 | dw | 0.0017 | 0.0032 | 0.0038 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 0.0038 |
| bf16 | none | 1.0 | y | 0.0017 | 0.0033 | 0.0039 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 0.0037 |
| bf16 | none | 1.0 | s | 0.0018 | 0.0018 | 0.0018 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 0.0 |
| bf16 | none | 1.0 | dx | 0.0017 | 0.0022 | 0.0034 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 0.0039 |
| bf16 | none | 1.0 | dr | 0.0017 | 0.0022 | 0.0034 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 0.0039 |
| bf16 | none | 1.0 | dw | 0.0017 | 0.0032 | 0.0038 | InductorError: InvalidCxxCompiler: Compiler: cl is not found. | 0.0038 |

## 4. Memory + latency (bf16; held after forward; extra peak through bwd; do_bench median)

### held after forward (MiB)

| shape | T | H | liger_fused_inplace | liger_fused_oop | separate_add_liger_rms | eager_hf | torch_compile |
|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | 2048 | 1024 | 8.0 | 8.0 | 8.0 | 20.0 | 8.0 |
| Qwen3-0.6B | 8192 | 1024 | 32.0 | 32.0 | 32.0 | 80.0 | 32.0 |
| Qwen3-0.6B | 16384 | 1024 | 64.1 | 64.1 | 64.1 | 160.1 | 64.1 |
| Llama-3.2-1B | 2048 | 2048 | 16.0 | 16.0 | 16.0 | 40.0 | 16.0 |
| Llama-3.2-1B | 8192 | 2048 | 64.0 | 64.0 | 64.0 | 160.0 | 64.0 |
| Llama-3.2-1B | 16384 | 2048 | 128.1 | 128.1 | 128.1 | 320.1 | 128.1 |

### extra peak (MiB)

| shape | T | H | liger_fused_inplace | liger_fused_oop | separate_add_liger_rms | eager_hf | torch_compile |
|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | 2048 | 1024 | 12.0 | 16.0 | 16.0 | 56.0 | 16.0 |
| Qwen3-0.6B | 8192 | 1024 | 48.0 | 64.0 | 64.0 | 224.0 | 64.0 |
| Qwen3-0.6B | 16384 | 1024 | 96.0 | 128.0 | 128.0 | 448.0 | 128.0 |
| Llama-3.2-1B | 2048 | 2048 | 24.0 | 32.0 | 32.0 | 112.0 | 32.0 |
| Llama-3.2-1B | 8192 | 2048 | 96.0 | 128.0 | 128.0 | 448.0 | 128.0 |
| Llama-3.2-1B | 16384 | 2048 | 192.0 | 256.0 | 256.0 | 896.0 | 256.0 |

### fwd + bwd (ms)

| shape | T | H | liger_fused_inplace | liger_fused_oop | separate_add_liger_rms | eager_hf | torch_compile |
|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | 2048 | 1024 | 0.379 | 0.332 | 0.385 | 1.166 | 0.315 |
| Qwen3-0.6B | 8192 | 1024 | 1.338 | 1.578 | 1.397 | 5.823 | 0.983 |
| Qwen3-0.6B | 16384 | 1024 | 2.759 | 2.753 | 3.829 | 12.145 | 2.779 |
| Llama-3.2-1B | 2048 | 2048 | 0.512 | 0.517 | 0.552 | 2.64 | 0.833 |
| Llama-3.2-1B | 8192 | 2048 | 2.26 | 2.789 | 2.81 | 12.134 | 3.153 |
| Llama-3.2-1B | 16384 | 2048 | 4.496 | 4.483 | 5.578 | 23.825 | 6.463 |

## 5. Bandwidth (bf16, T = 4096; fwd counted as 3·T·H·2 bytes (X+R→Y+S traffic), bwd as 4·T·H·2; do_bench median of 3 rounds)

| H | BLOCK | lanes | copy | fwd fused | fwd separate | fwd eager | fwd compile | bwd fused oop | bwd fused in-place | bwd eager | bwd graph | fused vs eager | fused vs separate | bwd vs eager |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 512 | 512 | 1.0 | 97.5 | 144.6 | 128.0 | 37.9 | 138.1 | 59.4 | 54.1 | 27.4 | 62.3 | 3.81 | 1.13 | 2.17 |
| 1024 | 1024 | 1.0 | 101.1 | 154.1 | 150.8 | 42.0 | 152.6 | 103.7 | 87.1 | 23.1 | 106.4 | 3.67 | 1.02 | 4.49 |
| 2048 | 2048 | 1.0 | 110.0 | 162.8 | 161.7 | 33.4 | 166.1 | 177.1 | 139.4 | 19.1 | 182.0 | 4.87 | 1.01 | 9.28 |
| 2049 | 4096 | 0.5 | 110.8 | 134.4 | 112.0 | 33.1 | 163.1 | 118.4 | 98.3 | 18.9 | 127.8 | 4.06 | 1.2 | 6.27 |
| 3072 | 4096 | 0.75 | 111.7 | 167.9 | 141.5 | 31.5 | 170.3 | 201.4 | 149.2 | 18.6 | 203.1 | 5.33 | 1.19 | 10.83 |
| 4096 | 4096 | 1.0 | 111.8 | 170.7 | 138.7 | 32.0 | 171.0 | 215.2 | 155.9 | 18.9 | 215.6 | 5.33 | 1.23 | 11.42 |
| 8192 | 8192 | 1.0 | 114.2 | 169.6 | 138.6 | 32.2 | 172.5 | 229.5 | 156.4 | 18.9 | 230.0 | 5.27 | 1.22 | 12.17 |
| 11008 | 16384 | 0.672 | 114.1 | 168.7 | 139.0 | 32.3 | 168.8 | 232.1 | 156.1 | 18.9 | 231.6 | 5.23 | 1.21 | 12.3 |

### Kernel resources (fused forward)

| H | kernel | BLOCK | warps | regs/thread | spills (bytes) | shared |
|---|---|---|---|---|---|---|
| 512 | fwd | 512 | 4 | 20 | 0 | 16 |
| 1024 | fwd | 1024 | 4 | 31 | 0 | 16 |
| 2048 | fwd | 2048 | 8 | 31 | 0 | 32 |
| 2049 | fwd | 4096 | 8 | 64 | 0 | 8192 |
| 3072 | fwd | 4096 | 8 | 40 | 0 | 32 |
| 4096 | fwd | 4096 | 8 | 40 | 0 | 32 |
| 8192 | fwd | 8192 | 16 | 39 | 0 | 64 |
| 11008 | fwd | 16384 | 16 | 64 | 0 | 64 |
