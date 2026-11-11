# 0018 results

## 1. Correctness (same routing, relative error vs fp32 per-expert reference)

| config | T | experts used | upstream-eager | upstream-compile | loop | grouped | dense | CUDA-graph capture |
|---|---|---|---|---|---|---|---|---|
| Mixtral-8x7B | 2 | 3 | 4.19e-03 | 2.20e-03 | 4.74e-03 | 4.27e-03 | 4.20e-03 | upstream-eager: ok; grouped: RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph capture unless the CPU tensor is pinned. Please use tensor.pin_memory() or allocate the tensor with pin_memory=True.; dense: ok |
| OLMoE-1B-7B | 16 | 58 | 4.32e-03 | 2.15e-03 | 5.59e-03 | 4.32e-03 | 4.32e-03 | upstream-eager: ok; grouped: RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph capture unless the CPU tensor is pinned. Please use tensor.pin_memory() or allocate the tensor with pin_memory=True.; dense: ok |

Tiny full `Transformer` (dim 64, 4 experts, top-2), batch 2 vs two batch-1 calls:

| S | B=1 output | B=2 output | MoE in -> out shape | h + moe(h) |
|---|---|---|---|---|
| 1 | [2, 1, 128] | RuntimeError: shape '[1, 2, 1, 8, 2]' is invalid for input of size 16 | [2, 1, 64] -> [2, 64] | [2, 2, 64] |
| 4 | [2, 4, 128] | RuntimeError: The size of tensor a (4) must match the size of tensor b (8) at non-singleton dimension 1 | [2, 4, 64] -> [8, 64] | RuntimeError: The size of tensor a (4) must match the size of tensor b (8) at non-singleton dimension 1 |

## 2. Routing numerics (65536 tokens, x ~ N(0, 1), synthetic bf16 router)

Top-A set mismatch vs fp64 routing / exact tie between the A-th and (A+1)-th probability / mismatch vs HF-style (fp32 softmax of bf16 logits) / mean |combine weight error| on matching tokens.

| config | logit std | mean max p | bf16 softmax (upstream) | fp32 softmax of bf16 logits (HF) | fp32 logits |
|---|---|---|---|---|---|
| Mixtral-8x7B | 0.5 | 0.233 | 0.72% / 1.41% / 0.54% / 1.1e-03 | 0.18% / 0.37% / 0.00% / 2.8e-04 | 0.00% / 0.00% / 0.18% / 3.6e-08 |
| Mixtral-8x7B | 1.0 | 0.361 | 0.33% / 0.67% / 0.16% / 1.2e-03 | 0.18% / 0.37% / 0.00% / 5.0e-04 | 0.00% / 0.00% / 0.18% / 6.0e-08 |
| Mixtral-8x7B | 2.0 | 0.566 | 0.22% / 0.44% / 0.04% / 1.3e-03 | 0.18% / 0.37% / 0.00% / 7.8e-04 | 0.00% / 0.00% / 0.18% / 9.4e-08 |
| Mixtral-8x7B | 4.0 | 0.759 | 0.19% / 0.39% / 0.01% / 1.4e-03 | 0.18% / 0.37% / 0.00% / 1.0e-03 | 0.00% / 0.00% / 0.18% / 1.3e-07 |
| OLMoE-1B-7B | 0.5 | 0.046 | 3.62% / 6.89% / 1.48% / 3.1e-04 | 2.23% / 4.31% / 0.00% / 1.4e-04 | 0.00% / 0.00% / 2.23% / 9.0e-09 |
| OLMoE-1B-7B | 1.0 | 0.107 | 2.33% / 4.54% / 0.12% / 3.9e-04 | 2.23% / 4.31% / 0.00% / 2.9e-04 | 0.00% / 0.00% / 2.23% / 1.4e-08 |
| OLMoE-1B-7B | 2.0 | 0.303 | 2.23% / 4.31% / 0.00% / 6.2e-04 | 2.23% / 4.31% / 0.00% / 5.6e-04 | 0.00% / 0.00% / 2.23% / 2.6e-08 |
| OLMoE-1B-7B | 4.0 | 0.594 | 2.23% / 4.31% / 0.00% / 9.0e-04 | 2.23% / 4.31% / 0.00% / 8.6e-04 | 0.00% / 0.00% / 2.23% / 3.8e-08 |

Output error vs fp32 FFN with fp64 routing (logit std 1):

| config | T | mismatched tokens | bf16 FFN, fp64 routing | bf16 FFN, upstream routing | upstream routing, mismatched tokens only | matched tokens only |
|---|---|---|---|---|---|---|
| Mixtral-8x7B | 1024 | 7 | 5.02e-03 | 5.33e-02 | 6.20e-01 | 5.71e-03 |
| OLMoE-1B-7B | 4096 | 95 | 5.23e-03 | 4.12e-02 | 2.61e-01 | 6.80e-03 |

## 3. Mixtral-8x7B layer (2.62 GiB bf16 experts)

### Extra peak memory per forward, MiB (cap: 60% of 8 GiB minus the weights)

| T | experts touched | gathered copies MiB (T·A experts) | upstream-eager | upstream-compile | upstream-compile-dyn | loop | grouped | dense |
|---|---|---|---|---|---|---|---|---|
| 1 | 2 | 672 | 672 | 0 | not run | 0 | 0 | 1 |
| 2 | 4 | 1344 | 1344 | 1 | 1 | 0 | 0 | 1 |
| 3 | 4 | 2016 | OOM | not run | not run | 0 | 1 | 2 |
| 4 | 5 | 2688 | OOM at smaller T | 1 | 1 | 0 | 1 | 3 |
| 8 | 7 | 5376 | OOM at smaller T | not run | 2 | 0 | 1 | 5 |
| 16 | 8 | 10752 | OOM at smaller T | not run | 4 | 1 | 3 | 11 |
| 64 | 8 | 43008 | OOM at smaller T | not run | 16 | 3 | 12 | 42 |
| 256 | 8 | 172032 | OOM at smaller T | not run | not run | 10 | 46 | 168 |
| 1024 | 8 | 688128 | OOM at smaller T | not run | not run | 41 | 184 | 672 |

### CUDA-graph replay (GPU time), µs

| T | upstream-eager | upstream-compile | upstream-compile-dyn | loop | grouped | dense |
|---|---|---|---|---|---|---|
| 1 | 8903 | 2795 | not run | - | - | 11150 |
| 2 | 17823 | 5590 | 5598 | - | - | 11311 |
| 3 | OOM | not run | not run | - | - | 11337 |
| 4 | OOM at smaller T | 11145 | 11145 | - | - | 11354 |
| 8 | OOM at smaller T | not run | 22245 | - | - | 11399 |
| 16 | OOM at smaller T | not run | 44448 | - | - | 11506 |
| 64 | OOM at smaller T | not run | 177599 | - | - | 12014 |
| 256 | OOM at smaller T | not run | not run | - | - | 26194 |
| 1024 | OOM at smaller T | not run | not run | - | - | 102967 |

### eager launches (wall per call incl. host syncs), µs

| T | upstream-eager | upstream-compile | upstream-compile-dyn | loop | grouped | dense |
|---|---|---|---|---|---|---|
| 1 | 8987 | 2854 | not run | 4273 | 3170 | 11347 |
| 2 | 18025 | 5690 | 7730 | 7474 | 5985 | 11508 |
| 3 | OOM | not run | not run | 7081 | 6119 | 11490 |
| 4 | OOM at smaller T | 11347 | 11309 | 8992 | 7717 | 11559 |
| 8 | OOM at smaller T | not run | 22487 | 12120 | 10490 | 11613 |
| 16 | OOM at smaller T | not run | 44703 | 13686 | 11994 | 11684 |
| 64 | OOM at smaller T | not run | 177774 | 13956 | 12434 | 12332 |
| 256 | OOM at smaller T | not run | not run | 15821 | 14543 | 26332 |
| 1024 | OOM at smaller T | not run | not run | 33125 | 31381 | 103303 |

### Bandwidth against the touched experts' bytes (best of graph / eager), GB/s

| T | upstream-eager | upstream-compile | upstream-compile-dyn | loop | grouped | dense |
|---|---|---|---|---|---|---|
| 1 | 79 | 252 | not run | 165 | 222 | 63 |
| 2 | 79 | 252 | 252 | 189 | 235 | 125 |
| 3 | OOM | not run | not run | 199 | 230 | 124 |
| 4 | OOM at smaller T | 158 | 158 | 196 | 228 | 155 |
| 8 | OOM at smaller T | not run | 111 | 203 | 235 | 216 |
| 16 | OOM at smaller T | not run | 63 | 206 | 235 | 245 |
| 64 | OOM at smaller T | not run | 16 | 202 | 227 | 235 |
| 256 | OOM at smaller T | not run | not run | 178 | 194 | 108 |
| 1024 | OOM at smaller T | not run | not run | 85 | 90 | 27 |

### torch.compile(upstream): first call (compile + tuning; ~0.3 s = Inductor cache hit) and extern kernel calls

| T | variant | first call s | extern kernel calls |
|---|---|---|---|
| 1 | upstream-compile | 0.5 | none |
| 2 | upstream-compile | 0.2 | mm x1 |
| 2 | upstream-compile-dyn | 18.3 | mm x1 |
| 4 | upstream-compile | 0.3 | mm x1 |
| 4 | upstream-compile-dyn | 0.6 | mm x1 |
| 8 | upstream-compile-dyn | 0.3 | mm x1 |
| 16 | upstream-compile-dyn | 0.4 | mm x1 |
| 64 | upstream-compile-dyn | 52.1 | mm x1 |

## 3. OLMoE-1B-7B layer (0.75 GiB bf16 experts)

### Extra peak memory per forward, MiB (cap: 60% of 8 GiB minus the weights)

| T | experts touched | gathered copies MiB (T·A experts) | upstream-eager | upstream-compile | upstream-compile-dyn | loop | grouped | dense |
|---|---|---|---|---|---|---|---|---|
| 1 | 8 | 96 | 96 | 0 | not run | 0 | 0 | 0 |
| 2 | 16 | 192 | 192 | 0 | 0 | 0 | 0 | 1 |
| 4 | 25 | 384 | 384 | 1 | 1 | 0 | 0 | 2 |
| 8 | 43 | 768 | 769 | not run | 1 | 0 | 1 | 3 |
| 16 | 56 | 1536 | 1537 | 2 | 2 | 0 | 2 | 6 |
| 32 | 64 | 3072 | 3075 | not run | 4 | 0 | 4 | 12 |
| 64 | 64 | 6144 | OOM | 8 | 8 | 0 | 7 | 24 |
| 256 | 64 | 24576 | OOM at smaller T | not run | 32 | 2 | 29 | 97 |
| 1024 | 64 | 98304 | OOM at smaller T | not run | not run | 6 | 116 | 388 |
| 4096 | 64 | 393216 | OOM at smaller T | not run | not run | 25 | 465 | 1553 |

### CUDA-graph replay (GPU time), µs

| T | upstream-eager | upstream-compile | upstream-compile-dyn | loop | grouped | dense |
|---|---|---|---|---|---|---|
| 1 | 1238 | 412 | not run | - | - | 3235 |
| 2 | 2585 | 816 | 1240 | - | - | 3749 |
| 4 | 4721 | 1609 | 1609 | - | - | 3274 |
| 8 | 8994 | not run | 3150 | - | - | 3304 |
| 16 | 16184 | 6363 | 6366 | - | - | 3348 |
| 32 | 30401 | not run | 12717 | - | - | 3455 |
| 64 | OOM | 25361 | 25365 | - | - | 3664 |
| 256 | OOM at smaller T | not run | 101841 | - | - | 7820 |
| 1024 | OOM at smaller T | not run | not run | - | - | 31576 |
| 4096 | OOM at smaller T | not run | not run | - | - | 126264 |

### eager launches (wall per call incl. host syncs), µs

| T | upstream-eager | upstream-compile | upstream-compile-dyn | loop | grouped | dense |
|---|---|---|---|---|---|---|
| 1 | 1271 | 432 | not run | 7296 | 1526 | 3300 |
| 2 | 2629 | 840 | 1304 | 13097 | 1679 | 3841 |
| 4 | 4801 | 1640 | 1645 | 10252 | 2202 | 3334 |
| 8 | 9142 | not run | 3197 | 18724 | 3061 | 3366 |
| 16 | 16368 | 6438 | 6479 | 20442 | 5376 | 3414 |
| 32 | 30529 | not run | 12879 | 23588 | 5494 | 3511 |
| 64 | OOM | 25598 | 25605 | 21328 | 6674 | 3742 |
| 256 | OOM at smaller T | not run | 102101 | 23643 | 7111 | 7983 |
| 1024 | OOM at smaller T | not run | not run | 24892 | 7879 | 31911 |
| 4096 | OOM at smaller T | not run | not run | 35025 | 22051 | 126463 |

### Bandwidth against the touched experts' bytes (best of graph / eager), GB/s

| T | upstream-eager | upstream-compile | upstream-compile-dyn | loop | grouped | dense |
|---|---|---|---|---|---|---|
| 1 | 81 | 244 | not run | 14 | 66 | 31 |
| 2 | 78 | 247 | 162 | 15 | 120 | 54 |
| 4 | 67 | 195 | 196 | 31 | 143 | 96 |
| 8 | 60 | not run | 172 | 29 | 177 | 164 |
| 16 | 44 | 111 | 111 | 34 | 131 | 210 |
| 32 | 26 | not run | 63 | 34 | 147 | 233 |
| 64 | OOM | 32 | 32 | 38 | 121 | 220 |
| 256 | OOM at smaller T | not run | 8 | 34 | 113 | 103 |
| 1024 | OOM at smaller T | not run | not run | 32 | 102 | 26 |
| 4096 | OOM at smaller T | not run | not run | 23 | 37 | 6 |

### torch.compile(upstream): first call (compile + tuning; ~0.3 s = Inductor cache hit) and extern kernel calls

| T | variant | first call s | extern kernel calls |
|---|---|---|---|
| 1 | upstream-compile | 0.5 | none |
| 2 | upstream-compile | 0.4 | mm x1 |
| 2 | upstream-compile-dyn | 18.3 | mm x1 |
| 4 | upstream-compile | 0.4 | mm x1 |
| 4 | upstream-compile-dyn | 0.5 | mm x1 |
| 8 | upstream-compile-dyn | 0.3 | mm x1 |
| 16 | upstream-compile | 0.2 | mm x1 |
| 16 | upstream-compile-dyn | 0.3 | mm x1 |
| 32 | upstream-compile-dyn | 0.4 | mm x1 |
| 64 | upstream-compile | 0.3 | mm x1 |
| 64 | upstream-compile-dyn | 0.3 | mm x1 |
| 256 | upstream-compile-dyn | 45.1 | mm x1 |
