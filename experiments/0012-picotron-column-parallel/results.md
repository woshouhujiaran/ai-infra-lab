# 0012 results: picotron ColumnParallelLinear on CPU (gloo)

torch 2.14.1+cu126, Intel64 Family 6 Model 186 Stepping 2, GenuineIntel, 2 threads per rank, total 224 s.

## 1. Shard storage (rank 0; RSS delta is the max over ranks)

| Module | TP | Shard shape | Shard MiB | Storage MiB | RSS delta MiB | torch.save MiB | Storage after .to(fp32) | after .to(bf16) |
|---|---|---|---|---|---|---|---|---|
| ColumnParallelLinear | 1 | [14336, 4096] | 224 | 224 | 224 | 224 | 224 | 112 |
| ColumnParallelLinear + clone | 1 | [14336, 4096] | 224 | 224 | 224 | 224 | 224 | 112 |
| RowParallelLinear | 1 | [4096, 14336] | 224 | 224 | 224 | 224 | 224 | 112 |
| VocabParallelEmbedding | 1 | [32000, 1024] | 125 | 125 | 125 | 125 | 125 | 62 |
| ColumnParallelLinear | 2 | [7168, 4096] | 112 | 224 | 224 | 224 | 224 | 56 |
| ColumnParallelLinear + clone | 2 | [7168, 4096] | 112 | 112 | 112 | 112 | 112 | 56 |
| RowParallelLinear | 2 | [4096, 7168] | 112 | 112 | 112 | 112 | 112 | 56 |
| VocabParallelEmbedding | 2 | [16000, 1024] | 62 | 125 | 125 | 125 | 125 | 31 |
| ColumnParallelLinear | 4 | [3584, 4096] | 56 | 224 | 224 | 224 | 224 | 28 |
| ColumnParallelLinear + clone | 4 | [3584, 4096] | 56 | 56 | 56 | 56 | 56 | 28 |
| RowParallelLinear | 4 | [4096, 3584] | 56 | 56 | 56 | 56 | 56 | 28 |
| VocabParallelEmbedding | 4 | [8000, 1024] | 31 | 125 | 125 | 125 | 125 | 16 |

## 2. Init vs nn.Linear (same seed on every rank / seed + rank)

| TP | Same seed: weight bitwise equal | Bias all zero | seed + rank: shard equal per rank |
|---|---|---|---|
| 1 | True | True | [True] |
| 2 | True | True | [True, False] |
| 4 | True | True | [True, False, False, False] |

## 3. Correctness (1024 -> 4096, 256 tokens, gather_output=True)

Relative L2 error vs fp64; `=` marks bitwise equality with single-process nn.Linear in the same dtype.

| dtype | Input | TP | Mode | y | dx | dW | db | dx max abs vs single | dx identical on all ranks |
|---|---|---|---|---|---|---|---|---|---|
| fp32 | 2d | - | single | 2.51e-07 = | 2.64e-07 = | 2.89e-07 = | 1.13e-07 = | 0.00e+00 | - |
| fp32 | 2d | 1 | sync | 2.51e-07 = | 2.64e-07 = | 2.89e-07 = | 1.13e-07 = | 0.00e+00 | True |
| fp32 | 2d | 1 | async | RuntimeError: inconsistent tensor size, expected tensor [1048576] and src [262144] to have the same number of elements, but got 1048576 and 262144 elements respectively | | | | | |
| fp32 | 3d | - | single | 2.51e-07 = | 2.64e-07 = | 2.89e-07 = | 1.13e-07 = | 0.00e+00 | - |
| fp32 | 3d | 1 | sync | 2.51e-07 = | 2.64e-07 = | 2.89e-07 = | 1.13e-07 = | 0.00e+00 | True |
| fp32 | 3d | 1 | async | 2.52e-07 | 2.64e-07 = | 2.89e-07 = | 1.13e-07 = | 0.00e+00 | True |
| fp32 | 4d | - | single | 2.51e-07 = | 2.64e-07 = | 2.89e-07 = | 1.13e-07 = | 0.00e+00 | - |
| fp32 | 4d | 1 | sync | 2.51e-07 = | 2.64e-07 = | 2.89e-07 = | 1.13e-07 = | 0.00e+00 | True |
| fp32 | 4d | 1 | async | RuntimeError: t() expects a tensor with <= 2 dimensions, but self is 3D | | | | | |
| bf16 | 2d | - | single | 2.77e-03 = | 2.77e-03 = | 2.88e-03 = | 2.36e-03 = | 0.00e+00 | - |
| bf16 | 2d | 1 | sync | 2.77e-03 = | 2.77e-03 = | 2.88e-03 = | 2.36e-03 = | 0.00e+00 | True |
| bf16 | 2d | 1 | async | RuntimeError: inconsistent tensor size, expected tensor [1048576] and src [262144] to have the same number of elements, but got 1048576 and 262144 elements respectively | | | | | |
| bf16 | 3d | - | single | 2.77e-03 = | 2.77e-03 = | 2.88e-03 = | 2.36e-03 = | 0.00e+00 | - |
| bf16 | 3d | 1 | sync | 2.77e-03 = | 2.77e-03 = | 2.88e-03 = | 2.36e-03 = | 0.00e+00 | True |
| bf16 | 3d | 1 | async | 3.24e-03 | 2.77e-03 = | 2.88e-03 = | 2.36e-03 = | 0.00e+00 | True |
| bf16 | 4d | - | single | 2.77e-03 = | 2.77e-03 = | 2.88e-03 = | 2.36e-03 = | 0.00e+00 | - |
| bf16 | 4d | 1 | sync | 2.77e-03 = | 2.77e-03 = | 2.88e-03 = | 2.36e-03 = | 0.00e+00 | True |
| bf16 | 4d | 1 | async | RuntimeError: t() expects a tensor with <= 2 dimensions, but self is 3D | | | | | |
| fp32 | 2d | 2 | sync | 2.51e-07 | 2.58e-07 | 2.88e-07 | 1.13e-07 = | 1.91e-06 | True |
| fp32 | 2d | 2 | async | RuntimeError: inconsistent tensor size, expected tensor [524288] and src [262144] to have the same number of elements, but got 524288 and 262144 elements respectively | | | | | |
| fp32 | 3d | 2 | sync | 2.51e-07 | 2.58e-07 | 2.88e-07 | 1.13e-07 = | 1.91e-06 | True |
| fp32 | 3d | 2 | async | 2.52e-07 | 2.58e-07 | 2.88e-07 | 1.13e-07 = | 1.91e-06 | True |
| fp32 | 4d | 2 | sync | 2.51e-07 | 2.58e-07 | 2.88e-07 | 1.13e-07 = | 1.91e-06 | True |
| fp32 | 4d | 2 | async | RuntimeError: t() expects a tensor with <= 2 dimensions, but self is 3D | | | | | |
| bf16 | 2d | 2 | sync | 2.77e-03 = | 3.31e-03 | 2.88e-03 = | 2.36e-03 = | 3.12e-02 | True |
| bf16 | 2d | 2 | async | RuntimeError: inconsistent tensor size, expected tensor [524288] and src [262144] to have the same number of elements, but got 524288 and 262144 elements respectively | | | | | |
| bf16 | 3d | 2 | sync | 2.77e-03 = | 3.31e-03 | 2.88e-03 = | 2.36e-03 = | 3.12e-02 | True |
| bf16 | 3d | 2 | async | 3.24e-03 | 3.31e-03 | 2.88e-03 = | 2.36e-03 = | 3.12e-02 | True |
| bf16 | 4d | 2 | sync | 2.77e-03 = | 3.31e-03 | 2.88e-03 = | 2.36e-03 = | 3.12e-02 | True |
| bf16 | 4d | 2 | async | RuntimeError: t() expects a tensor with <= 2 dimensions, but self is 3D | | | | | |
| fp32 | 2d | 4 | sync | 2.50e-07 | 2.54e-07 | 2.88e-07 | 1.13e-07 = | 1.91e-06 | True |
| fp32 | 2d | 4 | async | RuntimeError: Function LinearWithAsyncAllReduceBackward returned an invalid gradient at index 1 - got [] but expected shape compatible with [1024, 1024] | | | | | |
| fp32 | 3d | 4 | sync | 2.50e-07 | 2.54e-07 | 2.88e-07 | 1.13e-07 = | 1.91e-06 | True |
| fp32 | 3d | 4 | async | 2.51e-07 | 2.54e-07 | 2.88e-07 | 1.13e-07 = | 1.91e-06 | True |
| fp32 | 4d | 4 | sync | 2.50e-07 | 2.54e-07 | 2.88e-07 | 1.13e-07 = | 1.91e-06 | True |
| fp32 | 4d | 4 | async | RuntimeError: t() expects a tensor with <= 2 dimensions, but self is 3D | | | | | |
| bf16 | 2d | 4 | sync | 2.77e-03 = | 3.87e-03 | 2.88e-03 = | 2.36e-03 = | 3.12e-02 | True |
| bf16 | 2d | 4 | async | RuntimeError: Function LinearWithAsyncAllReduceBackward returned an invalid gradient at index 1 - got [] but expected shape compatible with [1024, 1024] | | | | | |
| bf16 | 3d | 4 | sync | 2.77e-03 = | 3.87e-03 | 2.88e-03 = | 2.36e-03 = | 3.12e-02 | True |
| bf16 | 3d | 4 | async | 3.24e-03 | 3.87e-03 | 2.88e-03 = | 2.36e-03 = | 3.12e-02 | True |
| bf16 | 4d | 4 | sync | 2.77e-03 = | 3.87e-03 | 2.88e-03 = | 2.36e-03 = | 3.12e-02 | True |
| bf16 | 4d | 4 | async | RuntimeError: t() expects a tensor with <= 2 dimensions, but self is 3D | | | | | |

## 4. CPU timing (1024 -> 4096, fp32, median ms, max over ranks)

| Tokens | TP | fwd | fwd + gather | local fwd+bwd | sync fwd+bwd | async fwd+bwd | dX all-reduce | up+gate picotron | up+gate shared copy | sync speedup vs TP 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| 256 | 1 | 11.6 | 11.1 | 33.8 | 33.6 | 33.5 | 0.0 | 76.4 | 74.9 | 1.00x |
| 256 | 2 | 5.6 | 22.6 | 32.8 | 27.3 | 30.1 | 14.5 | 69.1 | 48.6 | 1.23x |
| 256 | 4 | 7.5 | 16.3 | 23.4 | 25.1 | 23.6 | 2.7 | 54.4 | 49.4 | 1.34x |
| 1024 | 1 | 42.4 | 41.0 | 125.9 | 130.9 | 125.9 | 0.0 | 271.6 | 271.4 | 1.00x |
| 1024 | 2 | 36.5 | 93.8 | 67.1 | 94.4 | 91.1 | 29.0 | 241.6 | 236.9 | 1.39x |
| 1024 | 4 | 26.7 | 64.1 | 74.0 | 93.7 | 83.3 | 8.7 | 192.4 | 171.8 | 1.40x |
| 4096 | 1 | 168.1 | 221.5 | 494.9 | 514.6 | 532.9 | 0.0 | 1124.6 | 1104.1 | 1.00x |
| 4096 | 2 | 100.8 | 274.7 | 303.3 | 441.4 | 375.8 | 79.4 | 881.8 | 805.6 | 1.17x |
| 4096 | 4 | 92.8 | 261.2 | 289.5 | 320.3 | 297.4 | 43.7 | 693.2 | 634.9 | 1.61x |

up+gate dX, picotron vs shared copy (max abs diff / bitwise): TP 1 T 256: 0.0e+00 / True; TP 1 T 1024: 0.0e+00 / True; TP 1 T 4096: 0.0e+00 / True; TP 2 T 256: 2.4e-07 / False; TP 2 T 1024: 2.4e-07 / False; TP 2 T 4096: 2.4e-07 / False; TP 4 T 256: 2.4e-07 / False; TP 4 T 1024: 3.6e-07 / False; TP 4 T 4096: 4.8e-07 / False

sync vs async dX bitwise equal: TP 1 T 256: True; TP 1 T 1024: True; TP 1 T 4096: True; TP 2 T 256: True; TP 2 T 1024: True; TP 2 T 4096: True; TP 4 T 256: True; TP 4 T 1024: True; TP 4 T 4096: True

## 5. gloo fp32 all-reduce (median ms / algorithm bandwidth GB/s)

| TP | 0.25 MiB | 1 MiB | 4 MiB | 16 MiB |
|---|---|---|---|---|
| 2 | 0.39 / 0.67 | 1.35 / 0.78 | 30.22 / 0.14 | 101.39 / 0.17 |
| 4 | 1.89 / 0.14 | 2.38 / 0.44 | 7.63 / 0.55 | 45.37 / 0.37 |
