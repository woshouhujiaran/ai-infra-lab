triton c01b6774b1 tutorial 04, triton-windows 3.8.0, torch 2.14.1+cu126, NVIDIA GeForce RTX 4060 Laptop GPU (memory cap 0.6), runtime 74.8 s

## 1a. Keep rate vs 1 - p (4 seeds x 2^25 elements; z = binomial z-score)

| variant | p | keep rate | expected | z |
|---|---|---|---|---|
| tutorial | 0.01 | 0.990006 | 0.99 | 0.74 |
| tutorial | 0.1 | 0.900029 | 0.9 | 1.11 |
| tutorial | 0.5 | 0.499961 | 0.5 | -0.91 |
| tutorial | 0.9 | 0.099963 | 0.1 | -1.43 |
| tutorial | 0.99 | 0.010002 | 0.01 | 0.23 |
| x4 | 0.01 | 0.990003 | 0.99 | 0.32 |
| x4 | 0.1 | 0.900025 | 0.9 | 0.97 |
| x4 | 0.5 | 0.499967 | 0.5 | -0.76 |
| x4 | 0.9 | 0.099935 | 0.1 | -2.5 |
| x4 | 0.99 | 0.01 | 0.01 | -0.04 |
| torch | 0.01 | 0.989998 | 0.99 | -0.28 |
| torch | 0.1 | 0.900006 | 0.9 | 0.24 |
| torch | 0.5 | 0.499983 | 0.5 | -0.39 |
| torch | 0.9 | 0.100013 | 0.1 | 0.5 |
| torch | 0.99 | 0.009991 | 0.01 | -1.06 |

## 1b. Lag-k mask agreement (p = 0.5, 2^26 elements; expected 0.5)

| variant | lag | agree | z |
|---|---|---|---|
| tutorial | 1 | 0.499983 | -0.27 |
| tutorial | 2 | 0.500068 | 1.11 |
| tutorial | 3 | 0.499974 | -0.43 |
| tutorial | 4 | 0.50003 | 0.49 |
| tutorial | 5 | 0.499963 | -0.6 |
| tutorial | 8 | 0.500001 | 0.01 |
| tutorial | 64 | 0.499985 | -0.25 |
| tutorial | 1024 | 0.49988 | -1.96 |
| tutorial | 1048576 | 0.499919 | -1.31 |
| x4 | 1 | 0.500081 | 1.33 |
| x4 | 2 | 0.50005 | 0.82 |
| x4 | 3 | 0.500007 | 0.12 |
| x4 | 4 | 0.500025 | 0.41 |
| x4 | 5 | 0.499972 | -0.46 |
| x4 | 8 | 0.499818 | -2.98 |
| x4 | 64 | 0.500059 | 0.96 |
| x4 | 1024 | 0.49995 | -0.81 |
| x4 | 1048576 | 0.499853 | -2.38 |
| torch | 1 | 0.500107 | 1.75 |
| torch | 2 | 0.499958 | -0.69 |
| torch | 3 | 0.500028 | 0.46 |
| torch | 4 | 0.500038 | 0.62 |
| torch | 5 | 0.499903 | -1.59 |
| torch | 8 | 0.500043 | 0.71 |
| torch | 64 | 0.499909 | -1.49 |
| torch | 1024 | 0.499977 | -0.39 |
| torch | 1048576 | 0.499964 | -0.59 |

## 1c. Mask agreement between seeds (p = 0.5, 16777216 elements; expected 0.5)

| variant | seed a | seed b | agree | z |
|---|---|---|---|---|
| tutorial | 0 | 1 | 0.500057 | 0.47 |
| tutorial | 1 | 2 | 0.500069 | 0.57 |
| tutorial | 0 | 65536 | 0.500027 | 0.22 |
| tutorial | 0 | 2147483648 | 0.500103 | 0.84 |
| tutorial | 0 | 4294967296 | 0.499858 | -1.16 |
| tutorial | 7 | 4294967303 | 0.50003 | 0.25 |
| tutorial | -1 | 4294967295 | 0.500107 | 0.87 |
| tutorial | 123 | 1099511627899 | 0.499785 | -1.76 |
| x4 | 0 | 1 | 0.499985 | -0.13 |
| x4 | 1 | 2 | 0.499959 | -0.34 |
| x4 | 0 | 65536 | 0.49992 | -0.66 |
| x4 | 0 | 2147483648 | 0.500101 | 0.83 |
| x4 | 0 | 4294967296 | 0.499774 | -1.85 |
| x4 | 7 | 4294967303 | 0.500047 | 0.38 |
| x4 | -1 | 4294967295 | 0.499987 | -0.11 |
| x4 | 123 | 1099511627899 | 0.500005 | 0.04 |

## 2a. p = 0 is not the identity

uint8 ones, 2147479552 elements x 32 seeds = 68719345664 elements, tutorial kernel as written: **37 dropped** (expected 32.0 from P(rand == 0) = 2^-31); rand value at the dropped offsets: [0.0]; first hits (seed, offset): [(1, 91093890), (1, 1711450034), (2, 1139270164), (2, 1450321319), (3, 355990253), (3, 1136270787), (3, 1799755431), (4, 1088694098)].

## 2b. p out of range / n = 0 / non-contiguous

| case | tutorial | torch F.dropout |
|---|---|---|
| 0.0 | keep 1.0000, kept value 1 | ok |
| 1.0 | keep 0.0000, kept value 0 | ok |
| 1.5 | keep 0.0000, kept value 0 | ValueError: dropout probability has to be between 0 and 1, but got 1.5 |
| -0.5 | keep 1.0000, kept value 0.6667 | ValueError: dropout probability has to be between 0 and 1, but got -0.5 |
| n = 0 | returns 0 elements | ok |
| non-contiguous x | AssertionError:  | ok |

## 2c. NaN / inf inputs at dropped positions (p = 0.5)

3156 non-finite inputs: tutorial leaves 1598 (0.506), torch leaves 3156 (1.0).

## 2d. dtype rounding vs fp64 reference

| dtype | p | keep rate | max rel err vs rounded fp64 | mask == fp32 mask |
|---|---|---|---|---|
| fp32 | 0.1 | 0.89997 | 1.1920917586405195e-07 | True |
| fp32 | 0.5 | 0.50022 | 0.0 | True |
| fp16 | 0.1 | 0.89997 | 0.0 | True |
| fp16 | 0.5 | 0.50022 | 0.0 | True |
| bf16 | 0.1 | 0.89997 | 0.0 | True |
| bf16 | 0.5 | 0.50022 | 0.0 | True |

## 2e. Mask independence of launch config (n = 2^24 + 37)

| BLOCK_SIZE | num_warps | bit-identical to default |
|---|---|---|
| 128 | 1 | True |
| 128 | 2 | True |
| 128 | 4 | True |
| 128 | 8 | True |
| 256 | 1 | True |
| 256 | 2 | True |
| 256 | 4 | True |
| 256 | 8 | True |
| 1024 | 1 | True |
| 1024 | 2 | True |
| 1024 | 4 | True |
| 1024 | 8 | True |
| 4096 | 1 | True |
| 4096 | 2 | True |
| 4096 | 4 | True |
| 4096 | 8 | True |
| 8192 | 1 | True |
| 8192 | 2 | True |
| 8192 | 4 | True |
| 8192 | 8 | True |
| int64 rewrite | 4 | True |

## 3a. Stacked dropout layers (x = ones, p = 0.5)

| layers | seeds | mean output | keep rate | expected keep |
|---|---|---|---|---|
| 1 | same seed | 1.0001 | 0.5 | 0.5 |
| 1 | distinct seeds | 1.0001 | 0.5 | 0.5 |
| 2 | same seed | 2.0001 | 0.5 | 0.5 |
| 2 | distinct seeds | 1.0 | 0.25 | 0.25 |
| 3 | same seed | 4.0002 | 0.5 | 0.5 |
| 3 | distinct seeds | 1.0 | 0.125 | 0.125 |
| 4 | same seed | 8.0005 | 0.5 | 0.5 |
| 4 | distinct seeds | 1.0012 | 0.0626 | 0.0625 |
| 4 | torch (RNG advances) | 0.9994 | 0.0625 | 0.0625 |

## 3b. Memory held between forward and backward (2^27 fp16 elements)

| variant | extra held MiB | bytes / element | grad == forward mask |
|---|---|---|---|
| seeded autograd.Function | 0.0 | 0.0 |  |
| torch F.dropout | 128.0 | 1.0 |  |
| torch.compile F.dropout | 128.0 | 1.0 |  |
| seeded backward mask == forward mask |  |  | True |

## 3c. Gradient error under non-reentrant checkpointing (recompute after the RNG advanced)

| dropout | preserve_rng_state | grad rel err vs no checkpoint |
|---|---|---|
| torch F.dropout | False | 0.8139166235923767 |
| torch F.dropout | True | 0.0 |
| seeded | False | 0.0 |
| seeded | True | 0.0 |

## 4. int32 offsets past 2^31 (uint8, n = 2148532224, p = 0.5; one child process per kernel)

```
[
 {
  "n": 2148532224,
  "int32_limit": 2147483648,
  "mode": "tutorial",
  "result": "AcceleratorError: CUDA error: an illegal memory access was encountered\nSearch for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES."
 },
 {
  "n": 2148532224,
  "int32_limit": 2147483648,
  "mode": "int64",
  "tail_elements": 1048576,
  "tail_keep_rate": 0.5003,
  "tail_equals_head_first_block": false,
  "head_sample_matches_tutorial_below_limit": true,
  "result": "completed"
 }
]
```

## 5a. Philox throughput (no memory traffic)

| kernel | ms | G calls / s | G numbers / s |
|---|---|---|---|
| rand (1 of 4 outputs) | 3.244 | 82.7 | 82.7 |
| rand4x | 4.028 | 66.6 | 266.5 |

## 5b. Launch-config sweep (2^26 elements, GB/s = 2 * n * bytes / time)

| dtype | kernel | BLOCK | warps | ms | GB/s |
|---|---|---|---|---|---|
| fp32 | tutorial | 256 | 2 | 2.3327 | 230.2 |
| fp32 | tutorial | 256 | 4 | 2.3286 | 230.6 |
| fp32 | tutorial | 256 | 8 | 2.3439 | 229.0 |
| fp32 | tutorial | 256 | 16 | 2.3951 | 224.2 |
| fp32 | tutorial | 512 | 2 | 2.348 | 228.6 |
| fp32 | tutorial | 512 | 4 | 2.3337 | 230.1 |
| fp32 | tutorial | 512 | 8 | 2.3276 | 230.7 |
| fp32 | tutorial | 512 | 16 | 2.3214 | 231.3 |
| fp32 | tutorial | 1024 | 2 | 2.2958 | 233.8 |
| fp32 | tutorial | 1024 | 4 | 2.3439 | 229.0 |
| fp32 | tutorial | 1024 | 8 | 2.3368 | 229.7 |
| fp32 | tutorial | 1024 | 16 | 2.3347 | 230.0 |
| fp32 | tutorial | 2048 | 2 | 2.3122 | 232.2 |
| fp32 | tutorial | 2048 | 4 | 2.3276 | 230.7 |
| fp32 | tutorial | 2048 | 8 | 2.348 | 228.6 |
| fp32 | tutorial | 2048 | 16 | 2.3501 | 228.4 |
| fp32 | tutorial | 4096 | 2 | 2.3429 | 229.1 |
| fp32 | tutorial | 4096 | 4 | 2.3286 | 230.6 |
| fp32 | tutorial | 4096 | 8 | 2.3286 | 230.6 |
| fp32 | tutorial | 4096 | 16 | 2.346 | 228.8 |
| fp32 | x4 | 128 | 2 | 2.3286 | 230.6 |
| fp32 | x4 | 128 | 4 | 2.3368 | 229.7 |
| fp32 | x4 | 128 | 8 | 2.3173 | 231.7 |
| fp32 | x4 | 256 | 2 | 2.3521 | 228.2 |
| fp32 | x4 | 256 | 4 | 2.388 | 224.8 |
| fp32 | x4 | 256 | 8 | 2.3982 | 223.9 |
| fp32 | x4 | 512 | 2 | 2.3194 | 231.5 |
| fp32 | x4 | 512 | 4 | 2.3316 | 230.3 |
| fp32 | x4 | 512 | 8 | 2.3439 | 229.0 |
| fp32 | x4 | 1024 | 2 | 2.3296 | 230.5 |
| fp32 | x4 | 1024 | 4 | 2.304 | 233.0 |
| fp32 | x4 | 1024 | 8 | 2.3409 | 229.3 |
| fp16 | tutorial | 256 | 2 | 1.1612 | 231.2 |
| fp16 | tutorial | 256 | 4 | 1.1715 | 229.1 |
| fp16 | tutorial | 256 | 8 | 1.2196 | 220.1 |
| fp16 | tutorial | 256 | 16 | 2.1166 | 126.8 |
| fp16 | tutorial | 512 | 2 | 1.1679 | 229.9 |
| fp16 | tutorial | 512 | 4 | 1.1633 | 230.8 |
| fp16 | tutorial | 512 | 8 | 1.1745 | 228.5 |
| fp16 | tutorial | 512 | 16 | 1.236 | 217.2 |
| fp16 | tutorial | 1024 | 2 | 1.1766 | 228.1 |
| fp16 | tutorial | 1024 | 4 | 1.1658 | 230.3 |
| fp16 | tutorial | 1024 | 8 | 1.1643 | 230.6 |
| fp16 | tutorial | 1024 | 16 | 1.1254 | 238.5 |
| fp16 | tutorial | 2048 | 2 | 1.1786 | 227.8 |
| fp16 | tutorial | 2048 | 4 | 1.1715 | 229.1 |
| fp16 | tutorial | 2048 | 8 | 1.1684 | 229.7 |
| fp16 | tutorial | 2048 | 16 | 1.1633 | 230.8 |
| fp16 | tutorial | 4096 | 2 | 1.1766 | 228.1 |
| fp16 | tutorial | 4096 | 4 | 1.1735 | 228.7 |
| fp16 | tutorial | 4096 | 8 | 1.1776 | 228.0 |
| fp16 | tutorial | 4096 | 16 | 1.1735 | 228.7 |
| fp16 | x4 | 128 | 2 | 1.1674 | 230.0 |
| fp16 | x4 | 128 | 4 | 1.1766 | 228.1 |
| fp16 | x4 | 128 | 8 | 1.4479 | 185.4 |
| fp16 | x4 | 256 | 2 | 1.1694 | 229.5 |
| fp16 | x4 | 256 | 4 | 1.1694 | 229.5 |
| fp16 | x4 | 256 | 8 | 1.1848 | 226.6 |
| fp16 | x4 | 512 | 2 | 1.1858 | 226.4 |
| fp16 | x4 | 512 | 4 | 1.1909 | 225.4 |
| fp16 | x4 | 512 | 8 | 1.196 | 224.4 |
| fp16 | x4 | 1024 | 2 | 1.1786 | 227.8 |
| fp16 | x4 | 1024 | 4 | 1.1735 | 228.7 |
| fp16 | x4 | 1024 | 8 | 1.1745 | 228.5 |

## 5c. Throughput (GB/s of 2 * n * bytes; do_bench median, L2 flushed; us in parentheses; best configs (BLOCK x warps) from 5b: {'fp32/tutorial': (1024, 2), 'fp32/x4': (1024, 4), 'fp16/tutorial': (1024, 16), 'fp16/x4': (128, 2)})

| dtype | n | MiB | copy | tutorial | tutorial best config | int64 offsets | x4 rewrite best config | torch F.dropout | torch.compile F.dropout | tutorial mask kernel only (int32 mask) | tutorial mask incl. torch.rand mask |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fp32 | 1048576 | 4.0 | 204.8 (41.0) | 190.5 (44.0) | 204.7 (41.0) | 210.1 (39.9) | 178.1 (47.1) | 151.7 (55.3) | 199.8 (42.0) | 122.3 (68.6) | 118.7 (70.7) |
| fp32 | 16777216 | 64.0 | 225.6 (594.9) | 225.6 (594.9) | 228.7 (586.8) | 226.4 (592.9) | 227.2 (590.8) | 192.5 (697.3) | 227.2 (590.8) | 152.2 (881.7) | 72.7 (1846.3) |
| fp32 | 134217728 | 512.0 | 228.9 (4690.9) | 229.5 (4678.7) | 234.7 (4575.2) | 227.3 (4724.7) | 237.3 (4524.0) | 203.8 (5269.5) | 231.0 (4647.9) | 157.0 (6837.2) | 73.6 (14586.9) |
| fp16 | 1048576 | 2.0 | 185.9 (22.6) | 186.2 (22.5) | 170.7 (24.6) | 195.0 (21.5) | 163.8 (25.6) | 136.5 (30.7) | 124.1 (33.8) | 87.1 (48.1) | 74.5 (56.3) |
| fp16 | 16777216 | 32.0 | 225.2 (298.0) | 226.0 (297.0) | 222.2 (302.1) | 226.8 (295.9) | 224.4 (299.0) | 171.6 (391.2) | 187.8 (357.4) | 111.8 (600.1) | 43.0 (1559.0) |
| fp16 | 134217728 | 256.0 | 229.1 (2342.9) | 231.0 (2324.5) | 242.4 (2214.9) | 230.9 (2325.5) | 230.0 (2334.7) | 183.0 (2933.8) | 201.0 (2671.6) | 122.1 (4396.5) | 44.1 (12179.5) |
