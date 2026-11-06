# 0013 results — EPLB on synthetic DeepSeek-V3-style routing

Env: Python 3.13.3, torch 2.14.1+cu126, numpy 2.5.4, 4 torch threads. Total 326 s.

README example reproduced: **True**. Invariant violations: **0**. Heap packing identical to upstream: **True** (38 checks). Swap search iterations per layer: mean 33.4, max 89.

## Routing scenarios (history window, mean over 58 layers)

| scenario | σ expert | σ group | max/mean expert load | min/mean | CV | hottest group share | hist→eval L1 (δ=0 / 0.3 / 0.6) |
|---|---|---|---|---|---|---|---|
| mild | 0.3 | 0.0 | 2.16 | 0.40 | 0.30 | 0.139 | 0.035 / 0.076 / 0.130 |
| skewed | 1.0 | 0.0 | 9.32 | 0.04 | 1.18 | 0.181 | 0.031 / 0.224 / 0.406 |
| hot-groups | 0.5 | 0.7 | 6.92 | 0.07 | 1.00 | 0.335 | 0.031 / 0.186 / 0.335 |

## GPU imbalance (fluid model, max/mean GPU load, mean over 58 layers)

`stale` = placement from the history window evaluated on the eval window with drift δ (eval bias correlation with history bias 1, 0.958, 0.857 for δ = 0, 0.3, 0.6); `oracle` = placement built on the eval window itself. `LB` = replica lower bound max(1, heaviest replica / mean GPU load) of the stale global placement. 1.000 = perfect.

### mild

| config | replicas | policy | stale δ=0 | stale δ=0.3 | stale δ=0.6 | oracle δ=0 | oracle δ=0.6 | worst layer δ=0 | node imb δ=0 | LB δ=0 |
|---|---|---|---|---|---|---|---|---|---|---|
| 32gpu-4node | 256 | none | 1.262 | 1.251 | 1.248 | — | — | 1.509 | 1.066 | 1.000 |
| 32gpu-4node | 256 | hierarchical | 1.049 | 1.087 | 1.139 | 1.033 | 1.031 | 1.117 | 1.020 | 1.000 |
| 32gpu-4node | 256 | global | 1.037 | 1.074 | 1.123 | 1.016 | 1.016 | 1.066 | 1.006 | 1.000 |
| 32gpu-4node | 256 | global+swap | 1.031 | 1.075 | 1.129 | 1.000 | 1.000 | 1.044 | 1.005 | 1.000 |
| 32gpu-4node | 288 | hierarchical | 1.045 | 1.079 | 1.123 | 1.026 | 1.025 | 1.084 | 1.020 | 1.000 |
| 32gpu-4node | 288 | global | 1.034 | 1.076 | 1.124 | 1.008 | 1.008 | 1.052 | 1.006 | 1.000 |
| 32gpu-4node | 288 | global+swap | 1.030 | 1.071 | 1.122 | 1.000 | 1.000 | 1.049 | 1.005 | 1.000 |
| 32gpu-4node | 320 | hierarchical | 1.043 | 1.071 | 1.117 | 1.023 | 1.022 | 1.079 | 1.020 | 1.000 |
| 32gpu-4node | 320 | global | 1.030 | 1.065 | 1.114 | 1.004 | 1.004 | 1.047 | 1.005 | 1.000 |
| 32gpu-4node | 320 | global+swap | 1.029 | 1.065 | 1.114 | 1.000 | 1.000 | 1.052 | 1.005 | 1.000 |
| 64gpu-8node | 256 | none | 1.416 | 1.410 | 1.411 | — | — | 1.902 | 1.110 | 1.000 |
| 64gpu-8node | 256 | hierarchical | 1.171 | 1.208 | 1.258 | 1.156 | 1.157 | 1.376 | 1.110 | 1.000 |
| 64gpu-8node | 256 | global | 1.081 | 1.131 | 1.216 | 1.067 | 1.067 | 1.176 | 1.027 | 1.000 |
| 64gpu-8node | 256 | global+swap | 1.052 | 1.123 | 1.211 | 1.004 | 1.003 | 1.145 | 1.012 | 1.000 |
| 64gpu-8node | 320 | hierarchical | 1.142 | 1.182 | 1.246 | 1.119 | 1.119 | 1.236 | 1.110 | 1.000 |
| 64gpu-8node | 320 | global | 1.050 | 1.110 | 1.200 | 1.007 | 1.007 | 1.073 | 1.012 | 1.000 |
| 64gpu-8node | 320 | global+swap | 1.047 | 1.108 | 1.196 | 1.001 | 1.001 | 1.077 | 1.010 | 1.000 |

### skewed

| config | replicas | policy | stale δ=0 | stale δ=0.3 | stale δ=0.6 | oracle δ=0 | oracle δ=0.6 | worst layer δ=0 | node imb δ=0 | LB δ=0 |
|---|---|---|---|---|---|---|---|---|---|---|
| 32gpu-4node | 256 | none | 2.310 | 2.283 | 2.295 | — | — | 3.097 | 1.204 | 1.218 |
| 32gpu-4node | 256 | hierarchical | 1.327 | 1.441 | 1.698 | 1.324 | 1.289 | 1.906 | 1.085 | 1.218 |
| 32gpu-4node | 256 | global | 1.275 | 1.385 | 1.650 | 1.270 | 1.237 | 1.881 | 1.056 | 1.218 |
| 32gpu-4node | 256 | global+swap | 1.272 | 1.386 | 1.650 | 1.262 | 1.225 | 1.881 | 1.054 | 1.218 |
| 32gpu-4node | 288 | hierarchical | 1.107 | 1.306 | 1.550 | 1.089 | 1.079 | 1.371 | 1.085 | 1.000 |
| 32gpu-4node | 288 | global | 1.030 | 1.254 | 1.494 | 1.005 | 1.005 | 1.049 | 1.005 | 1.000 |
| 32gpu-4node | 288 | global+swap | 1.029 | 1.253 | 1.497 | 1.000 | 1.000 | 1.052 | 1.005 | 1.000 |
| 32gpu-4node | 320 | hierarchical | 1.105 | 1.281 | 1.489 | 1.089 | 1.079 | 1.390 | 1.085 | 1.000 |
| 32gpu-4node | 320 | global | 1.029 | 1.235 | 1.458 | 1.005 | 1.005 | 1.045 | 1.005 | 1.000 |
| 32gpu-4node | 320 | global+swap | 1.027 | 1.242 | 1.479 | 1.000 | 1.000 | 1.048 | 1.005 | 1.000 |
| 64gpu-8node | 256 | none | 3.405 | 3.314 | 3.339 | — | — | 5.147 | 1.444 | 2.331 |
| 64gpu-8node | 256 | hierarchical | 2.444 | 2.377 | 2.556 | 2.443 | 2.344 | 3.705 | 1.444 | 2.331 |
| 64gpu-8node | 256 | global | 2.374 | 2.290 | 2.413 | 2.370 | 2.275 | 3.673 | 1.529 | 2.331 |
| 64gpu-8node | 256 | global+swap | 2.374 | 2.290 | 2.413 | 2.370 | 2.275 | 3.673 | 1.529 | 2.331 |
| 64gpu-8node | 320 | hierarchical | 1.482 | 1.721 | 2.063 | 1.469 | 1.439 | 2.232 | 1.444 | 1.000 |
| 64gpu-8node | 320 | global | 1.050 | 1.393 | 1.802 | 1.016 | 1.016 | 1.072 | 1.015 | 1.000 |
| 64gpu-8node | 320 | global+swap | 1.046 | 1.390 | 1.779 | 1.002 | 1.002 | 1.066 | 1.011 | 1.000 |

### hot-groups

| config | replicas | policy | stale δ=0 | stale δ=0.3 | stale δ=0.6 | oracle δ=0 | oracle δ=0.6 | worst layer δ=0 | node imb δ=0 | LB δ=0 |
|---|---|---|---|---|---|---|---|---|---|---|
| 32gpu-4node | 256 | none | 3.160 | 3.180 | 3.124 | — | — | 4.897 | 1.733 | 1.055 |
| 32gpu-4node | 256 | hierarchical | 1.493 | 1.689 | 1.829 | 1.481 | 1.414 | 2.428 | 1.452 | 1.055 |
| 32gpu-4node | 256 | global | 1.108 | 1.313 | 1.510 | 1.096 | 1.166 | 1.607 | 1.023 | 1.055 |
| 32gpu-4node | 256 | global+swap | 1.101 | 1.312 | 1.512 | 1.079 | 1.146 | 1.607 | 1.020 | 1.055 |
| 32gpu-4node | 288 | hierarchical | 1.478 | 1.645 | 1.768 | 1.461 | 1.374 | 2.396 | 1.452 | 1.000 |
| 32gpu-4node | 288 | global | 1.031 | 1.214 | 1.404 | 1.005 | 1.005 | 1.051 | 1.006 | 1.000 |
| 32gpu-4node | 288 | global+swap | 1.029 | 1.203 | 1.388 | 1.000 | 1.000 | 1.051 | 1.006 | 1.000 |
| 32gpu-4node | 320 | hierarchical | 1.476 | 1.610 | 1.701 | 1.458 | 1.373 | 2.379 | 1.452 | 1.000 |
| 32gpu-4node | 320 | global | 1.027 | 1.193 | 1.370 | 1.004 | 1.005 | 1.040 | 1.004 | 1.000 |
| 32gpu-4node | 320 | global+swap | 1.025 | 1.190 | 1.366 | 1.000 | 1.000 | 1.041 | 1.005 | 1.000 |
| 64gpu-8node | 256 | none | 3.781 | 3.860 | 3.893 | — | — | 5.883 | 2.676 | 1.725 |
| 64gpu-8node | 256 | hierarchical | 2.882 | 3.170 | 3.358 | 2.877 | 2.764 | 4.811 | 2.676 | 1.725 |
| 64gpu-8node | 256 | global | 1.786 | 1.912 | 2.134 | 1.783 | 1.962 | 3.082 | 1.296 | 1.725 |
| 64gpu-8node | 256 | global+swap | 1.786 | 1.913 | 2.134 | 1.781 | 1.960 | 3.082 | 1.295 | 1.725 |
| 64gpu-8node | 320 | hierarchical | 2.744 | 3.032 | 3.211 | 2.726 | 2.506 | 4.711 | 2.676 | 1.021 |
| 64gpu-8node | 320 | global | 1.049 | 1.304 | 1.606 | 1.017 | 1.016 | 1.084 | 1.015 | 1.000 |
| 64gpu-8node | 320 | global+swap | 1.042 | 1.291 | 1.582 | 1.002 | 1.002 | 1.057 | 1.010 | 1.000 |

## Token-level GPU imbalance vs tokens per step (random replica choice, δ=0, first 16 layers)

| scenario | config | replicas | policy | B=256 | B=1024 | B=4096 | B=32768 | nodes/token |
|---|---|---|---|---|---|---|---|---|
| mild | 32gpu-4node | 256 | none | 1.394 | 1.309 | 1.288 | 1.280 | 3.13 |
| mild | 32gpu-4node | 256 | hierarchical | 1.274 | 1.140 | 1.077 | 1.050 | 3.13 |
| mild | 32gpu-4node | 256 | global | 1.264 | 1.132 | 1.070 | 1.038 | 3.61 |
| mild | 32gpu-4node | 256 | global+swap | 1.265 | 1.130 | 1.067 | 1.032 | 3.61 |
| mild | 32gpu-4node | 288 | hierarchical | 1.274 | 1.140 | 1.079 | 1.045 | 3.13 |
| mild | 32gpu-4node | 288 | global | 1.264 | 1.133 | 1.069 | 1.035 | 3.61 |
| mild | 32gpu-4node | 288 | global+swap | 1.264 | 1.131 | 1.067 | 1.030 | 3.62 |
| mild | 32gpu-4node | 320 | hierarchical | 1.276 | 1.141 | 1.076 | 1.044 | 3.13 |
| mild | 32gpu-4node | 320 | global | 1.264 | 1.133 | 1.070 | 1.029 | 3.61 |
| mild | 32gpu-4node | 320 | global+swap | 1.266 | 1.131 | 1.067 | 1.027 | 3.61 |
| mild | 64gpu-8node | 256 | none | 1.620 | 1.469 | 1.422 | 1.407 | 3.97 |
| mild | 64gpu-8node | 256 | hierarchical | 1.483 | 1.289 | 1.212 | 1.177 | 3.97 |
| mild | 64gpu-8node | 256 | global | 1.433 | 1.218 | 1.123 | 1.086 | 5.30 |
| mild | 64gpu-8node | 256 | global+swap | 1.433 | 1.213 | 1.107 | 1.051 | 5.30 |
| mild | 64gpu-8node | 320 | hierarchical | 1.486 | 1.285 | 1.198 | 1.153 | 3.97 |
| mild | 64gpu-8node | 320 | global | 1.431 | 1.214 | 1.111 | 1.052 | 5.29 |
| mild | 64gpu-8node | 320 | global+swap | 1.431 | 1.213 | 1.107 | 1.047 | 5.29 |
| skewed | 32gpu-4node | 256 | none | 2.305 | 2.282 | 2.277 | 2.276 | 3.13 |
| skewed | 32gpu-4node | 256 | hierarchical | 1.419 | 1.357 | 1.341 | 1.336 | 3.16 |
| skewed | 32gpu-4node | 256 | global | 1.384 | 1.311 | 1.290 | 1.282 | 3.63 |
| skewed | 32gpu-4node | 256 | global+swap | 1.384 | 1.311 | 1.290 | 1.280 | 3.63 |
| skewed | 32gpu-4node | 288 | hierarchical | 1.304 | 1.184 | 1.129 | 1.104 | 3.16 |
| skewed | 32gpu-4node | 288 | global | 1.263 | 1.130 | 1.068 | 1.031 | 3.63 |
| skewed | 32gpu-4node | 288 | global+swap | 1.265 | 1.133 | 1.069 | 1.034 | 3.63 |
| skewed | 32gpu-4node | 320 | hierarchical | 1.308 | 1.185 | 1.130 | 1.100 | 3.16 |
| skewed | 32gpu-4node | 320 | global | 1.265 | 1.132 | 1.069 | 1.031 | 3.63 |
| skewed | 32gpu-4node | 320 | global+swap | 1.265 | 1.131 | 1.069 | 1.028 | 3.63 |
| skewed | 64gpu-8node | 256 | none | 3.403 | 3.354 | 3.344 | 3.343 | 3.97 |
| skewed | 64gpu-8node | 256 | hierarchical | 2.542 | 2.493 | 2.482 | 2.479 | 3.97 |
| skewed | 64gpu-8node | 256 | global | 2.455 | 2.415 | 2.407 | 2.404 | 5.28 |
| skewed | 64gpu-8node | 256 | global+swap | 2.455 | 2.415 | 2.407 | 2.404 | 5.28 |
| skewed | 64gpu-8node | 320 | hierarchical | 1.779 | 1.602 | 1.519 | 1.480 | 3.97 |
| skewed | 64gpu-8node | 320 | global | 1.435 | 1.214 | 1.111 | 1.057 | 5.34 |
| skewed | 64gpu-8node | 320 | global+swap | 1.431 | 1.213 | 1.108 | 1.049 | 5.33 |
| hot-groups | 32gpu-4node | 256 | none | 3.282 | 3.249 | 3.239 | 3.235 | 3.05 |
| hot-groups | 32gpu-4node | 256 | hierarchical | 1.701 | 1.597 | 1.550 | 1.523 | 3.26 |
| hot-groups | 32gpu-4node | 256 | global | 1.272 | 1.156 | 1.105 | 1.083 | 3.63 |
| hot-groups | 32gpu-4node | 256 | global+swap | 1.272 | 1.156 | 1.102 | 1.075 | 3.63 |
| hot-groups | 32gpu-4node | 288 | hierarchical | 1.701 | 1.587 | 1.532 | 1.499 | 3.26 |
| hot-groups | 32gpu-4node | 288 | global | 1.262 | 1.132 | 1.069 | 1.031 | 3.63 |
| hot-groups | 32gpu-4node | 288 | global+swap | 1.264 | 1.131 | 1.067 | 1.030 | 3.63 |
| hot-groups | 32gpu-4node | 320 | hierarchical | 1.698 | 1.587 | 1.532 | 1.503 | 3.26 |
| hot-groups | 32gpu-4node | 320 | global | 1.265 | 1.133 | 1.067 | 1.028 | 3.62 |
| hot-groups | 32gpu-4node | 320 | global+swap | 1.263 | 1.131 | 1.067 | 1.028 | 3.62 |
| hot-groups | 64gpu-8node | 256 | none | 3.927 | 3.873 | 3.863 | 3.860 | 3.84 |
| hot-groups | 64gpu-8node | 256 | hierarchical | 3.243 | 3.073 | 3.018 | 2.996 | 3.84 |
| hot-groups | 64gpu-8node | 256 | global | 1.906 | 1.835 | 1.815 | 1.808 | 5.30 |
| hot-groups | 64gpu-8node | 256 | global+swap | 1.906 | 1.834 | 1.814 | 1.807 | 5.30 |
| hot-groups | 64gpu-8node | 320 | hierarchical | 3.203 | 2.991 | 2.893 | 2.838 | 3.84 |
| hot-groups | 64gpu-8node | 320 | global | 1.431 | 1.212 | 1.109 | 1.053 | 5.33 |
| hot-groups | 64gpu-8node | 320 | global+swap | 1.432 | 1.212 | 1.106 | 1.048 | 5.32 |

## rebalance_experts wall time, 58 layers, skewed loads (median of 3)

| GPUs | replicas | nodes | policy | upstream s | heap packing s | speedup | identical |
|---|---|---|---|---|---|---|---|
| 8 | 264 | 1 | global | 0.367 | 0.013 | 28.7× | True |
| 8 | 264 | 1 | hierarchical | 0.355 | 0.018 | 19.5× | True |
| 32 | 288 | 1 | global | 0.889 | 0.017 | 52.9× | True |
| 32 | 288 | 4 | hierarchical | 0.475 | 0.018 | 25.8× | True |
| 64 | 320 | 1 | global | 1.620 | 0.020 | 80.0× | True |
| 64 | 320 | 8 | hierarchical | 0.399 | 0.015 | 26.8× | True |
| 128 | 384 | 1 | global | 3.054 | 0.027 | 113.4× | True |
| 256 | 512 | 1 | global | 5.996 | 0.037 | 160.2× | True |
