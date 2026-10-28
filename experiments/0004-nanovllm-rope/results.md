cos/sin cache: 40960 positions x 128 fp32 = 20.00 MiB (shared by all layers via get_rope's lru_cache)

## cos/sin values vs fp64 reference

| method | positions | max abs | mean abs |
|---|---|---|---|
| cached_fp32 | [0, 1024) | 6.65e-05 | 1.24e-06 |
| cached_fp32 | [1024, 8192) | 0.000563 | 1.12e-05 |
| cached_fp32 | [8192, 32768) | 0.00233 | 4.97e-05 |
| cached_fp32 | [32768, 40960) | 0.0032 | 8.76e-05 |
| cached_fp64_built | [0, 1024) | 2.98e-08 | 1.6e-08 |
| cached_fp64_built | [1024, 8192) | 2.98e-08 | 1.65e-08 |
| cached_fp64_built | [8192, 32768) | 2.98e-08 | 1.67e-08 |
| cached_fp64_built | [32768, 40960) | 2.98e-08 | 1.68e-08 |
| on_the_fly_fp32 | [0, 1024) | 6.65e-05 | 1.24e-06 |
| on_the_fly_fp32 | [1024, 8192) | 0.000563 | 1.12e-05 |
| on_the_fly_fp32 | [8192, 32768) | 0.00233 | 4.97e-05 |
| on_the_fly_fp32 | [32768, 40960) | 0.0032 | 8.76e-05 |

## Rotated q/k (bf16) vs fp64 reference

| method | positions | max abs | mean abs | % elements != ref rounded to bf16 |
|---|---|---|---|---|
| cached_fp32_eager | [0, 1024) | 0.01549 | 0.00105 | 0.115 |
| cached_fp64_built_eager | [0, 1024) | 0.01549 | 0.00105 | 0.002 |
| on_the_fly_eager | [0, 1024) | 0.01549 | 0.00105 | 0.115 |
| cached_fp32_compiled | [0, 1024) | 0.01549 | 0.00105 | 0.115 |
| on_the_fly_compiled | [0, 1024) | 0.01549 | 0.00105 | 0.115 |
| cached_fp32_eager | [1024, 8192) | 0.01556 | 0.00112 | 0.718 |
| cached_fp64_built_eager | [1024, 8192) | 0.01556 | 0.00112 | 0.002 |
| on_the_fly_eager | [1024, 8192) | 0.01556 | 0.00112 | 0.718 |
| cached_fp32_compiled | [1024, 8192) | 0.01556 | 0.00112 | 0.718 |
| on_the_fly_compiled | [1024, 8192) | 0.01556 | 0.00112 | 0.719 |
| cached_fp32_eager | [8192, 32768) | 0.01558 | 0.00114 | 2.375 |
| cached_fp64_built_eager | [8192, 32768) | 0.01558 | 0.00112 | 0.002 |
| on_the_fly_eager | [8192, 32768) | 0.01558 | 0.00114 | 2.375 |
| cached_fp32_compiled | [8192, 32768) | 0.01558 | 0.00114 | 2.375 |
| on_the_fly_compiled | [8192, 32768) | 0.01558 | 0.00114 | 2.375 |
| cached_fp32_eager | [32768, 40960) | 0.0156 | 0.00115 | 3.583 |
| cached_fp64_built_eager | [32768, 40960) | 0.0156 | 0.00112 | 0.002 |
| on_the_fly_eager | [32768, 40960) | 0.0156 | 0.00115 | 3.583 |
| cached_fp32_compiled | [32768, 40960) | 0.0156 | 0.00115 | 3.583 |
| on_the_fly_compiled | [32768, 40960) | 0.0156 | 0.00115 | 3.583 |

## Throughput (bf16 q 16x128, k 8x128, random positions)

| method | tokens | ms/call | tokens/s | extra peak MiB |
|---|---|---|---|---|
| cached_fp32_eager | 1 | 0.2736 | 3655 | 0.03 |
| cached_fp64_built_eager | 1 | 0.4369 | 2289 | 0.03 |
| on_the_fly_eager | 1 | 0.4044 | 2473 | 0.03 |
| cached_fp32_compiled | 1 | 0.0611 | 16358 | 0.01 |
| on_the_fly_compiled | 1 | 0.1994 | 5015 | 0.01 |
| cached_fp32_eager | 16 | 0.6492 | 24647 | 0.45 |
| cached_fp64_built_eager | 16 | 0.3423 | 46737 | 0.45 |
| on_the_fly_eager | 16 | 0.4818 | 33211 | 0.45 |
| cached_fp32_compiled | 16 | 0.121 | 132191 | 0.09 |
| on_the_fly_compiled | 16 | 0.1253 | 127728 | 0.09 |
| cached_fp32_eager | 256 | 0.3474 | 736941 | 7.12 |
| cached_fp64_built_eager | 256 | 0.4488 | 570399 | 7.12 |
| on_the_fly_eager | 256 | 0.4802 | 533072 | 7.19 |
| cached_fp32_compiled | 256 | 0.1658 | 1543782 | 1.5 |
| on_the_fly_compiled | 256 | 0.191 | 1340195 | 1.5 |
| cached_fp32_eager | 2048 | 1.0844 | 1888628 | 57.0 |
| cached_fp64_built_eager | 2048 | 1.0784 | 1899155 | 57.0 |
| on_the_fly_eager | 2048 | 1.0672 | 1919109 | 57.5 |
| cached_fp32_compiled | 2048 | 0.4841 | 4230745 | 12.0 |
| on_the_fly_compiled | 2048 | 0.5346 | 3831197 | 12.0 |
| cached_fp32_eager | 8192 | 5.5045 | 1488239 | 228.0 |
| cached_fp64_built_eager | 8192 | 5.3251 | 1538379 | 228.0 |
| on_the_fly_eager | 8192 | 5.2403 | 1563272 | 230.0 |
| cached_fp32_compiled | 8192 | 1.1334 | 7227784 | 48.0 |
| on_the_fly_compiled | 8192 | 1.2514 | 6546163 | 48.0 |

compiled variants: first compile took 41.2 s
