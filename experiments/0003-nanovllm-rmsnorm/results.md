## Numerics (bf16 vs fp32 reference, hidden=4096)

| method | tokens | out max abs | out mean abs | out max rel | residual max abs | allclose 1e-2 |
|---|---|---|---|---|---|---|
| fused_eager | 1 | 0.0 | 0.0 | 0.0 | 0.0 | True |
| separate_eager | 1 | 0.03125 | 0.000973 | 0.013333 | 0.0 | True |
| fused_eager | 16 | 0.0 | 0.0 | 0.0 | 0.0 | True |
| separate_eager | 16 | 0.03125 | 0.001055 | 0.013793 | 0.0 | True |
| fused_eager | 256 | 0.0 | 0.0 | 0.0 | 0.0 | True |
| separate_eager | 256 | 0.03125 | 0.001052 | 0.013889 | 0.0 | True |
| fused_eager | 2048 | 0.0 | 0.0 | 0.0 | 0.0 | True |
| separate_eager | 2048 | 0.03125 | 0.00105 | 0.013889 | 0.0 | True |

## Throughput (bf16, hidden=4096)

| method | tokens | ms/call | tokens/s | extra peak MiB |
|---|---|---|---|---|
| fused_eager | 1 | 0.4635 | 2157 | 0.04 |
| separate_eager | 1 | 0.2654 | 3767 | 0.04 |
| fused_compiled | 1 | 0.1547 | 6464 | 0.02 |
| separate_compiled | 1 | 0.1156 | 8651 | 0.02 |
| fused_eager | 16 | 0.1613 | 99219 | 0.63 |
| separate_eager | 16 | 0.0935 | 171214 | 0.63 |
| fused_compiled | 16 | 0.0678 | 236099 | 0.38 |
| separate_compiled | 16 | 0.0858 | 186411 | 0.25 |
| fused_eager | 256 | 0.2607 | 982087 | 10.0 |
| separate_eager | 256 | 0.1896 | 1350184 | 10.0 |
| fused_compiled | 256 | 0.0874 | 2927401 | 6.0 |
| separate_compiled | 256 | 0.0982 | 2607426 | 4.0 |
| fused_eager | 2048 | 2.4365 | 840548 | 80.01 |
| separate_eager | 2048 | 1.7 | 1204732 | 80.01 |
| fused_compiled | 2048 | 0.38 | 5389673 | 48.0 |
| separate_compiled | 2048 | 0.3299 | 6207325 | 32.01 |

compiled variants: first compile took 24.3 s
