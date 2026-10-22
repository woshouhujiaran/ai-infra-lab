## Correctness (vocab 32, T=0.7)

| method | samples | chi2/dof | total variation |
|---|---|---|---|
| nanovllm_gumbel_eager | 1998848 | 0.705 | 0.00084 |
| torch_multinomial | 1998848 | 0.818 | 0.00099 |

## Throughput (vocab 151936, bf16 logits)

| method | batch | ms/call | tokens/s | extra peak MiB |
|---|---|---|---|---|
| nanovllm_gumbel_eager | 1 | 0.2699 | 3705 | 1.7 |
| torch_multinomial | 1 | 0.4158 | 2405 | 1.2 |
| nanovllm_gumbel_eager | 16 | 0.4857 | 32943 | 27.8 |
| torch_multinomial | 16 | 0.5697 | 28087 | 18.6 |
| nanovllm_gumbel_eager | 64 | 2.2911 | 27934 | 114.0 |
| torch_multinomial | 64 | 2.4308 | 26329 | 76.0 |
| nanovllm_gumbel_eager | 256 | 9.3782 | 27297 | 445.1 |
| torch_multinomial | 256 | 8.9689 | 28543 | 296.8 |

compiled variant skipped: triton not installed (torch.compile needs it on Windows)
