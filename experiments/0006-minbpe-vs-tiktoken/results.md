# 0006-minbpe-vs-tiktoken

minbpe `1acefe8`, tiktoken 0.14.0 (Rust CoreBPE, single thread), Python 3.13.3.
Training text: first half of taylorswift.txt (92874 bytes). GPT4Tokenizer init: 7.91 s.

## Training

| vocab_size | merges | train_seconds | ms_per_merge | duplicate_byte_tokens | merges_before_exhaustion | error |
|---|---|---|---|---|---|---|
| 512 | 256 | 2.9 | 11.35 | 0 |  |  |
| 1024 | 768 | 7.51 | 9.78 | 0 |  |  |
| 2048 | 1792 | 15.67 | 8.75 | 0 |  |  |
| 4096 | 3840 | 29.84 | 7.77 | 0 |  |  |
| 8192 | 7936 | 50.94 |  |  | 7252 | ValueError: max() iterable argument is empty |

## Encoding (best of N runs)

| tokenizer | corpus | impl | seconds | MB_per_s | tokens | bytes_per_token |
|---|---|---|---|---|---|---|
| trained_512 | heldout_prose | minbpe | 0.09313 | 0.9974 | 44032 | 2.1097 |
| trained_512 | heldout_prose | minbpe+cache | 0.03115 | 2.9824 | 44032 | 2.1097 |
| trained_512 | heldout_prose | tiktoken | 0.00517 | 17.9748 | 44032 | 2.1097 |
| trained_512 | code | minbpe | 0.01691 | 1.4078 | 17169 | 1.3865 |
| trained_512 | code | minbpe+cache | 0.00677 | 3.5186 | 17169 | 1.3865 |
| trained_512 | code | tiktoken | 0.00142 | 16.7787 | 17169 | 1.3865 |
| trained_1024 | heldout_prose | minbpe | 0.09969 | 0.9319 | 36637 | 2.5355 |
| trained_1024 | heldout_prose | minbpe+cache | 0.03422 | 2.715 | 36637 | 2.5355 |
| trained_1024 | heldout_prose | tiktoken | 0.00488 | 19.0349 | 36637 | 2.5355 |
| trained_1024 | code | minbpe | 0.01818 | 1.3092 | 15730 | 1.5133 |
| trained_1024 | code | minbpe+cache | 0.00733 | 3.2464 | 15730 | 1.5133 |
| trained_1024 | code | tiktoken | 0.00142 | 16.7977 | 15730 | 1.5133 |
| trained_2048 | heldout_prose | minbpe | 0.10116 | 0.9183 | 33150 | 2.8022 |
| trained_2048 | heldout_prose | minbpe+cache | 0.03679 | 2.5248 | 33150 | 2.8022 |
| trained_2048 | heldout_prose | tiktoken | 0.00461 | 20.1606 | 33150 | 2.8022 |
| trained_2048 | code | minbpe | 0.01967 | 1.2099 | 14655 | 1.6243 |
| trained_2048 | code | minbpe+cache | 0.00773 | 3.0805 | 14655 | 1.6243 |
| trained_2048 | code | tiktoken | 0.00141 | 16.844 | 14655 | 1.6243 |
| trained_4096 | heldout_prose | minbpe | 0.10561 | 0.8796 | 31086 | 2.9883 |
| trained_4096 | heldout_prose | minbpe+cache | 0.03762 | 2.4694 | 31086 | 2.9883 |
| trained_4096 | heldout_prose | tiktoken | 0.00447 | 20.7826 | 31086 | 2.9883 |
| trained_4096 | code | minbpe | 0.01976 | 1.2046 | 13903 | 1.7121 |
| trained_4096 | code | minbpe+cache | 0.00785 | 3.031 | 13903 | 1.7121 |
| trained_4096 | code | tiktoken | 0.00137 | 17.3473 | 13903 | 1.7121 |
| gpt4_cl100k | full_prose | minbpe | 0.24687 | 0.7525 | 49298 | 3.7683 |
| gpt4_cl100k | full_prose | tiktoken | 0.00849 | 21.8741 | 49298 | 3.7683 |
| gpt4_cl100k | code | minbpe | 0.02951 | 0.8067 | 5729 | 4.155 |
| gpt4_cl100k | code | tiktoken | 0.00106 | 22.4439 | 5729 | 4.155 |

## minbpe vs tiktoken token ids

| tokenizer | corpus | n_bytes | ids_identical | first_mismatch | tiktoken_speedup_vs_minbpe |
|---|---|---|---|---|---|
| trained_512 | heldout_prose | 92894 | True | None | 18.0 |
| trained_512 | code | 23804 | True | None | 11.9 |
| trained_1024 | heldout_prose | 92894 | True | None | 20.4 |
| trained_1024 | code | 23804 | True | None | 12.8 |
| trained_2048 | heldout_prose | 92894 | True | None | 22.0 |
| trained_2048 | code | 23804 | True | None | 13.9 |
| trained_4096 | heldout_prose | 92894 | True | None | 23.6 |
| trained_4096 | code | 23804 | True | None | 14.4 |
| gpt4_cl100k | full_prose | 185768 | True | None | 29.1 |
| gpt4_cl100k | code | 23804 | True | None | 27.8 |
