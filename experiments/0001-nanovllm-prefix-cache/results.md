| scenario | arrival | block_size | prefix_len | num_blocks | prompt_tokens | prefill_tokens_computed | prefix_hit_rate | steps | peak_used_blocks | preemptions |
|---|---|---|---|---|---|---|---|---|---|---|
| no_pressure | burst | 16 | 0 |  | 8192 | 8192 | 0.0 | 32 | 640 | 0 |
| no_pressure | burst | 16 | 256 |  | 24576 | 18944 | 0.2292 | 33 | 1312 | 0 |
| no_pressure | burst | 16 | 1024 |  | 73728 | 22528 | 0.6944 | 33 | 1536 | 0 |
| no_pressure | burst | 16 | 2048 |  | 139264 | 22528 | 0.8382 | 33 | 1536 | 0 |
| no_pressure | burst | 64 | 0 |  | 8192 | 8192 | 0.0 | 32 | 192 | 0 |
| no_pressure | burst | 64 | 256 |  | 24576 | 18944 | 0.2292 | 33 | 360 | 0 |
| no_pressure | burst | 64 | 1024 |  | 73728 | 22528 | 0.6944 | 33 | 416 | 0 |
| no_pressure | burst | 64 | 2048 |  | 139264 | 22528 | 0.8382 | 33 | 416 | 0 |
| no_pressure | burst | 256 | 0 |  | 8192 | 8192 | 0.0 | 32 | 64 | 0 |
| no_pressure | burst | 256 | 256 |  | 24576 | 18944 | 0.2292 | 33 | 106 | 0 |
| no_pressure | burst | 256 | 1024 |  | 73728 | 22528 | 0.6944 | 33 | 120 | 0 |
| no_pressure | burst | 256 | 2048 |  | 139264 | 22528 | 0.8382 | 33 | 120 | 0 |
| no_pressure | staggered | 16 | 0 |  | 8192 | 8192 | 0.0 | 95 | 640 | 0 |
| no_pressure | staggered | 16 | 256 |  | 24576 | 8448 | 0.6562 | 95 | 656 | 0 |
| no_pressure | staggered | 16 | 1024 |  | 73728 | 9216 | 0.875 | 95 | 704 | 0 |
| no_pressure | staggered | 16 | 2048 |  | 139264 | 10240 | 0.9265 | 95 | 768 | 0 |
| no_pressure | staggered | 64 | 0 |  | 8192 | 8192 | 0.0 | 95 | 192 | 0 |
| no_pressure | staggered | 64 | 256 |  | 24576 | 8448 | 0.6562 | 95 | 196 | 0 |
| no_pressure | staggered | 64 | 1024 |  | 73728 | 9216 | 0.875 | 95 | 208 | 0 |
| no_pressure | staggered | 64 | 2048 |  | 139264 | 10240 | 0.9265 | 95 | 224 | 0 |
| no_pressure | staggered | 256 | 0 |  | 8192 | 8192 | 0.0 | 95 | 64 | 0 |
| no_pressure | staggered | 256 | 256 |  | 24576 | 8448 | 0.6562 | 95 | 65 | 0 |
| no_pressure | staggered | 256 | 1024 |  | 73728 | 9216 | 0.875 | 95 | 68 | 0 |
| no_pressure | staggered | 256 | 2048 |  | 139264 | 10240 | 0.9265 | 95 | 72 | 0 |
| memory_pressure | burst | 256 | 1024 | 24 | 73728 | 20538 | 0.7214 | 1537 | 24 | 42 |
| memory_pressure | burst | 256 | 1024 | 48 | 73728 | 31262 | 0.576 | 898 | 48 | 26 |
| memory_pressure | burst | 256 | 1024 | 96 | 73728 | 27668 | 0.6247 | 513 | 96 | 20 |
