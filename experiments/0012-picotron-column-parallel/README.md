# 0012 — picotron `ColumnParallelLinear`: tensor parallelism simulated with gloo on CPU

- **Upstream**: [huggingface/picotron](https://github.com/huggingface/picotron) (submodule `upstream/picotron` at `59714b1`). `tensor_parallel/tensor_parallel.py`, `tensor_parallel/tp_communications.py` and `process_group_manager.py` are imported unmodified. `picotron/utils.py` is not imported, because it needs the Unix-only `fcntl` module.
- **Component**: `ColumnParallelLinear`. Each rank holds `out / TP` rows of W and computes `Y_i = X W_i^T + b_i`. With `gather_output=True` it all-gathers Y. Backward all-reduces dX, in one of two ways:
  - Sync (default): `CopyToModelParallelRegion` all-reduces dX after `F.linear`'s backward.
  - Async (`async_all_reduce=True`): `LinearWithAsyncAllReduce` launches the dX all-reduce, computes dW and db, and then waits.
- **Hardware**: i9-13900H (6 P-cores and 8 E-cores), CPU only, torch 2.14.1 Windows wheel. TP = 1, 2 and 4 run as `torch.multiprocessing.spawn` processes with a gloo process group over 127.0.0.1, with 2 intra-op threads per rank. This is picotron's `use_cpu` mode (gloo backend).
- **Run**: `.venv\Scripts\python.exe experiments\0012-picotron-column-parallel\run.py`. It takes about 4 minutes.

## Question

At TP 2 and 4, does picotron's column-parallel layer reproduce single-process `nn.Linear` (output, dX, dW and db) in fp32 and bf16, for every input shape `nn.Linear` accepts? Does sharding actually shrink each rank's weight memory? On CPU with gloo, how much do the dX all-reduce, the async overlap and the output all-gather cost or save?

## Setup

- **Storage**: one Llama-3-8B MLP projection (4096 → 14336, fp32, no bias), plus `RowParallelLinear` (14336 → 4096) and `VocabParallelEmbedding` (32000 × 1024) for contrast. The script reports:
  - the parameter's storage bytes;
  - process RSS growth, with `MIMALLOC_PURGE_DELAY=0` so that torch's Windows allocator returns freed pages and RSS reflects them;
  - the `torch.save(state_dict())` size;
  - the storage after `.to(float32)` and after `.to(bfloat16)`, the two casts `train.py` can apply.
  `ClonedColumnParallelLinear` in `run.py` is the same layer, except that `reset_parameters` clones the shard.
- **Init**: compares picotron's gathered weight with `nn.Linear` built from the same seed, using either the same seed on every rank (as in `train.py`'s `set_all_seed`) or seed + rank.
- **Correctness**: 1024 → 4096 with bias and `gather_output=True`, over 256 tokens shaped as 2D `(256, 1024)`, 3D `(4, 64, 1024)` or 4D `(2, 2, 64, 1024)`. Weights, inputs and output gradients are random and identical in every run (seed 0). The script reports:
  - the relative L2 error of y, dX, dW and db against fp64;
  - bitwise equality with single-process `nn.Linear` in the same dtype;
  - whether dX is identical on all ranks.
- **Timing**: 1024 → 4096 in fp32 over 256, 1024 and 4096 tokens (batch 4). Each cell is the median of 5–10 reps of the slowest rank. Variants:
  - forward, with and without `gather_output`;
  - local-only forward + backward (`F.linear` on the shard, no communication);
  - sync and async forward + backward;
  - the bare dX all-reduce;
  - an MLP `silu(gate(x)) * up(x)` pair, built either as picotron does it (two layers, two dX all-reduces) or with one shared `CopyToModelParallelRegion` (one all-reduce);
  - a gloo all-reduce size sweep.
- **Repeatability**: two full runs. Storage, init and correctness were bit-identical. Individual timing cells moved by up to ±35% (hybrid P/E-core CPU, no pinning), so the timing claims below are only the ones that hold in both runs. The committed numbers are from the second run.

## Findings

1. **Column shards keep the full master weight alive.** `reset_parameters` builds the full `out × in` master on every rank and keeps `torch.split(master, n, dim=0)[rank].contiguous()`. A dim-0 slice of a contiguous tensor is already contiguous, so `.contiguous()` returns a view, and the parameter's storage is the whole master. For the 4096 → 14336 projection, the shard is 224, 112 and 56 MiB at TP 1, 2 and 4. Yet every rank's RSS grows by 224 MiB, and `torch.save(state_dict())` writes 224 MiB, at every TP. `VocabParallelEmbedding` also splits on dim 0 and has the same problem (125 MiB at every TP). `RowParallelLinear` splits on dim 1, where `.contiguous()` really copies, so it is fine.
   - **Why it matters in `train.py`**: `train.py` calls `reset_parameters()` and then `model.to(dtype).to(device)`. With bf16 on GPU, that cast copies only the shard and frees the master. In CPU/gloo mode, however, dtype is fp32 and the device is CPU, so `.to()` is a no-op. Every rank then keeps, and checkpoints, full-size column, embedding and LM-head weights.
   - **Fix**: replacing `.contiguous()` with `.clone()` brings storage, RSS and checkpoint size down to the shard (224, 112 and 56 MiB).
2. **Init reproduces `nn.Linear`'s weights bit for bit.** With the same seed on every rank, the gathered weight is bitwise equal to `nn.Linear(in, out).weight` at TP 1, 2 and 4. The reason is that the master is drawn with the same `uniform(±1/√in)` on the same shape, so a TP-N run starts from exactly the same weights as a TP-1 run. The bias is zero rather than `nn.Linear`'s uniform init. With seed + rank, only rank 0's shard matches. This reproducibility is what finding 1 pays for: building the full master on every rank is how the shards line up.
3. **The sync path is correct for every input shape. In fp32 it differs from single-process only in the last bits.** In fp32 at TP 2 and 4, y, dX and dW are not bitwise equal to `nn.Linear` (oneDNN blocks the narrower matmuls differently). db is bitwise equal, the maximum absolute difference in dX is 1.9e-6, and every relative error against fp64 stays at 2.5–2.9e-7. dX error even falls slightly with TP (2.64, 2.58 and 2.54e-7 at TP 1, 2 and 4), because each rank sums fewer terms. After gloo's all-reduce, dX is bitwise identical on all ranks.
4. **In bf16, dX error grows with TP: +19% at TP 2 and +40% at TP 4.** Relative error is 2.77e-3, 3.31e-3 and 3.87e-3 at TP 1, 2 and 4, with a maximum absolute difference of 0.031 against single-process. Each rank's partial dX is rounded to bf16 before the all-reduce, and gloo also sums in bf16, so every extra rank adds a rounding. y, dW and db have no cross-rank reduction and stay bitwise equal to single-process bf16.
5. **The async path only works for 3D inputs.** `merge_first_two_dims` reshapes with `view(-1, *shape[2:])`, then computes dW as `grad_output.t() @ input_`.
   - **2D input `(tokens, in)`**: the view flattens to 1D. The result is either "inconsistent tensor size" or, at TP 4 where `out / TP == in == 1024`, a scalar dW that autograd rejects ("returned an invalid gradient at index 1 - got []").
   - **4D input**: it calls `.t()` on a 3D tensor and fails.
   - The sync path handles all three shapes. picotron's own model always passes `(B, S, H)`, so its training is unaffected. With `async_all_reduce=True`, though, the layer is not a drop-in replacement for `nn.Linear`.
   - **Forward numerics**: the async forward computes `input_ @ weight.t() + bias` instead of `F.linear`. In bf16 the matmul output is rounded before the bias is added, so y error is 3.24e-3 instead of 2.77e-3 (+17%), even at TP 1. Its dX is bitwise equal to the sync path's.
6. **On this CPU, TP is far from linear.** Each extra rank brings 2 more threads. Even so, sync forward + backward is only 0.99–1.39× faster than TP 1 at TP 2, and 1.30–1.88× at TP 4, against ideals of 2× and 4×. Communication is only part of the gap. Local compute alone, with no communication, is only about 2× faster at TP 4 (245–290 ms vs 495–583 ms at 4096 tokens), as the extra ranks share memory bandwidth and E-cores. The dX all-reduce then adds 30–140 ms at 4096 tokens.
7. **gloo with 2 ranks has a cliff above 1 MiB on Windows.** A 2-rank all-reduce takes 0.39 ms at 0.25 MiB and 1.35 ms at 1 MiB, then 30 ms at 4 MiB and 101 ms at 16 MiB (0.17 GB/s). With 4 ranks, the same 4 MiB and 16 MiB take 7.6 ms and 45 ms. Both runs show this: the 16 MiB dX all-reduce took 79–120 ms at TP 2 and 40–44 ms at TP 4. I did not isolate the cause.
8. **Async overlap gives a modest gain.** At 4096 tokens, async forward + backward is 7–17% faster than sync at TP 2 and 4 in both runs (23–65 ms), which is less than the all-reduce itself. gloo runs the collective on its own thread while dW is computed, but they compete for the same cores. At 1024 tokens or fewer, the difference is within noise. dX is bitwise identical to the sync path.
9. **picotron does one dX all-reduce per projection, and one shared all-reduce is 8–23% faster.** picotron's MLP calls `gate_proj(x)` and `up_proj(x)`, two `ColumnParallelLinear` layers that each have their own `CopyToModelParallelRegion`. That makes two `(tokens × hidden)` all-reduces in backward. Attention makes three, for q, k and v. With one shared copy, the local branch gradients are summed first and all-reduced once. At 4096 tokens this is 8–23% faster for the pair at TP 2 and 4 in both runs, and dX stays within 4.8e-7 (fp32) of picotron's.
10. **`gather_output=True` can cost more than TP saves.** At TP 4, all-gathering the 4096 × 4096 fp32 output (64 MiB) takes 150–170 ms. A gathered forward (244–261 ms) is therefore slower than TP 1's forward (168–181 ms). picotron gathers only at the LM head, whose output (tokens × vocab) is the widest in the model.

Takeaway: picotron's column-parallel math is right. The sync path matches `nn.Linear` to the last bits in fp32 for any input shape, and its init reproduces `nn.Linear`'s weights exactly.

The problems are in memory and in the optional path:
- Dim-0 shards are views of the full master weight. On the CPU/fp32 path, TP therefore saves no memory or checkpoint size for column, embedding and LM-head layers. A one-line `.clone()` fixes it.
- `async_all_reduce=True` crashes on anything but 3D inputs, and adds bf16 error in the forward pass.
- bf16 dX error grows by 19–40% from TP 2 to TP 4.

On a laptop CPU, gloo TP is a correctness harness rather than a speedup: it is at most 1.9× faster with 4× the cores. If you use it, all-reduce once per shared input and avoid 2-rank all-reduces over 1 MiB.

Full numbers: [results.md](results.md) / [results.json](results.json).
