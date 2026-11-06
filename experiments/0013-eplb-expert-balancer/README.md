# 0013 — DeepSeek EPLB: expert-parallel load balancing on synthetic DeepSeek-V3-style routing

- **Upstream**: [deepseek-ai/EPLB](https://github.com/deepseek-ai/EPLB) (submodule `upstream/EPLB` at `d52c72d`). `eplb.py` is imported unmodified.
- **Component**: `rebalance_experts(weight, num_replicas, num_groups, num_nodes, num_gpus)`. It has three steps:
  - `replicate_experts` gives each redundant slot to the expert with the largest load per replica.
  - `balanced_packing` greedily places items, in descending load order, on the lightest pack that is not yet full. Every pack holds the same number of items.
  - Two policies combine them. **Hierarchical** (used when `num_groups % num_nodes == 0`) packs whole expert groups onto nodes, replicates only inside a node, then packs replicas onto that node's GPUs. **Global** ignores groups and nodes.
- **Hardware**: i9-13900H, CPU only (the balancer runs on CPU by design: `weight.float().cpu()`), torch 2.14.1, 4 threads.
- **Run**: `.venv\Scripts\python.exe experiments\0013-eplb-expert-balancer\run.py`. It takes about 6 minutes.

## Question

How close to perfect GPU balance does EPLB get on a DeepSeek-V3-shaped MoE? When does the hierarchical policy (which limits inter-node traffic) cost balance? How much do redundant experts, stale load statistics and small per-step batches matter? Is there headroom left in the greedy heuristics, and what does the algorithm cost to run?

## Setup

- **Routing model** (synthetic, seeded; no model download). 256 routed experts in 8 groups of 32, 58 MoE layers. Each token picks its top 4 groups, scored by the sum of each group's top-2 expert scores, then its top 8 experts inside those groups. A token's score for an expert is a per-layer expert bias plus Gumbel noise, so popularity is roughly proportional to `exp(bias)`. Three scenarios (mean over layers, 32k-token history window):

  | scenario | bias | hottest expert / mean | hottest group's share |
  |---|---|---|---|
  | mild | 0.3·N(0,1) per expert | 2.2× | 13.9% |
  | skewed | 1.0·N(0,1) per expert | 9.3× | 18.1% |
  | hot-groups | 0.5·N(0,1) per expert + 0.7·N(0,1) per group | 6.9× | 33.5% |

- **Configs**: 32 GPUs on 4 nodes (2 groups per node, like DeepSeek's prefill deployment) with 256, 288 or 320 replicas (0, 1 or 2 redundant experts per GPU). 64 GPUs on 8 nodes (1 group per node) with 256 or 320 replicas.
- **Policies**:
  - `none`: no EPLB, so expert e sits on GPU `e // (256 / GPUs)`.
  - `hierarchical` and `global`: upstream.
  - `global+swap`: run.py's pairwise-swap local search on top of global's output. Replica counts stay fixed; it swaps replicas between the most loaded GPU and any other while that lowers the pair's maximum.
- **Statistics**. Placements are built from a 32k-token history window and evaluated on a fresh 32k-token window. In the fresh window the bias drifts in a variance-preserving way, with correlation 1, 0.958 and 0.857 to the history bias at δ = 0, 0.3 and 0.6. As a reference, an `oracle` placement is built on the evaluation window itself.
- **Metrics**:
  - **GPU imbalance** = max / mean GPU load, averaged over the 58 layers. MoE layers synchronize, so this is the MoE step time relative to perfect balance. The "fluid" version assumes each replica gets exactly load / replica count, as EPLB does.
  - **Token-level**: each token-expert pair goes to a uniformly random replica of its expert. GPU imbalance per step vs tokens per step, and distinct nodes per token (first 16 layers).
  - **Replica lower bound**: max(1, heaviest single replica / mean GPU load).
- **Checks**: the upstream README example is reproduced exactly. For every placement, run.py checks:
  - every expert has a replica;
  - `logcnt`, `log2phy` and `phy2log` are consistent;
  - under hierarchical, every group and all of its replicas stay on one node.
  There were 0 violations.
- **Repeatability**: two full runs. All balance numbers at δ = 0 and the routing statistics were identical; the second run fixed the drift model to be variance-preserving. Wall times moved by up to 30% between runs.

## Findings

1. **With one redundant expert per GPU, global EPLB is within 3–5% of perfect, and within 0.4–1.7% when its statistics are exact.** At δ = 0 with 288–320 replicas, global imbalance is 1.027–1.050 in every scenario and config. Without EPLB it is 1.26–3.78. The 3–5% that remains comes from the history window itself: 32k tokens give about 3% L1 error in expert loads, and the oracle placement on the same data reaches 1.004–1.017. More redundancy helps little once there is some: 288 → 320 replicas improves global imbalance by at most 0.004 at 32 GPUs.
2. **Without redundant experts, packing alone cannot fix a hot expert.** At 64 GPUs with 256 replicas (4 experts per GPU) in the skewed scenario, the hottest expert alone carries 2.33× a GPU's mean load. Global EPLB only reaches 2.37 (none: 3.41), and the swap search cannot improve it. One redundant expert per GPU (320 replicas) brings it to 1.05.
3. **The hierarchical policy is capped by node balance whenever whole groups do not split evenly across nodes.** It replicates only inside a node, so node imbalance is fixed by how groups pack onto nodes, and extra redundancy cannot fix it.
   - **Hot groups at 32 GPUs / 4 nodes**: node imbalance is 1.452 and GPU imbalance 1.476–1.493 at 256, 288 and 320 replicas. Global reaches 1.027–1.108.
   - **Skewed at 64 GPUs / 8 nodes**: with one group per node, node imbalance equals group imbalance (1.444). Hierarchical gets 1.482 at 320 replicas, where global gets 1.050.
   - **Even in the mild scenario at 8 nodes**: hierarchical sits at 1.14–1.17 vs 1.05–1.08 for global.
4. **What hierarchical buys is inter-node traffic.** Group-limited routing (4 of 8 groups) plus "a group's experts all live on one node" means a token touches at most 4 nodes. Measured with random replica choice:
   - at 8 nodes, 3.84–3.97 nodes per token under hierarchical vs 5.28–5.34 under global (+33–39% inter-node dispatch);
   - at 4 nodes, 3.13–3.26 vs 3.61–3.63 (+11–15%).
   So the trade-off is real. For mild skew at 4 nodes, hierarchical costs 1.1–1.3% balance for 13% less traffic. With hot groups at 4 nodes, it costs 35–44%.
5. **Stale statistics erase most of the gain quickly.** In the skewed scenario, global EPLB at 64 GPUs with 320 replicas goes from 1.050 at δ = 0 to 1.39 at bias correlation 0.958 and 1.80 at correlation 0.857. A fresh (oracle) placement gives 1.016. Without EPLB the imbalance is 3.3–3.4 at every δ, so stale EPLB is still far better than none. In the mild scenario, the same config goes 1.050 → 1.110 → 1.200 (none: 1.41). How often to rebalance matters as much as the algorithm.
6. **The greedy heuristics leave little headroom, and only on exact data.** On oracle data with 288–320 replicas, the swap search reaches 1.000–1.002 where global gives 1.004–1.017. Without redundancy the gaps are larger: 1.067 → 1.004 (mild, 64 GPUs) and 1.096 → 1.079 (hot groups, 32 GPUs). On stale data, its gain is at most 0.03 and is sometimes negative (it overfits the history window). It needs 33 swaps per layer on average (max 89).
7. **At small per-step batches, sampling noise dominates.** With random replica choice, even a near-perfect fluid placement shows imbalance of about 1.26 at 32 GPUs and 1.43 at 64 GPUs with 256 tokens per step. That falls to about 1.07 / 1.11 at 4096 tokens and about 1.03 / 1.05 at 32k tokens. In the mild scenario, EPLB therefore gains only 9–12% at 256 tokens per step (64 GPUs: 1.62 → 1.43). In the skewed and hot-groups scenarios, it still wins at every batch size: at 256 tokens, 320 replicas bring imbalance from 2.31–3.93 down to 1.26–1.44.
8. **Upstream's `balanced_packing` dominates the runtime and grows with EP size. A heap version gives identical output 20–175× faster.** For 58 layers:
   - global policy: 0.37–0.54 s at 8 GPUs, rising to 6.0–6.4 s at 256 GPUs (512 replicas);
   - hierarchical policy: stays at 0.36–0.60 s, because it packs per node.
   The cost is the Python loop. Every item scans all packs with `min(...)`, and pack weights are accumulated as 0-d torch tensors. run.py's `balanced_packing_heap` keeps the same order, float32 sums and lowest-index tie-break, and produces bitwise-identical `phy2log` / `log2phy` / `logcnt` in all 38 checks. It runs in 13–37 ms. For the global policy in the committed run, that is 29× faster at 8 GPUs and 160× at 256 GPUs.

Takeaway: EPLB's algorithm is sound. It is correct on every invariant, and with one redundant expert per GPU and fresh statistics it is within about 1% of perfect balance. The limits come from how it is used:
- Without redundancy, a single hot expert bounds the result.
- The hierarchical policy cannot fix node imbalance caused by hot groups (up to 44% slower MoE steps). Use it only when group loads are even or inter-node bandwidth is the bottleneck.
- Statistics a few percent stale cost more than the entire gap between the greedy heuristic and the best placement, and at small decode batches sampling noise sets the floor.
- For large-EP global rebalancing, replacing the linear `min` scan with a heap is a drop-in 100×+ speedup.

Full numbers: [results.md](results.md) / [results.json](results.json).
