"""DeepSeek EPLB (expert-parallel load balancer) on synthetic DeepSeek-V3-style routing, CPU only.

eplb.rebalance_experts(weight, num_replicas, num_groups, num_nodes, num_gpus) replicates
hot experts (replicate_experts: give each extra slot to the expert with the largest
load / replica count) and packs the replicas onto GPUs (balanced_packing: greedy, items
in descending load order, each to the lightest non-full pack). The hierarchical policy
first packs whole expert groups onto nodes and replicates only within a node; the global
policy ignores groups and nodes.

Routing model (DeepSeek-V3 shape): 256 routed experts in 8 groups of 32, each token picks
its top 4 groups (by the sum of the group's top-2 scores) and its top 8 experts inside
them, 58 MoE layers. Per-token scores are a per-layer expert bias plus Gumbel noise, so
expert popularity is ~ exp(bias). Three skew scenarios: mild, skewed, hot-groups (a
per-group bias on top of the per-expert one).

1. Balance quality (fluid model: each replica gets load / replica count): GPU and node
   imbalance (max / mean) for no EPLB (contiguous experts per GPU), hierarchical, global,
   and global + a pairwise-swap local search (run.py, not upstream), at 32 GPUs / 4 nodes
   and 64 GPUs / 8 nodes with 0, 1 or 2 redundant experts per GPU. Placements are built
   from a history window and evaluated on a fresh window whose bias drifted
   (variance-preserving, correlation 1 / sqrt(1 + delta^2) with the history bias), against
   an oracle placement built on the evaluation window itself.
2. Token-level: each token-expert pair goes to a uniformly random replica; GPU imbalance
   per step vs tokens per step, and distinct nodes each token is sent to.
3. Invariants of every placement, the upstream README example, and rebalance_experts wall
   time for 58 layers vs EP size, upstream vs a heap-based balanced_packing (run.py) that
   must give identical outputs.

eplb.py is imported unmodified from the submodule.
"""
import heapq
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "upstream" / "EPLB"))

import eplb  # noqa: E402

SEED = 0
NUM_EXPERTS, NUM_GROUPS, TOP_K, TOPK_GROUP, NUM_LAYERS = 256, 8, 8, 4, 58
GROUP_SIZE = NUM_EXPERTS // NUM_GROUPS
HIST_TOKENS = 32768
EVAL_TOKENS = 32768
SCENARIOS = {  # (sigma_expert, sigma_group)
    "mild": (0.3, 0.0),
    "skewed": (1.0, 0.0),
    "hot-groups": (0.5, 0.7),
}
DRIFTS = [0.0, 0.3, 0.6]
CONFIGS = [
    {"name": "32gpu-4node", "gpus": 32, "nodes": 4, "replicas": [256, 288, 320]},
    {"name": "64gpu-8node", "gpus": 64, "nodes": 8, "replicas": [256, 320]},
]
POLICIES = ["hierarchical", "global", "global+swap"]
TOKEN_LAYERS = 16
BATCHES = [256, 1024, 4096, 32768]
RUNTIME_GPUS = [8, 32, 64, 128, 256]
RUNTIME_REPS = 3
SWAP_MAX_ITERS = 5000

torch.set_num_threads(4)


# ---------------------------------------------------------------- routing

def make_biases(s_idx: int, sigma_e: float, sigma_g: float):
    rng = np.random.default_rng((SEED, s_idx))
    z_e = rng.standard_normal((NUM_LAYERS, NUM_EXPERTS))
    z_g = rng.standard_normal((NUM_LAYERS, NUM_GROUPS))
    drift = rng.standard_normal((NUM_LAYERS, NUM_EXPERTS))
    bias = sigma_e * z_e + sigma_g * np.repeat(z_g, GROUP_SIZE, axis=1)
    return bias, drift


def drifted(bias: np.ndarray, drift: np.ndarray, delta: float) -> np.ndarray:
    """Variance-preserving drift: corr(bias, result) = 1 / sqrt(1 + delta^2) per layer."""
    return (bias + delta * bias.std(1, keepdims=True) * drift) / np.sqrt(1 + delta**2)


def route(bias: np.ndarray, n_tokens: int, seed: int) -> np.ndarray:
    """Group-limited top-k routing with Gumbel-perturbed scores. Returns [n_tokens, TOP_K] uint8."""
    gen = torch.Generator().manual_seed(seed)
    u = torch.rand(n_tokens, NUM_EXPERTS, generator=gen).clamp_(1e-12, 1 - 1e-7)
    scores = torch.from_numpy(bias).float() - torch.log(-torch.log(u))
    group_scores = scores.view(n_tokens, NUM_GROUPS, GROUP_SIZE).topk(2, dim=-1).values.sum(-1)
    top_groups = group_scores.topk(TOPK_GROUP, dim=-1).indices
    mask = torch.full((n_tokens, NUM_GROUPS), float("-inf"))
    mask.scatter_(1, top_groups, 0.0)
    scores = scores + mask.repeat_interleave(GROUP_SIZE, dim=1)
    return scores.topk(TOP_K, dim=-1).indices.to(torch.uint8).numpy()


def counts_of(ids: np.ndarray) -> np.ndarray:
    return np.bincount(ids.reshape(-1), minlength=NUM_EXPERTS).astype(np.float64)


# ---------------------------------------------------------------- placements

def identity_placement(num_layers: int) -> np.ndarray:
    return np.tile(np.arange(NUM_EXPERTS), (num_layers, 1))


def eplb_placement(weight: np.ndarray, replicas: int, gpus: int, nodes: int, policy: str):
    w = torch.from_numpy(weight)
    if policy == "hierarchical":
        out = eplb.rebalance_experts(w, replicas, NUM_GROUPS, nodes, gpus)
    else:
        out = eplb.rebalance_experts(w, replicas, 1, 1, gpus)
    return out


def swap_refine(phy2log: np.ndarray, weight: np.ndarray, gpus: int):
    """Pairwise-swap local search on replica -> GPU assignment (replica counts fixed)."""
    out = phy2log.copy()
    iters = []
    per_gpu = phy2log.shape[1] // gpus
    for layer in range(phy2log.shape[0]):
        p2l = phy2log[layer]
        cnt = np.bincount(p2l, minlength=NUM_EXPERTS)
        loads = (weight[layer, p2l] / cnt[p2l]).reshape(gpus, per_gpu).copy()
        ids = p2l.reshape(gpus, per_gpu).copy()
        it = 0
        for it in range(SWAP_MAX_ITERS):
            gl = loads.sum(1)
            a = int(gl.argmax())
            cur = gl[a]
            d = loads[a][:, None, None] - loads[None, :, :]  # x_i - y_bj, shape [k, G, k]
            new_max = np.maximum(cur - d, gl[None, :, None] + d)
            new_max[:, a, :] = np.inf
            flat = int(new_max.argmin())
            i, b, j = np.unravel_index(flat, new_max.shape)
            if new_max[i, b, j] >= cur * (1 - 1e-12):
                break
            loads[a, i], loads[b, j] = loads[b, j], loads[a, i]
            ids[a, i], ids[b, j] = ids[b, j], ids[a, i]
        iters.append(it)
        out[layer] = ids.reshape(-1)
    return out, iters


def check_invariants(phy2log, log2phy, logcnt, replicas, gpus, nodes, policy) -> list[str]:
    errs = []
    p2l = phy2log.numpy()
    l2p = log2phy.numpy()
    cnt = logcnt.numpy()
    per_node = replicas // nodes
    for layer in range(p2l.shape[0]):
        bc = np.bincount(p2l[layer], minlength=NUM_EXPERTS)
        if bc.min() < 1:
            errs.append(f"layer {layer}: expert without replica")
        if not np.array_equal(bc, cnt[layer]):
            errs.append(f"layer {layer}: logcnt != bincount(phy2log)")
        for e in range(NUM_EXPERTS):
            phys = l2p[layer, e][l2p[layer, e] >= 0]
            if len(phys) != cnt[layer, e] or not np.all(p2l[layer, phys] == e):
                errs.append(f"layer {layer} expert {e}: log2phy inconsistent")
                break
        if policy == "hierarchical":
            node_of_phys = np.arange(replicas) // per_node
            exp_nodes = [set() for _ in range(NUM_EXPERTS)]
            for p, e in enumerate(p2l[layer]):
                exp_nodes[e].add(int(node_of_phys[p]))
            if any(len(s) != 1 for s in exp_nodes):
                errs.append(f"layer {layer}: expert replicated across nodes")
            for g in range(NUM_GROUPS):
                ns = set().union(*exp_nodes[g * GROUP_SIZE:(g + 1) * GROUP_SIZE])
                if len(ns) != 1:
                    errs.append(f"layer {layer}: group {g} split across nodes")
    return errs


# ---------------------------------------------------------------- evaluation

def fluid_metrics(phy2log: np.ndarray, counts: np.ndarray, gpus: int, nodes: int) -> dict:
    L, R = phy2log.shape
    per_gpu = R // gpus
    gpu_imb, node_imb, lb = [], [], []
    for layer in range(L):
        p2l = phy2log[layer]
        cnt = np.bincount(p2l, minlength=NUM_EXPERTS)
        phys_load = counts[layer, p2l] / cnt[p2l]
        gl = phys_load.reshape(gpus, per_gpu).sum(1)
        nl = gl.reshape(nodes, -1).sum(1)
        gpu_imb.append(gl.max() / gl.mean())
        node_imb.append(nl.max() / nl.mean())
        lb.append(max(1.0, phys_load.max() / gl.mean()))
    return {
        "gpu_imbalance_mean": float(np.mean(gpu_imb)),
        "gpu_imbalance_max": float(np.max(gpu_imb)),
        "node_imbalance_mean": float(np.mean(node_imb)),
        "replica_lower_bound_mean": float(np.mean(lb)),
    }


def token_metrics(p2l: np.ndarray, ids: np.ndarray, u: np.ndarray, gpus: int, nodes: int) -> dict:
    """One layer: each token-expert pair goes to a uniformly random replica (common random u)."""
    R = p2l.shape[0]
    per_gpu = R // gpus
    gpus_per_node = gpus // nodes
    cnt = np.bincount(p2l, minlength=NUM_EXPERTS)
    table = np.full((NUM_EXPERTS, cnt.max()), -1, dtype=np.int64)
    fill = np.zeros(NUM_EXPERTS, dtype=np.int64)
    for p, e in enumerate(p2l):
        table[e, fill[e]] = p
        fill[e] += 1
    e = ids.astype(np.int64)
    r = np.minimum((u * cnt[e]).astype(np.int64), cnt[e] - 1)
    gpu = table[e, r] // per_gpu
    n = ids.shape[0]
    out = {}
    for B in BATCHES:
        steps = n // B
        step = (np.arange(steps * B) // B)[:, None]
        loads = np.bincount((step * gpus + gpu[: steps * B]).reshape(-1),
                            minlength=steps * gpus).reshape(steps, gpus)
        out[B] = float((loads.max(1) / loads.mean(1)).mean())
    node = np.sort(gpu // gpus_per_node, axis=1)
    distinct = 1 + (np.diff(node, axis=1) != 0).sum(1)
    return {"gpu_imbalance_by_batch": out, "nodes_per_token": float(distinct.mean())}


# ---------------------------------------------------------------- fast packing

def balanced_packing_heap(weight: torch.Tensor, num_packs: int):
    """Same greedy as eplb.balanced_packing (float32 sums, lowest index wins ties), with a heap."""
    num_layers, num_groups = weight.shape
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs
    if groups_per_pack == 1:
        pack_index = torch.arange(num_groups, dtype=torch.int64).expand(weight.shape)
        return pack_index, torch.zeros_like(weight, dtype=torch.int64)
    w = weight.float().cpu().numpy()
    order = weight.float().sort(-1, descending=True).indices.cpu().numpy()
    pack_index = np.full((num_layers, num_groups), -1, dtype=np.int64)
    rank_in_pack = np.full((num_layers, num_groups), -1, dtype=np.int64)
    for i in range(num_layers):
        heap = [(np.float32(0.0), p) for p in range(num_packs)]
        items = [0] * num_packs
        wi = w[i]
        for g in order[i]:
            pw, p = heapq.heappop(heap)
            pack_index[i, g] = p
            rank_in_pack[i, g] = items[p]
            items[p] += 1
            if items[p] < groups_per_pack:
                heapq.heappush(heap, (np.float32(pw + wi[g]), p))
    return torch.from_numpy(pack_index), torch.from_numpy(rank_in_pack)


def time_rebalance(weight: torch.Tensor, replicas, groups, nodes, gpus):
    ts = []
    out = None
    for _ in range(RUNTIME_REPS):
        t0 = time.perf_counter()
        out = eplb.rebalance_experts(weight, replicas, groups, nodes, gpus)
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), out


# ---------------------------------------------------------------- main

def main():
    t_start = time.time()
    torch.manual_seed(SEED)
    results = {
        "env": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cpu": platform.processor(),
            "torch_threads": torch.get_num_threads(),
        },
        "params": {
            "num_experts": NUM_EXPERTS, "num_groups": NUM_GROUPS, "top_k": TOP_K,
            "topk_group": TOPK_GROUP, "num_layers": NUM_LAYERS, "hist_tokens": HIST_TOKENS,
            "eval_tokens": EVAL_TOKENS, "scenarios": SCENARIOS, "drifts": DRIFTS,
            "configs": CONFIGS, "token_layers": TOKEN_LAYERS, "batches": BATCHES, "seed": SEED,
        },
    }

    # README example
    ex_w = torch.tensor([[90, 132, 40, 61, 104, 165, 39, 4, 73, 56, 183, 86],
                         [20, 107, 104, 64, 19, 197, 187, 157, 172, 86, 16, 27]])
    ex = eplb.rebalance_experts(ex_w, 16, 4, 2, 8)[0]
    ex_expected = torch.tensor([[5, 6, 5, 7, 8, 4, 3, 4, 10, 9, 10, 2, 0, 1, 11, 1],
                                [7, 10, 6, 8, 6, 11, 8, 9, 2, 4, 5, 1, 5, 0, 3, 1]])
    results["readme_example"] = {"phy2log": ex.tolist(), "matches_readme": bool(torch.equal(ex, ex_expected))}
    print("README example matches:", results["readme_example"]["matches_readme"])

    scen_stats = {}
    quality = []
    token_rows = []
    invariant_errors = []
    swap_iters = []
    hist_weights = {}

    for s_idx, (sname, (sig_e, sig_g)) in enumerate(SCENARIOS.items()):
        t0 = time.time()
        bias, drift = make_biases(s_idx, sig_e, sig_g)
        hist = np.stack([counts_of(route(bias[l], HIST_TOKENS, 10_000 * s_idx + l)) for l in range(NUM_LAYERS)])
        hist_weights[sname] = hist
        eval_ids, eval_counts = {}, {}
        for d_idx, delta in enumerate(DRIFTS):
            b = drifted(bias, drift, delta)
            ids = [route(b[l], EVAL_TOKENS, 1_000_000 + 10_000 * s_idx + 100 * l + d_idx) for l in range(NUM_LAYERS)]
            eval_ids[delta] = ids[:TOKEN_LAYERS]
            eval_counts[delta] = np.stack([counts_of(x) for x in ids])
        mean_load = HIST_TOKENS * TOP_K / NUM_EXPERTS
        grp = hist.reshape(NUM_LAYERS, NUM_GROUPS, GROUP_SIZE).sum(-1)
        scen_stats[sname] = {
            "sigma_expert": sig_e, "sigma_group": sig_g,
            "expert_max_over_mean": float((hist.max(1) / mean_load).mean()),
            "expert_min_over_mean": float((hist.min(1) / mean_load).mean()),
            "expert_cv": float((hist.std(1) / hist.mean(1)).mean()),
            "group_max_share": float((grp.max(1) / grp.sum(1)).mean()),
            "hist_vs_eval_l1": {str(d): float(np.abs(hist - eval_counts[d]).sum(1).mean() / (HIST_TOKENS * TOP_K))
                                for d in DRIFTS},
        }
        print(f"[{sname}] routing done in {time.time() - t0:.1f}s: {scen_stats[sname]}")

        # placements from history (stale) and from each eval window (oracle)
        for cfg in CONFIGS:
            gpus, nodes = cfg["gpus"], cfg["nodes"]
            entries = [("none", NUM_EXPERTS)] + [(p, r) for r in cfg["replicas"] for p in POLICIES]
            for policy, replicas in entries:
                placements = {}
                for source in ["history"] + [f"oracle@{d}" for d in DRIFTS]:
                    if policy == "none" and source != "history":
                        continue
                    w = hist if source == "history" else eval_counts[float(source.split("@")[1])]
                    if policy == "none":
                        p2l = identity_placement(NUM_LAYERS)
                    else:
                        base = "hierarchical" if policy == "hierarchical" else "global"
                        out = eplb_placement(w, replicas, gpus, nodes, base)
                        if source == "history" and policy != "global+swap":
                            invariant_errors += [f"{sname}/{cfg['name']}/{replicas}/{policy}: {e}"
                                                 for e in check_invariants(*out, replicas, gpus, nodes, base)]
                        p2l = out[0].numpy()
                        if policy == "global+swap":
                            p2l, iters = swap_refine(p2l, w, gpus)
                            if source == "history":
                                swap_iters += iters
                    placements[source] = p2l
                for delta in DRIFTS:
                    row = {"scenario": sname, "drift": delta, "config": cfg["name"], "gpus": gpus,
                           "nodes": nodes, "replicas": replicas, "policy": policy}
                    row["history"] = fluid_metrics(placements["history"], eval_counts[delta], gpus, nodes)
                    key = f"oracle@{delta}"
                    if key in placements:
                        row["oracle"] = fluid_metrics(placements[key], eval_counts[delta], gpus, nodes)
                    quality.append(row)
                    u_rng = np.random.default_rng((SEED, s_idx, int(delta * 10), 7))
                    tm = [token_metrics(placements["history"][l], eval_ids[delta][l],
                                        u_rng.random((EVAL_TOKENS, TOP_K)), gpus, nodes)
                          for l in range(TOKEN_LAYERS)]
                    token_rows.append({
                        "scenario": sname, "drift": delta, "config": cfg["name"], "replicas": replicas,
                        "policy": policy,
                        "gpu_imbalance_by_batch": {B: float(np.mean([t["gpu_imbalance_by_batch"][B] for t in tm]))
                                                   for B in BATCHES},
                        "nodes_per_token": float(np.mean([t["nodes_per_token"] for t in tm])),
                    })
        print(f"[{sname}] placements + evaluation done, total {time.time() - t_start:.1f}s")

    results["scenario_stats"] = scen_stats
    results["quality"] = quality
    results["token_level"] = token_rows
    results["invariant_errors"] = invariant_errors
    results["swap_iters"] = {"mean": float(np.mean(swap_iters)), "max": int(np.max(swap_iters)),
                             "cap": SWAP_MAX_ITERS}

    # runtime + heap packing equivalence
    runtime = []
    weight = torch.from_numpy(hist_weights["skewed"])
    orig_packing = eplb.balanced_packing
    for gpus in RUNTIME_GPUS:
        replicas = NUM_EXPERTS + gpus
        nodes = gpus // 8
        modes = [("global", 1, 1)]
        if NUM_GROUPS % nodes == 0:
            modes.append(("hierarchical", NUM_GROUPS, nodes))
        for mode, groups, n in modes:
            eplb.balanced_packing = orig_packing
            t_up, out_up = time_rebalance(weight, replicas, groups, n, gpus)
            eplb.balanced_packing = balanced_packing_heap
            t_heap, out_heap = time_rebalance(weight, replicas, groups, n, gpus)
            eplb.balanced_packing = orig_packing
            same = all(torch.equal(a, b) for a, b in zip(out_up, out_heap))
            runtime.append({"gpus": gpus, "replicas": replicas, "nodes": n, "policy": mode,
                            "upstream_s": t_up, "heap_s": t_heap, "speedup": t_up / t_heap,
                            "identical_output": same})
            print(f"runtime gpus={gpus} {mode}: upstream {t_up:.3f}s heap {t_heap:.3f}s same={same}")
    # equivalence on every scenario / config used above
    eq_checks = 0
    eq_fail = []
    for sname, hist in hist_weights.items():
        w = torch.from_numpy(hist)
        for cfg in CONFIGS:
            for replicas in cfg["replicas"]:
                for groups, n in [(NUM_GROUPS, cfg["nodes"]), (1, 1)]:
                    eplb.balanced_packing = orig_packing
                    a = eplb.rebalance_experts(w, replicas, groups, n, cfg["gpus"])
                    eplb.balanced_packing = balanced_packing_heap
                    b = eplb.rebalance_experts(w, replicas, groups, n, cfg["gpus"])
                    eplb.balanced_packing = orig_packing
                    eq_checks += 1
                    if not all(torch.equal(x, y) for x, y in zip(a, b)):
                        eq_fail.append(f"{sname}/{cfg['name']}/{replicas}/{groups}")
    results["runtime"] = runtime
    results["heap_equivalence"] = {"checks": eq_checks + len(runtime), "failures": eq_fail}
    results["total_seconds"] = time.time() - t_start

    (HERE / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    (HERE / "results.md").write_text(render_md(results), encoding="utf-8")
    print(f"done in {results['total_seconds']:.1f}s, invariant errors: {len(invariant_errors)}, "
          f"heap mismatches: {len(eq_fail)}")


def render_md(r: dict) -> str:
    L = ["# 0013 results — EPLB on synthetic DeepSeek-V3-style routing", ""]
    L.append(f"Env: Python {r['env']['python']}, torch {r['env']['torch']}, numpy {r['env']['numpy']}, "
             f"{r['env']['torch_threads']} torch threads. Total {r['total_seconds']:.0f} s.")
    L.append("")
    L.append(f"README example reproduced: **{r['readme_example']['matches_readme']}**. "
             f"Invariant violations: **{len(r['invariant_errors'])}**. "
             f"Heap packing identical to upstream: **{not r['heap_equivalence']['failures']}** "
             f"({r['heap_equivalence']['checks']} checks). Swap search iterations per layer: "
             f"mean {r['swap_iters']['mean']:.1f}, max {r['swap_iters']['max']}.")
    L.append("")
    L.append("## Routing scenarios (history window, mean over 58 layers)")
    L.append("")
    L.append("| scenario | σ expert | σ group | max/mean expert load | min/mean | CV | hottest group share | hist→eval L1 (δ=0 / 0.3 / 0.6) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for s, v in r["scenario_stats"].items():
        l1 = " / ".join(f"{x:.3f}" for x in v["hist_vs_eval_l1"].values())
        L.append(f"| {s} | {v['sigma_expert']} | {v['sigma_group']} | {v['expert_max_over_mean']:.2f} | "
                 f"{v['expert_min_over_mean']:.2f} | {v['expert_cv']:.2f} | {v['group_max_share']:.3f} | {l1} |")
    L.append("")
    L.append("## GPU imbalance (fluid model, max/mean GPU load, mean over 58 layers)")
    L.append("")
    L.append("`stale` = placement from the history window evaluated on the eval window with drift δ "
             "(eval bias correlation with history bias 1, 0.958, 0.857 for δ = 0, 0.3, 0.6); "
             "`oracle` = placement built on the eval window itself. `LB` = replica lower bound "
             "max(1, heaviest replica / mean GPU load) of the stale global placement. 1.000 = perfect.")
    for s in r["scenario_stats"]:
        L.append("")
        L.append(f"### {s}")
        L.append("")
        L.append("| config | replicas | policy | stale δ=0 | stale δ=0.3 | stale δ=0.6 | oracle δ=0 | oracle δ=0.6 | worst layer δ=0 | node imb δ=0 | LB δ=0 |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        rows = [q for q in r["quality"] if q["scenario"] == s]
        keys = []
        for q in rows:
            k = (q["config"], q["replicas"], q["policy"])
            if k not in keys:
                keys.append(k)
        for k in keys:
            by_d = {q["drift"]: q for q in rows if (q["config"], q["replicas"], q["policy"]) == k}

            def o(d):
                return f"{by_d[d]['oracle']['gpu_imbalance_mean']:.3f}" if "oracle" in by_d[d] else "—"
            h0 = by_d[0.0]["history"]
            L.append(f"| {k[0]} | {k[1]} | {k[2]} | {h0['gpu_imbalance_mean']:.3f} | "
                     f"{by_d[0.3]['history']['gpu_imbalance_mean']:.3f} | {by_d[0.6]['history']['gpu_imbalance_mean']:.3f} | "
                     f"{o(0.0)} | {o(0.6)} | {h0['gpu_imbalance_max']:.3f} | {h0['node_imbalance_mean']:.3f} | "
                     f"{h0['replica_lower_bound_mean']:.3f} |")
    L.append("")
    L.append(f"## Token-level GPU imbalance vs tokens per step (random replica choice, δ=0, first {r['params']['token_layers']} layers)")
    L.append("")
    L.append("| scenario | config | replicas | policy | " + " | ".join(f"B={b}" for b in r["params"]["batches"]) + " | nodes/token |")
    L.append("|---|---|---|---|" + "---|" * len(r["params"]["batches"]) + "---|")
    for t in r["token_level"]:
        if t["drift"] != 0.0:
            continue
        cells = " | ".join(f"{t['gpu_imbalance_by_batch'][b]:.3f}" for b in r["params"]["batches"])
        L.append(f"| {t['scenario']} | {t['config']} | {t['replicas']} | {t['policy']} | {cells} | {t['nodes_per_token']:.2f} |")
    L.append("")
    L.append("## rebalance_experts wall time, 58 layers, skewed loads (median of 3)")
    L.append("")
    L.append("| GPUs | replicas | nodes | policy | upstream s | heap packing s | speedup | identical |")
    L.append("|---|---|---|---|---|---|---|---|")
    for t in r["runtime"]:
        L.append(f"| {t['gpus']} | {t['replicas']} | {t['nodes']} | {t['policy']} | {t['upstream_s']:.3f} | "
                 f"{t['heap_s']:.3f} | {t['speedup']:.1f}× | {t['identical_output']} |")
    if r["invariant_errors"]:
        L.append("")
        L.append("## Invariant violations (first 20)")
        L.append("")
        L += [f"- {e}" for e in r["invariant_errors"][:20]]
    L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
