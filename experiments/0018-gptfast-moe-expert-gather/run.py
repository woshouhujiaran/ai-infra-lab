"""gpt-fast mixtral-moe MOEFeedForward / ConditionalFeedForward on a laptop GPU.

Component: mixtral-moe/model.py. The router is a bf16 nn.Linear -> softmax (in bf16) ->
topk -> renormalize; ConditionalFeedForward then gathers every token's expert weights,
w1[expert_indices] -> [T, A, I, D], and runs three einsums over the gathered copies.

1. Correctness: upstream eager / torch.compile vs an fp32 per-expert reference with the same
   routing, next to three rewrites (HF-style per-expert loop, sort + F.grouped_mm, dense
   all-expert bmm); whether F.grouped_mm can be CUDA-graph captured on sm89; what the
   un-reshaped [T, D] MoE output does inside TransformerBlock for batch > 1.
2. Routing numerics: top-A set mismatch vs fp64 routing and exact ties at the selection
   boundary for bf16 softmax (upstream) vs fp32 softmax of bf16 logits (HF) vs fp32 logits,
   vs logit scale, for a Mixtral-8x7B (8 experts, top-2) and an OLMoE-1B-7B (64, top-8)
   router; output error of the routing difference.
3. One MoE layer vs token count T (Mixtral-8x7B and OLMoE-1B-7B shapes, synthetic bf16
   weights): extra peak memory (incl. OOM under the 60% cap) and latency, eager launches and
   CUDA-graph replay, for upstream eager / torch.compile (gpt-fast's inductor flags, static
   shapes) / loop / grouped / dense; which extern kernels Inductor emits; bandwidth against
   the bytes of the experts actually touched.

model.py is imported unmodified.
"""
import importlib.util
import json
import math
import platform
import re
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MOE_DIR = ROOT / "upstream" / "gpt-fast" / "mixtral-moe"
sys.path.insert(0, str(ROOT))

from lab import gpu_guard  # noqa: E402

GPU_INFO = gpu_guard.acquire()

_spec = importlib.util.spec_from_file_location("mixtral_moe_model", MOE_DIR / "model.py")
MX = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(MX)

SEED = 0
DEVICE = "cuda"
CONFIGS = {
    "Mixtral-8x7B": dict(dim=4096, intermediate_size=14336, num_experts=8, num_activated_experts=2),
    "OLMoE-1B-7B": dict(dim=2048, intermediate_size=1024, num_experts=64, num_activated_experts=8),
}
CHECK_T = {"Mixtral-8x7B": 2, "OLMoE-1B-7B": 16}
T_LIST = {
    "Mixtral-8x7B": [1, 2, 3, 4, 8, 16, 64, 256, 1024],
    "OLMoE-1B-7B": [1, 2, 4, 8, 16, 32, 64, 256, 1024, 4096],
}
COMPILE_T = {"Mixtral-8x7B": [1, 2, 4], "OLMoE-1B-7B": [1, 2, 4, 16, 64]}
ROUTE_TOKENS = 65536
ROUTE_CHUNK = 8192
LOGIT_STD = [0.5, 1.0, 2.0, 4.0]
IMPACT_T = {"Mixtral-8x7B": 1024, "OLMoE-1B-7B": 4096}
METHODS = ["upstream-eager", "upstream-compile", "loop", "grouped", "dense"]
GRAPHABLE = {"upstream-eager", "upstream-compile", "dense"}
TARGET_S = 0.05
MAX_REPS = 200

for _k in ("cache_size_limit", "recompile_limit"):
    if hasattr(torch._dynamo.config, _k):
        setattr(torch._dynamo.config, _k, 512)
torch._inductor.config.coordinate_descent_tuning = True  # as in mixtral-moe/generate.py
if hasattr(torch._inductor.config.triton, "unique_kernel_names"):
    torch._inductor.config.triton.unique_kernel_names = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sync():
    torch.cuda.synchronize()


def err_text(e: BaseException) -> str:
    lines = str(e).strip().splitlines()
    return f"{type(e).__name__}: {lines[0][:200] if lines else ''}"


def rel(y, ref) -> float:
    y, ref = y.double(), ref.double()
    return ((y - ref).norm() / ref.norm()).item()


def args_for(name: str, **extra) -> "MX.ModelArgs":
    return MX.ModelArgs(**CONFIGS[name], **extra)


def build_moe(name: str, logit_std: float = 1.0) -> "MX.MOEFeedForward":
    cfg = args_for(name)
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device(DEVICE):
            m = MX.MOEFeedForward(cfg)
    finally:
        torch.set_default_dtype(prev)
    g = torch.Generator(device=DEVICE).manual_seed(SEED)
    D, I = cfg.dim, cfg.intermediate_size
    with torch.no_grad():
        m.gate.weight.normal_(0, logit_std / math.sqrt(D), generator=g)
        m.cond_ffn.w1.normal_(0, 1 / math.sqrt(D), generator=g)
        m.cond_ffn.w3.normal_(0, 1 / math.sqrt(D), generator=g)
        m.cond_ffn.w2.normal_(0, 1 / math.sqrt(I), generator=g)
    m.requires_grad_(False)
    return m


def expert_bytes(m) -> int:
    cf = m.cond_ffn
    return 3 * cf.w1[0].numel() * cf.w1.element_size()


# ---------------------------------------------------------------- MoE variants

def route(m, x):
    """MOEFeedForward.forward's routing, unchanged (bf16 softmax, topk, renormalize)."""
    scores = m.gate(x)
    expert_weights = F.softmax(scores, dim=-1)
    expert_weights, expert_indices = torch.topk(expert_weights, m.num_activated_experts, dim=-1)
    expert_weights /= expert_weights.sum(dim=-1, keepdim=True)
    return expert_weights, expert_indices


def ffn_loop(cf, x, w, idx):
    """HF Mixtral-style: one dense GEMM chain per expert over the tokens routed to it."""
    out = torch.zeros_like(x)
    for e in range(cf.w1.shape[0]):
        tok, slot = torch.where(idx == e)
        if tok.numel() == 0:
            continue
        xe = x[tok]
        h = F.silu(xe @ cf.w1[e].T) * (xe @ cf.w3[e].T)
        out.index_add_(0, tok, (h @ cf.w2[e].T) * w[tok, slot, None])
    return out


def ffn_grouped(cf, x, w, idx):
    """Sort the T*A (token, expert) pairs by expert and run three F.grouped_mm calls."""
    T, A = idx.shape
    E = cf.w1.shape[0]
    flat = idx.reshape(-1)
    order = flat.argsort(stable=True)
    offs = torch.searchsorted(flat[order], torch.arange(E, device=x.device), right=True).to(torch.int32)
    xs = x[order // A]
    h = F.silu(F.grouped_mm(xs, cf.w1.transpose(1, 2), offs=offs)) * F.grouped_mm(xs, cf.w3.transpose(1, 2), offs=offs)
    y = F.grouped_mm(h, cf.w2.transpose(1, 2), offs=offs)
    ys = torch.empty_like(y)
    ys[order] = y
    return torch.einsum('tai,ta -> ti', ys.view(T, A, -1), w)


def ffn_dense(cf, x, w, idx):
    """Every expert on every token (reads each expert once), then a dense routing-weight combine."""
    T, _ = idx.shape
    E = cf.w1.shape[0]
    comb = torch.zeros(T, E, dtype=x.dtype, device=x.device).scatter_(1, idx, w)
    xe = x.unsqueeze(0).expand(E, -1, -1)
    h = F.silu(torch.bmm(xe, cf.w1.transpose(1, 2))) * torch.bmm(xe, cf.w3.transpose(1, 2))
    y = torch.bmm(h, cf.w2.transpose(1, 2))
    return torch.einsum('eti,te -> ti', y, comb)


def moe_with(ffn):
    def fwd(m, x):
        x = x.view(-1, m.dim)
        w, idx = route(m, x)
        return ffn(m.cond_ffn, x, w, idx)
    return fwd


VARIANTS = {"loop": moe_with(ffn_loop), "grouped": moe_with(ffn_grouped), "dense": moe_with(ffn_dense)}


def ref_ffn(cf, x, w, idx):
    """fp32 (TF32 off) per-expert reference for a given routing."""
    x32 = x.float()
    out = torch.zeros(x.shape[0], cf.w1.shape[2], dtype=torch.float32, device=x.device)
    for e in idx.unique().tolist():
        tok, slot = torch.where(idx == e)
        w1, w3, w2 = cf.w1[e].float(), cf.w3[e].float(), cf.w2[e].float()
        h = F.silu(x32[tok] @ w1.T) * (x32[tok] @ w3.T)
        out.index_add_(0, tok, (h @ w2.T) * w[tok, slot, None].float())
    return out


# ---------------------------------------------------------------- 1. correctness

def graph_capture_error(fn) -> str | None:
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            fn()
        g.replay()
        sync()
        return None
    except Exception as e:
        return err_text(e)
    finally:
        del g


def batch_shape_check():
    """Tiny full Transformer: B = 1 vs B = 2 for decode (S = 1) and prefill (S = 4)."""
    torch.manual_seed(SEED)
    cfg = MX.ModelArgs(block_size=64, vocab_size=128, n_layer=2, n_head=4, dim=64,
                       intermediate_size=128, num_experts=4, num_activated_experts=2)
    with torch.device(DEVICE):
        model = MX.Transformer(cfg)
        for p in model.parameters():
            torch.nn.init.normal_(p, 0, 0.1)
        model.setup_caches(max_batch_size=2, max_seq_length=16)
    model = model.to(torch.bfloat16)
    model.requires_grad_(False)
    rows = []
    for S in (1, 4):
        idx = torch.randint(0, 128, (2, S), device=DEVICE)
        pos = torch.arange(S, device=DEVICE)
        row = {"B": 2, "S": S}
        try:
            single = torch.cat([model(idx[b:b + 1], pos) for b in range(2)])
            row["B1_out_shape"] = list(single.shape)
        except Exception as e:
            row["B1_error"] = err_text(e)
        try:
            out = model(idx, pos)
            row["B2_out_shape"] = list(out.shape)
            if "B1_out_shape" in row and out.shape == single.shape:
                row["B2_matches_B1"] = rel(out, single)
        except Exception as e:
            row["B2_error"] = err_text(e)
        x = torch.randn(2, S, cfg.dim, device=DEVICE, dtype=torch.bfloat16)
        moe = model.layers[0].block_sparse_moe
        row["moe_in_shape"] = list(x.shape)
        row["moe_out_shape"] = list(moe(x).shape)
        try:
            row["h_plus_moe_shape"] = list((x + moe(x)).shape)
        except Exception as e:
            row["h_plus_moe_error"] = err_text(e)
        rows.append(row)
        print(f"  batch check S={S}: {row}")
    return rows


def part_correctness():
    out = {"configs": [], "batch_shapes": batch_shape_check()}
    for name in CONFIGS:
        torch.manual_seed(SEED)
        m = build_moe(name)
        T = CHECK_T[name]
        x = torch.randn(T, m.dim, device=DEVICE, dtype=torch.bfloat16)
        w, idx = route(m, x)
        ref = ref_ffn(m.cond_ffn, x, w, idx)
        bf16_ref = None
        row = {"config": name, "T": T, "unique_experts": int(idx.unique().numel()), "errors_vs_fp32_ref": {}}
        compiled = torch.compile(m, fullgraph=True, dynamic=False)
        fns = {"upstream-eager": lambda: m(x), "upstream-compile": lambda: compiled(x)}
        fns.update({k: (lambda f=f: f(m, x)) for k, f in VARIANTS.items()})
        for k, fn in fns.items():
            try:
                y = fn()
                row["errors_vs_fp32_ref"][k] = rel(y, ref)
                if k == "upstream-eager":
                    bf16_ref = y
                else:
                    row.setdefault("max_abs_diff_vs_upstream_eager", {})[k] = (y.float() - bf16_ref.float()).abs().max().item()
            except Exception as e:
                row["errors_vs_fp32_ref"][k] = err_text(e)
        row["graph_capture"] = {k: graph_capture_error(fns[k]) or "ok" for k in ("upstream-eager", "grouped", "dense")}
        out["configs"].append(row)
        print(f"  {name}: {row}")
        del m, compiled
        torch._dynamo.reset()
        torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------- 2. routing numerics

def topk_sets(probs, A):
    v, i = torch.topk(probs, A + 1, dim=-1)
    return i[:, :A].sort(dim=-1).values, v[:, :A], v[:, A - 1] == v[:, A], i[:, :A]


def part_routing():
    rows = []
    for name, c in CONFIGS.items():
        E, A, D = c["num_experts"], c["num_activated_experts"], c["dim"]
        for s in LOGIT_STD:
            g = torch.Generator(device=DEVICE).manual_seed(SEED)
            gate = (torch.randn(E, D, device=DEVICE, generator=g) * (s / math.sqrt(D))).bfloat16()
            acc = {k: {"mismatch": 0, "ties": 0, "mismatch_vs_hf": 0, "w_err": 0.0, "w_n": 0} for k in ("bf16-softmax", "fp32-softmax", "fp32-logits")}
            maxp, ent = 0.0, 0.0
            for c0 in range(0, ROUTE_TOKENS, ROUTE_CHUNK):
                x = torch.randn(ROUTE_CHUNK, D, device=DEVICE, generator=g).bfloat16()
                p64 = torch.softmax(x.double() @ gate.double().T, dim=-1)
                maxp += p64.max(-1).values.sum().item()
                ent += -(p64 * p64.clamp_min(1e-300).log()).sum(-1).sum().item()
                set64, v64, _, _ = topk_sets(p64, A)
                lb = F.linear(x, gate)
                probs = {"bf16-softmax": F.softmax(lb, dim=-1), "fp32-softmax": F.softmax(lb.float(), dim=-1),
                         "fp32-logits": F.softmax(x.float() @ gate.float().T, dim=-1)}
                sets = {}
                for k, p in probs.items():
                    st, v, tie, _ = topk_sets(p, A)
                    sets[k] = st
                    mm = (st != set64).any(-1)
                    a = acc[k]
                    a["mismatch"] += mm.sum().item()
                    a["ties"] += tie.sum().item()
                    ok = ~mm
                    wq = (v / v.sum(-1, keepdim=True)).double()
                    w64 = v64 / v64.sum(-1, keepdim=True)
                    a["w_err"] += (wq[ok] - w64[ok]).abs().sum().item()
                    a["w_n"] += ok.sum().item() * A
                for k in probs:
                    acc[k]["mismatch_vs_hf"] += (sets[k] != sets["fp32-softmax"]).any(-1).sum().item()
            row = {"config": name, "E": E, "A": A, "logit_std": s, "tokens": ROUTE_TOKENS,
                   "mean_max_prob": maxp / ROUTE_TOKENS, "mean_entropy_nats": ent / ROUTE_TOKENS}
            for k, a in acc.items():
                row[k] = {"set_mismatch_vs_fp64": a["mismatch"] / ROUTE_TOKENS, "boundary_tie": a["ties"] / ROUTE_TOKENS,
                          "set_mismatch_vs_hf": a["mismatch_vs_hf"] / ROUTE_TOKENS,
                          "combine_weight_abs_err": a["w_err"] / max(1, a["w_n"])}
            rows.append(row)
            print(f"  {name} std={s}: " + " ".join(f"{k} mm={row[k]['set_mismatch_vs_fp64']:.4f} tie={row[k]['boundary_tie']:.4f}" for k in acc))
    return rows


def part_impact():
    """Output error of upstream (bf16) routing vs fp64 routing, same FFN arithmetic."""
    rows = []
    for name in CONFIGS:
        m = build_moe(name)
        T, A = IMPACT_T[name], m.num_activated_experts
        g = torch.Generator(device=DEVICE).manual_seed(SEED + 1)
        x = torch.randn(T, m.dim, device=DEVICE, generator=g).bfloat16()
        w_up, idx_up = route(m, x)
        p64 = torch.softmax(x.double() @ m.gate.weight.double().T, dim=-1)
        v64, idx64 = torch.topk(p64, A, dim=-1)
        w64 = v64 / v64.sum(-1, keepdim=True)
        ref = ref_ffn(m.cond_ffn, x, w64, idx64)
        mm = (idx_up.sort(-1).values != idx64.sort(-1).values).any(-1)
        y_floor = ffn_grouped(m.cond_ffn, x, w64.bfloat16(), idx64)
        y_up = ffn_grouped(m.cond_ffn, x, w_up, idx_up)
        row = {"config": name, "T": T, "mismatch_tokens": int(mm.sum().item()),
               "bf16_ffn_fp64_routing": rel(y_floor, ref), "bf16_ffn_upstream_routing": rel(y_up, ref),
               "upstream_routing_on_mismatched": rel(y_up[mm], ref[mm]) if mm.any() else None,
               "upstream_routing_on_matched": rel(y_up[~mm], ref[~mm])}
        rows.append(row)
        print(f"  impact {row}")
        del m
        torch.cuda.empty_cache()
    return rows


# ---------------------------------------------------------------- 3. memory and speed

def bench(call, graph: bool):
    call()
    sync()
    t0 = time.perf_counter()
    call()
    sync()
    est = time.perf_counter() - t0
    reps = max(3, min(MAX_REPS, int(TARGET_S / max(est, 1e-6))))
    eager = []
    for _ in range(3):
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        st.record()
        for _ in range(reps):
            call()
        en.record()
        sync()
        eager.append(st.elapsed_time(en) * 1e3 / reps)
    if not graph:
        return sorted(eager)[1], None
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        call()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            call()
    g.replay()
    sync()
    times = []
    for _ in range(3):
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        st.record()
        g.replay()
        en.record()
        sync()
        times.append(st.elapsed_time(en) * 1e3 / reps)
    del g
    return sorted(eager)[1], sorted(times)[1]


def peak_extra(call) -> int:
    sync()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    y = call()
    sync()
    peak = torch.cuda.max_memory_allocated() - base
    del y
    return peak


def extern_kernels(fn, *args) -> list[str] | None:
    try:
        from torch._inductor.utils import run_and_get_code
        _, codes = run_and_get_code(fn, *args)
        return sorted({k for c in codes for k in re.findall(r"extern_kernels\.(\w+)", c)})
    except Exception:
        return None


def warm_gpu(seconds=3.0):
    """Laptop GPU clocks ramp up under load; without this the first cell is measured cold."""
    a = torch.randn(4096, 4096, device=DEVICE, dtype=torch.bfloat16)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        for _ in range(10):
            a @ a
        sync()


def short(r) -> str:
    if r.get("status") != "ok":
        return r.get("status", "")[:24]
    t = r["graph_us"] if r["graph_us"] is not None else r["eager_us"]
    return f"{t:.0f}us/{r['peak_extra_bytes'] / 2**20:.0f}MiB"


def part_speed():
    rows = []
    warm_gpu()
    for name in CONFIGS:
        torch._dynamo.reset()
        m = build_moe(name)
        eb = expert_bytes(m)
        A = m.num_activated_experts
        compiled = torch.compile(m, fullgraph=True, dynamic=False)
        dead = set()
        for T in T_LIST[name]:
            g = torch.Generator(device=DEVICE).manual_seed(SEED + T)
            x = torch.randn(T, m.dim, device=DEVICE, generator=g).bfloat16()
            _, idx = route(m, x)
            u = int(idx.unique().numel())
            row = {"config": name, "T": T, "unique_experts": u, "touched_expert_bytes": u * eb,
                   "gathered_bytes": T * A * eb, "weights_bytes": m.cond_ffn.w1.shape[0] * eb}
            fns = {"upstream-eager": lambda: m(x), "upstream-compile": lambda: compiled(x)}
            fns.update({k: (lambda f=f: f(m, x)) for k, f in VARIANTS.items()})
            for k in METHODS:
                if k in dead:
                    row[k] = {"status": "OOM at smaller T"}
                    continue
                if k == "upstream-compile" and T not in COMPILE_T[name]:
                    row[k] = {"status": "not compiled at this T"}
                    continue
                r = {}
                try:
                    if k == "upstream-compile":
                        t0 = time.perf_counter()
                        r["extern_kernels"] = extern_kernels(compiled, x)
                        sync()
                        r["first_call_s"] = round(time.perf_counter() - t0, 1)
                    r["peak_extra_bytes"] = peak_extra(fns[k])
                    e_us, g_us = bench(fns[k], k in GRAPHABLE)
                    r.update(status="ok", eager_us=e_us, graph_us=g_us)
                    best = g_us if g_us is not None else e_us
                    r["touched_GBps"] = u * eb / best / 1e3
                except torch.OutOfMemoryError:
                    r = {"status": "OOM"}
                    dead.add(k)
                except Exception as e:
                    r = {"status": err_text(e)}
                    print("   ", k, traceback.format_exc().splitlines()[-1])
                row[k] = r
                sync()
                torch.cuda.empty_cache()
            rows.append(row)
            print(f"  {name} T={T:5d} u={u:2d} " + " ".join(f"{k}:{short(row[k])}" for k in METHODS))
        del m, compiled
        torch._dynamo.reset()
        torch.cuda.empty_cache()
    return rows


# ---------------------------------------------------------------- report

def fmt_cell(r, key):
    if r.get("status") != "ok":
        return r.get("status", "")
    v = r.get(key)
    return "-" if v is None else f"{v:.0f}"


def write_md(R):
    L = ["# 0018 results", ""]
    c = R["correctness"]
    L += ["## 1. Correctness (same routing, relative error vs fp32 per-expert reference)", "",
          "| config | T | experts used | " + " | ".join(["upstream-eager", "upstream-compile", *VARIANTS]) + " | CUDA-graph capture |",
          "|---|---|---|" + "---|" * (2 + len(VARIANTS)) + "---|"]
    for r in c["configs"]:
        errs = r["errors_vs_fp32_ref"]
        cells = [f"{errs[k]:.2e}" if isinstance(errs[k], float) else errs[k] for k in ["upstream-eager", "upstream-compile", *VARIANTS]]
        L.append(f"| {r['config']} | {r['T']} | {r['unique_experts']} | " + " | ".join(cells) + " | " +
                 "; ".join(f"{k}: {v}" for k, v in r["graph_capture"].items()) + " |")
    L += ["", "Tiny full `Transformer` (dim 64, 4 experts, top-2), batch 2 vs two batch-1 calls:", "",
          "| S | B=1 output | B=2 output | MoE in -> out shape | h + moe(h) |", "|---|---|---|---|---|"]
    for r in c["batch_shapes"]:
        L.append(f"| {r['S']} | {r.get('B1_out_shape', r.get('B1_error'))} | {r.get('B2_out_shape', r.get('B2_error'))} | "
                 f"{r['moe_in_shape']} -> {r['moe_out_shape']} | {r.get('h_plus_moe_shape', r.get('h_plus_moe_error'))} |")

    L += ["", f"## 2. Routing numerics ({ROUTE_TOKENS} tokens, x ~ N(0, 1), synthetic bf16 router)", "",
          "Top-A set mismatch vs fp64 routing / exact tie between the A-th and (A+1)-th probability / mismatch vs HF-style (fp32 softmax of bf16 logits) / mean |combine weight error| on matching tokens.", "",
          "| config | logit std | mean max p | bf16 softmax (upstream) | fp32 softmax of bf16 logits (HF) | fp32 logits |", "|---|---|---|---|---|---|"]
    for r in R["routing"]:
        cells = [f"{100 * r[k]['set_mismatch_vs_fp64']:.2f}% / {100 * r[k]['boundary_tie']:.2f}% / {100 * r[k]['set_mismatch_vs_hf']:.2f}% / {r[k]['combine_weight_abs_err']:.1e}"
                 for k in ("bf16-softmax", "fp32-softmax", "fp32-logits")]
        L.append(f"| {r['config']} | {r['logit_std']} | {r['mean_max_prob']:.3f} | " + " | ".join(cells) + " |")
    L += ["", "Output error vs fp32 FFN with fp64 routing (logit std 1):", "",
          "| config | T | mismatched tokens | bf16 FFN, fp64 routing | bf16 FFN, upstream routing | upstream routing, mismatched tokens only | matched tokens only |",
          "|---|---|---|---|---|---|---|"]
    for r in R["impact"]:
        mmv = f"{r['upstream_routing_on_mismatched']:.2e}" if r["upstream_routing_on_mismatched"] is not None else "-"
        L.append(f"| {r['config']} | {r['T']} | {r['mismatch_tokens']} | {r['bf16_ffn_fp64_routing']:.2e} | {r['bf16_ffn_upstream_routing']:.2e} | {mmv} | {r['upstream_routing_on_matched']:.2e} |")

    for name in CONFIGS:
        rows = [r for r in R["speed"] if r["config"] == name]
        if not rows:
            continue
        L += ["", f"## 3. {name} layer ({rows[0]['weights_bytes'] / 2**30:.2f} GiB bf16 experts)", "",
              "### Extra peak memory per forward, MiB (cap: 60% of 8 GiB minus the weights)", "",
              "| T | experts touched | gathered copies MiB (T·A experts) | " + " | ".join(METHODS) + " |", "|---|---|---|" + "---|" * len(METHODS)]
        for r in rows:
            L.append(f"| {r['T']} | {r['unique_experts']} | {r['gathered_bytes'] / 2**20:.0f} | " + " | ".join(
                (f"{r[k]['peak_extra_bytes'] / 2**20:.0f}" if r[k].get("status") == "ok" else r[k].get("status", "")) for k in METHODS) + " |")
        for key, title in (("graph_us", "CUDA-graph replay (GPU time), µs"), ("eager_us", "eager launches (wall per call incl. host syncs), µs")):
            L += ["", f"### {title}", "", "| T | " + " | ".join(METHODS) + " |", "|---|" + "---|" * len(METHODS)]
            for r in rows:
                L.append(f"| {r['T']} | " + " | ".join(fmt_cell(r[k], key) for k in METHODS) + " |")
        L += ["", "### Bandwidth against the touched experts' bytes (best of graph / eager), GB/s", "", "| T | " + " | ".join(METHODS) + " |", "|---|" + "---|" * len(METHODS)]
        for r in rows:
            L.append(f"| {r['T']} | " + " | ".join(fmt_cell(r[k], "touched_GBps") for k in METHODS) + " |")
        L += ["", "### torch.compile(upstream): first call (compile + tuning) and Inductor extern kernels", "", "| T | first call s | extern kernels |", "|---|---|---|"]
        for r in rows:
            k = r["upstream-compile"]
            if "extern_kernels" in k:
                L.append(f"| {r['T']} | {k.get('first_call_s', '')} | {', '.join(k['extern_kernels'] or []) or 'none'} |")
            elif k.get("status") == "OOM":
                L.append(f"| {r['T']} | | OOM |")
    (HERE / "results.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    t_start = time.time()
    torch.manual_seed(SEED)
    R = {"meta": {"gpu": GPU_INFO, "python": platform.python_version(), "torch": torch.__version__,
                  "platform": platform.platform(), "seed": SEED, "configs": CONFIGS}}
    print("1. correctness")
    R["correctness"] = part_correctness()
    print("2. routing numerics")
    t0 = time.time()
    R["routing"] = part_routing()
    R["impact"] = part_impact()
    print(f"   {time.time() - t0:.0f}s")
    print("3. memory and speed")
    t0 = time.time()
    R["speed"] = part_speed()
    print(f"   {time.time() - t0:.0f}s")
    R["meta"]["wall_seconds"] = round(time.time() - t_start, 1)
    R["meta"]["peak_memory_mib"] = gpu_guard.peak_memory_mib()
    (HERE / "results.json").write_text(json.dumps(R, indent=1), encoding="utf-8")
    write_md(R)
    print(f"done in {R['meta']['wall_seconds']}s")


if __name__ == "__main__":
    main()
