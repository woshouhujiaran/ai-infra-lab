"""picotron ColumnParallelLinear simulated with gloo on CPU (TP = 1, 2, 4 processes).

picotron's ColumnParallelLinear keeps W_i = rows of W, computes Y_i = X W_i^T + b_i
and, in backward, all-reduces dX over the TP group (sync path: CopyToModelParallelRegion;
async path: LinearWithAsyncAllReduce, which launches the all-reduce before computing dW).

1. Shard storage: reset_parameters() builds the full master weight on every rank and
   keeps `torch.split(master, n, dim=0)[rank].contiguous()`. A dim-0 slice is already
   contiguous, so this is a view of the master. Measured: storage bytes behind the
   shard, RSS growth, torch.save size, and what `.to(float32)` / `.to(bfloat16)` (as in
   train.py) do to it. RowParallelLinear (dim-1 split) and VocabParallelEmbedding
   are measured for contrast; a variant that clones the shard is the fix.
2. Init: with the same seed on every rank (train.py's set_all_seed), is the gathered
   weight bitwise equal to nn.Linear's? And with per-rank seeds?
3. Correctness vs single-process nn.Linear and an fp64 reference: output, dX, dW, db
   for sync and async paths, 2D / 3D / 4D inputs, fp32 and bf16, TP = 1, 2, 4.
4. CPU timing (2 threads per rank): forward, forward+backward with sync / async
   all-reduce vs a local-only (no communication) baseline, the bare dX all-reduce, and
   an MLP up+gate pair where picotron all-reduces dX once per projection vs a single
   shared CopyToModelParallelRegion (one all-reduce).

picotron is imported unmodified from the submodule (picotron/utils.py is not imported:
it needs the Unix-only fcntl module).
"""
import gc
import io
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PICOTRON_DIR = ROOT / "upstream" / "picotron"
sys.path.insert(0, str(PICOTRON_DIR))
os.environ.setdefault("USE_LIBUV", "0")

import psutil  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import picotron.process_group_manager as pgm  # noqa: E402
from picotron.tensor_parallel.tensor_parallel import (  # noqa: E402
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from picotron.tensor_parallel.tp_communications import CopyToModelParallelRegion  # noqa: E402

SEED = 0
TPS = [1, 2, 4]
THREADS_PER_RANK = 2
BASE_PORT = 29610

# Llama-3-8B MLP projection and a 32k x 1024 embedding for the storage check.
MLP_HIDDEN, MLP_INTER = 4096, 14336
VOCAB, VOCAB_DIM = 32000, 1024

C_IN, C_OUT = 1024, 4096
C_SHAPES = {"2d": (256,), "3d": (4, 64), "4d": (2, 2, 64)}
C_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}

P_IN, P_OUT = 1024, 4096
P_TOKENS = [256, 1024, 4096]
P_BATCH = 4
WARMUP = 2

MiB = 2**20


class ClonedColumnParallelLinear(ColumnParallelLinear):
    """Upstream init, but the shard owns its storage so the master weight is freed."""

    def reset_parameters(self):
        super().reset_parameters()
        self.weight.data = self.weight.data.clone()


def tp_info():
    m = pgm.process_group_manager
    return m.tp_world_size, m.tp_rank, m.tp_group


def all_gather_cat(t, dim):
    tp, _, group = tp_info()
    if tp == 1:
        return t.clone()
    parts = [torch.empty_like(t) for _ in range(tp)]
    dist.all_gather(parts, t.contiguous(), group=group)
    return torch.cat(parts, dim=dim)


def rss():
    return psutil.Process().memory_info().rss


def storage_bytes(t):
    return t.untyped_storage().nbytes()


# ---------------------------------------------------------------- 1. storage

def measure_storage():
    cases = [
        ("ColumnParallelLinear", lambda: ColumnParallelLinear(MLP_HIDDEN, MLP_INTER, bias=False)),
        ("ColumnParallelLinear + clone", lambda: ClonedColumnParallelLinear(MLP_HIDDEN, MLP_INTER, bias=False)),
        ("RowParallelLinear", lambda: RowParallelLinear(MLP_INTER, MLP_HIDDEN, bias=False)),
        ("VocabParallelEmbedding", lambda: VocabParallelEmbedding(VOCAB, VOCAB_DIM)),
    ]
    rows = []
    for name, build in cases:
        gc.collect()
        dist.barrier()
        before = rss()
        m = build()
        after = rss()
        w = m.weight
        buf = io.BytesIO()
        torch.save(m.state_dict(), buf)
        row = {
            "module": name,
            "shape": list(w.shape),
            "shard_bytes": w.numel() * w.element_size(),
            "storage_bytes": storage_bytes(w),
            "rss_delta_bytes": after - before,
            "torch_save_bytes": buf.getbuffer().nbytes,
        }
        m.to(torch.float32)
        row["storage_after_to_fp32"] = storage_bytes(m.weight)
        m.to(torch.bfloat16)
        row["storage_after_to_bf16"] = storage_bytes(m.weight)
        rows.append(row)
        del m, w, buf
        gc.collect()
    return rows


# ---------------------------------------------------------------- 2. init

def init_equivalence():
    tp, rank, _ = tp_info()
    torch.manual_seed(1234)
    ref = nn.Linear(C_IN, C_OUT)
    torch.manual_seed(1234)
    col = ColumnParallelLinear(C_IN, C_OUT, bias=True)
    full = all_gather_cat(col.weight.data, 0)
    torch.manual_seed(1234 + rank)
    col2 = ColumnParallelLinear(C_IN, C_OUT, bias=True)
    full2 = all_gather_cat(col2.weight.data, 0)
    return {
        "same_seed_weight_bitwise_equal_nn_linear": torch.equal(full, ref.weight.data),
        "bias_all_zero": bool((col.bias == 0).all()),
        "rank_offset_seed_shard_equal_nn_linear": [
            torch.equal(a, b) for a, b in zip(full2.chunk(tp, 0), ref.weight.data.chunk(tp, 0))
        ],
    }


# ---------------------------------------------------------------- 3. correctness

def fp64_problem(lead):
    g = torch.Generator().manual_seed(SEED)
    bound = 1 / math.sqrt(C_IN)
    W = (torch.rand(C_OUT, C_IN, generator=g, dtype=torch.float64) * 2 - 1) * bound
    b = torch.randn(C_OUT, generator=g, dtype=torch.float64) * 0.02
    x = torch.randn(*lead, C_IN, generator=g, dtype=torch.float64)
    gy = torch.randn(*lead, C_OUT, generator=g, dtype=torch.float64)
    gf, xf = gy.reshape(-1, C_OUT), x.reshape(-1, C_IN)
    ref = {"y": x @ W.T + b, "dx": gy @ W, "dW": gf.T @ xf, "db": gf.sum(0)}
    return W, b, x, gy, ref


def run_single(W, b, x, gy, dtype):
    lin = nn.Linear(C_IN, C_OUT, dtype=dtype)
    with torch.no_grad():
        lin.weight.copy_(W)
        lin.bias.copy_(b)
    xi = x.to(dtype).requires_grad_(True)
    y = lin(xi)
    y.backward(gy.to(dtype))
    return {"y": y.detach(), "dx": xi.grad, "dW": lin.weight.grad, "db": lin.bias.grad}


def run_tp(W, b, x, gy, dtype, async_ar):
    tp, rank, _ = tp_info()
    layer = ColumnParallelLinear(C_IN, C_OUT, bias=True, gather_output=True, async_all_reduce=async_ar)
    layer.weight = nn.Parameter(W.to(dtype).chunk(tp, 0)[rank].clone())
    layer.bias = nn.Parameter(b.to(dtype).chunk(tp, 0)[rank].clone())
    xi = x.to(dtype).requires_grad_(True)
    y = layer(xi)
    y.backward(gy.to(dtype))
    dx_all = all_gather_cat(xi.grad.unsqueeze(0), 0)
    out = {
        "y": y.detach(),
        "dx": xi.grad,
        "dW": all_gather_cat(layer.weight.grad, 0),
        "db": all_gather_cat(layer.bias.grad, 0),
    }
    return out, all(torch.equal(dx_all[0], d) for d in dx_all)


def rel_l2(a, ref):
    return ((a.double() - ref).norm() / ref.norm()).item()


def compare(res, single, ref):
    return {
        k: {
            "rel_err_vs_fp64": rel_l2(res[k], ref[k]),
            "max_abs_vs_single": (res[k].double() - single[k].double()).abs().max().item(),
            "bitwise_equal_single": torch.equal(res[k], single[k]),
        }
        for k in ("y", "dx", "dW", "db")
    }


def correctness():
    rows = []
    for dname, dtype in C_DTYPES.items():
        for sname, lead in C_SHAPES.items():
            W, b, x, gy, ref = fp64_problem(lead)
            single = run_single(W, b, x, gy, dtype)
            rows.append({"dtype": dname, "shape": sname, "mode": "single", "status": "ok",
                         "metrics": compare(single, single, ref)})
            for mode in ("sync", "async"):
                row = {"dtype": dname, "shape": sname, "mode": mode}
                try:
                    res, dx_same = run_tp(W, b, x, gy, dtype, mode == "async")
                    row.update(status="ok", metrics=compare(res, single, ref), dx_ranks_identical=dx_same)
                except Exception as e:  # noqa: BLE001
                    row.update(status="error", error=f"{type(e).__name__}: {str(e).splitlines()[0][:200]}")
                dist.barrier()
                rows.append(row)
    return rows


# ---------------------------------------------------------------- 4. timing

def timed(fn, reps):
    _, _, group = tp_info()
    for _ in range(WARMUP):
        fn()
    ts = []
    for _ in range(reps):
        dist.barrier(group=group)
        t0 = time.perf_counter()
        fn()
        t = torch.tensor([time.perf_counter() - t0])
        dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group)
        ts.append(t.item())
    return statistics.median(ts) * 1e3


def perf():
    tp, rank, group = tp_info()
    torch.manual_seed(SEED)
    col = ColumnParallelLinear(P_IN, P_OUT, bias=False)
    col_g = ColumnParallelLinear(P_IN, P_OUT, bias=False, gather_output=True)
    col_a = ColumnParallelLinear(P_IN, P_OUT, bias=False, async_all_reduce=True)
    up = ColumnParallelLinear(P_IN, P_OUT, bias=False)
    gate = ColumnParallelLinear(P_IN, P_OUT, bias=False)
    for m in (col_g, col_a):
        m.weight = nn.Parameter(col.weight.data.clone())
    rows = []
    for T in P_TOKENS:
        reps = 5 if T >= 4096 else 10
        g = torch.Generator().manual_seed(SEED + T)
        lead = (P_BATCH, T // P_BATCH)
        x = torch.randn(*lead, P_IN, generator=g).requires_grad_(True)
        gy = torch.randn(*lead, P_OUT // tp, generator=g)
        dx_buf = torch.randn(*lead, P_IN, generator=g)

        def reset():
            x.grad = None
            for m in (col, col_a, up, gate):
                m.weight.grad = None

        def fwd():
            with torch.no_grad():
                col(x)

        def fwd_gather():
            with torch.no_grad():
                col_g(x)

        def local_fwd_bwd():
            reset()
            F.linear(x, col.weight).backward(gy)

        def sync_fwd_bwd():
            reset()
            col(x).backward(gy)

        def async_fwd_bwd():
            reset()
            col_a(x).backward(gy)

        def allreduce_dx():
            if tp > 1:
                dist.all_reduce(dx_buf, group=group)

        def up_gate_picotron():
            reset()
            (F.silu(gate(x)) * up(x)).backward(gy)

        def up_gate_shared_copy():
            reset()
            xc = CopyToModelParallelRegion.apply(x)
            (F.silu(F.linear(xc, gate.weight)) * F.linear(xc, up.weight)).backward(gy)

        row = {"tokens": T}
        for name, fn in [("fwd", fwd), ("fwd_gather", fwd_gather), ("local_fwd_bwd", local_fwd_bwd),
                         ("sync_fwd_bwd", sync_fwd_bwd), ("async_fwd_bwd", async_fwd_bwd),
                         ("allreduce_dx", allreduce_dx), ("up_gate_picotron", up_gate_picotron),
                         ("up_gate_shared_copy", up_gate_shared_copy)]:
            row[name + "_ms"] = timed(fn, reps)

        up_gate_picotron()
        dx_a = x.grad.clone()
        up_gate_shared_copy()
        row["up_gate_dx_max_abs_diff"] = (dx_a - x.grad).abs().max().item()
        row["up_gate_dx_bitwise_equal"] = torch.equal(dx_a, x.grad)
        reset()
        sync_fwd_bwd()
        dx_s = x.grad.clone()
        async_fwd_bwd()
        row["sync_async_dx_bitwise_equal"] = torch.equal(dx_s, x.grad)
        rows.append(row)
    return rows


# ---------------------------------------------------------------- driver

def worker(rank, tp, port):
    torch.set_num_threads(THREADS_PER_RANK)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=tp)
    pgm.setup_process_group_manager(tp_size=tp, cp_size=1, pp_size=1, dp_size=1)
    try:
        t0 = time.time()
        res = {"storage": measure_storage()}
        res["init"] = init_equivalence()
        res["correctness"] = correctness()
        res["perf"] = perf()
        res["seconds"] = time.time() - t0
        gathered = [None] * tp
        dist.all_gather_object(gathered, res)
        if rank == 0:
            (HERE / f"_tp{tp}.json").write_text(json.dumps(gathered), encoding="utf-8")
        dist.barrier()
    finally:
        dist.destroy_process_group()


def fmt(v, spec=".3g"):
    return format(v, spec) if isinstance(v, float) else str(v)


def write_md(result):
    runs = result["runs"]
    lines = [
        "# 0012 results: picotron ColumnParallelLinear on CPU (gloo)",
        "",
        f"torch {result['torch']}, {result['cpu']}, {THREADS_PER_RANK} threads per rank, total {result['seconds']:.0f} s.",
        "",
        "## 1. Shard storage (rank 0; RSS delta is the max over ranks)",
        "",
        "| Module | TP | Shard shape | Shard MiB | Storage MiB | RSS delta MiB | torch.save MiB | Storage after .to(fp32) | after .to(bf16) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for tp, ranks in runs.items():
        for i, r in enumerate(ranks[0]["storage"]):
            rss_max = max(rk["storage"][i]["rss_delta_bytes"] for rk in ranks)
            lines.append(
                f"| {r['module']} | {tp} | {r['shape']} | {r['shard_bytes'] / MiB:.0f} | {r['storage_bytes'] / MiB:.0f} | "
                f"{rss_max / MiB:.0f} | {r['torch_save_bytes'] / MiB:.0f} | {r['storage_after_to_fp32'] / MiB:.0f} | "
                f"{r['storage_after_to_bf16'] / MiB:.0f} |"
            )
    lines += ["", "## 2. Init vs nn.Linear (same seed on every rank / seed + rank)", "",
              "| TP | Same seed: weight bitwise equal | Bias all zero | seed + rank: shard equal per rank |", "|---|---|---|---|"]
    for tp, ranks in runs.items():
        r = ranks[0]["init"]
        lines.append(f"| {tp} | {r['same_seed_weight_bitwise_equal_nn_linear']} | {r['bias_all_zero']} | "
                     f"{r['rank_offset_seed_shard_equal_nn_linear']} |")
    lines += ["", f"## 3. Correctness ({C_IN} -> {C_OUT}, 256 tokens, gather_output=True)", "",
              "Relative L2 error vs fp64; `=` marks bitwise equality with single-process nn.Linear in the same dtype.", "",
              "| dtype | Input | TP | Mode | y | dx | dW | db | dx max abs vs single | dx identical on all ranks |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for tp, ranks in runs.items():
        for r in ranks[0]["correctness"]:
            if r["mode"] == "single" and tp != "1":
                continue
            if r["status"] != "ok":
                lines.append(f"| {r['dtype']} | {r['shape']} | {tp} | {r['mode']} | {r['error']} | | | | | |")
                continue
            m = r["metrics"]
            cells = [f"{m[k]['rel_err_vs_fp64']:.2e}{' =' if m[k]['bitwise_equal_single'] else ''}" for k in ("y", "dx", "dW", "db")]
            lines.append(f"| {r['dtype']} | {r['shape']} | {tp if r['mode'] != 'single' else '-'} | {r['mode']} | "
                         + " | ".join(cells) + f" | {m['dx']['max_abs_vs_single']:.2e} | {r.get('dx_ranks_identical', '-')} |")
    lines += ["", f"## 4. CPU timing ({P_IN} -> {P_OUT}, fp32, median ms, max over ranks)", "",
              "| Tokens | TP | fwd | fwd + gather | local fwd+bwd | sync fwd+bwd | async fwd+bwd | dX all-reduce | up+gate picotron | up+gate shared copy | sync speedup vs TP 1 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    base = {r["tokens"]: r["sync_fwd_bwd_ms"] for r in runs["1"][0]["perf"]}
    for T in P_TOKENS:
        for tp, ranks in runs.items():
            r = next(p for p in ranks[0]["perf"] if p["tokens"] == T)
            lines.append(
                f"| {T} | {tp} | {r['fwd_ms']:.1f} | {r['fwd_gather_ms']:.1f} | {r['local_fwd_bwd_ms']:.1f} | "
                f"{r['sync_fwd_bwd_ms']:.1f} | {r['async_fwd_bwd_ms']:.1f} | {r['allreduce_dx_ms']:.1f} | "
                f"{r['up_gate_picotron_ms']:.1f} | {r['up_gate_shared_copy_ms']:.1f} | {base[T] / r['sync_fwd_bwd_ms']:.2f}x |"
            )
    lines += ["", "up+gate dX, picotron vs shared copy (max abs diff / bitwise): "
              + "; ".join(f"TP {tp} T {p['tokens']}: {p['up_gate_dx_max_abs_diff']:.1e} / {p['up_gate_dx_bitwise_equal']}"
                          for tp, ranks in runs.items() for p in ranks[0]["perf"]),
              "", "sync vs async dX bitwise equal: "
              + "; ".join(f"TP {tp} T {p['tokens']}: {p['sync_async_dx_bitwise_equal']}"
                          for tp, ranks in runs.items() for p in ranks[0]["perf"])]
    (HERE / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    t0 = time.time()
    runs = {}
    for tp in TPS:
        mp.spawn(worker, args=(tp, BASE_PORT + tp), nprocs=tp, join=True)
        path = HERE / f"_tp{tp}.json"
        runs[str(tp)] = json.loads(path.read_text(encoding="utf-8"))
        path.unlink()
        print(f"TP {tp} done in {runs[str(tp)][0]['seconds']:.0f} s", flush=True)
    result = {
        "torch": torch.__version__,
        "cpu": platform.processor(),
        "threads_per_rank": THREADS_PER_RANK,
        "seconds": time.time() - t0,
        "runs": runs,
    }
    (HERE / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_md(result)


if __name__ == "__main__":
    main()
