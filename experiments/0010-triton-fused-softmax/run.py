"""Triton tutorial fused softmax vs torch.softmax on an RTX 4060 Laptop.

The tutorial (`python/tutorials/02-fused-softmax.py`) loads each row into one
block of next_power_of_2(n_cols) elements with 8 warps and launches a
persistent grid sized from the compiled kernel's register / shared-memory use.

1. Effective bandwidth (2 * M * N * 4 bytes / time) vs row length N at
   M = 4096 fp32 rows: tutorial wrapper as-is, the same kernel with its compiled
   kernel + grid cached, the best num_warps in {4, 8, 16, 32}, torch.softmax,
   torch.compile(torch.softmax), the tutorial's naive_softmax, and a plain copy
   as the achievable-bandwidth ceiling.
2. Kernel registers / spills / occupancy per (N, num_warps).
3. Per-call wall time on small inputs, where the tutorial calls `warmup` on
   every invocation.
4. Error vs an fp64 reference.

The tutorial is imported unmodified from the submodule; only its top-level
unit test and benchmark (everything from `torch.manual_seed(0)` on) are skipped.
"""
import ast
import gc
import json
import subprocess
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TRITON_DIR = ROOT / "upstream" / "triton"
TUTORIAL = TRITON_DIR / "python" / "tutorials" / "02-fused-softmax.py"
sys.path.insert(0, str(ROOT))

from lab import gpu_guard  # noqa: E402

import torch  # noqa: E402
import triton  # noqa: E402
from triton.runtime.errors import OutOfResources  # noqa: E402
from triton.testing import do_bench  # noqa: E402

SEED = 0
M = 4096
NS = [256, 512, 1024, 2048, 4096, 4097, 8192, 8193, 12288, 16384, 32768, 65536]
WARPS = [4, 8, 16, 32]
NUMERICS_ROWS = 512
HOST_SHAPES = [(64, 1024), (4096, 1024)]
HOST_CALLS = 2000
DTYPE = torch.float32

torch._dynamo.config.recompile_limit = 64


def load_tutorial() -> types.ModuleType:
    tree = ast.parse(TUTORIAL.read_text(encoding="utf-8"), filename=str(TUTORIAL))
    body = []
    for node in tree.body:
        if isinstance(node, ast.Expr) and ast.unparse(node).startswith("torch.manual_seed"):
            break
        body.append(node)
    tree.body = body
    mod = types.ModuleType("fused_softmax_tutorial")
    mod.__file__ = str(TUTORIAL)
    sys.modules[mod.__name__] = mod
    exec(compile(tree, str(TUTORIAL), "exec"), mod.__dict__)
    return mod


# The tutorial queries the CUDA device at import time.
GPU = gpu_guard.acquire()
tut = load_tutorial()


def launch_config(x, num_warps):
    """The tutorial's NVIDIA occupancy heuristic for an arbitrary num_warps."""
    n_rows, n_cols = x.shape
    block = triton.next_power_of_2(n_cols)
    num_stages = 4 if tut.SIZE_SMEM > 200000 else 2
    # warmup() only compiles, so x can stand in for the output buffer.
    kernel = tut.softmax_kernel.warmup(x, x, x.stride(0), x.stride(0), n_rows, n_cols, BLOCK_SIZE=block,
                                       num_stages=num_stages, num_warps=num_warps, grid=(1, ))
    cfg = {"kernel": kernel, "block_size": block, "num_stages": num_stages, "num_warps": num_warps,
           "shared_bytes": kernel.metadata.shared}
    try:
        kernel._init_handles()
    except OutOfResources:
        return dict(cfg, error=f"needs {kernel.metadata.shared} B shared memory > {tut.SIZE_SMEM} B limit",
                    n_regs=None, n_spills=None, occupancy=None, num_programs=None)
    occupancy = tut.NUM_REGS // (kernel.n_regs * tut.WARP_SIZE * num_warps)
    occupancy = min(occupancy, tut.SIZE_SMEM // max(kernel.metadata.shared, 1))
    return dict(cfg, error=None, n_regs=kernel.n_regs, n_spills=kernel.n_spills, occupancy=occupancy,
                num_programs=min(tut.NUM_SM * max(occupancy, 1), n_rows))


class CachedSoftmax:
    """Same kernel and grid as the tutorial, but compiled kernel + grid looked up once per shape."""

    def __init__(self, num_warps):
        self.num_warps = num_warps
        self.configs = {}

    def __call__(self, x):
        key = (x.shape, x.stride(0), x.dtype)
        cfg = self.configs.get(key)
        if cfg is None:
            cfg = self.configs[key] = launch_config(x, self.num_warps)
        if cfg["error"]:
            raise OutOfResources(cfg["shared_bytes"], tut.SIZE_SMEM, "shared memory")
        n_rows, n_cols = x.shape
        y = torch.empty_like(x)
        cfg["kernel"][(cfg["num_programs"], 1, 1)](y, x, x.stride(0), y.stride(0), n_rows, n_cols,
                                                    cfg["block_size"], cfg["num_stages"])
        return y


def torch_softmax(x):
    return torch.softmax(x, dim=-1)


torch_compiled = torch.compile(torch_softmax, dynamic=False)
cached = {w: CachedSoftmax(w) for w in WARPS}

VARIANTS = {
    "tutorial": tut.softmax,
    "cached_w8": cached[8],
    "torch_softmax": torch_softmax,
    "torch_compile": torch_compiled,
    "naive_softmax": tut.naive_softmax,
}
DESCRIPTIONS = {
    "tutorial": "tutorial softmax() as-is: warmup() + occupancy math on every call, 8 warps, one row per block",
    "cached_w8": "same kernel / grid, compiled kernel and grid cached per shape",
    "best_warps": "cached launch with the fastest num_warps in {4, 8, 16, 32} for that N",
    "torch_softmax": "torch.softmax(x, -1)",
    "torch_compile": "torch.compile(torch.softmax, dynamic=False) via triton-windows (Inductor-generated kernel)",
    "naive_softmax": "tutorial naive_softmax: max, sub, exp, sum, div as separate eager ops",
    "copy": "y.copy_(x): same bytes moved, no compute (achievable-bandwidth ceiling)",
}


def gbps(ms, numel):
    return round(2 * numel * 4 * 1e-9 / (ms * 1e-3), 1)


def bench(fn):
    """(median ms, status); status is 'ok', 'OOM' or 'smem' (kernel exceeds the shared-memory limit)."""
    try:
        ms, status = do_bench(fn, return_mode="median"), "ok"
    except torch.OutOfMemoryError:
        ms, status = None, "OOM"
    except OutOfResources:
        ms, status = None, "smem"
    if status != "ok":
        gc.collect()  # the failed call's temporaries are kept alive by the traceback cycle
    torch.cuda.empty_cache()
    return ms, status


def numerics(n, g):
    x = torch.randn(NUMERICS_ROWS, n, generator=g).cuda()
    ref = torch.softmax(x.double(), dim=-1)
    rows = []
    fns = dict(VARIANTS, **{f"cached_w{w}": cached[w] for w in WARPS if w != 8})
    for name, fn in fns.items():
        try:
            y = fn(x).double()
        except OutOfResources:
            continue
        rows.append({
            "N": n, "variant": name,
            "max_abs_err": float(f"{(y - ref).abs().max().item():.3g}"),
            "max_rel_err": float(f"{((y - ref).abs() / ref).max().item():.3g}"),
            "max_rowsum_err": float(f"{(y.sum(-1) - 1).abs().max().item():.3g}"),
            "allclose_torch_fp32": bool(torch.allclose(y.float(), torch.softmax(x, -1))),
        })
    return rows


def precompile():
    """Compile every torch.compile shape up front so no compilation work overlaps a measurement."""
    for shape in [(M, n) for n in NS] + [(NUMERICS_ROWS, n) for n in NS] + HOST_SHAPES:
        torch_compiled(torch.empty(*shape, device="cuda"))
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()


def sweep(g):
    precompile()
    perf, kernels, numeric_rows = [], [], []
    for n in NS:
        x = torch.randn(M, n, generator=g).cuda()
        numel = x.numel()
        row = {"N": n, "block_size": triton.next_power_of_2(n), "MiB_per_tensor": round(numel * 4 / 2**20, 1)}
        y = torch.empty_like(x)
        row["copy_gbps"] = gbps(bench(lambda: y.copy_(x))[0], numel)
        del y
        for name, fn in VARIANTS.items():
            ms, row[name + "_status"] = bench(lambda: fn(x))
            row[name + "_ms"] = round(ms, 4) if ms else None
            row[name + "_gbps"] = gbps(ms, numel) if ms else None
        warp_ms = {}
        for w in WARPS:
            cfg = launch_config(x, w)
            ms, status = bench(lambda: cached[w](x))
            if ms:
                warp_ms[w] = ms
            kernels.append({"N": n, **{k: v for k, v in cfg.items() if k != "kernel"},
                            "status": status, "ms": round(ms, 4) if ms else None,
                            "gbps": gbps(ms, numel) if ms else None})
        best = min(warp_ms, key=warp_ms.get) if warp_ms else None
        row["best_warps"] = best
        row["best_warps_status"] = "ok" if best else "smem"
        row["best_warps_ms"] = round(warp_ms[best], 4) if best else None
        row["best_warps_gbps"] = gbps(warp_ms[best], numel) if best else None
        base = row["torch_softmax_ms"]
        for name in ("tutorial", "cached_w8", "best_warps", "torch_compile", "naive_softmax"):
            row[name + "_speedup_vs_torch"] = round(base / row[name + "_ms"], 2) if row[name + "_ms"] else None
        perf.append(row)
        print(f"N={n}: " + ", ".join(f"{k}={row[k + '_gbps']}" for k in
                                      ("copy", "tutorial", "cached_w8", "best_warps", "torch_softmax",
                                       "torch_compile", "naive_softmax")) + f" GB/s (best warps {best})", flush=True)
        del x
        gc.collect()
        torch.cuda.empty_cache()
        numeric_rows += numerics(n, g)
    return perf, kernels, numeric_rows


def host_overhead(g):
    rows = []
    for shape in HOST_SHAPES:
        x = torch.randn(*shape, generator=g).cuda()
        for name, fn in VARIANTS.items():
            for _ in range(50):
                fn(x)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(HOST_CALLS):
                fn(x)
            t_issue = time.perf_counter() - t0
            torch.cuda.synchronize()
            t_wall = time.perf_counter() - t0
            rows.append({"shape": list(shape), "variant": name,
                         "issue_us_per_call": round(t_issue / HOST_CALLS * 1e6, 1),
                         "wall_us_per_call": round(t_wall / HOST_CALLS * 1e6, 1),
                         "gpu_us_per_call": round(do_bench(lambda: fn(x), return_mode="median") * 1e3, 1)})
            print(f"host {shape} {name}: {rows[-1]}", flush=True)
    return rows


def upstream_commit() -> str:
    return subprocess.run(["git", "-C", str(TRITON_DIR), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def main():
    t_start = time.time()
    torch.manual_seed(SEED)
    g = torch.Generator().manual_seed(SEED)
    props = tut.properties
    peak_gbps = round(2 * props["mem_clock_rate"] * 1e3 * props["mem_bus_width"] / 8 / 1e9, 1)

    perf, kernels, numeric_rows = sweep(g)
    host_rows = host_overhead(g)

    result = {
        "experiment": HERE.name,
        "upstream": "triton-lang/triton",
        "upstream_commit": upstream_commit(),
        "component": "python/tutorials/02-fused-softmax.py (softmax_kernel + softmax launcher)",
        "variants": DESCRIPTIONS,
        "dtype": str(DTYPE),
        "rows": M,
        "numerics_rows": NUMERICS_ROWS,
        "host_calls": HOST_CALLS,
        "triton": triton.__version__,
        "device_properties": props,
        "theoretical_dram_gbps": peak_gbps,
        "gpu": GPU,
        "perf": perf,
        "kernels": kernels,
        "numerics": numeric_rows,
        "host_overhead": host_rows,
        "peak_memory_mib": gpu_guard.peak_memory_mib(),
        "runtime_s": round(time.time() - t_start, 1),
    }
    (HERE / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    def cell(r, name):
        return r[name + "_gbps"] if r[name + "_status"] == "ok" else r[name + "_status"]

    def best_cell(r):
        return f"{r['best_warps_gbps']} ({r['best_warps']})" if r["best_warps"] else "smem"

    lines = [
        f"triton {upstream_commit()[:10]} tutorial, triton-windows {triton.__version__}, torch {torch.__version__}, "
        f"{GPU.get('name', 'cuda')} (theoretical DRAM {peak_gbps} GB/s, memory cap {GPU.get('memory_fraction_cap')})",
        "",
        f"## Effective bandwidth, GB/s ({M} fp32 rows; 2*M*N*4 bytes / median time from `triton.testing.do_bench`)",
        "",
        "| N | block | MiB | copy | tutorial | cached_w8 | best warps (w) | torch.softmax | torch.compile | naive | "
        "tutorial / torch | best / torch | compile / torch |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    lines += [f"| {r['N']} | {r['block_size']} | {r['MiB_per_tensor']} | {r['copy_gbps']} | {cell(r, 'tutorial')} | "
              f"{cell(r, 'cached_w8')} | {best_cell(r)} | "
              f"{cell(r, 'torch_softmax')} | {cell(r, 'torch_compile')} | {cell(r, 'naive_softmax')} | "
              f"{r['tutorial_speedup_vs_torch'] or ''} | {r['best_warps_speedup_vs_torch'] or ''} | "
              f"{r['torch_compile_speedup_vs_torch'] or ''} |" for r in perf]
    lines += [
        "",
        "## Kernel resources per num_warps (tutorial occupancy heuristic; occupancy = programs per SM)",
        "",
        "| N | warps | regs/thread | spills (bytes) | shared B | occupancy | programs | GB/s |",
        "|---|---|---|---|---|---|---|---|",
    ]
    lines += [f"| {k['N']} | {k['num_warps']} | {k['n_regs'] or ''} | {'' if k['n_spills'] is None else k['n_spills']} | "
              f"{k['shared_bytes']} | {'' if k['occupancy'] is None else k['occupancy']} | {k['num_programs'] or ''} | "
              f"{k['gbps'] if k['status'] == 'ok' else k['status']} |" for k in kernels]
    lines += [
        "",
        f"## Per-call cost on small inputs ({HOST_CALLS} back-to-back calls; issue = CPU time to enqueue, "
        "wall = until the GPU finishes, gpu = do_bench median)",
        "",
        "| shape | variant | issue us/call | wall us/call | gpu us/call |",
        "|---|---|---|---|---|",
    ]
    lines += [f"| {r['shape'][0]}x{r['shape'][1]} | {r['variant']} | {r['issue_us_per_call']} | "
              f"{r['wall_us_per_call']} | {r['gpu_us_per_call']} |" for r in host_rows]
    lines += [
        "",
        f"## Error vs fp64 softmax ({NUMERICS_ROWS} rows of N(0,1) logits)",
        "",
        "| N | variant | max abs err | max rel err | max abs(row sum - 1) | allclose(torch fp32) |",
        "|---|---|---|---|---|---|",
    ]
    lines += [f"| {r['N']} | {r['variant']} | {r['max_abs_err']} | {r['max_rel_err']} | {r['max_rowsum_err']} | "
              f"{r['allclose_torch_fp32']} |" for r in numeric_rows]
    (HERE / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
