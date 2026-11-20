"""Triton tutorial low-memory (seeded) dropout on an RTX 4060 Laptop.

The tutorial (`python/tutorials/04-low-memory-dropout.py`) replaces the stored
dropout mask with one int32 seed: `random = tl.rand(seed, offsets)`,
`keep = random > p`, `out = where(keep, x / (1 - p), 0)`.

1. Statistics: keep rate vs 1 - p, lag-k mask correlation and seed-to-seed
   mask agreement for the tutorial kernel, an int64 / rand4x rewrite and torch.
2. Edge semantics: p = 0 drops elements (rand can be exactly 0; counted on a
   2^31 - 4096 element uint8 tensor, the largest the int32 offsets allow), p
   outside [0, 1], n = 0, NaN / inf under dropped positions vs torch, dtype
   rounding vs fp64, mask independence of BLOCK_SIZE / num_warps.
3. Training semantics: reusing one seed across stacked layers (bias), memory
   held between forward and backward, recompute under checkpointing.
4. int32 offset overflow past 2^31 elements (child process).
5. Throughput: Philox rate (1 vs 4 outputs per call), launch-config sweep, GB/s
   vs size against torch eager / torch.compile / the mask-based tutorial kernel.

The tutorial is imported unmodified from the submodule; only its top-level
demo code is skipped (kernels and wrappers are executed as written).
"""
import ast
import gc
import json
import math
import subprocess
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TRITON_DIR = ROOT / "upstream" / "triton"
TUTORIAL = TRITON_DIR / "python" / "tutorials" / "04-low-memory-dropout.py"
sys.path.insert(0, str(ROOT))

from lab import gpu_guard  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from torch.utils.checkpoint import checkpoint  # noqa: E402
from triton.testing import do_bench  # noqa: E402

SEED = 0
KEEP_PS = [0.01, 0.1, 0.5, 0.9, 0.99]
KEEP_N = 2**25
KEEP_SEEDS = 4
LAG_N = 2**26
LAGS = [1, 2, 3, 4, 5, 8, 64, 1024, 2**20]
SEED_PAIRS = [(0, 1), (1, 2), (0, 2**16), (0, 2**31), (0, 2**32), (7, 2**32 + 7), (-1, 2**32 - 1), (123, 2**40 + 123)]
SEED_N = 2**24
ZERO_N = 2**31 - 4096
ZERO_SEEDS = 32
ZERO_CHUNK = 2**28
OVERFLOW_N = 2**31 + 2**20
CONFIG_N = 2**26
PERF_NS = [2**20, 2**24, 2**27]
PERF_ROUNDS = 3
TUT_BLOCKS = [256, 512, 1024, 2048, 4096]
TUT_WARPS = [2, 4, 8, 16]
X4_BLOCKS = [128, 256, 512, 1024]
X4_WARPS = [2, 4, 8]
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def load_tutorial() -> types.ModuleType:
    tree = ast.parse(TUTORIAL.read_text(encoding="utf-8"), filename=str(TUTORIAL))
    body = []
    for node in tree.body:
        if isinstance(node, ast.Import) and any(a.name == "tabulate" for a in node.names):
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)):
            body.append(node)
        elif isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "DEVICE":
            body.append(node)
    tree.body = body
    mod = types.ModuleType("dropout_tutorial")
    mod.__file__ = str(TUTORIAL)
    sys.modules[mod.__name__] = mod
    exec(compile(tree, str(TUTORIAL), "exec"), mod.__dict__)
    return mod


GPU = gpu_guard.acquire()
tut = load_tutorial()


@triton.jit
def _seeded_dropout_i64(x_ptr, output_ptr, n_elements, p, seed, BLOCK_SIZE: tl.constexpr):
    """The tutorial kernel with 64-bit offsets (same Philox counters below 2^32)."""
    pid = tl.program_id(axis=0).to(tl.int64)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.rand(seed, offsets)
    x_keep = random > p
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def _drop_chunk(x_ptr, output_ptr, offsets, n_elements, r, p):
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, tl.where(r > p, x / (1 - p), 0.0), mask=mask)


@triton.jit
def _seeded_dropout_x4(x_ptr, output_ptr, n_elements, p, seed, BLOCK_SIZE: tl.constexpr):
    """One Philox call per 4 elements (`rand4x`), 64-bit offsets."""
    pid = tl.program_id(axis=0).to(tl.int64)
    lane = tl.arange(0, BLOCK_SIZE)
    r0, r1, r2, r3 = tl.rand4x(seed, pid * BLOCK_SIZE + lane)
    base = pid * (4 * BLOCK_SIZE) + lane
    _drop_chunk(x_ptr, output_ptr, base, n_elements, r0, p)
    _drop_chunk(x_ptr, output_ptr, base + BLOCK_SIZE, n_elements, r1, p)
    _drop_chunk(x_ptr, output_ptr, base + 2 * BLOCK_SIZE, n_elements, r2, p)
    _drop_chunk(x_ptr, output_ptr, base + 3 * BLOCK_SIZE, n_elements, r3, p)


@triton.jit
def _rand_at(offsets_ptr, out_ptr, seed, n):
    idx = tl.arange(0, 32)
    m = idx < n
    off = tl.load(offsets_ptr + idx, mask=m, other=0)
    tl.store(out_ptr + idx, tl.rand(seed, off), mask=m)


@triton.jit
def _philox_rate1(out_ptr, seed, iters, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for it in range(iters):
        acc += tl.rand(seed, (pid * iters + it) * BLOCK + tl.arange(0, BLOCK))
    tl.store(out_ptr + pid, tl.sum(acc, axis=0))


@triton.jit
def _philox_rate4(out_ptr, seed, iters, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for it in range(iters):
        r0, r1, r2, r3 = tl.rand4x(seed, (pid * iters + it) * BLOCK + tl.arange(0, BLOCK))
        acc += r0 + r1 + r2 + r3
    tl.store(out_ptr + pid, tl.sum(acc, axis=0))


def launch_tut(x, out, p, seed, block=1024, warps=4):
    n = x.numel()
    tut._seeded_dropout[(triton.cdiv(n, block), )](x, out, n, p, seed, BLOCK_SIZE=block, num_warps=warps)
    return out


def launch_i64(x, out, p, seed, block=1024, warps=4):
    n = x.numel()
    _seeded_dropout_i64[(triton.cdiv(n, block), )](x, out, n, p, seed, BLOCK_SIZE=block, num_warps=warps)
    return out


def launch_x4(x, out, p, seed, block=256, warps=4):
    n = x.numel()
    _seeded_dropout_x4[(triton.cdiv(n, 4 * block), )](x, out, n, p, seed, BLOCK_SIZE=block, num_warps=warps)
    return out


def tut_dropout(x, p, seed):
    return tut.seeded_dropout(x, p, seed)


def i64_dropout(x, p, seed):
    return launch_i64(x, torch.empty_like(x), p, seed)


def x4_dropout(x, p, seed):
    return launch_x4(x, torch.empty_like(x), p, seed)


def torch_dropout(x, p, seed):
    torch.manual_seed(seed)
    return F.dropout(x, p, True)


STAT_VARIANTS = {"tutorial": tut_dropout, "x4": x4_dropout, "torch": torch_dropout}


def z_binom(k, n, q):
    return (k - n * q) / math.sqrt(n * q * (1 - q))


def release():
    gc.collect()
    torch.cuda.empty_cache()


def mask_of(fn, n, p, seed):
    x = torch.ones(n, device="cuda")
    m = fn(x, p, seed) != 0
    del x
    return m


# ---------------------------------------------------------------
# 1. statistics
def keep_rates():
    rows = []
    for name, fn in STAT_VARIANTS.items():
        for p in KEEP_PS:
            kept = sum(int(mask_of(fn, KEEP_N, p, s).sum()) for s in range(KEEP_SEEDS))
            total = KEEP_N * KEEP_SEEDS
            rows.append({"variant": name, "p": p, "keep_rate": round(kept / total, 6), "expected": round(1 - p, 6),
                         "z": round(z_binom(kept, total, 1 - p), 2)})
            release()
    return rows


def lag_correlation():
    rows = []
    for name, fn in STAT_VARIANTS.items():
        m = mask_of(fn, LAG_N, 0.5, 11)
        for lag in LAGS:
            agree = int((m[:-lag] == m[lag:]).sum())
            n = LAG_N - lag
            rows.append({"variant": name, "lag": lag, "agree": round(agree / n, 6), "z": round(z_binom(agree, n, 0.5), 2)})
        del m
        release()
    return rows


def seed_agreement():
    rows = []
    for name in ("tutorial", "x4"):
        fn = STAT_VARIANTS[name]
        for a, b in SEED_PAIRS:
            ma, mb = mask_of(fn, SEED_N, 0.5, a), mask_of(fn, SEED_N, 0.5, b)
            agree = int((ma == mb).sum())
            rows.append({"variant": name, "seed_a": a, "seed_b": b, "agree": round(agree / SEED_N, 6),
                         "z": round(z_binom(agree, SEED_N, 0.5), 2)})
            del ma, mb
        release()
    return rows


# ---------------------------------------------------------------
# 2. edge semantics
def zero_drop_at_p0():
    """p = 0 should be the identity, but rand() == 0 exactly drops the element."""
    x = torch.ones(ZERO_N, dtype=torch.uint8, device="cuda")
    hits = []
    t0 = time.time()
    for seed in range(ZERO_SEEDS):
        out = tut.seeded_dropout(x, 0.0, seed)
        for i in range(0, ZERO_N, ZERO_CHUNK):
            bad = (out[i:i + ZERO_CHUNK] != 1).nonzero().flatten()
            for o in bad.tolist():
                hits.append((seed, i + o))
        del out
    offs = torch.zeros(32, dtype=torch.int64, device="cuda")
    vals = torch.full((32, ), -1.0, device="cuda")
    rand_values = []
    for seed, o in hits:
        offs[0] = o
        _rand_at[(1, )](offs, vals, seed, 1)
        rand_values.append(float(vals[0]))
    n_total = ZERO_N * ZERO_SEEDS
    out = {"n_per_seed": ZERO_N, "seeds": ZERO_SEEDS, "elements": n_total, "dropped": len(hits),
           "expected_dropped": round(n_total * 2.0**-31, 2), "drop_rate": len(hits) / n_total,
           "rand_value_at_dropped": sorted(set(rand_values)), "first_hits": hits[:8],
           "seconds": round(time.time() - t0, 1)}
    del x
    release()
    return out


def attempt(fn):
    try:
        r = fn()
        torch.cuda.synchronize()
        return {"ok": True, "result": r}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:120]}"}


def p_edge_cases():
    rows = []
    n = 2**20
    x = torch.ones(n, device="cuda")
    for p in (0.0, 1.0, 1.5, -0.5):
        r = attempt(lambda: tut.seeded_dropout(x, p, 3))
        row = {"p": p, "tutorial": r.get("error", "")}
        if r["ok"]:
            o = r["result"]
            row["tutorial"] = f"keep {float((o != 0).float().mean()):.4f}, kept value {float(o.max()):.4g}"
        t = attempt(lambda: F.dropout(x, p, True))
        row["torch"] = t.get("error", "ok")
        rows.append(row)
    r = attempt(lambda: tut.seeded_dropout(torch.empty(0, device="cuda"), 0.5, 3).numel())
    rows.append({"p": "n = 0", "tutorial": r.get("error", f"returns {r.get('result')} elements"),
                 "torch": attempt(lambda: F.dropout(torch.empty(0, device="cuda"), 0.5, True).numel()).get("error", "ok")})
    xt = torch.ones(64, 64, device="cuda").t()
    rows.append({"p": "non-contiguous x", "tutorial": attempt(lambda: tut.seeded_dropout(xt, 0.5, 3)).get("error", "ok"),
                 "torch": attempt(lambda: F.dropout(xt, 0.5, True)).get("error", "ok")})
    del x
    return rows


def nonfinite_propagation():
    n = 2**20
    g = torch.Generator(device="cuda").manual_seed(SEED)
    x = torch.randn(n, device="cuda", generator=g)
    x[::997], x[1::997], x[2::997] = float("nan"), float("inf"), float("-inf")
    bad = ~torch.isfinite(x)
    out = {"nonfinite_inputs": int(bad.sum())}
    for name in ("tutorial", "torch"):
        o = STAT_VARIANTS[name](x, 0.5, 3)
        out[f"{name}_nonfinite_outputs"] = int((~torch.isfinite(o)).sum())
        out[f"{name}_nonfinite_at_dropped"] = int(((o == 0) & bad).sum()) if name == "tutorial" else int(
            (~torch.isfinite(o)).sum())
    out["tutorial_fraction_still_nonfinite"] = round(out["tutorial_nonfinite_outputs"] / out["nonfinite_inputs"], 3)
    out["torch_fraction_still_nonfinite"] = round(out["torch_nonfinite_outputs"] / out["nonfinite_inputs"], 3)
    return out


def dtype_rounding():
    rows = []
    n = 2**22
    for dname, dt in DTYPES.items():
        g = torch.Generator(device="cuda").manual_seed(SEED)
        x = (1 + 0.1 * torch.randn(n, device="cuda", generator=g)).to(dt)
        for p in (0.1, 0.5):
            o = tut.seeded_dropout(x, p, 5)
            keep = o != 0
            ref = (x.double() / (1 - p)).to(dt).double()
            err = (o.double() - ref)[keep].abs() / ref[keep].abs()
            rows.append({"dtype": dname, "p": p, "keep_rate": round(float(keep.float().mean()), 5),
                         "max_rel_err_vs_rounded_fp64": float(err.max()),
                         "mask_equal_to_fp32_mask": bool(torch.equal(keep, mask_of(tut_dropout, n, p, 5)))})
    return rows


def launch_independence():
    n = 2**24 + 37
    x = torch.randn(n, device="cuda")
    ref = tut.seeded_dropout(x, 0.5, 9)
    rows = []
    for block in (128, 256, 1024, 4096, 8192):
        for warps in (1, 2, 4, 8):
            r = attempt(lambda: launch_tut(x, torch.empty_like(x), 0.5, 9, block, warps))
            rows.append({"block": block, "warps": warps,
                         "identical": bool(torch.equal(r["result"], ref)) if r["ok"] else r["error"]})
    r = attempt(lambda: torch.equal(i64_dropout(x, 0.5, 9), ref))
    rows.append({"block": "int64 rewrite", "warps": 4, "identical": r.get("result", r.get("error"))})
    return rows


# ---------------------------------------------------------------
# 3. training semantics
class SeededDropout(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, p, seed):
        ctx.p, ctx.seed = p, seed
        return tut.seeded_dropout(x, p, seed)

    @staticmethod
    def backward(ctx, dy):
        return tut.seeded_dropout(dy.contiguous(), ctx.p, ctx.seed), None, None


def stacked_layers():
    n, p = 2**24, 0.5
    x = torch.ones(n, device="cuda")
    rows = []
    for layers in (1, 2, 3, 4):
        for mode in ("same seed", "distinct seeds"):
            h = x
            for i in range(layers):
                h = tut.seeded_dropout(h, p, 7 if mode == "same seed" else 7 + i)
            rows.append({"layers": layers, "seeds": mode, "mean_output": round(float(h.mean()), 4),
                         "keep_rate": round(float((h != 0).float().mean()), 4),
                         "expected_keep": round((1 - p)**layers if mode == "distinct seeds" else 1 - p, 4)})
    torch.manual_seed(1)
    h = x
    for _ in range(4):
        h = F.dropout(h, p, True)
    rows.append({"layers": 4, "seeds": "torch (RNG advances)", "mean_output": round(float(h.mean()), 4),
                 "keep_rate": round(float((h != 0).float().mean()), 4), "expected_keep": round((1 - p)**4, 4)})
    return rows


def held_memory():
    n, p = 2**27, 0.5
    rows = []
    compiled = torch.compile(lambda t: F.dropout(t, p, True))
    fns = {"seeded autograd.Function": lambda t: SeededDropout.apply(t, p, 5),
           "torch F.dropout": lambda t: F.dropout(t, p, True),
           "torch.compile F.dropout": compiled}
    for name, fn in fns.items():
        x = torch.randn(n, device="cuda", dtype=torch.float16, requires_grad=True)
        fn(x).sum().backward()  # warm up / compile
        x.grad = None
        release()
        base = torch.cuda.memory_allocated()
        y = fn(x)
        held = torch.cuda.memory_allocated() - base - y.numel() * y.element_size()
        y.backward(torch.ones_like(y))
        rows.append({"variant": name, "n": n, "dtype": "fp16", "extra_held_MiB_after_forward": round(held / 2**20, 1),
                     "bytes_per_element": round(held / n, 3)})
        del x, y
        release()
    del compiled
    # gradient is exactly the forward mask times dy / (1 - p)
    x = torch.randn(2**22, device="cuda", requires_grad=True)
    y = SeededDropout.apply(x, p, 5)
    dy = torch.randn_like(y)
    y.backward(dy)
    keep = y != 0
    ref = torch.where(keep, dy / (1 - p), torch.zeros_like(dy))
    rows.append({"variant": "seeded backward mask == forward mask", "grad_matches": bool(torch.equal(x.grad, ref))})
    return rows


def checkpoint_recompute():
    rows = []
    g = torch.Generator(device="cuda").manual_seed(SEED)
    x = torch.randn(512, 512, device="cuda", generator=g)
    w0 = torch.randn(512, 512, device="cuda", generator=g)
    p = 0.5

    def block(x, w, drop):
        return (drop(x @ w)**2).sum()

    drops = {"torch F.dropout": lambda h: F.dropout(h, p, True), "seeded": lambda h: SeededDropout.apply(h, p, 21)}
    for name, drop in drops.items():
        w = w0.clone().requires_grad_()
        torch.manual_seed(1234)
        block(x, w, drop).backward()
        ref = w.grad.clone()
        for preserve in (False, True):
            w = w0.clone().requires_grad_()
            torch.manual_seed(1234)
            checkpoint(block, x, w, drop, use_reentrant=False, preserve_rng_state=preserve).backward()
            rows.append({"dropout": name, "preserve_rng_state": preserve,
                         "grad_rel_err_vs_no_checkpoint": float((w.grad - ref).norm() / ref.norm())})
    return rows


# ---------------------------------------------------------------
# 4. int32 overflow (child process)
def overflow_child():
    n = OVERFLOW_N
    x = torch.ones(n, dtype=torch.uint8, device="cuda")
    out = torch.zeros(n, dtype=torch.uint8, device="cuda")
    res = {"n": n, "int32_limit": 2**31}
    try:
        tut._seeded_dropout[(triton.cdiv(n, 1024), )](x, out, n, 0.5, 5, BLOCK_SIZE=1024)
        torch.cuda.synchronize()
        res["tutorial"] = "completed"
        tail = out[2**31:].clone()
        head = out[::4093].clone()
        x_bad = sum(int((x[i:i + ZERO_CHUNK] != 1).sum()) for i in range(0, n, ZERO_CHUNK))
        res["tail_elements"] = int(tail.numel())
        res["tail_nonzero"] = int((tail != 0).sum())
        res["x_elements_overwritten"] = x_bad
        out.zero_()
        x.fill_(1)
        launch_i64(x, out, 0.5, 5, 1024)
        torch.cuda.synchronize()
        res["i64_tail_nonzero"] = int((out[2**31:] != 0).sum())
        res["tail_matches_i64"] = bool(torch.equal(tail, out[2**31:].clone()))
        res["head_matches_i64"] = bool(torch.equal(head, out[::4093].clone()))
        res["i64_tail_keep_rate"] = round(res["i64_tail_nonzero"] / res["tail_elements"], 4)
    except Exception as e:  # noqa: BLE001
        res["tutorial"] = f"{type(e).__name__}: {str(e)[:160]}"
    print("RESULT " + json.dumps(res), flush=True)


def overflow():
    p = subprocess.run([sys.executable, __file__, "--overflow"], capture_output=True, text=True, timeout=900)
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")]
    if lines:
        return json.loads(lines[-1][7:])
    tail = (p.stderr or p.stdout).strip().splitlines()
    return {"n": OVERFLOW_N, "tutorial": f"child exited {p.returncode}: {(tail[-1] if tail else '')[:200]}"}


# ---------------------------------------------------------------
# 5. throughput
def bench(fn):
    try:
        ts = [do_bench(fn, return_mode="median") for _ in range(PERF_ROUNDS)]
        return sorted(ts)[len(ts) // 2]
    except torch.OutOfMemoryError:
        release()
        return None


def philox_rate():
    rows = []
    out = torch.empty(2**14, device="cuda")
    iters, block = 16, 1024
    calls = out.numel() * iters * block
    for name, kern, per_call in (("rand (1 of 4 outputs)", _philox_rate1, 1), ("rand4x", _philox_rate4, 4)):
        ms = bench(lambda: kern[(out.numel(), )](out, 1, iters, BLOCK=block))
        rows.append({"kernel": name, "philox_calls": calls, "ms": round(ms, 3),
                     "G_calls_per_s": round(calls / ms * 1e-6, 1), "G_numbers_per_s": round(calls * per_call / ms * 1e-6, 1)})
    return rows


def config_sweep():
    rows, best = [], {}
    for dname in ("fp32", "fp16"):
        x = torch.randn(CONFIG_N, device="cuda").to(DTYPES[dname])
        out = torch.empty_like(x)
        es = x.element_size()
        for kname, fn, blocks, warps_list in (("tutorial", launch_tut, TUT_BLOCKS, TUT_WARPS),
                                              ("x4", launch_x4, X4_BLOCKS, X4_WARPS)):
            for block in blocks:
                for warps in warps_list:
                    r = attempt(lambda: fn(x, out, 0.5, 1, block, warps))
                    if not r["ok"]:
                        rows.append({"dtype": dname, "kernel": kname, "block": block, "warps": warps, "error": r["error"]})
                        continue
                    ms = bench(lambda: fn(x, out, 0.5, 1, block, warps))
                    gbps = 2 * CONFIG_N * es * 1e-9 / (ms * 1e-3)
                    rows.append({"dtype": dname, "kernel": kname, "block": block, "warps": warps, "ms": round(ms, 4),
                                 "gbps": round(gbps, 1)})
                    if gbps > best.get((dname, kname), (0, 0, 0))[0]:
                        best[(dname, kname)] = (gbps, block, warps)
        del x, out
        release()
    return rows, {k: v[1:] for k, v in best.items()}


def perf(best):
    rows = []
    compiled = torch.compile(lambda t: F.dropout(t, 0.5, True))
    p = 0.5
    for dname in ("fp32", "fp16"):
        dt = DTYPES[dname]
        es = torch.empty(0, dtype=dt).element_size()
        tb, tw = best[(dname, "tutorial")]
        xb, xw = best[(dname, "x4")]
        for n in PERF_NS:
            x = torch.randn(n, device="cuda").to(dt)
            out = torch.empty_like(x)
            keep32 = (torch.rand(n, device="cuda") > p).to(torch.int32)
            compiled(x)
            fns = {
                "copy": lambda: out.copy_(x),
                "tutorial": lambda: tut.seeded_dropout(x, p, 1),
                f"tutorial best (BLOCK {tb}, {tw} warps)": lambda: launch_tut(x, out, p, 1, tb, tw),
                "int64 offsets": lambda: launch_i64(x, out, p, 1),
                f"x4 rewrite (BLOCK {xb}, {xw} warps)": lambda: launch_x4(x, out, p, 1, xb, xw),
                "torch F.dropout": lambda: F.dropout(x, p, True),
                "torch.compile F.dropout": lambda: compiled(x),
                "tutorial mask kernel only (int32 mask)": lambda: tut.dropout(x, keep32, p),
                "tutorial mask incl. torch.rand mask": lambda: tut.dropout(x, (torch.rand(n, device="cuda") > p).to(torch.int32), p),
            }
            row = {"dtype": dname, "n": n, "MiB": round(n * es / 2**20, 1)}
            for name, fn in fns.items():
                ms = bench(fn)
                row[name] = {"us": round(ms * 1e3, 1), "gbps": round(2 * n * es * 1e-9 / (ms * 1e-3), 1)} if ms else None
            rows.append(row)
            print("perf", dname, n, {k: (v or {}).get("gbps") for k, v in row.items() if isinstance(v, dict) or v is None},
                  flush=True)
            del x, out, keep32
            release()
    return rows


# ---------------------------------------------------------------
def upstream_commit() -> str:
    return subprocess.run(["git", "-C", str(TRITON_DIR), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def write_md(res):
    gpu = res["gpu"]
    L = [f"triton {res['upstream_commit'][:10]} tutorial 04, triton-windows {triton.__version__}, torch {torch.__version__}, "
         f"{gpu.get('name', 'cuda')} (memory cap {gpu.get('memory_fraction_cap')}), runtime {res['runtime_s']} s", ""]
    L += [f"## 1a. Keep rate vs 1 - p ({KEEP_SEEDS} seeds x 2^25 elements; z = binomial z-score)", ""]
    L += table(["variant", "p", "keep rate", "expected", "z"],
               [[r["variant"], r["p"], r["keep_rate"], r["expected"], r["z"]] for r in res["keep_rates"]])
    L += ["", "## 1b. Lag-k mask agreement (p = 0.5, 2^26 elements; expected 0.5)", ""]
    L += table(["variant", "lag", "agree", "z"], [[r["variant"], r["lag"], r["agree"], r["z"]] for r in res["lag"]])
    L += ["", f"## 1c. Mask agreement between seeds (p = 0.5, {SEED_N} elements; expected 0.5)", ""]
    L += table(["variant", "seed a", "seed b", "agree", "z"],
               [[r["variant"], r["seed_a"], r["seed_b"], r["agree"], r["z"]] for r in res["seeds"]])
    z = res["zero_drop"]
    L += ["", "## 2a. p = 0 is not the identity", "",
          f"uint8 ones, {z['n_per_seed']} elements x {z['seeds']} seeds = {z['elements']} elements, tutorial kernel as written: "
          f"**{z['dropped']} dropped** (expected {z['expected_dropped']} from P(rand == 0) = 2^-31); rand value at the dropped "
          f"offsets: {z['rand_value_at_dropped']}; first hits (seed, offset): {z['first_hits']}.", ""]
    L += ["## 2b. p out of range / n = 0 / non-contiguous", ""]
    L += table(["case", "tutorial", "torch F.dropout"], [[r["p"], r["tutorial"], r["torch"]] for r in res["p_edge"]])
    nf = res["nonfinite"]
    L += ["", "## 2c. NaN / inf inputs at dropped positions (p = 0.5)", "",
          f"{nf['nonfinite_inputs']} non-finite inputs: tutorial leaves {nf['tutorial_nonfinite_outputs']} "
          f"({nf['tutorial_fraction_still_nonfinite']}), torch leaves {nf['torch_nonfinite_outputs']} "
          f"({nf['torch_fraction_still_nonfinite']}).", "", "## 2d. dtype rounding vs fp64 reference", ""]
    L += table(["dtype", "p", "keep rate", "max rel err vs rounded fp64", "mask == fp32 mask"],
               [[r["dtype"], r["p"], r["keep_rate"], r["max_rel_err_vs_rounded_fp64"], r["mask_equal_to_fp32_mask"]]
                for r in res["dtype_rounding"]])
    L += ["", "## 2e. Mask independence of launch config (n = 2^24 + 37)", ""]
    L += table(["BLOCK_SIZE", "num_warps", "bit-identical to default"],
               [[r["block"], r["warps"], r["identical"]] for r in res["launch_independence"]])
    L += ["", "## 3a. Stacked dropout layers (x = ones, p = 0.5)", ""]
    L += table(["layers", "seeds", "mean output", "keep rate", "expected keep"],
               [[r["layers"], r["seeds"], r["mean_output"], r["keep_rate"], r["expected_keep"]] for r in res["stacked"]])
    L += ["", "## 3b. Memory held between forward and backward (2^27 fp16 elements)", ""]
    L += table(["variant", "extra held MiB", "bytes / element", "grad == forward mask"],
               [[r["variant"], r.get("extra_held_MiB_after_forward", ""), r.get("bytes_per_element", ""),
                 r.get("grad_matches", "")] for r in res["held_memory"]])
    L += ["", "## 3c. Gradient error under non-reentrant checkpointing (recompute after the RNG advanced)", ""]
    L += table(["dropout", "preserve_rng_state", "grad rel err vs no checkpoint"],
               [[r["dropout"], r["preserve_rng_state"], r["grad_rel_err_vs_no_checkpoint"]] for r in res["checkpoint"]])
    o = res["overflow"]
    L += ["", f"## 4. int32 offsets past 2^31 (uint8, n = {o['n']}, p = 0.5)", "", "```", json.dumps(o, indent=1), "```", "",
          "## 5a. Philox throughput (no memory traffic)", ""]
    L += table(["kernel", "ms", "G calls / s", "G numbers / s"],
               [[r["kernel"], r["ms"], r["G_calls_per_s"], r["G_numbers_per_s"]] for r in res["philox_rate"]])
    L += ["", f"## 5b. Launch-config sweep (2^26 elements, GB/s = 2 * n * bytes / time)", ""]
    L += table(["dtype", "kernel", "BLOCK", "warps", "ms", "GB/s"],
               [[r["dtype"], r["kernel"], r["block"], r["warps"], r.get("ms", r.get("error")), r.get("gbps", "")]
                for r in res["config_sweep"]])
    names = [k for k in res["perf"][0] if k not in ("dtype", "n", "MiB")]
    L += ["", "## 5c. Throughput (GB/s of 2 * n * bytes; do_bench median, L2 flushed; us in parentheses)", ""]
    L += table(["dtype", "n", "MiB"] + names,
               [[r["dtype"], r["n"], r["MiB"]] + [f"{r[k]['gbps']} ({r[k]['us']})" if r[k] else "OOM" for k in names]
                for r in res["perf"]])
    (HERE / "results.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main():
    t_start = time.time()
    torch.manual_seed(SEED)
    res = {"experiment": HERE.name, "upstream": "triton-lang/triton", "upstream_commit": upstream_commit(),
           "component": "python/tutorials/04-low-memory-dropout.py (_seeded_dropout, seeded_dropout, _dropout, dropout)",
           "triton": triton.__version__, "gpu": GPU}
    for key, fn in (("keep_rates", keep_rates), ("lag", lag_correlation), ("seeds", seed_agreement),
                    ("zero_drop", zero_drop_at_p0), ("p_edge", p_edge_cases), ("nonfinite", nonfinite_propagation),
                    ("dtype_rounding", dtype_rounding), ("launch_independence", launch_independence),
                    ("stacked", stacked_layers), ("held_memory", held_memory), ("checkpoint", checkpoint_recompute),
                    ("overflow", overflow), ("philox_rate", philox_rate)):
        res[key] = fn()
        print(key, json.dumps(res[key])[:600], flush=True)
        release()
    res["config_sweep"], best = config_sweep()
    res["best_configs"] = {f"{k[0]}/{k[1]}": v for k, v in best.items()}
    res["perf"] = perf(best)
    res["peak_memory_mib"] = gpu_guard.peak_memory_mib()
    res["runtime_s"] = round(time.time() - t_start, 1)
    (HERE / "results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    write_md(res)


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--overflow":
        overflow_child()
    else:
        main()
