"""Triton tutorial layer norm (fwd + lock-based bwd reduction) on an RTX 4060 Laptop.

The tutorial (`python/tutorials/05-layer-norm.py`) runs one program per row in
both passes. The backward pass accumulates per-row dW / db into GROUP_SIZE_M
partial buffers of x.dtype, each guarded by a spin lock (`tl.atomic_cas`), and a
second kernel sums the buffers in fp32.

1. Layouts: contiguous 2D / 3D inputs, a row-strided x (`big[:, :N]`), a
   permuted x, and grads the wrapper receives from autograd: stride-0 `dy` from
   `y.sum()` / `y.mean()` and a permuted `dy`. Each case runs in a child process
   (out-of-bounds accesses can kill the CUDA context), as-is and with a fixed
   wrapper; the row-strided case checks a canary tensor placed after `y`.
2. Limits: largest supported N per dtype (64 KB per row).
3. Numerics: y / dx / dW / db error vs fp64 as the token count grows, zero-mean
   and biased dy, as-is vs fp32 partial buffers vs a lock-free rewrite vs torch.
4. Determinism: distinct dW / db bit patterns over repeated backward calls.
5. GROUP_SIZE_M sweep: backward time, determinism and error per group count.
6. Bandwidth: fwd / bwd GB/s vs N against torch, torch.compile and the rewrite,
   with registers / spills of each kernel.

The tutorial is imported unmodified from the submodule; only its top-level
unit test and benchmark (everything from `test_layer_norm(...)` on) are skipped.
"""
import ast
import gc
import hashlib
import json
import subprocess
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TRITON_DIR = ROOT / "upstream" / "triton"
TUTORIAL = TRITON_DIR / "python" / "tutorials" / "05-layer-norm.py"
sys.path.insert(0, str(ROOT))

from lab import gpu_guard  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from triton.testing import do_bench  # noqa: E402

SEED = 0
EPS = 1e-5
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
LAYOUT_CASES = ["contiguous_2d", "contiguous_3d", "row_strided_x", "permuted_x",
                "sum_loss_dy", "mean_loss_dy", "permuted_dy"]
LAYOUT_M, LAYOUT_N = 1024, 2048
NUMERICS_SHAPES = [(1024, 1024), (8192, 1024), (65536, 1024), (1024, 4096), (8192, 4096), (32768, 4096)]
REF_CHUNK = 2048
DET_SHAPES = [(4096, 1024), (32768, 1024), (4096, 4096), (32768, 4096)]
DET_REPEATS = 20
GROUP_SHAPES = [(16384, 1024), (16384, 4096)]
GROUPS = [1, 4, 16, 64, 256, 1024]
GROUP_REPEATS = 10
PERF_M = 4096
PERF_NS = [512, 1024, 2048, 3072, 4096, 8192, 16384, 32768]

torch._dynamo.config.recompile_limit = 64


def load_tutorial() -> types.ModuleType:
    tree = ast.parse(TUTORIAL.read_text(encoding="utf-8"), filename=str(TUTORIAL))
    body = []
    for node in tree.body:
        if isinstance(node, ast.Expr) and ast.unparse(node).startswith("test_layer_norm("):
            break
        body.append(node)
    tree.body = body
    mod = types.ModuleType("layer_norm_tutorial")
    mod.__file__ = str(TUTORIAL)
    sys.modules[mod.__name__] = mod
    exec(compile(tree, str(TUTORIAL), "exec"), mod.__dict__)
    return mod


# The tutorial queries the CUDA device at import time.
GPU = gpu_guard.acquire()
tut = load_tutorial()
NUM_SM = torch.cuda.get_device_properties(0).multi_processor_count


def tutorial_group_size(n):
    g = 64
    if n <= 8192: g = 96
    if n <= 4096: g = 128
    if n <= 1024: g = 256
    return g


def launch_params(n, element_size):
    block = min(65536 // element_size, triton.next_power_of_2(n))
    if n > block:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    return block, min(max(block // 256, 1), 8)


def ln_fwd(x2, w, b, eps):
    """The tutorial forward launch on a contiguous 2D x; returns y, mean, rstd and launch params."""
    m, n = x2.shape
    block, warps = launch_params(n, x2.element_size())
    y = torch.empty_like(x2)
    mean = torch.empty(m, dtype=torch.float32, device=x2.device)
    rstd = torch.empty(m, dtype=torch.float32, device=x2.device)
    tut._layer_norm_fwd_fused[(m, )](x2, y, w, b, mean, rstd, x2.stride(0), n, eps,
                                     BLOCK_SIZE=block, num_warps=warps, num_ctas=1)
    return y, mean, rstd, block, warps


def locked_bwd(x2, dy2, w, mean, rstd, block, warps, group=None, buf_dtype=None):
    """The tutorial backward with GROUP_SIZE_M and the partial-buffer dtype exposed."""
    m, n = x2.shape
    group = group or tutorial_group_size(n)
    buf_dtype = buf_dtype or x2.dtype
    locks = torch.zeros(2 * group, dtype=torch.int32, device=w.device)
    _dw = torch.zeros((group, n), dtype=buf_dtype, device=w.device)
    _db = torch.zeros((group, n), dtype=buf_dtype, device=w.device)
    dw = torch.empty((n, ), dtype=w.dtype, device=w.device)
    db = torch.empty((n, ), dtype=w.dtype, device=w.device)
    dx = torch.empty_like(dy2)
    tut._layer_norm_bwd_dx_fused[(m, )](dx, dy2, _dw, _db, x2, w, mean, rstd, locks, x2.stride(0), n,
                                        BLOCK_SIZE_N=block, GROUP_SIZE_M=group, num_warps=warps)
    tut._layer_norm_bwd_dwdb[(triton.cdiv(n, 128), )](_dw, _db, dw, db, min(group, m), n,
                                                      BLOCK_SIZE_M=32, BLOCK_SIZE_N=128, num_ctas=1)
    return dx, dw, db


@triton.jit
def _bwd_rows_fp32(DX, DY, PDW, PDB, X, W, Mean, Rstd, N, M, ROWS_PER_PROG, BLOCK_SIZE_N: tl.constexpr):
    """Each program owns a contiguous run of rows and keeps its dW / db partials in fp32 registers."""
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N
    w = tl.load(W + cols, mask=mask, other=0.).to(tl.float32)
    dw_acc = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)
    db_acc = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)
    row_start = pid * ROWS_PER_PROG
    row_end = tl.minimum(row_start + ROWS_PER_PROG, M)
    for row in range(row_start, row_end):
        off = row.to(tl.int64) * N
        x = tl.load(X + off + cols, mask=mask, other=0.).to(tl.float32)
        dy = tl.load(DY + off + cols, mask=mask, other=0.).to(tl.float32)
        mean = tl.load(Mean + row)
        rstd = tl.load(Rstd + row)
        xhat = tl.where(mask, (x - mean) * rstd, 0.)
        wdy = tl.where(mask, w * dy, 0.)
        c1 = tl.sum(xhat * wdy, axis=0) / N
        c2 = tl.sum(wdy, axis=0) / N
        dx = (wdy - (xhat * c1 + c2)) * rstd
        tl.store(DX + off + cols, dx, mask=mask)
        dw_acc += dy * xhat
        db_acc += dy
    tl.store(PDW + pid * N + cols, dw_acc, mask=mask)
    tl.store(PDB + pid * N + cols, db_acc, mask=mask)


def lockfree_bwd(x2, dy2, w, mean, rstd, block, warps):
    m, n = x2.shape
    rows_per_prog = triton.cdiv(m, min(m, 4 * NUM_SM))
    progs = triton.cdiv(m, rows_per_prog)
    pdw = torch.empty((progs, n), dtype=torch.float32, device=w.device)
    pdb = torch.empty((progs, n), dtype=torch.float32, device=w.device)
    dw = torch.empty((n, ), dtype=w.dtype, device=w.device)
    db = torch.empty((n, ), dtype=w.dtype, device=w.device)
    dx = torch.empty_like(x2)
    _bwd_rows_fp32[(progs, )](dx, dy2, pdw, pdb, x2, w, mean, rstd, n, m, rows_per_prog,
                              BLOCK_SIZE_N=block, num_warps=warps)
    tut._layer_norm_bwd_dwdb[(triton.cdiv(n, 128), )](pdw, pdb, dw, db, progs, n,
                                                      BLOCK_SIZE_M=32, BLOCK_SIZE_N=128, num_ctas=1)
    return dx, dw, db


class LockedLayerNorm(torch.autograd.Function):
    """Tutorial kernels on contiguous copies of x / dy, with GROUP_SIZE_M and buffer dtype configurable."""

    @staticmethod
    def forward(ctx, x, w, b, eps, group, buf_dtype):
        x2 = x.reshape(-1, x.shape[-1]).contiguous()
        y, mean, rstd, block, warps = ln_fwd(x2, w, b, eps)
        ctx.save_for_backward(x2, w, mean, rstd)
        ctx.cfg = (block, warps, group, buf_dtype, x.shape)
        return y.view(x.shape)

    @staticmethod
    def backward(ctx, dy):
        x2, w, mean, rstd = ctx.saved_tensors
        block, warps, group, buf_dtype, shape = ctx.cfg
        dy2 = dy.reshape(-1, shape[-1]).contiguous()
        dx, dw, db = locked_bwd(x2, dy2, w, mean, rstd, block, warps, group, buf_dtype)
        return dx.view(shape), dw, db, None, None, None


class LockFreeLayerNorm(torch.autograd.Function):
    """Contiguous x / y / dy, tutorial forward kernel, lock-free fp32-partial backward."""

    @staticmethod
    def forward(ctx, x, w, b, eps):
        x2 = x.reshape(-1, x.shape[-1]).contiguous()
        y, mean, rstd, block, warps = ln_fwd(x2, w, b, eps)
        ctx.save_for_backward(x2, w, mean, rstd)
        ctx.cfg = (block, warps, x.shape)
        return y.view(x.shape)

    @staticmethod
    def backward(ctx, dy):
        x2, w, mean, rstd = ctx.saved_tensors
        block, warps, shape = ctx.cfg
        dy2 = dy.reshape(-1, shape[-1]).contiguous()
        dx, dw, db = lockfree_bwd(x2, dy2, w, mean, rstd, block, warps)
        return dx.view(shape), dw, db, None


def tutorial_ln(x, w, b):
    return tut.layer_norm(x, (x.shape[-1], ), w, b, EPS)


def torch_ln(x, w, b):
    return F.layer_norm(x, (x.shape[-1], ), w, b, EPS)


def fp32_buf_ln(x, w, b):
    return LockedLayerNorm.apply(x, w, b, EPS, None, torch.float32)


def lockfree_ln(x, w, b):
    return LockFreeLayerNorm.apply(x, w, b, EPS)


def grouped_ln(group):
    return lambda x, w, b: LockedLayerNorm.apply(x, w, b, EPS, group, None)


torch_compiled = torch.compile(torch_ln, dynamic=False)

VARIANTS = {
    "tutorial": "tutorial layer_norm() as-is: GROUP_SIZE_M partial dW/db buffers in x.dtype behind spin locks",
    "fp32_buffers": "same kernels and GROUP_SIZE_M, partial buffers allocated in fp32 (contiguous x / dy)",
    "lockfree": "tutorial forward kernel; backward with 4 * num_SMs programs over contiguous row runs, "
                "fp32 register partials, no locks (contiguous x / y / dy)",
    "torch": "torch.nn.functional.layer_norm (native CUDA kernels)",
    "torch_compile": "torch.compile(F.layer_norm, dynamic=False) via triton-windows (Inductor fwd + bwd)",
}
LAYOUT_FNS = {"tutorial": tutorial_ln, "fixed": lockfree_ln}
NUMERIC_FNS = {"tutorial": tutorial_ln, "fp32_buffers": fp32_buf_ln, "lockfree": lockfree_ln, "torch": torch_ln}


def cuda_gen():
    return torch.Generator(device="cuda").manual_seed(SEED)


def params(n, dtype, g):
    w = torch.rand(n, device="cuda", generator=g).to(dtype).requires_grad_()
    b = torch.rand(n, device="cuda", generator=g).to(dtype).requires_grad_()
    return w, b


def rel(err_sq, ref_sq):
    return float(f"{(err_sq / ref_sq) ** 0.5:.3g}") if ref_sq > 0 else None


# ---------------------------------------------------------------- 1. layouts (child processes)

def layout_child(case, variant):
    fn = LAYOUT_FNS[variant]
    g = cuda_gen()
    m, n, dtype = LAYOUT_M, LAYOUT_N, torch.float16
    w, b = params(n, dtype, g)
    out = {"case": case, "variant": variant}
    canary = None
    if case == "contiguous_2d":
        x = (-2.3 + 0.5 * torch.randn(m, n, device="cuda", generator=g)).to(dtype)
    elif case == "row_strided_x":
        x = (-2.3 + 0.5 * torch.randn(m, 2 * n, device="cuda", generator=g)).to(dtype)[:, :n]
    elif case == "permuted_x":
        x = (-2.3 + 0.5 * torch.randn(m // 8, 8, n, device="cuda", generator=g)).to(dtype).transpose(0, 1)
    else:
        x = (-2.3 + 0.5 * torch.randn(8, m // 8, n, device="cuda", generator=g)).to(dtype)
    x.requires_grad_()
    gy = torch.randn(x.shape, device="cuda", generator=g).to(dtype)
    gy_t = torch.randn(x.transpose(0, 1).shape, device="cuda", generator=g).to(dtype)

    def loss_of(y):
        if case == "sum_loss_dy":
            return y.sum()
        if case == "mean_loss_dy":
            return y.mean() * y.numel()
        if case == "permuted_dy":
            return (y.transpose(0, 1) * gy_t).sum()
        return (y * gy).sum()

    xd = x.detach().double().requires_grad_()
    wd, bd = w.detach().double().requires_grad_(), b.detach().double().requires_grad_()
    y_ref = torch_ln(xd, wd, bd)
    loss_of(y_ref).backward()
    torch.cuda.synchronize()
    if case == "row_strided_x":
        torch.cuda.empty_cache()
        nbytes = m * n * x.element_size()
        a = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
        canary = torch.full((nbytes // 2, ), 7.0, dtype=dtype, device="cuda")
        a_ptr = a.data_ptr()
        out["canary_adjacent_to_freed_block"] = canary.data_ptr() == a_ptr + nbytes
        del a
    try:
        y = fn(x, w, b)
        out["y_strides"] = list(y.stride())
        if canary is not None:
            out["y_reuses_freed_block"] = y.data_ptr() == a_ptr
        loss_of(y).backward()
        torch.cuda.synchronize()
        out["error"] = None
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"
        print(json.dumps(out))
        return
    yd = y.detach().double()
    row_err = (yd - y_ref.detach()).abs().reshape(-1, n).amax(-1)
    out["y_rows_wrong"] = int((row_err > 0.05).sum().item())
    out["rows"] = row_err.numel()
    out["y_rel_err"] = rel(((yd - y_ref) ** 2).sum().item(), (y_ref ** 2).sum().item())
    for name, got, ref in (("dx", x.grad, xd.grad), ("dw", w.grad, wd.grad), ("db", b.grad, bd.grad)):
        got = got.double()
        out[name + "_rel_err"] = rel(((got - ref) ** 2).sum().item(), (ref ** 2).sum().item())
        out[name + "_finite"] = bool(torch.isfinite(got).all().item())
    if canary is not None:
        out["canary_elements_overwritten"] = int((canary != 7.0).sum().item())
        out["canary_elements"] = canary.numel()
    print(json.dumps(out))


def layouts():
    rows = []
    for case in LAYOUT_CASES:
        for variant in LAYOUT_FNS:
            p = subprocess.run([sys.executable, __file__, "--layout", case, variant],
                               capture_output=True, text=True, timeout=600)
            line = next((ln for ln in reversed(p.stdout.splitlines()) if ln.startswith("{")), None)
            if line:
                row = json.loads(line)
            else:
                tail = [ln for ln in (p.stderr or "").splitlines() if ln.strip()]
                row = {"case": case, "variant": variant,
                       "error": f"child exited {p.returncode}: {(tail[-1] if tail else '')[:160]}"}
            row["exit_code"] = p.returncode
            rows.append(row)
            print("layout", row, flush=True)
    return rows


# ---------------------------------------------------------------- 2. limits

def limits():
    rows = []
    g = cuda_gen()
    for name, n in (("fp32", 16384), ("fp32", 16385), ("fp16", 32768), ("fp16", 32769),
                    ("bf16", 32768), ("bf16", 32769)):
        dtype = DTYPES[name]
        w, b = params(n, dtype, g)
        x = (-2.3 + 0.5 * torch.randn(64, n, device="cuda", generator=g)).to(dtype).requires_grad_()
        dy = (0.1 * torch.randn(64, n, device="cuda", generator=g)).to(dtype)
        row = {"dtype": name, "N": n, "row_bytes": n * x.element_size()}
        try:
            y = tutorial_ln(x, w, b)
            gx = torch.autograd.grad(y, x, dy)[0]
            yr = torch_ln(x.float(), w.float(), b.float())
            gr = torch.autograd.grad(yr, x, dy.float())[0]
            row["error"] = None
            row["y_max_abs_err_vs_torch_fp32"] = float(f"{(y.float() - yr).abs().max().item():.3g}")
            row["dx_max_abs_err_vs_torch_fp32"] = float(f"{(gx.float() - gr).abs().max().item():.3g}")
        except RuntimeError as e:
            row["error"] = str(e).splitlines()[0][:160]
        rows.append(row)
        print("limit", row, flush=True)
    return rows


# ---------------------------------------------------------------- 3. numerics

def reference_errors(x, w, b, dy, outs):
    """fp64 reference computed in row chunks; returns rel L2 errors for every output set plus the dtype floor."""
    m, n = x.shape
    wd, bd = w.detach().double(), b.detach().double()
    keys = list(outs) + ["floor"]
    acc = {k: {f: [0.0, 0.0] for f in ("y", "dx")} for k in keys}
    dw_ref = torch.zeros(n, dtype=torch.float64, device="cuda")
    db_ref = torch.zeros(n, dtype=torch.float64, device="cuda")
    for r0 in range(0, m, REF_CHUNK):
        xc = x.detach()[r0:r0 + REF_CHUNK].double()
        dyc = dy[r0:r0 + REF_CHUNK].double()
        mean = xc.mean(-1, keepdim=True)
        rstd = ((xc - mean).pow(2).mean(-1, keepdim=True) + EPS).rsqrt()
        xhat = (xc - mean) * rstd
        y_ref = xhat * wd + bd
        wdy = wd * dyc
        dx_ref = (wdy - xhat * (xhat * wdy).mean(-1, keepdim=True) - wdy.mean(-1, keepdim=True)) * rstd
        dw_ref += (dyc * xhat).sum(0)
        db_ref += dyc.sum(0)
        for k in keys:
            for f, ref in (("y", y_ref), ("dx", dx_ref)):
                got = ref.to(x.dtype).double() if k == "floor" else outs[k][f][r0:r0 + REF_CHUNK].double()
                acc[k][f][0] += ((got - ref) ** 2).sum().item()
                acc[k][f][1] += (ref ** 2).sum().item()
        del xc, dyc, mean, rstd, xhat, y_ref, wdy, dx_ref
    res = {}
    for k in keys:
        r = {f: rel(*acc[k][f]) for f in ("y", "dx")}
        for f, ref in (("dw", dw_ref), ("db", db_ref)):
            got = ref.to(x.dtype).double() if k == "floor" else outs[k][f].double()
            r[f] = rel(((got - ref) ** 2).sum().item(), (ref ** 2).sum().item())
        res[k] = r
    return res


def run_variant(fn, x, w, b, dy):
    y = fn(x, w, b)
    gx, gw, gb = torch.autograd.grad(y, (x, w, b), dy)
    return {"y": y.detach(), "dx": gx, "dw": gw, "db": gb}


def variant_errors(fns, x, w, b, dy):
    """One variant's outputs alive at a time; the fp64 reference is recomputed per variant."""
    errs = {}
    for name, fn in fns.items():
        outs = run_variant(fn, x, w, b, dy)
        e = reference_errors(x, w, b, dy, {name: outs})
        del outs
        torch.cuda.empty_cache()
        errs.update(e)
    return errs


def numerics():
    rows = []
    for dname, dtype in DTYPES.items():
        for m, n in NUMERICS_SHAPES:
            for dy_mode in ("zero_mean", "biased"):
                g = cuda_gen()
                w, b = params(n, dtype, g)
                x = (-2.3 + 0.5 * torch.randn(m, n, device="cuda", generator=g)).to(dtype).requires_grad_()
                dy = 0.1 * torch.randn(m, n, device="cuda", generator=g)
                if dy_mode == "biased":
                    dy += 0.05
                dy = dy.to(dtype)
                errs = variant_errors(NUMERIC_FNS, x, w, b, dy)
                group = tutorial_group_size(n)
                for name, e in errs.items():
                    rows.append({"dtype": dname, "N": n, "M": m, "dy": dy_mode, "variant": name,
                                 "group_size_m": group, "rows_per_buffer": triton.cdiv(m, group),
                                 **{f + "_rel_err": e[f] for f in ("y", "dx", "dw", "db")}})
                print(f"numerics {dname} N={n} M={m} {dy_mode}: " +
                      ", ".join(f"{k} dw={e['dw']} db={e['db']}" for k, e in errs.items()), flush=True)
                del x, dy, w, b
                gc.collect()
                torch.cuda.empty_cache()
    return rows


# ---------------------------------------------------------------- 4. determinism

def digest(*ts):
    h = hashlib.sha1()
    for t in ts:
        h.update(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()


def repeat_bwd(fn, x, w, b, dy, repeats):
    y = fn(x, w, b)
    seen, dws, dbs = set(), [], []
    for _ in range(repeats):
        gx, gw, gb = torch.autograd.grad(y, (x, w, b), dy, retain_graph=True)
        seen.add(digest(gw, gb))
        dws.append(gw.float())
        dbs.append(gb.float())
    dw, db = torch.stack(dws), torch.stack(dbs)
    spread = lambda t: float(f"{((t.amax(0) - t.amin(0)).abs().max() / t.abs().max()).item():.3g}")  # noqa: E731
    return len(seen), spread(dw), spread(db)


def determinism():
    rows = []
    fns = {"tutorial": tutorial_ln, "fp32_buffers": fp32_buf_ln, "lockfree": lockfree_ln, "torch": torch_ln}
    for dname, dtype in DTYPES.items():
        for m, n in DET_SHAPES:
            g = cuda_gen()
            w, b = params(n, dtype, g)
            x = (-2.3 + 0.5 * torch.randn(m, n, device="cuda", generator=g)).to(dtype).requires_grad_()
            dy = (0.1 * torch.randn(m, n, device="cuda", generator=g)).to(dtype)
            for name, fn in fns.items():
                distinct, sw, sb = repeat_bwd(fn, x, w, b, dy, DET_REPEATS)
                rows.append({"dtype": dname, "M": m, "N": n, "variant": name, "repeats": DET_REPEATS,
                             "distinct_dw_db": distinct, "dw_spread_rel": sw, "db_spread_rel": sb})
                print("determinism", rows[-1], flush=True)
            del x, dy
            torch.cuda.empty_cache()
    return rows


# ---------------------------------------------------------------- 5. GROUP_SIZE_M sweep

def bwd_ms(fn, x, w, b, dy):
    y = fn(x, w, b)
    return do_bench(lambda: torch.autograd.grad(y, (x, w, b), dy, retain_graph=True), return_mode="median")


def group_sweep():
    rows = []
    for dname in ("fp16", "bf16"):
        dtype = DTYPES[dname]
        for m, n in GROUP_SHAPES:
            g = cuda_gen()
            w, b = params(n, dtype, g)
            x = (-2.3 + 0.5 * torch.randn(m, n, device="cuda", generator=g)).to(dtype).requires_grad_()
            dy = (0.1 * torch.randn(m, n, device="cuda", generator=g) + 0.05).to(dtype)
            cands = {f"G={grp}": grouped_ln(grp) for grp in GROUPS}
            cands["lockfree"] = lockfree_ln
            cands["torch"] = torch_ln
            errs = variant_errors(cands, x, w, b, dy)
            for k, fn in cands.items():
                ms = bwd_ms(fn, x, w, b, dy)
                distinct, _, _ = repeat_bwd(fn, x, w, b, dy, GROUP_REPEATS)
                rows.append({"dtype": dname, "M": m, "N": n, "variant": k,
                             "tutorial_default": k == f"G={tutorial_group_size(n)}",
                             "bwd_ms": round(ms, 4), "bwd_gbps": round(3 * m * n * 2 * 1e-9 / (ms * 1e-3), 1),
                             "distinct_dw_db": distinct, "repeats": GROUP_REPEATS,
                             "dw_rel_err": errs[k]["dw"], "db_rel_err": errs[k]["db"]})
                print("group", rows[-1], flush=True)
            del x, dy
            torch.cuda.empty_cache()
    return rows


# ---------------------------------------------------------------- 6. bandwidth

def kernel_snapshot():
    out = {}
    for kname, k in (("fwd", tut._layer_norm_fwd_fused), ("bwd_locked", tut._layer_norm_bwd_dx_fused),
                     ("bwd_lockfree", _bwd_rows_fp32)):
        for cache in k.device_caches.values():
            for key, ck in cache[0].items():
                out[(kname, str(key))] = ck
    return out


def safe_bench(fn):
    try:
        return do_bench(fn, return_mode="median"), "ok"
    except torch.OutOfMemoryError:
        gc.collect()
        torch.cuda.empty_cache()
        return None, "OOM"
    except Exception as e:  # noqa: BLE001
        gc.collect()
        return None, f"{type(e).__name__}"


def perf():
    rows, kernels = [], []
    fns = {"tutorial": tutorial_ln, "lockfree": lockfree_ln, "torch": torch_ln, "torch_compile": torch_compiled}
    for dname in ("fp16", "bf16"):
        dtype = DTYPES[dname]
        for n in PERF_NS:
            g = cuda_gen()
            w, b = params(n, dtype, g)
            x = (-2.3 + 0.5 * torch.randn(PERF_M, n, device="cuda", generator=g)).to(dtype).requires_grad_()
            dy = (0.1 * torch.randn(PERF_M, n, device="cuda", generator=g)).to(dtype)
            numel, es = x.numel(), x.element_size()
            row = {"dtype": dname, "M": PERF_M, "N": n, "MiB_per_tensor": round(numel * es / 2**20, 1)}
            out = torch.empty_like(x)
            ms = do_bench(lambda: out.copy_(x), return_mode="median")
            row["copy_gbps"] = round(2 * numel * es * 1e-9 / (ms * 1e-3), 1)
            del out
            before = set(kernel_snapshot())
            for name, fn in fns.items():
                if name == "torch_compile":
                    for _ in range(2):
                        torch.autograd.grad(fn(x, w, b), (x, w, b), dy)
                ms, status = safe_bench(lambda: fn(x, w, b))
                row[name + "_fwd_gbps"] = round(2 * numel * es * 1e-9 / (ms * 1e-3), 1) if ms else status
                try:
                    y = fn(x, w, b)
                    ms, status = safe_bench(lambda: torch.autograd.grad(y, (x, w, b), dy, retain_graph=True))
                    del y
                except Exception as e:  # noqa: BLE001
                    ms, status = None, type(e).__name__
                row[name + "_bwd_gbps"] = round(3 * numel * es * 1e-9 / (ms * 1e-3), 1) if ms else status
                row[name + "_bwd_ms"] = round(ms, 4) if ms else None
            for name in ("tutorial", "lockfree", "torch_compile"):
                t, base = row[name + "_bwd_ms"], row["torch_bwd_ms"]
                row[name + "_bwd_speedup_vs_torch"] = round(base / t, 2) if t and base else None
            rows.append(row)
            print("perf", row, flush=True)
            for (kname, _), ck in kernel_snapshot().items():
                if (kname, _) in before:
                    continue
                kernels.append({"dtype": dname, "N": n, "kernel": kname, "num_warps": ck.metadata.num_warps,
                                "n_regs": ck.n_regs, "n_spills": ck.n_spills, "shared_bytes": ck.metadata.shared})
            del x, dy, w, b
            gc.collect()
            torch.cuda.empty_cache()
    return rows, kernels


def upstream_commit() -> str:
    return subprocess.run(["git", "-C", str(TRITON_DIR), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def write_md(result):
    gpu = result["gpu"]
    lines = [
        f"triton {result['upstream_commit'][:10]} tutorial, triton-windows {triton.__version__}, "
        f"torch {torch.__version__}, {gpu.get('name', 'cuda')} ({NUM_SM} SMs, memory cap {gpu.get('memory_fraction_cap')}), "
        f"runtime {result['runtime_s']} s",
        "",
        f"## 1. Layouts ({LAYOUT_M} x {LAYOUT_N} fp16 rows; rel L2 error vs fp64 autograd on the same logical tensors; "
        "`fixed` = contiguous x / y / dy + lock-free backward)",
        "",
        "| case | variant | error | y strides | y rows wrong | y rel err | dx rel err | dw rel err | db rel err | canary |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in result["layouts"]:
        canary = ""
        if "canary_elements_overwritten" in r:
            canary = (f"{r['canary_elements_overwritten']} / {r['canary_elements']} overwritten "
                      f"(adjacent {r.get('canary_adjacent_to_freed_block')}, y reuses block {r.get('y_reuses_freed_block')})")
        lines.append(f"| {r['case']} | {r['variant']} | {r.get('error') or ''} | {r.get('y_strides', '')} | "
                     f"{'' if 'y_rows_wrong' not in r else str(r['y_rows_wrong']) + ' / ' + str(r['rows'])} | "
                     f"{r.get('y_rel_err', '')} | {r.get('dx_rel_err', '')} | {r.get('dw_rel_err', '')} | "
                     f"{r.get('db_rel_err', '')} | {canary} |")
    lines += ["", "## 2. Row-size limit (64 rows, tutorial as-is; error vs torch fp32)", "",
              "| dtype | N | row bytes | error | y max abs err | dx max abs err |", "|---|---|---|---|---|---|"]
    lines += [f"| {r['dtype']} | {r['N']} | {r['row_bytes']} | {r['error'] or ''} | "
              f"{r.get('y_max_abs_err_vs_torch_fp32', '')} | {r.get('dx_max_abs_err_vs_torch_fp32', '')} |"
              for r in result["limits"]]
    lines += ["", "## 3. Error vs fp64 (rel L2; x = -2.3 + 0.5 N(0,1), w, b ~ U(0,1), dy = 0.1 N(0,1) "
              "(+ 0.05 when biased); `floor` = fp64 reference rounded to the dtype)", "",
              "| dtype | N | M | dy | rows / buffer | variant | y | dx | dW | db |", "|---|---|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r['dtype']} | {r['N']} | {r['M']} | {r['dy']} | {r['rows_per_buffer']} | {r['variant']} | "
              f"{r['y_rel_err']} | {r['dx_rel_err']} | {r['dw_rel_err']} | {r['db_rel_err']} |" for r in result["numerics"]]
    lines += ["", f"## 4. Determinism ({DET_REPEATS} backward calls on one graph; spread = max(max - min) / max|g|)", "",
              "| dtype | M | N | variant | distinct (dW, db) | dW spread | db spread |", "|---|---|---|---|---|---|---|"]
    lines += [f"| {r['dtype']} | {r['M']} | {r['N']} | {r['variant']} | {r['distinct_dw_db']} | {r['dw_spread_rel']} | "
              f"{r['db_spread_rel']} |" for r in result["determinism"]]
    lines += ["", f"## 5. GROUP_SIZE_M sweep (biased dy; * = tutorial default; {GROUP_REPEATS} repeats for determinism)", "",
              "| dtype | M | N | variant | bwd ms | bwd GB/s | distinct | dW rel err | db rel err |",
              "|---|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r['dtype']} | {r['M']} | {r['N']} | {r['variant']}{' *' if r['tutorial_default'] else ''} | {r['bwd_ms']} | "
              f"{r['bwd_gbps']} | {r['distinct_dw_db']} | {r['dw_rel_err']} | {r['db_rel_err']} |" for r in result["group_sweep"]]
    lines += ["", f"## 6. Bandwidth (M = {PERF_M}; fwd 2*M*N*bytes, bwd 3*M*N*bytes / do_bench median; "
              "bwd = autograd.grad for x, w, b)", "",
              "| dtype | N | MiB | copy | fwd tutorial | fwd torch | fwd compile | bwd tutorial | bwd lockfree | bwd torch | "
              "bwd compile | tutorial / torch | lockfree / torch | compile / torch |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r['dtype']} | {r['N']} | {r['MiB_per_tensor']} | {r['copy_gbps']} | {r['tutorial_fwd_gbps']} | "
              f"{r['torch_fwd_gbps']} | {r['torch_compile_fwd_gbps']} | {r['tutorial_bwd_gbps']} | {r['lockfree_bwd_gbps']} | "
              f"{r['torch_bwd_gbps']} | {r['torch_compile_bwd_gbps']} | {r['tutorial_bwd_speedup_vs_torch']} | "
              f"{r['lockfree_bwd_speedup_vs_torch']} | {r['torch_compile_bwd_speedup_vs_torch']} |" for r in result["perf"]]
    lines += ["", "### Kernel resources", "", "| dtype | N | kernel | warps | regs/thread | spills (bytes) | shared B |",
              "|---|---|---|---|---|---|---|"]
    lines += [f"| {k['dtype']} | {k['N']} | {k['kernel']} | {k['num_warps']} | {k['n_regs']} | {k['n_spills']} | "
              f"{k['shared_bytes']} |" for k in result["kernels"]]
    (HERE / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    t_start = time.time()
    torch.manual_seed(SEED)
    layout_rows = layouts()
    limit_rows = limits()
    numeric_rows = numerics()
    det_rows = determinism()
    group_rows = group_sweep()
    perf_rows, kernel_rows = perf()
    result = {
        "experiment": HERE.name,
        "upstream": "triton-lang/triton",
        "upstream_commit": upstream_commit(),
        "component": "python/tutorials/05-layer-norm.py (_layer_norm_fwd_fused, _layer_norm_bwd_dx_fused, "
                     "_layer_norm_bwd_dwdb, LayerNorm autograd wrapper)",
        "variants": VARIANTS,
        "eps": EPS,
        "triton": triton.__version__,
        "num_sms": NUM_SM,
        "gpu": GPU,
        "layouts": layout_rows,
        "limits": limit_rows,
        "numerics": numeric_rows,
        "determinism": det_rows,
        "group_sweep": group_rows,
        "perf": perf_rows,
        "kernels": kernel_rows,
        "peak_memory_mib": gpu_guard.peak_memory_mib(),
        "runtime_s": round(time.time() - t_start, 1),
    }
    (HERE / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_md(result)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--layout":
        layout_child(sys.argv[2], sys.argv[3])
    else:
        main()
