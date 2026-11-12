"""bitsandbytes 8-bit blockwise Adam / AdamW optimizer states vs 32-bit Adam.

bitsandbytes keeps Adam's m and v as uint8 codes into a "dynamic" (exponent +
fraction) code book, scaled per 256-element block by an fp32 absmax. Each step
(csrc/kernels.cu kOptimizerStatic8bit2StateBlockwise) dequantizes, updates the
parameter with fp32 m / v, then re-quantizes them to the nearest code.

1. Code books: range, smallest non-zero code, sign asymmetry.
2. A pure-torch re-implementation of the 8-bit blockwise step, checked code by
   code against the CUDA kernel (teacher-forced and free-running); bnb Adam's
   weight_decay semantics vs torch Adam (L2) / AdamW (decoupled).
3. Optimizer-state memory on GPT-2-124M shapes (nanoGPT): torch-allocator view
   vs real bytes (paged states live in CUDA managed memory), checkpoint size,
   leftover managed memory after the optimizer is deleted.
4. step() latency vs parameter count (fp32 / bf16 params) and on the 148 GPT-2
   tensors, incl. the per-parameter torch.cuda.synchronize() in bnb's step().
5. Update error vs the in-block gradient dynamic range on synthetic gradient
   streams, vs ablations: tensor-wise dynamic, blockwise linear, tensor-wise linear.
6. Byte-level nanoGPT (6 x 384) training on torch's own .py sources: val loss of
   torch AdamW vs bnb AdamW 32-bit / 8-bit vs blockwise linear int8 states.
"""
import gc
import hashlib
import importlib.util
import io
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NANOGPT_DIR = ROOT / "upstream" / "nanoGPT"
sys.path.insert(0, str(ROOT))

from lab import gpu_guard  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import bitsandbytes as bnb  # noqa: E402
import bitsandbytes.functional as bnbF  # noqa: E402
import bitsandbytes.optim.optimizer as bnb_optimizer  # noqa: E402

SEED = 0
BLOCK = 256
LR = 1e-3
EPS = 1e-8
BETAS_SYN = (0.9, 0.999)

MATCH_N = (1 << 20) + 100
MATCH_STEPS = 50
DR_N = 1 << 20
DR_STEPS = 300
DR_LAST = 50
DR_RANGES = [0, 2, 3, 4, 5, 6]
LAT_SIZES = [1 << 18, 1 << 20, 1 << 22, 1 << 24, 1 << 26]
LAT_ITERS = 20

TRAIN_STEPS = 1200
TRAIN_BATCH = 32
TRAIN_BLOCK = 256
TRAIN_WARMUP = 100
TRAIN_EVAL_EVERY = 200
TRAIN_EVAL_BATCHES = 16
TRAIN_BETAS = (0.9, 0.95)
TRAIN_WD = 0.1
CORPUS_BYTES = 8_000_000

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

QMAP_S = bnbF.create_dynamic_map(signed=True)
QMAP_U = bnbF.create_dynamic_map(signed=False)


def load_nanogpt():
    spec = importlib.util.spec_from_file_location("nanogpt_model", NANOGPT_DIR / "model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def git_head(path: Path) -> str:
    return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def free_cuda():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def smi_used_mib() -> int:
    exe = shutil.which("nvidia-smi")
    out = subprocess.run([exe, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=30).stdout
    return int(out.strip().splitlines()[0])


def r4(x: float) -> float:
    return float(f"{x:.4g}")


# ---------------------------------------------------------------- reference step

def blocks_of(x: torch.Tensor, block: int | None):
    n = x.numel()
    if block is None:
        return x.view(1, n), n
    pad = (-n) % block
    return F.pad(x, (0, pad)).view(-1, block), n


def quantize(x: torch.Tensor, kind: str, signed: bool, block: int | None):
    """Round-trip x through a blockwise quantizer; returns (dequantized, codes, absmax)."""
    xb, n = blocks_of(x, block)
    absmax = xb.abs().amax(1, keepdim=True)
    scaled = xb / torch.where(absmax > 0, absmax, torch.ones_like(absmax))
    if kind == "dynamic":
        code = (QMAP_S if signed else QMAP_U).to(x.device)
        idx = torch.searchsorted(code, scaled.contiguous()).clamp_(1, 255)
        lo, hi = code[idx - 1], code[idx]
        c = torch.where(scaled - lo <= hi - scaled, idx - 1, idx)
        if signed:
            flip = torch.signbit(code[c]) != torch.signbit(scaled)
            c = c + torch.where(flip, torch.where(scaled > 0, 1, -1), 0)
        dq = code[c] * absmax
    elif kind == "linear":
        levels = 127 if signed else 255
        c = torch.round(scaled * levels)
        dq = c / levels * absmax
    else:
        raise ValueError(kind)
    return dq.reshape(-1)[:n], c.reshape(-1)[:n], absmax.reshape(-1)


def ref_adam_update(p, g, st, step, lr, betas, eps, wd, quant, block=BLOCK):
    """bnb kOptimizerStatic8bit2StateBlockwise in torch (fp32); quant=None keeps fp32 states."""
    b1, b2 = betas
    s2 = st["s2"] * b2 + (1 - b2) * g * g
    s1 = st["s1"] * b1 + (1 - b1) * g
    c1 = 1 - b1 ** step
    c2 = math.sqrt(1 - b2 ** step)
    p.add_((-lr * c2 / c1) * s1 / (s2.sqrt() + c2 * eps))
    if wd > 0:
        p.mul_(1 - lr * wd)
    if quant is None:
        st["s1"], st["s2"] = s1, s2
    elif quant[0] == "dynamic_sqrtv":
        st["s1"], st["c1"], st["a1"] = quantize(s1, "dynamic", True, quant[1])
        root, st["c2"], st["a2"] = quantize(s2.sqrt(), "dynamic", False, quant[1])
        st["s2"] = root * root
    else:
        kind, qblock = quant
        st["s1"], st["c1"], st["a1"] = quantize(s1, kind, True, qblock)
        st["s2"], st["c2"], st["a2"] = quantize(s2, kind, False, qblock)


class RefAdamW(torch.optim.Optimizer):
    """AdamW with simulated quantized states (bnb update order, min_8bit_size like bnb)."""

    def __init__(self, params, lr, betas, eps, weight_decay, quant, min_8bit_size=4096):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
        self.quant = quant
        self.min_8bit_size = min_8bit_size

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st["step"] = 0
                    st["s1"] = torch.zeros(p.numel(), device=p.device)
                    st["s2"] = torch.zeros(p.numel(), device=p.device)
                st["step"] += 1
                quant = self.quant if p.numel() >= self.min_8bit_size else None
                ref_adam_update(p.data.view(-1), p.grad.view(-1).float(), st, st["step"], group["lr"],
                                group["betas"], group["eps"], group["weight_decay"], quant)


def bnb_dequant(state: dict, idx: int, n: int) -> torch.Tensor:
    codes, absmax, qmap = state[f"state{idx}"], state[f"absmax{idx}"], state[f"qmap{idx}"]
    return qmap[codes.long()] * absmax.repeat_interleave(BLOCK)[:n]


# ---------------------------------------------------------------- 1. code books

def codebooks():
    rows = []
    for name, code in (("signed (m)", QMAP_S), ("unsigned (v)", QMAP_U)):
        nz = code[code != 0].abs()
        rows.append({"map": name, "min": r4(code.min().item()), "max": r4(code.max().item()),
                     "smallest_nonzero_abs": r4(nz.min().item()),
                     "codes_below_1e-3": int((code.abs() < 1e-3).sum().item()) - 1,
                     "codes_above_0.1": int((code.abs() > 0.1).sum().item())})
    return rows


# ---------------------------------------------------------------- 2. reference match

def synthetic_grad_stream(n: int, r: float, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    scale = 10 ** (-r * torch.rand(n, generator=gen, device="cuda"))
    mu = 0.5 * torch.randn(n, generator=gen, device="cuda")
    return scale, mu, gen


def reference_match():
    rows = []
    for name, cls, wd in (("Adam8bit", bnb.optim.Adam8bit, 0.0), ("AdamW8bit", bnb.optim.AdamW8bit, 1e-2)):
        scale, mu, gen = synthetic_grad_stream(MATCH_N, 3.0, SEED)
        p = torch.nn.Parameter(0.1 * torch.randn(MATCH_N, generator=gen, device="cuda"))
        p0 = p.detach().clone()
        opt = cls([p], lr=LR, betas=BETAS_SYN, eps=EPS, weight_decay=wd)
        p_free = p0.clone()
        st_free = {"s1": torch.zeros_like(p0), "s2": torch.zeros_like(p0)}
        mism1 = mism2 = max_cdiff = 0
        absmax_rel = p_rel = 0.0
        for t in range(1, MATCH_STEPS + 1):
            g = scale * (mu + torch.randn(MATCH_N, generator=gen, device="cuda"))
            if t == 1:
                st = {"s1": torch.zeros_like(p0), "s2": torch.zeros_like(p0)}
            else:
                s = opt.state[p]
                st = {"s1": bnb_dequant(s, 1, MATCH_N), "s2": bnb_dequant(s, 2, MATCH_N)}
            p_in = p.detach().clone()
            p.grad = g.clone()
            opt.step()
            s = opt.state[p]
            p_ref = p_in.clone()
            ref_adam_update(p_ref, g, st, t, LR, BETAS_SYN, EPS, wd, ("dynamic", BLOCK))
            c1, c2 = s["state1"].long(), s["state2"].long()
            mism1 += int((c1 != st["c1"]).sum().item())
            mism2 += int((c2 != st["c2"]).sum().item())
            max_cdiff = max(max_cdiff, int((c1 - st["c1"]).abs().max().item()), int((c2 - st["c2"]).abs().max().item()))
            absmax_rel = max(absmax_rel, ((s["absmax1"] - st["a1"]).abs() / st["a1"]).max().item(),
                             ((s["absmax2"] - st["a2"]).abs() / st["a2"]).max().item())
            upd = (p.detach() - p_in)
            p_rel = max(p_rel, ((p_ref - p.detach()).abs().max() / upd.abs().max()).item())
            ref_adam_update(p_free, g, st_free, t, LR, BETAS_SYN, EPS, wd, ("dynamic", BLOCK))
        total = MATCH_N * MATCH_STEPS
        drift = ((p_free - p.detach()).norm() / (p.detach() - p0).norm()).item()
        rows.append({"optimizer": name, "weight_decay": wd, "n": MATCH_N, "steps": MATCH_STEPS,
                     "m_code_mismatch_frac": r4(mism1 / total), "v_code_mismatch_frac": r4(mism2 / total),
                     "max_code_diff": max_cdiff, "absmax_max_rel_diff": r4(absmax_rel),
                     "param_max_diff_over_max_update": r4(p_rel), "free_running_drift_rel": r4(drift)})
        del opt, p
        free_cuda()
    return rows


def weight_decay_semantics():
    """Which torch optimizer does bnb Adam(weight_decay=0.1) match after 20 steps?"""
    n, wd, steps = 1 << 16, 0.1, 20
    gen = torch.Generator(device="cuda").manual_seed(SEED)
    p0 = torch.randn(n, generator=gen, device="cuda")
    grads = [torch.randn(n, generator=gen, device="cuda") for _ in range(steps)]
    lr = 1e-2

    def run(make):
        p = torch.nn.Parameter(p0.clone())
        opt = make([p])
        for g in grads:
            p.grad = g.clone()
            opt.step()
        return p.detach()

    outs = {
        "torch.optim.Adam (L2)": run(lambda ps: torch.optim.Adam(ps, lr=lr, weight_decay=wd)),
        "torch.optim.AdamW (decoupled)": run(lambda ps: torch.optim.AdamW(ps, lr=lr, weight_decay=wd)),
        "bnb.optim.Adam 32-bit": run(lambda ps: bnb.optim.Adam(ps, lr=lr, weight_decay=wd)),
        "bnb.optim.AdamW 32-bit": run(lambda ps: bnb.optim.AdamW(ps, lr=lr, weight_decay=wd)),
        "bnb.optim.Adam8bit": run(lambda ps: bnb.optim.Adam8bit(ps, lr=lr, weight_decay=wd)),
    }
    moved = {k: (v - p0).norm().item() for k, v in outs.items()}
    rows = []
    for k, v in outs.items():
        if not k.startswith("bnb"):
            continue
        rows.append({"optimizer": k,
                     "rel_diff_vs_torch_Adam_L2": r4((v - outs["torch.optim.Adam (L2)"]).norm().item()
                                                     / moved["torch.optim.Adam (L2)"]),
                     "rel_diff_vs_torch_AdamW": r4((v - outs["torch.optim.AdamW (decoupled)"]).norm().item()
                                                   / moved["torch.optim.AdamW (decoupled)"])})
    return {"lr": lr, "weight_decay": wd, "steps": steps, "rows": rows}


# ---------------------------------------------------------------- 3. memory

OPTIMIZERS = {
    "torch AdamW fused": lambda ps: torch.optim.AdamW(ps, lr=LR, fused=True),
    "bnb AdamW 32-bit": lambda ps: bnb.optim.AdamW(ps, lr=LR),
    "bnb AdamW8bit": lambda ps: bnb.optim.AdamW8bit(ps, lr=LR),
    "bnb PagedAdamW8bit": lambda ps: bnb.optim.PagedAdamW8bit(ps, lr=LR),
}


def state_bytes(opt) -> tuple[int, int, int]:
    total = paged = fp32_state_params = 0
    for st in opt.state.values():
        for k, v in st.items():
            if torch.is_tensor(v) and k not in ("qmap1", "qmap2", "step"):
                total += v.nbytes
                if getattr(v, "is_paged", False):
                    paged += v.nbytes
        s1 = st.get("state1", st.get("exp_avg"))
        if s1 is not None and s1.dtype == torch.float32:
            fp32_state_params += s1.numel()
    return total, paged, fp32_state_params


def gpt2_params(nanogpt):
    torch.manual_seed(SEED)
    model = nanogpt.GPT(nanogpt.GPTConfig()).cuda()
    params = list(model.parameters())
    gen = torch.Generator(device="cuda").manual_seed(SEED)
    for p in params:
        p.grad = 1e-3 * torch.randn(p.shape, generator=gen, device="cuda")
    return model, params


def time_steps(opt, iters: int) -> tuple[float, float]:
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0 = time.perf_counter()
    start.record()
    for _ in range(iters):
        opt.step()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters, (time.perf_counter() - t0) * 1e3 / iters


def memory_and_gpt2_latency(nanogpt):
    model, params = gpt2_params(nanogpt)
    n_params = sum(p.numel() for p in params)
    small = [p for p in params if p.numel() < 4096]
    rows = []
    page_mng = bnbF.GlobalPageManager.get_instance()
    for name, make in OPTIMIZERS.items():
        free_cuda()
        base_alloc, base_smi = torch.cuda.memory_allocated(), smi_used_mib()
        opt = make(params)
        opt.step()
        torch.cuda.synchronize()
        alloc_delta = (torch.cuda.memory_allocated() - base_alloc) / 2**20
        smi_delta = smi_used_mib() - base_smi
        total, paged, fp32_elems = state_bytes(opt)
        buf = io.BytesIO()
        torch.save(opt.state_dict(), buf)
        time_steps(opt, 3)
        ms_gpu, ms_wall = time_steps(opt, 10)
        row = {"optimizer": name, "state_mib": round(total / 2**20, 1), "paged_state_mib": round(paged / 2**20, 1),
               "bytes_per_param": round(total / n_params, 3),
               "torch_allocated_delta_mib": round(alloc_delta, 1), "nvidia_smi_delta_mib": smi_delta,
               "params_with_fp32_state": fp32_elems, "checkpoint_mib": round(buf.tell() / 2**20, 1),
               "step_ms_gpu_events": round(ms_gpu, 3), "step_ms_wall": round(ms_wall, 3)}
        if name in ("bnb AdamW 32-bit", "bnb AdamW8bit"):
            saved = bnb_optimizer.sync_gpu
            bnb_optimizer.sync_gpu = lambda t: None
            try:
                row["step_ms_wall_no_per_param_sync"] = round(time_steps(opt, 10)[1], 3)
            finally:
                bnb_optimizer.sync_gpu = saved
        del opt, buf
        free_cuda()
        row["nvidia_smi_after_delete_mib"] = smi_used_mib() - base_smi
        row["page_manager_tensors_after_delete"] = len(page_mng.paged_tensors)
        rows.append(row)
        print(f"  memory/latency: {name}", flush=True)

    sync_check = no_sync_matches_sync(params)
    info = {"model": "nanoGPT GPTConfig() (GPT-2 124M shapes, vocab 50304, tied wte / lm_head)",
            "params": n_params, "tensors": len(params), "tensors_below_min_8bit_size": len(small),
            "elements_below_min_8bit_size": sum(p.numel() for p in small),
            "no_sync_bitwise_equal": sync_check}
    del model, params
    free_cuda()
    return info, rows


def no_sync_matches_sync(params) -> bool:
    sub = [p for p in params if p.numel() >= 4096][:8]
    outs = []
    for patch in (False, True):
        ps = [torch.nn.Parameter(p.detach().clone()) for p in sub]
        for p, src in zip(ps, sub):
            p.grad = src.grad.clone()
        opt = bnb.optim.AdamW8bit(ps, lr=LR)
        saved = bnb_optimizer.sync_gpu
        if patch:
            bnb_optimizer.sync_gpu = lambda t: None
        try:
            for _ in range(3):
                opt.step()
        finally:
            bnb_optimizer.sync_gpu = saved
        torch.cuda.synchronize()
        outs.append([p.detach().clone() for p in ps])
    return all(torch.equal(a, b) for a, b in zip(*outs))


# ---------------------------------------------------------------- 4. latency vs size

LAT_OPTIMIZERS = {
    "torch Adam fused": lambda ps: torch.optim.Adam(ps, lr=LR, fused=True),
    "bnb Adam 32-bit": lambda ps: bnb.optim.Adam(ps, lr=LR),
    "bnb Adam8bit": lambda ps: bnb.optim.Adam8bit(ps, lr=LR),
    "bnb PagedAdam8bit": lambda ps: bnb.optim.PagedAdam8bit(ps, lr=LR),
}


def bytes_per_elem(name: str, dtype: torch.dtype) -> int:
    pb = 2 if dtype == torch.bfloat16 else 4
    if name.startswith("torch"):
        state = 4 * pb
    elif "8bit" in name:
        state = 4
    else:
        state = 16
    return 3 * pb + state


def latency_vs_size():
    rows = []
    for dtype in (torch.float32, torch.bfloat16):
        for n in LAT_SIZES:
            for name, make in LAT_OPTIMIZERS.items():
                free_cuda()
                p = torch.nn.Parameter(torch.randn(n, device="cuda", dtype=dtype))
                p.grad = 1e-3 * torch.randn(n, device="cuda", dtype=dtype)
                opt = make([p])
                for _ in range(5):
                    opt.step()
                ms_gpu, ms_wall = time_steps(opt, LAT_ITERS)
                gbs = bytes_per_elem(name, dtype) * n / (ms_wall * 1e-3) / 1e9
                rows.append({"dtype": str(dtype).replace("torch.", ""), "n": n, "optimizer": name,
                             "us_per_step": round(ms_wall * 1e3, 1), "us_gpu_events": round(ms_gpu * 1e3, 1),
                             "effective_GBps": round(gbs, 1)})
                del opt, p
        print(f"  latency: {dtype}", flush=True)
    free_cuda()
    return rows


# ---------------------------------------------------------------- 5. dynamic range

DR_VARIANTS = {
    "bnb Adam8bit": "bnb",
    "dynamic / tensor-wise": ("dynamic", None),
    "linear / block 256": ("linear", BLOCK),
    "linear / tensor-wise": ("linear", None),
    "dynamic / block 256, sqrt(v) (fix)": ("dynamic_sqrtv", BLOCK),
}


def run_stream(r: float, variant):
    scale, mu, gen = synthetic_grad_stream(DR_N, r, SEED + 1)
    p = torch.zeros(DR_N, device="cuda")
    updates = []
    if variant == "bnb":
        param = torch.nn.Parameter(p)
        opt = bnb.optim.Adam8bit([param], lr=LR, betas=BETAS_SYN, eps=EPS)
    else:
        st = {"s1": torch.zeros_like(p), "s2": torch.zeros_like(p)}
    for t in range(1, DR_STEPS + 1):
        g = scale * (mu + torch.randn(DR_N, generator=gen, device="cuda"))
        if variant == "bnb":
            before = param.detach().clone()
            param.grad = g
            opt.step()
            u = param.detach() - before
        else:
            before = p.clone()
            ref_adam_update(p, g, st, t, LR, BETAS_SYN, EPS, 0.0, variant)
            u = p - before
        if t > DR_STEPS - DR_LAST:
            updates.append(u)
    if variant == "bnb":
        v_zero = (opt.state[param]["state2"] == 0).float().mean().item()
        m_zero = (bnb_dequant(opt.state[param], 1, DR_N) == 0).float().mean().item()
    elif variant is None:
        v_zero = m_zero = 0.0
    else:
        v_zero = (st["s2"] == 0).float().mean().item()
        m_zero = (st["s1"] == 0).float().mean().item()
    return scale, torch.stack(updates), v_zero, m_zero


def dynamic_range():
    rows = []
    for r in DR_RANGES:
        scale, u_ref, _, _ = run_stream(r, None)
        order = torch.argsort(scale)
        lo, hi = order[: DR_N // 10], order[-DR_N // 10:]
        for name, variant in DR_VARIANTS.items():
            _, u, v_zero, m_zero = run_stream(r, variant)
            err = (u - u_ref).norm() / u_ref.norm()
            cos = F.cosine_similarity(u, u_ref, dim=1).mean()
            row = {"range_decades": r, "variant": name, "update_rel_err": r4(err.item()), "update_cos": r4(cos.item()),
                   "v_zero_frac": r4(v_zero), "m_zero_frac": r4(m_zero),
                   "max_abs_update_over_lr": r4(u.abs().max().item() / LR),
                   "ref_max_abs_update_over_lr": r4(u_ref.abs().max().item() / LR)}
            for tag, idx in (("smallest_decile", lo), ("largest_decile", hi)):
                ub, rb = u[:, idx], u_ref[:, idx]
                row[f"{tag}_rel_err"] = r4(((ub - rb).norm() / rb.norm()).item())
                row[f"{tag}_mag_ratio"] = r4((ub.abs().mean() / rb.abs().mean()).item())
            rows.append(row)
            del u
        del u_ref
        free_cuda()
        print(f"  dynamic range: 10^{r}", flush=True)
    return rows


# ---------------------------------------------------------------- 6. training

def load_corpus() -> tuple[torch.Tensor, str, int]:
    root = Path(torch.__file__).parent
    files = sorted(root.rglob("*.py"), key=lambda f: f.relative_to(root).as_posix())
    chunks, size = [], 0
    for f in files:
        b = f.read_bytes()
        chunks.append(b)
        size += len(b) + 1
        if size >= CORPUS_BYTES:
            break
    data = b"\n".join(chunks)[:CORPUS_BYTES]
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).cuda(), hashlib.sha256(data).hexdigest()[:16], len(chunks)


def batch(data: torch.Tensor, gen: torch.Generator):
    ix = torch.randint(0, data.numel() - TRAIN_BLOCK - 1, (TRAIN_BATCH,), generator=gen, device="cuda")
    idx = ix[:, None] + torch.arange(TRAIN_BLOCK + 1, device="cuda")
    tok = data[idx].long()
    return tok[:, :-1].contiguous(), tok[:, 1:].contiguous()


TRAIN_RUNS = [
    # (name, optimizer, peak lr, seed)
    ("torch AdamW fused", "torch", 1e-3, 0),
    ("torch AdamW fused (seed 1)", "torch", 1e-3, 1),
    ("bnb AdamW 32-bit", "bnb32", 1e-3, 0),
    ("bnb AdamW8bit", "bnb8", 1e-3, 0),
    ("linear int8 / block 256 (ref)", "linear", 1e-3, 0),
    ("torch AdamW fused, lr 3e-3", "torch", 3e-3, 0),
    ("bnb AdamW8bit, lr 3e-3", "bnb8", 3e-3, 0),
    ("linear int8 / block 256 (ref), lr 3e-3", "linear", 3e-3, 0),
    ("dynamic / block 256, sqrt(v) (ref)", "sqrtv", 1e-3, 0),
    ("dynamic / block 256, sqrt(v) (ref), lr 3e-3", "sqrtv", 3e-3, 0),
]
V_ZERO_THRESHOLD = 0.5 * QMAP_U[QMAP_U > 0].min().item()


def make_train_optimizer(kind: str, groups, lr: float):
    kw = dict(lr=lr, betas=TRAIN_BETAS, eps=EPS)
    if kind == "torch":
        return torch.optim.AdamW(groups, fused=True, **kw)
    if kind == "bnb32":
        return bnb.optim.AdamW(groups, **kw)
    if kind == "bnb8":
        return bnb.optim.AdamW8bit(groups, **kw)
    if kind == "linear":
        return RefAdamW(groups, weight_decay=0.0, quant=("linear", BLOCK), **kw)
    if kind == "sqrtv":
        return RefAdamW(groups, weight_decay=0.0, quant=("dynamic_sqrtv", BLOCK), **kw)
    raise ValueError(kind)


@torch.no_grad()
def v_block_stats(opt, kind: str) -> dict:
    """How much of v would round to code 0 (fp32 runs) / is code 0 (bnb 8-bit) in 8-bit-eligible tensors."""
    below = total = 0
    spans = []
    for p, st in opt.state.items():
        if p.numel() < 4096:
            continue
        if kind == "bnb8":
            below += int((st["state2"] == 0).sum().item())
            total += p.numel()
            continue
        vb, _ = blocks_of(st["exp_avg_sq"].float().reshape(-1), BLOCK)
        rel = vb / vb.amax(1, keepdim=True).clamp_min(1e-38)
        below += int((rel < V_ZERO_THRESHOLD).sum().item())
        total += p.numel()
        root = vb.sqrt()
        lo = torch.where(root > 0, root, torch.full_like(root, float("inf"))).amin(1)
        spans.append(torch.log10(root.amax(1) / lo))
    out = {"v_zero_frac": r4(below / total)}
    if spans:
        s = torch.cat(spans)
        s = s[torch.isfinite(s)]
        out.update({"block_grad_span_decades_median": r4(s.median().item()),
                    "block_grad_span_decades_p99": r4(s.quantile(0.99).item()),
                    "block_grad_span_decades_max": r4(s.max().item())})
    return out


def lr_at(step: int, peak: float) -> float:
    if step < TRAIN_WARMUP:
        return peak * (step + 1) / TRAIN_WARMUP
    prog = (step - TRAIN_WARMUP) / (TRAIN_STEPS - TRAIN_WARMUP)
    return 0.1 * peak + 0.45 * peak * (1 + math.cos(math.pi * prog))


@torch.no_grad()
def evaluate(model, val, n_batches: int) -> float:
    model.eval()
    gen = torch.Generator(device="cuda").manual_seed(1234)
    losses = []
    for _ in range(n_batches):
        x, y = batch(val, gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def train_one(nanogpt, train, val, name, kind, peak_lr, seed):
    torch.manual_seed(seed)
    cfg = nanogpt.GPTConfig(block_size=TRAIN_BLOCK, vocab_size=256, n_layer=6, n_head=6, n_embd=384,
                            dropout=0.0, bias=False)
    model = nanogpt.GPT(cfg).cuda()
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    groups = [{"params": decay, "weight_decay": TRAIN_WD}, {"params": no_decay, "weight_decay": 0.0}]
    opt = make_train_optimizer(kind, groups, peak_lr)
    gen = torch.Generator(device="cuda").manual_seed(100 + seed)
    curve, train_losses, spikes = [], [], 0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    status = "ok"
    for step in range(TRAIN_STEPS):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, peak_lr)
        x, y = batch(train, gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        lv = loss.item()
        if not math.isfinite(lv):
            status = f"non-finite loss at step {step}"
            break
        if len(train_losses) >= 50 and step >= TRAIN_WARMUP and lv > 1.3 * min(train_losses[-50:]):
            spikes += 1
        train_losses.append(lv)
        if (step + 1) % TRAIN_EVAL_EVERY == 0:
            curve.append({"step": step + 1, "val_loss": round(evaluate(model, val, TRAIN_EVAL_BATCHES), 4)})
    torch.cuda.synchronize()
    secs = time.perf_counter() - t0
    row = {"run": name, "optimizer": kind, "peak_lr": peak_lr, "seed": seed, "status": status,
           "final_val_loss": curve[-1]["val_loss"] if curve and status == "ok" else None,
           "final_train_loss_avg50": round(sum(train_losses[-50:]) / max(1, len(train_losses[-50:])), 4),
           "loss_spikes": spikes, "ms_per_step": round(secs * 1e3 / max(1, len(train_losses)), 2),
           "val_curve": curve}
    if kind in ("torch", "bnb8") and status == "ok":
        row.update(v_block_stats(opt, kind))
    del model, opt
    free_cuda()
    print(f"  train: {name}: {row['final_val_loss']} ({row['ms_per_step']} ms/step)", flush=True)
    return row


def training(nanogpt):
    data, digest, n_files = load_corpus()
    split = int(0.9 * data.numel())
    train, val = data[:split], data[split:]
    rows = [train_one(nanogpt, train, val, *run) for run in TRAIN_RUNS]
    info = {"corpus": f"first {CORPUS_BYTES} bytes of torch {torch.__version__} package .py files "
                      f"({n_files} files, sorted by path), byte-level, 90/10 split",
            "corpus_sha256_16": digest, "model": "nanoGPT GPT 6 layers x 384, 6 heads, block 256, vocab 256, no bias",
            "steps": TRAIN_STEPS, "batch": TRAIN_BATCH, "warmup": TRAIN_WARMUP, "betas": TRAIN_BETAS,
            "weight_decay": TRAIN_WD, "grad_clip": 1.0, "autocast": "bf16", "eval_batches": TRAIN_EVAL_BATCHES}
    return info, rows


# ---------------------------------------------------------------- report

def md_table(rows, cols):
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    return lines


def main():
    gpu = gpu_guard.acquire()
    torch.manual_seed(SEED)
    t_start = time.perf_counter()
    nanogpt = load_nanogpt()

    print("codebooks / reference match", flush=True)
    cb = codebooks()
    match = reference_match()
    wd_sem = weight_decay_semantics()
    print("memory", flush=True)
    mem_info, mem_rows = memory_and_gpt2_latency(nanogpt)
    print("latency", flush=True)
    lat = latency_vs_size()
    print("dynamic range", flush=True)
    dr = dynamic_range()
    print("training", flush=True)
    train_info, train_rows = training(nanogpt)
    runtime = round(time.perf_counter() - t_start, 1)

    result = {
        "experiment": HERE.name,
        "upstream": "bitsandbytes-foundation/bitsandbytes",
        "upstream_commit": git_head(ROOT / "upstream" / "bitsandbytes"),
        "nanogpt_commit": git_head(NANOGPT_DIR),
        "bitsandbytes": bnb.__version__,
        "components": ["bitsandbytes/optim/optimizer.py (Optimizer2State, Optimizer8bit.step)",
                       "csrc/kernels.cu (kOptimizerStatic8bit2StateBlockwise)",
                       "bitsandbytes/functional.py (create_dynamic_map, get_paged)"],
        "gpu": gpu,
        "runtime_s": runtime,
        "codebooks": cb,
        "reference_match": match,
        "weight_decay_semantics": wd_sem,
        "memory_info": mem_info,
        "memory": mem_rows,
        "latency": lat,
        "dynamic_range": {"n": DR_N, "steps": DR_STEPS, "last_steps_scored": DR_LAST, "betas": BETAS_SYN,
                          "grad": "g_t = 10^(-r*u) * (mu + noise_t), u ~ U(0,1) per element, mu ~ N(0, 0.25), "
                                  "noise ~ N(0, 1)", "rows": dr},
        "training_info": train_info,
        "training": train_rows,
        "peak_memory_mib": gpu_guard.peak_memory_mib(),
    }
    (HERE / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [f"bitsandbytes {bnb.__version__}, torch {torch.__version__}, {gpu.get('name', 'cuda')} "
             f"(memory cap {gpu.get('memory_fraction_cap')}), runtime {runtime} s", ""]
    lines += ["## Code books (create_dynamic_map)", ""]
    lines += md_table(cb, ["map", "min", "max", "smallest_nonzero_abs", "codes_below_1e-3", "codes_above_0.1"])
    lines += ["", f"## torch re-implementation vs CUDA kernel ({MATCH_N} elements, {MATCH_STEPS} steps, "
                  "grad range 10^3, teacher-forced per step)", ""]
    lines += md_table(match, ["optimizer", "weight_decay", "m_code_mismatch_frac", "v_code_mismatch_frac",
                              "max_code_diff", "absmax_max_rel_diff", "param_max_diff_over_max_update",
                              "free_running_drift_rel"])
    lines += ["", f"## weight_decay={wd_sem['weight_decay']} semantics (lr {wd_sem['lr']}, {wd_sem['steps']} steps): "
                  "rel. diff of final params", ""]
    lines += md_table(wd_sem["rows"], ["optimizer", "rel_diff_vs_torch_Adam_L2", "rel_diff_vs_torch_AdamW"])
    lines += ["", f"## Optimizer state on {mem_info['model']}: {mem_info['params']} params in {mem_info['tensors']} "
                  f"tensors ({mem_info['tensors_below_min_8bit_size']} tensors / "
                  f"{mem_info['elements_below_min_8bit_size']} elements below min_8bit_size); "
                  f"no-sync step bitwise equal: {mem_info['no_sync_bitwise_equal']}", ""]
    lines += md_table(mem_rows, ["optimizer", "state_mib", "paged_state_mib", "bytes_per_param",
                                 "torch_allocated_delta_mib", "nvidia_smi_delta_mib", "nvidia_smi_after_delete_mib",
                                 "page_manager_tensors_after_delete", "params_with_fp32_state", "checkpoint_mib",
                                 "step_ms_wall", "step_ms_wall_no_per_param_sync"])
    lines += ["", f"## step() latency, one tensor ({LAT_ITERS} steps after 5 warmup, wall clock)", ""]
    lines += md_table(lat, ["dtype", "n", "optimizer", "us_per_step", "us_gpu_events", "effective_GBps"])
    lines += ["", f"## Update error vs in-block gradient dynamic range ({DR_N} elements, {DR_STEPS} steps, "
                  f"last {DR_LAST} scored vs fp32 Adam)", ""]
    lines += md_table(dr, ["range_decades", "variant", "update_rel_err", "update_cos", "v_zero_frac", "m_zero_frac",
                           "smallest_decile_rel_err", "smallest_decile_mag_ratio", "largest_decile_rel_err",
                           "max_abs_update_over_lr", "ref_max_abs_update_over_lr"])
    lines += ["", f"## Training: {train_info['model']}, {TRAIN_STEPS} steps x {TRAIN_BATCH} x {TRAIN_BLOCK} tokens", ""]
    lines += md_table(train_rows, ["run", "status", "final_val_loss", "final_train_loss_avg50", "loss_spikes",
                                   "ms_per_step", "v_zero_frac", "block_grad_span_decades_median",
                                   "block_grad_span_decades_p99", "block_grad_span_decades_max"])
    lines += ["", f"v_zero_frac: fraction of v in 8-bit-eligible tensors at the end that is code 0 (bnb) or below "
                  f"{V_ZERO_THRESHOLD:.3g} x block absmax, i.e. would round to code 0 (torch fp32). Block grad span: "
                  "log10(max / min sqrt(v)) per 256-element block.", ""]
    lines += ["", "Val loss curves:", ""]
    steps = [c["step"] for c in train_rows[0]["val_curve"]]
    lines += ["| run | " + " | ".join(str(s) for s in steps) + " |", "|---|" + "---|" * len(steps)]
    lines += [f"| {r['run']} | " + " | ".join(str(c["val_loss"]) for c in r["val_curve"]) + " |" for r in train_rows]
    (HERE / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
