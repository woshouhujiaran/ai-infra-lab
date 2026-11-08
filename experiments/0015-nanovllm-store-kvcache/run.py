"""nano-vllm `store_kvcache` (Triton KV-cache write kernel) on an RTX 4060 Laptop.

`nanovllm/layers/attention.py` writes each layer's new K/V rows into the paged
cache with `store_kvcache_kernel`: one program per token, `tl.arange(0, D)` over
D = num_kv_heads * head_dim, int32 slot ids, and `cache_offsets = slot * D`.

1. Exactness vs a torch reference for fp32 / fp16 / bf16, contiguous K and the
   strided V view that Qwen3Attention passes (a slice of the fused qkv output),
   decode batches padded with slot -1 (CUDA-graph padding) and prefill runs;
   also what a plain torch `index_put_` replacement does with the -1 slots.
2. Non-power-of-two D (shapes such as 5 x 64 or 32 x 80): does it compile?
3. int32 `slot * D` overflow: with nano-vllm's layout (all layers are views
   into one kv_cache tensor), where does a write past 2**31 elements per layer
   land? Run in a child process with an int8 cache so 2**31 elements fit in 8 GB.
4. Bandwidth and per-call cost vs D and token count for upstream, a masked
   int64-offset rewrite (1 and several tokens per program), torch `index_put_`,
   `index_copy_`, `torch.compile(index_put_)` and a contiguous copy baseline,
   both eager (Python launch) and replayed from a CUDA graph (GPU time only,
   which is how nano-vllm runs decode).

The upstream module is imported unmodified from the submodule; `flash_attn`
(not installable on Windows) is stubbed because only the kernel is exercised.
"""
import json
import statistics
import subprocess
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from lab import gpu_guard, nanovllm_shim  # noqa: E402

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

SEED = 0
BLOCK_SIZE = 256  # nano-vllm kvcache_block_size
PERF_CACHE_BYTES = 128 * 2**20  # per K (and per V) cache, well above the 32 MB L2
PERF_DTYPE = torch.bfloat16
PERF_SHAPES = [(2, 64), (8, 32), (8, 64), (8, 128), (32, 128), (5, 64), (32, 80)]  # (num_kv_heads, head_dim)
PERF_TOKENS = [1, 16, 64, 256, 1024, 4096, 16384]  # <= 256: decode (random slots), else prefill (block runs)
GRAPH_REPLAYS = 15
ROUNDS = 3  # methods interleaved per cell; medians over rounds
L2_BYTES = 32 * 2**20
CORRECT_SHAPES = [(2, 64), (8, 64), (8, 128), (32, 128)]
CORRECT_DTYPES = [torch.float32, torch.float16, torch.bfloat16]
CORRECT_SLOTS = 8192
NONPOW2_SHAPES = [(3, 64), (5, 64), (12, 64), (3, 128), (32, 80), (32, 96)]
OVERFLOW_D = 1024
QWEN3_LAYERS = {"Qwen3-0.6B / 1.7B": 28, "Qwen3-4B / 8B": 36, "Qwen3-14B": 40, "Qwen3-30B-A3B": 48, "Qwen3-32B": 64}

torch._dynamo.config.recompile_limit = 128


def load_upstream() -> types.ModuleType:
    nanovllm_shim.install()
    if "nanovllm.utils" not in sys.modules:
        pkg = types.ModuleType("nanovllm.utils")
        pkg.__path__ = [str(nanovllm_shim.NANOVLLM_SRC / "utils")]
        sys.modules["nanovllm.utils"] = pkg
    if "flash_attn" not in sys.modules:
        def unavailable(*args, **kwargs):
            raise RuntimeError("flash_attn is stubbed; only store_kvcache is exercised")
        flash = types.ModuleType("flash_attn")
        flash.flash_attn_varlen_func = flash.flash_attn_with_kvcache = unavailable
        sys.modules["flash_attn"] = flash
    from nanovllm.layers import attention
    return attention


up = load_upstream()


@triton.jit
def store_kvcache_masked_kernel(
    key_ptr, key_stride, value_ptr, value_stride, k_cache_ptr, v_cache_ptr, slot_mapping_ptr, N,
    D: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    slots = tl.load(slot_mapping_ptr + rows, mask=rows < N, other=-1).to(tl.int64)
    cols = tl.arange(0, BLOCK_D)
    mask = (slots != -1)[:, None] & (cols < D)[None, :]
    rows = rows.to(tl.int64)
    key = tl.load(key_ptr + rows[:, None] * key_stride + cols[None, :], mask=mask)
    value = tl.load(value_ptr + rows[:, None] * value_stride + cols[None, :], mask=mask)
    cache_offsets = slots[:, None] * D + cols[None, :]
    tl.store(k_cache_ptr + cache_offsets, key, mask=mask)
    tl.store(v_cache_ptr + cache_offsets, value, mask=mask)


def tiled_block_n(D: int) -> int:
    return max(1, min(64, 4096 // triton.next_power_of_2(D)))


def store_kvcache_masked(key, value, k_cache, v_cache, slot_mapping, block_n=1):
    """Same contract as upstream store_kvcache; any D, int64 offsets, BLOCK_N tokens per program."""
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_masked_kernel[(triton.cdiv(N, block_n),)](
        key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, N,
        D, triton.next_power_of_2(D), block_n)


def store_masked_n1(key, value, k_cache, v_cache, slot_mapping):
    store_kvcache_masked(key, value, k_cache, v_cache, slot_mapping, 1)


def store_masked_tiled(key, value, k_cache, v_cache, slot_mapping):
    store_kvcache_masked(key, value, k_cache, v_cache, slot_mapping, tiled_block_n(key.shape[1] * key.shape[2]))


def torch_index_put(key, value, k_cache, v_cache, slot_mapping):
    N, D = key.shape[0], key.shape[1] * key.shape[2]
    k_cache.view(-1, D)[slot_mapping] = key.reshape(N, D)
    v_cache.view(-1, D)[slot_mapping] = value.reshape(N, D)


def torch_index_copy(key, value, k_cache, v_cache, slot_mapping):
    N, D = key.shape[0], key.shape[1] * key.shape[2]
    idx = slot_mapping.long()
    k_cache.view(-1, D).index_copy_(0, idx, key.reshape(N, D))
    v_cache.view(-1, D).index_copy_(0, idx, value.reshape(N, D))


torch_compiled = torch.compile(torch_index_put, dynamic=False)


def copy_baseline(key, value, k_cache, v_cache, slot_mapping):
    """Moves the same bytes into the first N cache rows (no scatter); bandwidth ceiling."""
    N, D = key.shape[0], key.shape[1] * key.shape[2]
    k_cache.view(-1, D)[:N].copy_(key.reshape(N, D))
    v_cache.view(-1, D)[:N].copy_(value.reshape(N, D))


METHODS = {
    "upstream": up.store_kvcache,
    "masked_n1": store_masked_n1,
    "masked_tiled": store_masked_tiled,
    "torch_index_put": torch_index_put,
    "torch_index_copy": torch_index_copy,
    "torch_compile": torch_compiled,
    "copy": copy_baseline,
}
DESCRIPTIONS = {
    "upstream": "nanovllm.layers.attention.store_kvcache as-is (1 token per program, int32 slot * D)",
    "masked_n1": "rewrite: masked BLOCK_D = next_pow2(D), int64 offsets, -1 handled, 1 token per program",
    "masked_tiled": "same rewrite with BLOCK_N = min(64, 4096 // BLOCK_D) tokens per program",
    "torch_index_put": "k_cache.view(-1, D)[slot_mapping] = key (int32 index, no -1 handling)",
    "torch_index_copy": "index_copy_ with slot_mapping.long() (conversion included)",
    "torch_compile": "torch.compile(torch_index_put, dynamic=False)",
    "copy": "copy_ of the same bytes into the first N rows (no scatter)",
}


def make_qkv(N, H, hd, dtype, gen):
    """Qwen3-style fused qkv (16:8 q:kv heads ratio -> q width 2 * D): contiguous K, strided V view."""
    D = H * hd
    qkv = torch.randn(N, 4 * D, generator=gen, dtype=torch.float32).to(dtype).cuda()
    key = qkv[:, 2 * D:3 * D].contiguous().view(N, H, hd)
    value = qkv[:, 3 * D:].view(N, H, hd)
    return key, value


def decode_slots(N, num_slots, gen):
    return torch.randperm(num_slots, generator=gen)[:N].to(torch.int32)


def prefill_slots(N, num_slots, gen):
    nblocks = -(-N // BLOCK_SIZE)
    blocks = torch.randperm(num_slots // BLOCK_SIZE, generator=gen)[:nblocks]
    return (blocks[:, None] * BLOCK_SIZE + torch.arange(BLOCK_SIZE)).flatten()[:N].to(torch.int32)


def new_cache(num_slots, H, hd, dtype, fill=None, gen=None):
    shape = (num_slots // BLOCK_SIZE, BLOCK_SIZE, H, hd)
    if fill == "randn":
        return torch.randn(shape, generator=gen, dtype=torch.float32).to(dtype).cuda()
    return torch.zeros(shape, dtype=dtype, device="cuda")


def short_error(e: Exception) -> str:
    lines = [ln.strip() for ln in str(e).splitlines() if ln.strip()]
    hit = [ln for ln in lines if "power of 2" in ln or "power of two" in ln]
    return f"{type(e).__name__}: {(hit or lines or [''])[-1][:160]}"


# ---------------------------------------------------------------- 1. exactness

def correctness(gen):
    rows = []
    methods = ["upstream", "masked_n1", "masked_tiled", "torch_index_put"]
    for dtype in CORRECT_DTYPES:
        for H, hd in CORRECT_SHAPES:
            D = H * hd
            for pattern in ["decode_padded", "prefill"]:
                if pattern == "decode_padded":
                    N = 64
                    slots = decode_slots(N, CORRECT_SLOTS, gen)
                    slots[3 * N // 4:] = -1  # graph bucket padding, as in ModelRunner.run_model
                else:
                    N = 1000
                    slots = prefill_slots(N, CORRECT_SLOTS, gen)
                slots = slots.cuda()
                key, value = make_qkv(N, H, hd, dtype, gen)
                k0 = new_cache(CORRECT_SLOTS, H, hd, dtype, "randn", gen)
                v0 = new_cache(CORRECT_SLOTS, H, hd, dtype, "randn", gen)
                valid = slots != -1
                k_ref, v_ref = k0.clone(), v0.clone()
                k_ref.view(-1, D)[slots[valid].long()] = key.reshape(N, D)[valid]
                v_ref.view(-1, D)[slots[valid].long()] = value.reshape(N, D)[valid]
                for name in methods:
                    kc, vc = k0.clone(), v0.clone()
                    METHODS[name](key, value, kc, vc, slots)
                    torch.cuda.synchronize()
                    bad_k = (kc.view(-1, D) != k_ref.view(-1, D)).any(1)
                    bad_v = (vc.view(-1, D) != v_ref.view(-1, D)).any(1)
                    bad = (bad_k | bad_v).nonzero().flatten().tolist()
                    rows.append({"dtype": str(dtype).replace("torch.", ""), "kv_heads": H, "head_dim": hd, "D": D,
                                 "pattern": pattern, "tokens": N, "padded": int((~valid).sum()), "method": name,
                                 "exact": not bad, "wrong_rows": len(bad), "wrong_slots": bad[:4]})
                    print(f"correct {rows[-1]}", flush=True)
                del k0, v0, k_ref, v_ref
    return rows


# ---------------------------------------------------------------- 2. non-power-of-two D

def nonpow2(gen):
    rows = []
    for H, hd in NONPOW2_SHAPES:
        D = H * hd
        N = 300
        slots = prefill_slots(N, CORRECT_SLOTS, gen).cuda()
        slots[-7:] = -1
        key, value = make_qkv(N, H, hd, PERF_DTYPE, gen)
        valid = slots != -1
        row = {"kv_heads": H, "head_dim": hd, "D": D}
        for name in ["upstream", "masked_n1", "masked_tiled"]:
            kc = new_cache(CORRECT_SLOTS, H, hd, PERF_DTYPE)
            vc = new_cache(CORRECT_SLOTS, H, hd, PERF_DTYPE)
            try:
                METHODS[name](key, value, kc, vc, slots)
                torch.cuda.synchronize()
                ok = (torch.equal(kc.view(-1, D)[slots[valid].long()], key.reshape(N, D)[valid])
                      and torch.equal(vc.view(-1, D)[slots[valid].long()], value.reshape(N, D)[valid])
                      and int((kc.view(-1, D) != 0).any(1).sum()) == int(valid.sum()))
                row[name] = "exact" if ok else "WRONG"
            except Exception as e:  # noqa: BLE001
                row[name] = short_error(e)
        rows.append(row)
        print(f"nonpow2 {row}", flush=True)
    return rows


# ---------------------------------------------------------------- 3. int32 overflow (child process)

def overflow_child(case: str) -> dict:
    """int8 kv_cache[2, S, D] like ModelRunner.allocate_kv_cache; write one token to layer 1."""
    gpu_guard.acquire()
    D = OVERFLOW_D
    S = 2**31 // D + BLOCK_SIZE  # slots per layer: one block past the int32 boundary
    kv = torch.zeros(2, S // BLOCK_SIZE, BLOCK_SIZE, 8, D // 8, dtype=torch.int8, device="cuda")
    k_cache = v_cache = kv[1]
    slot = {"below": 2**31 // D - 1}.get(case.split(":")[1], 2**31 // D + 3)
    slot_dtype = torch.int64 if case.endswith("int64") else torch.int32
    key = torch.full((1, 8, D // 8), 7, dtype=torch.int8, device="cuda")
    slots = torch.tensor([slot], dtype=slot_dtype, device="cuda")
    fn = METHODS[case.split(":")[0]]
    fn(key, key, k_cache, v_cache, slots)
    torch.cuda.synchronize()
    flat = kv.view(-1, D)
    hits = []
    chunk = 65536
    for start in range(0, flat.shape[0], chunk):
        nz = flat[start:start + chunk].ne(0).any(1).nonzero().flatten()
        hits += [start + int(i) for i in nz]
    found = [{"layer": h // S, "slot": h % S} for h in hits]
    return {"case": case, "slot": slot, "slot_dtype": str(slot_dtype).replace("torch.", ""),
            "slot_times_D": slot * D, "expected": {"layer": 1, "slot": slot}, "found": found,
            "correct": found == [{"layer": 1, "slot": slot}],
            "kv_cache_mib": round(kv.numel() / 2**20, 1)}


def overflow_tests():
    cases = ["upstream:below:int32", "upstream:above:int32", "upstream:above:int64",
             "masked_n1:above:int32", "torch_index_put:above:int32"]
    rows = []
    for case in cases:
        proc = subprocess.run([sys.executable, __file__, "--overflow", case], capture_output=True, text=True,
                              timeout=600)
        line = [ln for ln in proc.stdout.splitlines() if ln.startswith("OVERFLOW ")]
        if proc.returncode == 0 and line:
            rows.append(json.loads(line[-1][len("OVERFLOW "):]))
        else:
            rows.append({"case": case, "error": (proc.stderr.strip().splitlines() or ["?"])[-1][:200],
                         "returncode": proc.returncode})
        print(f"overflow {rows[-1]}", flush=True)
    return rows


def overflow_thresholds():
    """Overflow starts at 2**31 cache elements per layer per K (or V), independent of D and TP."""
    rows = []
    for name, layers in QWEN3_LAYERS.items():
        for dtype, nbytes in [("bf16", 2), ("fp8", 1)]:
            rows.append({"model": name, "layers": layers, "kv_dtype": dtype,
                         "kv_cache_gib_at_overflow": 2 * layers * 2**31 * nbytes / 2**30})
    return rows


# ---------------------------------------------------------------- 4. performance

def time_eager(fn, args, calls):
    for _ in range(5):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(calls):
        fn(*args)
    t_issue = time.perf_counter() - t0
    torch.cuda.synchronize()
    t_wall = time.perf_counter() - t0
    return t_issue / calls * 1e6, t_wall / calls * 1e6


def graph_reps(N):
    return 200 if N <= 256 else 100 if N <= 4096 else 20


def time_graph(fn, args, reps):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn(*args)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn(*args)
    g.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(GRAPH_REPLAYS):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        g.replay()
        e1.record()
        e1.synchronize()
        times.append(e0.elapsed_time(e1))
    del g
    return statistics.median(times) / reps * 1e3


def warm_gpu(seconds=2.0):
    a = torch.randn(2048, 2048, device="cuda")
    t0 = time.time()
    while time.time() - t0 < seconds:
        for _ in range(20):
            a @ a
        torch.cuda.synchronize()


def perf(gen):
    rows = []
    item = torch.finfo(PERF_DTYPE).bits // 8
    warm_gpu()
    for H, hd in PERF_SHAPES:
        D = H * hd
        num_slots = PERF_CACHE_BYTES // (D * item) // BLOCK_SIZE * BLOCK_SIZE
        kc = new_cache(num_slots, H, hd, PERF_DTYPE)
        vc = new_cache(num_slots, H, hd, PERF_DTYPE)
        for N in PERF_TOKENS:
            pattern = "decode" if N <= 256 else "prefill"
            slots = (decode_slots if pattern == "decode" else prefill_slots)(N, num_slots, gen).cuda()
            key, value = make_qkv(N, H, hd, PERF_DTYPE, gen)
            nbytes = 4 * N * D * item + 4 * N
            args = (key, value, kc, vc, slots)
            calls = 200 if N <= 256 else 30
            reps = graph_reps(N)
            cell = {}
            for name, fn in METHODS.items():
                row = {"kv_heads": H, "head_dim": hd, "D": D, "tokens": N, "pattern": pattern, "method": name,
                       "bytes": nbytes, "l2_resident": 4 * N * D * item <= L2_BYTES, "graph_reps": reps,
                       "issue": [], "wall": [], "graph": []}
                kc.zero_()
                vc.zero_()
                try:
                    fn(*args)
                    torch.cuda.synchronize()
                    if name != "copy":
                        idx = slots.long()
                        row["exact"] = bool(torch.equal(kc.view(-1, D)[idx], key.reshape(N, D))
                                            and torch.equal(vc.view(-1, D)[idx], value.reshape(N, D)))
                    row["status"] = "ok"
                except Exception as e:  # noqa: BLE001
                    row["status"] = short_error(e)
                cell[name] = row
            warm_gpu(0.3)  # torch.compile just compiled this shape with the GPU idle; restore clocks
            for _ in range(ROUNDS):
                for name, fn in METHODS.items():
                    row = cell[name]
                    if row["status"] != "ok":
                        continue
                    issue, wall = time_eager(fn, args, calls)
                    row["issue"].append(issue)
                    row["wall"].append(wall)
                    try:
                        row["graph"].append(time_graph(fn, args, reps))
                    except Exception as e:  # noqa: BLE001
                        row["graph_error"] = short_error(e)
                        torch.cuda.synchronize()
            for name, row in cell.items():
                issue, wall, graph = row.pop("issue"), row.pop("wall"), row.pop("graph")
                if row["status"] == "ok":
                    row.update(issue_us=round(statistics.median(issue), 2),
                               wall_us=round(statistics.median(wall), 2),
                               wall_us_best=round(min(wall), 2),
                               wall_us_rounds=[round(w, 2) for w in wall])
                    if graph:
                        gpu = statistics.median(graph)
                        row.update(graph_us=round(gpu, 2), graph_gbps=round(nbytes / gpu / 1e3, 1),
                                   graph_us_rounds=[round(x, 2) for x in graph])
                rows.append(row)
                print(f"perf {row}", flush=True)
        del kc, vc
        torch.cuda.empty_cache()
    return rows


# ---------------------------------------------------------------- report

def upstream_commit() -> str:
    return nanovllm_shim.upstream_commit()


def fmt(v, digits=1):
    return "" if v is None else (f"{v:.{digits}f}" if isinstance(v, float) else str(v))


def write_report(result):
    perf_rows = result["perf"]
    by = {(r["D"], r["kv_heads"], r["tokens"], r["method"]): r for r in perf_rows}
    gpu = result["gpu"]
    names = list(METHODS)
    lines = [
        f"nano-vllm {result['upstream_commit'][:10]} `store_kvcache`, triton-windows {triton.__version__}, "
        f"torch {torch.__version__}, {gpu.get('name', 'cuda')} (memory cap {gpu.get('memory_fraction_cap')})",
        "",
        "## 1. Exactness vs torch reference (whole cache compared; caches start as random data)",
        "",
        "decode_padded = 64 random slots with the last 16 set to -1; prefill = 1000 tokens in 256-slot blocks. "
        "K is contiguous, V is the strided slice of a fused qkv tensor.",
        "",
        "| dtype | D (heads x dim) | pattern | upstream | masked_n1 | masked_tiled | torch_index_put |",
        "|---|---|---|---|---|---|---|",
    ]
    corr = {}
    for r in result["correctness"]:
        corr.setdefault((r["dtype"], r["kv_heads"], r["head_dim"], r["pattern"]), {})[r["method"]] = r
    for (dt, H, hd, pat), m in corr.items():
        cells = []
        for name in ["upstream", "masked_n1", "masked_tiled", "torch_index_put"]:
            r = m[name]
            cells.append("exact" if r["exact"] else f"{r['wrong_rows']} wrong rows (slots {r['wrong_slots']})")
        lines.append(f"| {dt} | {H * hd} ({H}x{hd}) | {pat} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## 2. Non-power-of-two D (bf16, 300 prefill tokens, 7 padded)",
        "",
        "| D (heads x dim) | upstream | masked_n1 | masked_tiled |",
        "|---|---|---|---|",
    ]
    lines += [f"| {r['D']} ({r['kv_heads']}x{r['head_dim']}) | {r['upstream']} | {r['masked_n1']} | "
              f"{r['masked_tiled']} |" for r in result["nonpow2"]]
    lines += [
        "",
        f"## 3. int32 `slot * D` overflow (int8 kv_cache[2 layers, S, D={OVERFLOW_D}], one token written to layer 1)",
        "",
        "| case | slot | slot dtype | slot*D | landed at | correct |",
        "|---|---|---|---|---|---|",
    ]
    for r in result["overflow"]:
        if "error" in r:
            lines.append(f"| {r['case']} | | | | error: {r['error']} | |")
        else:
            found = ", ".join(f"layer {f['layer']} slot {f['slot']}" for f in r["found"]) or "nowhere"
            lines.append(f"| {r['case']} | {r['slot']} | {r['slot_dtype']} | {r['slot_times_D']} | {found} | "
                         f"{r['correct']} |")
    lines += [
        "",
        "Total KV cache per GPU at which the first overflowing slot exists (2^31 elements per layer per K/V; "
        "independent of D and TP size):",
        "",
        "| model | layers | KV dtype | KV cache GiB |",
        "|---|---|---|---|",
    ]
    lines += [f"| {r['model']} | {r['layers']} | {r['kv_dtype']} | {r['kv_cache_gib_at_overflow']:.0f} |"
              for r in result["overflow_thresholds"]]
    lines += [
        "",
        f"## 4a. GPU time per call from CUDA-graph replay, us (200 / 100 / 20 calls per graph for N <= 256 / "
        f"<= 4096 / larger; median of {GRAPH_REPLAYS} replays, then median of {ROUNDS} interleaved rounds; "
        "bf16; K+V read and written)",
        "",
        "| D (heads x dim) | tokens | " + " | ".join(names) + " | masked_tiled / upstream |",
        "|---|---|" + "---|" * (len(names) + 1),
    ]
    for H, hd in PERF_SHAPES:
        D = H * hd
        for N in PERF_TOKENS:
            cells = []
            for name in names:
                r = by[(D, H, N, name)]
                cells.append(fmt(r.get("graph_us"), 2) if r["status"] == "ok" else "compile error")
            u, t = by[(D, H, N, "upstream")].get("graph_us"), by[(D, H, N, "masked_tiled")].get("graph_us")
            sp = f"{u / t:.2f}x" if u and t else ""
            lines.append(f"| {D} ({H}x{hd}) | {N} | " + " | ".join(cells) + f" | {sp} |")
    lines += [
        "",
        "## 4b. Effective bandwidth from CUDA-graph replay, GB/s (4 * N * D * 2 bytes + slot ids)",
        "",
        "Back-to-back calls rewrite the same rows, so working sets <= 32 MB (the L2 size) are served from L2 "
        "and can exceed the 256 GB/s DRAM peak.",
        "",
        "| D (heads x dim) | tokens | MiB moved | L2-resident | " + " | ".join(names) + " |",
        "|---|---|---|---|" + "---|" * len(names),
    ]
    for H, hd in PERF_SHAPES:
        D = H * hd
        for N in PERF_TOKENS:
            if N < 1024:
                continue
            first = by[(D, H, N, "copy")]
            cells = [fmt(by[(D, H, N, n)].get("graph_gbps")) if by[(D, H, N, n)]["status"] == "ok" else "-"
                     for n in names]
            lines.append(f"| {D} ({H}x{hd}) | {N} | {first['bytes'] / 2**20:.0f} | "
                         f"{'yes' if first['l2_resident'] else 'no'} | " + " | ".join(cells) + " |")
    lines += [
        "",
        f"## 4c. Eager per-call wall time, us (Python launch included; median / best of {ROUNDS} rounds)",
        "",
        "| D (heads x dim) | tokens | " + " | ".join(names) + " |",
        "|---|---|" + "---|" * len(names),
    ]
    for H, hd in PERF_SHAPES:
        D = H * hd
        for N in PERF_TOKENS:
            cells = []
            for n in names:
                r = by[(D, H, N, n)]
                cells.append(f"{fmt(r['wall_us'])} / {fmt(r['wall_us_best'])}" if r["status"] == "ok" else "-")
            lines.append(f"| {D} ({H}x{hd}) | {N} | " + " | ".join(cells) + " |")
    bad = [r for r in perf_rows if r.get("exact") is False or r.get("graph_error")]
    lines += ["", f"Perf cells with a wrong result or a CUDA-graph error: {len(bad)}"]
    lines += [f"- {r['method']} D={r['D']} N={r['tokens']}: exact={r.get('exact')} "
              f"graph_error={r.get('graph_error')}" for r in bad]
    (HERE / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    t_start = time.time()
    torch.manual_seed(SEED)
    gen = torch.Generator().manual_seed(SEED)
    gpu = gpu_guard.acquire()
    timings = {}

    t = time.time()
    overflow = overflow_tests()
    timings["overflow_s"] = round(time.time() - t, 1)
    t = time.time()
    corr = correctness(gen)
    timings["correctness_s"] = round(time.time() - t, 1)
    t = time.time()
    np2 = nonpow2(gen)
    timings["nonpow2_s"] = round(time.time() - t, 1)
    torch.cuda.empty_cache()
    t = time.time()
    perf_rows = perf(gen)
    timings["perf_s"] = round(time.time() - t, 1)

    result = {
        "experiment": HERE.name,
        "upstream": "GeeeekExplorer/nano-vllm",
        "upstream_commit": upstream_commit(),
        "component": "nanovllm/layers/attention.py (store_kvcache_kernel + store_kvcache)",
        "methods": DESCRIPTIONS,
        "seed": SEED,
        "block_size": BLOCK_SIZE,
        "perf_cache_bytes_per_tensor": PERF_CACHE_BYTES,
        "perf_dtype": str(PERF_DTYPE),
        "graph_replays": GRAPH_REPLAYS,
        "rounds": ROUNDS,
        "triton": triton.__version__,
        "gpu": gpu,
        "correctness": corr,
        "nonpow2": np2,
        "overflow": overflow,
        "overflow_thresholds": overflow_thresholds(),
        "perf": perf_rows,
        "peak_memory_mib": gpu_guard.peak_memory_mib(),
        "timings": timings,
        "runtime_s": round(time.time() - t_start, 1),
    }
    (HERE / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_report(result)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--overflow":
        print("OVERFLOW " + json.dumps(overflow_child(sys.argv[2])), flush=True)
    else:
        main()
