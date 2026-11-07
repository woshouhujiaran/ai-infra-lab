"""gpt-fast speculative decoding (generate.py: speculative_decode) on TinyStories Llamas.

speculative_decode(model, draft_model, cur_token, input_pos, speculate_k) lets the draft
decode k tokens one by one, runs the target once over [cur_token, d1..dk], accepts draft
token i with probability min(1, q_i / p_i), and on the first rejection samples a
replacement from normalize(max(q - p, 0)); if all k are accepted it samples a bonus token
from the target and feeds d_k into the draft's KV cache.

Models: karpathy/tinyllamas stories15M / stories42M (drafts) and stories110M (target),
llama2.c .bin checkpoints (same Llama-2 32k vocab) loaded into gpt-fast's Transformer.

1. Upstream generate(draft_model=...) at HEAD, as-is and with each failure shimmed in
   turn, to list what breaks.
2. The verify forward needs a BlockMask for k+1 queries at offset input_pos. A
   prefill-style mask (create_block_mask(causal, S, kv_len), then Transformer.forward
   shifts mask_mod by input_pos[0]) vs an offset-aware mask: verify logits vs k+1
   sequential decode_one_token calls, eager and torch.compile, around the 128 block edge.
3. Exactness of upstream speculative_decode (unmodified) with toy trigram models whose
   token buffer acts as the KV cache: chi-square of the joint of 3 generated tokens vs the
   analytic target distribution, with and without the draft cache fill; greedy equality.
4. Acceptance on the real models: tokens per call vs k for two drafts and three sampling
   settings (bf16), against (1 - a^(k+1)) / (1 - a) from the measured per-position
   acceptance a = sum(min(p, q)); self-drafting in bf16 vs fp32.
5. bf16 wall clock with only flex_attention compiled (the rest eager): tokens/s vs plain
   decoding, and per-step costs that explain it.

generate.py / model.py are imported unmodified; run.py patches module attributes
(decode_n_tokens output shapes, model_forward's mask) and uses its own copy of generate()'s
speculative loop with the seq indexing fixed.
"""
import copy
import hashlib
import json
import math
import platform
import struct
import sys
import time
import traceback
import types
import warnings
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "upstream" / "gpt-fast"))
# tokenizer.py imports sentencepiece at module level; no tokenizer from it is used here
sys.modules.setdefault("sentencepiece", types.ModuleType("sentencepiece"))
warnings.filterwarnings("ignore", message=".*flex_attention called without torch.compile.*")

from lab import gpu_guard  # noqa: E402

GPU_INFO = gpu_guard.acquire()

import generate as G  # noqa: E402
import model as M  # noqa: E402
from model import ModelArgs, Transformer  # noqa: E402
from torch.nn.attention.flex_attention import create_block_mask as create_block_mask_eager  # noqa: E402

SEED = 0
DEVICE = "cuda"
CKPT_DIR = Path.home() / ".cache" / "ai-infra-lab" / "tinyllamas"
CKPTS = ["stories15M", "stories42M", "stories110M"]
TARGET = "stories110M"
DRAFTS = ["stories15M", "stories42M"]
SAMPLINGS = {
    "greedy": dict(temperature=0.0, top_k=None),
    "t0.8-top200": dict(temperature=0.8, top_k=200),
    "t1.0": dict(temperature=1.0, top_k=None),
}
MAX_SEQ = 256  # stories15M was trained (and its RoPE table built) for 256 positions
PROMPT_LEN = 16
NEW_TOKENS = 200
NUM_PROMPTS = 4
KS = [1, 2, 4, 8]
ALPHA_K = 4
TOY_V = 8
TOY_TRIALS = 20000
TOY_LONG_TRIALS = 1000
TOY_LONG_TOKENS = 24
TOY_SAMPLINGS = {"t1.0": dict(temperature=1.0, top_k=None), "t0.7-top3": dict(temperature=0.7, top_k=3)}
VERIFY_OFFSETS = [64, 124, 200, 380]
VERIFY_S = 9
TIMING_REPS = 40

torch._dynamo.config.cache_size_limit = 64
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sync():
    torch.cuda.synchronize()


# ---------------------------------------------------------------- checkpoints

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def load_llama2c(name: str, dtype: torch.dtype) -> Transformer:
    """llama2.c export format v0: 7 int32 header, then fp32 weights in a fixed order."""
    raw = (CKPT_DIR / f"{name}.bin").read_bytes()
    dim, hidden, n_layer, n_head, n_kv, vocab, seq_len = struct.unpack("7i", raw[:28])
    shared = vocab > 0
    vocab = abs(vocab)
    w = torch.frombuffer(bytearray(raw[28:]), dtype=torch.float32)
    head = dim // n_head
    off = 0

    def take(*shape):
        nonlocal off
        n = math.prod(shape)
        t = w[off:off + n].view(*shape)
        off += n
        return t

    emb = take(vocab, dim)
    rms_att = take(n_layer, dim)
    wq = take(n_layer, dim, dim)
    wk = take(n_layer, n_kv * head, dim)
    wv = take(n_layer, n_kv * head, dim)
    wo = take(n_layer, dim, dim)
    rms_ffn = take(n_layer, dim)
    w1 = take(n_layer, hidden, dim)
    w2 = take(n_layer, dim, hidden)
    w3 = take(n_layer, hidden, dim)
    rms_final = take(dim)
    take(seq_len, head // 2)
    take(seq_len, head // 2)
    wcls = emb if shared else take(vocab, dim)
    assert off == w.numel(), (name, off, w.numel())

    args = ModelArgs(block_size=seq_len, vocab_size=vocab, n_layer=n_layer, n_head=n_head, dim=dim,
                     intermediate_size=hidden, n_local_heads=n_kv)
    sd = {"tok_embeddings.weight": emb, "norm.weight": rms_final, "output.weight": wcls}
    for i in range(n_layer):
        p = f"layers.{i}."
        sd[p + "attention_norm.weight"] = rms_att[i]
        sd[p + "attention.wqkv.weight"] = torch.cat([wq[i], wk[i], wv[i]])
        sd[p + "attention.wo.weight"] = wo[i]
        sd[p + "ffn_norm.weight"] = rms_ffn[i]
        sd[p + "feed_forward.w1.weight"] = w1[i]
        sd[p + "feed_forward.w2.weight"] = w2[i]
        sd[p + "feed_forward.w3.weight"] = w3[i]
    with torch.device("meta"):
        model = Transformer(args)
    model.load_state_dict(sd, assign=True)
    model = model.to(device=DEVICE, dtype=dtype).eval()
    with torch.device(DEVICE):
        model.setup_caches(max_batch_size=1, max_seq_length=MAX_SEQ)
    model.n_params = sum(p.numel() for p in model.parameters())
    model.ckpt = name
    return model


def load_vocab() -> list[str]:
    raw = (CKPT_DIR / "tokenizer.bin").read_bytes()
    off = 4
    vocab = []
    for _ in range(32000):
        _, n = struct.unpack_from("fi", raw, off)
        off += 8
        vocab.append(raw[off:off + n].decode("utf-8", "replace"))
        off += n
    return vocab


def detok(vocab, ids) -> str:
    out = []
    for i in ids:
        if i in (1, 2):
            continue
        s = vocab[i]
        if s.startswith("<0x") and s.endswith(">"):
            s = chr(int(s[3:-1], 16))
        out.append(s)
    return "".join(out).lstrip()


# ---------------------------------------------------------------- masks / shims

_mask_cache: dict = {}


def build_offset_mask(off: int, S: int, kv_len: int, device) -> "G.BlockMask":
    bm = create_block_mask_eager(lambda b, h, q, kv: q + off >= kv, 1, 1, S, kv_len, device=device, _compile=False)
    # Transformer.forward shifts mask_mod by input_pos[0] itself, so hand it the unshifted one.
    bm.mask_mod = G.causal_mask
    return bm


def verify_forward_fixed(model, x, input_pos):
    off, S = int(input_pos[0]), x.shape[-1]
    key = (off, S, model.max_seq_length, str(x.device))
    if key not in _mask_cache:
        _mask_cache[key] = build_offset_mask(off, S, model.max_seq_length, x.device)
    bm = copy.copy(_mask_cache[key])  # forward overwrites bm.mask_mod
    return model(bm, x, input_pos)


def verify_forward_prefill_style(model, x, input_pos):
    bm = create_block_mask_eager(G.causal_mask, 1, 1, x.shape[-1], model.max_seq_length, device=x.device, _compile=False)
    return model(bm, x, input_pos)


_upstream = {"decode_n_tokens": G.decode_n_tokens, "model_forward": G.model_forward,
             "create_block_mask": G.create_block_mask}


def decode_n_tokens_flat(*args, **kwargs):
    """decode_one_token returns [B, 1] tokens and [B, V] probs; speculative_decode wants [1] / [V]."""
    toks, probs = _upstream["decode_n_tokens"](*args, **kwargs)
    return [t.view(-1) for t in toks], [p.view(-1, p.shape[-1])[0] for p in probs]


def set_shims(flat: bool, verify):
    G.decode_n_tokens = decode_n_tokens_flat if flat else _upstream["decode_n_tokens"]
    G.model_forward = verify if verify is not None else _upstream["model_forward"]


def speculative_generate(target, draft, prompt, max_new, k, sampling):
    """generate()'s speculative loop for batch 1, writing seq[pos] instead of seq[batch rows]."""
    device = prompt.device
    T = prompt.numel()
    T_new = T + max_new
    seq = torch.empty(T_new, dtype=prompt.dtype, device=device)
    seq[:T] = prompt
    input_pos = torch.arange(0, T, device=device)
    next_token = G.prefill(target, prompt.view(1, -1), input_pos, **sampling).clone()
    if draft is not target:
        G.prefill(draft, prompt.view(1, -1), input_pos, **sampling)
    seq[T] = next_token.squeeze()
    pos = T
    accept_counts = [0] * (k + 1)
    while pos < T_new - 1:
        cur = next_token.view(())
        nxt = G.speculative_decode(target, draft, cur, pos, k, **sampling)
        accept_counts[len(nxt) - 1] += 1
        n = min(T_new - pos - 1, len(nxt))
        seq[pos + 1: pos + n + 1] = nxt[:n]
        pos += n
        next_token = nxt[-1]
    return seq, accept_counts


def plain_generate(target, prompt, max_new, sampling):
    T = prompt.numel()
    input_pos = torch.arange(0, T, device=prompt.device)
    nt = G.prefill(target, prompt.view(1, -1), input_pos, **sampling).clone()
    toks, _ = _upstream["decode_n_tokens"](target, nt.view(1, -1), torch.tensor([T], device=prompt.device),
                                           max_new - 1, **sampling)
    return torch.cat([prompt, nt.view(-1)] + [t.view(-1) for t in toks])


def mean_tokens_per_call(counts) -> float:
    return sum((i + 1) * c for i, c in enumerate(counts)) / max(1, sum(counts))


# ---------------------------------------------------------------- 1. upstream as-is

def part_upstream(target, draft, prompts):
    out = []
    attempts = [
        ("as-is", False, None),
        ("+ flat decode_n_tokens", True, None),
        ("+ flat decode_n_tokens + offset verify mask", True, verify_forward_fixed),
    ]
    with torch.no_grad():
        torch.manual_seed(SEED)
        y = G.generate(target, prompts[0], 32, 1, interactive=False, draft_model=None, **SAMPLINGS["t0.8-top200"])[0]
        out.append({"attempt": "generate(draft_model=None)", "ok": True, "shape": list(y.shape)})
        for name, flat, verify in attempts:
            set_shims(flat, verify)
            torch.manual_seed(SEED)
            try:
                y, stats = G.generate(target, prompts[0], 32, 1, interactive=False, draft_model=draft, speculate_k=4,
                                      **SAMPLINGS["t0.8-top200"])
                out.append({"attempt": name, "ok": True, "stats": stats})
            except Exception as e:  # noqa: BLE001
                tb = traceback.extract_tb(e.__traceback__)
                frame = next((f for f in reversed(tb) if "gpt-fast" in f.filename), tb[-1])
                out.append({"attempt": name, "ok": False, "error": f"{type(e).__name__}: {str(e).splitlines()[0]}",
                            "where": f"{Path(frame.filename).name}:{frame.lineno} {frame.line}"})
        set_shims(False, None)
    for r in out:
        print("  upstream", r)
    return out


# ---------------------------------------------------------------- 2. verify mask

def part_verify_mask(target32):
    res = []
    m = target32
    with torch.device(DEVICE):
        m.setup_caches(max_batch_size=1, max_seq_length=512)
    kv_len = m.max_seq_length
    eager_refs = {}
    g = torch.Generator().manual_seed(SEED)
    toks = torch.randint(3, 32000, (1, max(VERIFY_OFFSETS) + VERIFY_S), generator=g).to(DEVICE, torch.int)
    full = create_block_mask_eager(G.causal_mask, 1, 1, kv_len, kv_len, device=DEVICE, _compile=False)
    modes = {"eager": m}
    try:
        modes["compiled"] = torch.compile(m)
    except Exception as e:  # noqa: BLE001
        print("  compile failed", e)

    def prefix(T0):
        mk = create_block_mask_eager(G.causal_mask, 1, 1, T0, kv_len, device=DEVICE, _compile=False)
        m(mk, toks[:, :T0], torch.arange(T0, device=DEVICE))

    with torch.no_grad():
        for mode, fwd in modes.items():
            for T0 in VERIFY_OFFSETS:
                t0 = time.perf_counter()
                prefix(T0)
                ref = []
                for i in range(T0, T0 + VERIFY_S):
                    ip = torch.tensor([i], device=DEVICE)
                    mk = full[:, :, ip // full.BLOCK_SIZE[0]]
                    mk.mask_mod = full.mask_mod
                    mk.seq_lengths = (1, kv_len)
                    ref.append(fwd(mk, toks[:, i:i + 1], ip)[0, 0])
                ref = torch.stack(ref)
                row = {"mode": mode, "offset": T0, "positions": [T0, T0 + VERIFY_S - 1]}
                if mode == "eager":
                    eager_refs[T0] = ref
                else:
                    row["ref_vs_eager_ref"] = float((ref - eager_refs[T0]).abs().max())
                for vname in ["prefill-style", "offset-aware"]:
                    prefix(T0)
                    ip = torch.arange(T0, T0 + VERIFY_S, device=DEVICE)
                    if vname == "offset-aware":
                        bm = build_offset_mask(T0, VERIFY_S, kv_len, DEVICE)
                    else:
                        bm = create_block_mask_eager(G.causal_mask, 1, 1, VERIFY_S, kv_len, device=DEVICE, _compile=False)
                    out = fwd(bm, toks[:, T0:T0 + VERIFY_S], ip)[0]
                    err = (out - ref).abs().amax(-1)
                    pr, po = torch.softmax(ref.double(), -1), torch.softmax(out.double(), -1)
                    row[vname] = {
                        "max_abs_logit_err": float(err.max()),
                        "per_pos_err": [float(e) for e in err],
                        "wrong_positions": int((err > 1e-3).sum()),
                        "max_tv_t1": float(0.5 * (pr - po).abs().sum(-1).max()),
                        "argmax_changed": int((ref.argmax(-1) != out.argmax(-1)).sum()),
                    }
                row["seconds"] = round(time.perf_counter() - t0, 2)
                res.append(row)
                print(f"  verify {mode:8s} off {T0:3d}: prefill-style err {row['prefill-style']['max_abs_logit_err']:.2e}"
                      f" ({row['prefill-style']['wrong_positions']} pos, TV {row['prefill-style']['max_tv_t1']:.3f})"
                      f"  offset-aware err {row['offset-aware']['max_abs_logit_err']:.2e}  [{row['seconds']}s]")
    torch._dynamo.reset()
    return res


# ---------------------------------------------------------------- 3. toy exactness

class DummyBlockMask:
    BLOCK_SIZE = (128, 128)
    mask_mod = None
    seq_lengths = None

    def __getitem__(self, idx):
        return DummyBlockMask()


class ToyTrigram:
    """Logits depend on the previous two tokens; buf plays the KV cache (written at input_pos)."""

    def __init__(self, table, max_seq=64, drop_fill=False):
        self.table = table
        self.max_seq_length = max_seq
        self.buf = torch.zeros(max_seq, dtype=torch.long)
        self.drop_fill = drop_fill
        self.fills = 0

    def __call__(self, *args):
        x, input_pos = args[-2], args[-1]
        x = x.reshape(-1).long()
        pos = input_pos.reshape(-1).long()
        if len(args) == 2 and x.numel() == 1:  # only speculative_decode's draft-cache fill
            self.fills += 1
            if self.drop_fill:
                return None
        self.buf[pos] = x
        prev = self.buf[(pos - 1).clamp(min=0)]
        return self.table[prev, x].unsqueeze(0)


def chi_square(counts: torch.Tensor, probs: torch.Tensor):
    n = counts.sum()
    exp = probs * n
    big = exp >= 5
    obs_b = torch.cat([counts[big], counts[~big].sum().view(1)])
    exp_b = torch.cat([exp[big], exp[~big].sum().view(1)])
    keep = exp_b > 0
    chi2 = float(((obs_b[keep] - exp_b[keep]) ** 2 / exp_b[keep]).sum())
    df = int(keep.sum()) - 1
    p = float(torch.special.gammaincc(torch.tensor(df / 2, dtype=torch.float64), torch.tensor(chi2 / 2, dtype=torch.float64)))
    tv = float(0.5 * (counts / n - probs).abs().sum())
    return {"chi2": round(chi2, 1), "df": df, "p_value": p, "tv": tv}


def exact_joint(probs_tab, a, b):
    """probs_tab[x, y] = target next-token distribution after (x, y). Joint of (t2, t3, t4) after prompt (a, b)."""
    p1 = probs_tab[a, b]
    V = p1.numel()
    joint = torch.zeros(V, V, V, dtype=torch.float64)
    for t1 in range(V):
        p2 = probs_tab[b, t1]
        for t2 in range(V):
            p3 = probs_tab[t1, t2]
            joint[t2] += p1[t1] * p2[t2] * p3[:, None] * probs_tab[t2]
    return joint.reshape(-1)


def part_toy():
    G.create_block_mask = lambda *a, **k: DummyBlockMask()
    set_shims(True, None)
    g = torch.Generator().manual_seed(SEED)
    V = TOY_V
    t_logits = 2.0 * torch.randn(V, V, V, generator=g)
    d_logits = t_logits + 1.0 * torch.randn(V, V, V, generator=g)
    prompt = torch.tensor([3, 5], dtype=torch.int)
    res = {"vocab": V, "trials": TOY_TRIALS, "runs": [], "greedy": []}
    with torch.no_grad():
        for sname, sk in TOY_SAMPLINGS.items():
            pt = G.logits_to_probs(t_logits, **sk).double()
            pd = G.logits_to_probs(d_logits, **sk).double()
            pt, pd = pt / pt.sum(-1, keepdim=True), pd / pd.sum(-1, keepdim=True)
            exact = exact_joint(pt, 3, 5)
            alpha = float(torch.minimum(pt, pd).sum(-1).mean())
            for label, probs_tab in [("target-only", pt), ("draft-only", pd)]:
                torch.manual_seed(SEED)
                cnt = torch.zeros(V ** 3, dtype=torch.float64)
                cum = probs_tab.cumsum(-1)
                for _ in range(TOY_TRIALS):
                    s = [3, 5]
                    for _ in range(4):
                        s.append(int(torch.searchsorted(cum[s[-2], s[-1]], torch.rand(1, dtype=torch.float64)).clamp(max=V - 1)))
                    cnt[s[3] * V * V + s[4] * V + s[5]] += 1
                r = {"sampling": sname, "method": label, **chi_square(cnt, exact), "mean_alpha": alpha}
                res["runs"].append(r)
                print(f"  toy {sname:9s} {label:22s} chi2 {r['chi2']:7.1f}/{r['df']} p {r['p_value']:.3f} TV {r['tv']:.4f}")
            for k, drop in [(1, False), (3, False), (1, True)]:
                tgt = ToyTrigram(t_logits)
                dft = ToyTrigram(d_logits, drop_fill=drop)
                torch.manual_seed(SEED + 1)
                cnt = torch.zeros(V ** 3, dtype=torch.float64)
                acc = [0] * (k + 1)
                t0 = time.perf_counter()
                for _ in range(TOY_TRIALS):
                    seq, c = speculative_generate(tgt, dft, prompt, 4, k, sk)
                    seq = seq.tolist()
                    cnt[seq[3] * V * V + seq[4] * V + seq[5]] += 1
                    for i, ci in enumerate(c):
                        acc[i] += ci
                label = f"speculative k={k}" + (" no fill" if drop else "")
                r = {"sampling": sname, "method": label, "k": k, "drop_fill": drop, **chi_square(cnt, exact),
                     "mean_alpha": alpha, "tokens_per_call": mean_tokens_per_call(acc), "accept_counts": acc,
                     "fills": dft.fills, "seconds": round(time.perf_counter() - t0, 1)}
                res["runs"].append(r)
                print(f"  toy {sname:9s} {label:22s} chi2 {r['chi2']:7.1f}/{r['df']} p {r['p_value']:.3f} TV {r['tv']:.4f}"
                      f" tok/call {r['tokens_per_call']:.3f} fills {dft.fills} [{r['seconds']}s]")
        # draft-cache fill: only matters when decoding continues after an all-accepted call
        res["fill"] = []
        sk = TOY_SAMPLINGS["t1.0"]
        for k in [1, 3]:
            for drop in [False, True]:
                torch.manual_seed(SEED + 2)
                acc = [0] * (k + 1)
                for _ in range(TOY_LONG_TRIALS):
                    _, c = speculative_generate(ToyTrigram(t_logits), ToyTrigram(d_logits, drop_fill=drop), prompt,
                                                TOY_LONG_TOKENS, k, sk)
                    acc = [a + b for a, b in zip(acc, c)]
                r = {"k": k, "drop_fill": drop, "tokens": TOY_LONG_TOKENS, "trials": TOY_LONG_TRIALS,
                     "tokens_per_call": mean_tokens_per_call(acc), "all_accepted_share": acc[-1] / sum(acc)}
                res["fill"].append(r)
                print(f"  toy fill k={k} drop={drop}: {r['tokens_per_call']:.3f} tok/call")
        # greedy: every prompt pair must reproduce target-only argmax decoding
        for k in [1, 3]:
            mism = 0
            for a in range(V):
                for b in range(V):
                    tgt, dft = ToyTrigram(t_logits), ToyTrigram(d_logits)
                    seq, _ = speculative_generate(tgt, dft, torch.tensor([a, b], dtype=torch.int), 12, k,
                                                  dict(temperature=0.0, top_k=None))
                    ref = [a, b]
                    for _ in range(12):
                        ref.append(int(t_logits[ref[-2], ref[-1]].argmax()))
                    mism += int(seq.tolist() != ref)
            res["greedy"].append({"k": k, "prompts": V * V, "mismatches": mism})
            print(f"  toy greedy k={k}: {mism} / {V * V} sequences differ from target argmax decoding")
    G.create_block_mask = _upstream["create_block_mask"]
    set_shims(False, None)
    return res


# ---------------------------------------------------------------- 4/5. real models

def make_prompts(target):
    prompts = []
    with torch.no_grad():
        for i in range(NUM_PROMPTS):
            torch.manual_seed(1000 + i)
            bos = torch.tensor([1], dtype=torch.int, device=DEVICE)
            s = plain_generate(target, bos, PROMPT_LEN - 1, SAMPLINGS["t1.0"])
            prompts.append(s.to(torch.int))
    return prompts


def per_position_alpha(target, draft, seq, sampling):
    """sum(min(p, q)) at every generated position, from full-sequence (prefill) forwards."""
    x = seq[:-1].view(1, -1)
    S = x.shape[-1]
    ip = torch.arange(S, device=DEVICE)
    out = []
    for m in (target, draft):
        mk = create_block_mask_eager(G.causal_mask, 1, 1, S, m.max_seq_length, device=DEVICE, _compile=False)
        out.append(G.logits_to_probs(m(mk, x, ip)[0].float(), **sampling))
    a = torch.minimum(out[0], out[1]).sum(-1)
    return a[PROMPT_LEN - 1:].double()


def predicted_tokens_per_call(alphas: list[torch.Tensor], k: int):
    """iid formula with the mean acceptance, and the position-wise sum_j prod_{i<j} a_{t+i}."""
    a_bar = float(torch.cat(alphas).mean())
    iid = (k + 1) if a_bar >= 1 else (1 - a_bar ** (k + 1)) / (1 - a_bar)
    vals = []
    for a in alphas:
        for t in range(len(a) - k):
            prods = torch.cumprod(a[t:t + k], 0)
            vals.append(1 + float(prods.sum()))
    return a_bar, iid, sum(vals) / len(vals)


def timed(fn):
    sync()
    t0 = time.perf_counter()
    out = fn()
    sync()
    return out, time.perf_counter() - t0


def part_real(models, prompts, vocab):
    target = models[TARGET]
    set_shims(True, verify_forward_fixed)
    res = {"plain": [], "spec": [], "alpha": [], "self": [], "samples": {}}
    with torch.no_grad():
        # warm up compiled create_block_mask and caches
        for d in DRAFTS:
            speculative_generate(target, models[d], prompts[0], 16, 2, SAMPLINGS["t1.0"])
        plain_generate(target, prompts[0], 16, SAMPLINGS["t1.0"])
        for sname, sk in SAMPLINGS.items():
            times, seqs = [], []
            for i, pr in enumerate(prompts):
                torch.manual_seed(SEED + i)
                s, dt = timed(lambda: plain_generate(target, pr, NEW_TOKENS, sk))
                times.append(dt)
                seqs.append(s)
            r = {"sampling": sname, "tokens_per_s": NUM_PROMPTS * NEW_TOKENS / sum(times)}
            res["plain"].append(r)
            res["samples"][sname] = detok(vocab, seqs[0].tolist())[:400]
            print(f"  plain {sname:12s} {r['tokens_per_s']:.1f} tok/s")
        plain_tps = {r["sampling"]: r["tokens_per_s"] for r in res["plain"]}
        for d in DRAFTS:
            draft = models[d]
            for sname, sk in SAMPLINGS.items():
                alpha_seqs = []
                for k in KS:
                    acc = [0] * (k + 1)
                    tt = 0.0
                    for i, pr in enumerate(prompts):
                        torch.manual_seed(SEED + i)
                        (s, c), dt = timed(lambda: speculative_generate(target, draft, pr, NEW_TOKENS, k, sk))
                        tt += dt
                        acc = [a + b for a, b in zip(acc, c)]
                        if k == ALPHA_K:
                            alpha_seqs.append(s)
                    tps = NUM_PROMPTS * NEW_TOKENS / tt
                    r = {"draft": d, "sampling": sname, "k": k, "tokens_per_call": mean_tokens_per_call(acc),
                         "accept_counts": acc, "tokens_per_s": tps, "speedup": tps / plain_tps[sname]}
                    res["spec"].append(r)
                    print(f"  spec {d} {sname:12s} k={k}: {r['tokens_per_call']:.2f} tok/call  {tps:.1f} tok/s"
                          f"  x{r['speedup']:.2f}")
                alphas = [per_position_alpha(target, draft, s, sk) for s in alpha_seqs]
                for k in KS:
                    a_bar, iid, pos = predicted_tokens_per_call(alphas, k)
                    meas = next(r["tokens_per_call"] for r in res["spec"]
                                if r["draft"] == d and r["sampling"] == sname and r["k"] == k)
                    res["alpha"].append({"draft": d, "sampling": sname, "k": k, "mean_alpha": a_bar,
                                         "pred_iid": iid, "pred_positionwise": pos, "measured": meas})
                a_all = torch.cat(alphas)
                print(f"  alpha {d} {sname}: mean {float(a_all.mean()):.3f}, P(a<0.5) {float((a_all < 0.5).double().mean()):.3f}")
        # self-drafting: acceptance should be 1 up to numerics
        for prec, m in [("bf16", target), ("fp32", models[TARGET + "-fp32"])]:
            for sname, sk in SAMPLINGS.items():
                acc = [0] * (ALPHA_K + 1)
                for i, pr in enumerate(prompts):
                    torch.manual_seed(SEED + i)
                    _, c = speculative_generate(m, m, pr, NEW_TOKENS, ALPHA_K, sk)
                    acc = [a + b for a, b in zip(acc, c)]
                calls = sum(acc)
                r = {"precision": prec, "sampling": sname, "k": ALPHA_K, "tokens_per_call": mean_tokens_per_call(acc),
                     "calls_with_rejection": calls - acc[-1], "calls": calls}
                res["self"].append(r)
                print(f"  self {prec} {sname:12s}: {r['tokens_per_call']:.3f} tok/call, "
                      f"{r['calls_with_rejection']}/{calls} calls rejected a token")
    set_shims(False, None)
    return res


def part_step_costs(models):
    """Per-step eager bf16 costs at position ~100 (median of TIMING_REPS)."""
    res = {}
    sk = SAMPLINGS["t1.0"]
    pos = 100
    with torch.no_grad():
        for name in DRAFTS + [TARGET]:
            m = models[name]
            full = create_block_mask_eager(G.causal_mask, 1, 1, m.max_seq_length, m.max_seq_length, device=DEVICE, _compile=False)
            x1 = torch.tensor([[5]], dtype=torch.int, device=DEVICE)
            ip1 = torch.tensor([pos], device=DEVICE)

            def step():
                return G.decode_one_token(m, x1, ip1, full, **sk)

            ts = []
            for r in range(5 + TIMING_REPS):
                _, dt = timed(step)
                if r >= 5:
                    ts.append(dt)
            row = {"params_M": m.n_params / 1e6, "layers": m.config.n_layer, "decode_step_ms": 1e3 * sorted(ts)[len(ts) // 2]}
            if name == TARGET:
                for S in sorted({k + 1 for k in KS}):
                    x = torch.full((1, S), 5, dtype=torch.int, device=DEVICE)
                    ip = torch.arange(pos, pos + S, device=DEVICE)
                    ts = []
                    for r in range(5 + TIMING_REPS):
                        _, dt = timed(lambda: G.logits_to_probs(verify_forward_fixed(m, x, ip)[0], **sk))
                        if r >= 5:
                            ts.append(dt)
                    row[f"verify_{S}_ms"] = 1e3 * sorted(ts)[len(ts) // 2]
                ts = []
                for r in range(10):
                    _, dt = timed(lambda: build_offset_mask(pos + r, 5, m.max_seq_length, DEVICE))
                    ts.append(dt)
                row["offset_mask_build_ms"] = 1e3 * sorted(ts)[len(ts) // 2]
            res[name] = row
            print(f"  cost {name}: {row}")
    return res


# ---------------------------------------------------------------- report

def write_md(R):
    L = ["# 0014 results", ""]
    L += ["## 1. Upstream generate(draft_model=...)", "", "| attempt | result | where |", "|---|---|---|"]
    for r in R["upstream"]:
        L.append(f"| {r['attempt']} | {'ok' if r['ok'] else r['error']} | {('`' + r['where'] + '`') if 'where' in r else ''} |")
    L += ["", "## 2. Verify-forward mask vs sequential decode (stories110M fp32, 9 queries)", "",
          "| mode | positions | prefill-style max logit err | wrong positions | max TV (T=1) | argmax changed | offset-aware max err |",
          "|---|---|---|---|---|---|---|"]
    for r in R["verify"]:
        a, b = r["prefill-style"], r["offset-aware"]
        L.append(f"| {r['mode']} | {r['positions'][0]}–{r['positions'][1]} | {a['max_abs_logit_err']:.2e} | {a['wrong_positions']}/9 | "
                 f"{a['max_tv_t1']:.3f} | {a['argmax_changed']} | {b['max_abs_logit_err']:.2e} |")
    t = R["toy"]
    L += ["", f"## 3. Toy trigram exactness (V={t['vocab']}, {t['trials']} trials, joint of 3 tokens)", "",
          "| sampling | method | chi2 / df | p | TV | tokens/call | draft fills |", "|---|---|---|---|---|---|---|"]
    for r in t["runs"]:
        L.append(f"| {r['sampling']} | {r['method']} | {r['chi2']} / {r['df']} | {r['p_value']:.3f} | {r['tv']:.4f} | "
                 f"{r.get('tokens_per_call', float('nan')):.3f} | {r.get('fills', '')} |")
    L += ["", f"Draft-cache fill ablation (t1.0, {TOY_LONG_TOKENS} new tokens, {TOY_LONG_TRIALS} trials):", "",
          "| k | fill | tokens/call | all-accepted calls |", "|---|---|---|---|"]
    L += [f"| {r['k']} | {'dropped' if r['drop_fill'] else 'upstream'} | {r['tokens_per_call']:.3f} | {r['all_accepted_share']:.3f} |"
          for r in t["fill"]]
    L += [""] + [f"- greedy k={g['k']}: {g['mismatches']} / {g['prompts']} sequences differ from target argmax decoding"
                 for g in t["greedy"]]
    rr = R["real"]
    L += ["", "## 4. Acceptance on TinyStories (target stories110M, bf16, compiled flex_attention)", "",
          "| draft | sampling | k | tokens/call | pred. iid (mean a) | pred. position-wise | mean a | tok/s | speedup vs plain |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rr["spec"]:
        a = next(x for x in rr["alpha"] if x["draft"] == r["draft"] and x["sampling"] == r["sampling"] and x["k"] == r["k"])
        L.append(f"| {r['draft']} | {r['sampling']} | {r['k']} | {r['tokens_per_call']:.2f} | {a['pred_iid']:.2f} | "
                 f"{a['pred_positionwise']:.2f} | {a['mean_alpha']:.3f} | {r['tokens_per_s']:.1f} | {r['speedup']:.2f}× |")
    L += ["", "| plain decoding | tok/s |", "|---|---|"] + [f"| {r['sampling']} | {r['tokens_per_s']:.1f} |" for r in rr["plain"]]
    L += ["", "### Self-drafting (draft = target, k = 4)", "", "| precision | sampling | tokens/call | calls with a rejection |", "|---|---|---|---|"]
    L += [f"| {r['precision']} | {r['sampling']} | {r['tokens_per_call']:.3f} | {r['calls_with_rejection']} / {r['calls']} |" for r in rr["self"]]
    L += ["", "## 5. bf16 per-step costs, compiled flex_attention (position 100, median)", "", "| model | params | layers | decode step ms | other |", "|---|---|---|---|---|"]
    for n, c in R["costs"].items():
        other = ", ".join(f"{k} {v:.2f}" for k, v in c.items() if k not in ("params_M", "layers", "decode_step_ms"))
        L.append(f"| {n} | {c['params_M']:.1f}M | {c['layers']} | {c['decode_step_ms']:.2f} | {other} |")
    L += ["", "## Samples (plain decoding, prompt 0)", ""] + [f"- {k}: {v!r}" for k, v in rr["samples"].items()]
    (HERE / "results.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    t_start = time.time()
    torch.manual_seed(SEED)
    for n in CKPTS + ["tokenizer"]:
        if not (CKPT_DIR / f"{n}.bin").exists():
            raise SystemExit(f"missing {CKPT_DIR / (n + '.bin')}; see README for the download commands")
    R = {"meta": {"gpu": GPU_INFO, "python": platform.python_version(), "torch": torch.__version__,
                  "platform": platform.platform(), "seed": SEED,
                  "checkpoints": {n: sha256(CKPT_DIR / f"{n}.bin")[:16] for n in CKPTS}}}
    vocab = load_vocab()
    models = {n: load_llama2c(n, torch.bfloat16) for n in CKPTS}
    models[TARGET + "-fp32"] = load_llama2c(TARGET, torch.float32)
    R["meta"]["models"] = {n: {"params_M": round(m.n_params / 1e6, 2), "n_layer": m.config.n_layer, "dim": m.config.dim,
                               "block_size": m.config.block_size} for n, m in models.items()}
    print("models", R["meta"]["models"])
    prompts = make_prompts(models[TARGET])
    R["meta"]["prompts"] = [detok(vocab, p.tolist()) for p in prompts]
    print("prompts", R["meta"]["prompts"])

    print("1. upstream generate()")
    R["upstream"] = part_upstream(models[TARGET], models[DRAFTS[0]], prompts)
    print("2. verify mask")
    t0 = time.time()
    R["verify"] = part_verify_mask(models[TARGET + "-fp32"])
    print(f"   {time.time() - t0:.0f}s")
    print("3. toy exactness")
    t0 = time.time()
    R["toy"] = part_toy()
    print(f"   {time.time() - t0:.0f}s")
    # Eager flex_attention runs an unfused, vmap-based reference path (~20 tok/s for stories110M);
    # compile just the attention op so parts 4-5 measure the model, not that fallback.
    M.flex_attention = torch.compile(M.flex_attention)
    print("4. real models")
    t0 = time.time()
    R["real"] = part_real(models, prompts, vocab)
    print(f"   {time.time() - t0:.0f}s")
    print("5. step costs")
    R["costs"] = part_step_costs(models)
    R["meta"]["wall_seconds"] = round(time.time() - t_start, 1)
    R["meta"]["peak_memory_mib"] = gpu_guard.peak_memory_mib()
    (HERE / "results.json").write_text(json.dumps(R, indent=1), encoding="utf-8")
    write_md(R)
    print(f"done in {R['meta']['wall_seconds']}s")


if __name__ == "__main__":
    main()
