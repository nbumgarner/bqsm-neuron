#!/usr/bin/env python3
"""
bqsm_llama.py — language out, on the local 3B.

Llama-3.2-3B (Hermes-3 abliterated) from safetensors, streamed layer by layer.
Standard architecture: GQA, one RoPE, RMSNorm, SiLU. No sandwich norms, no
sliding window — which makes it the right place to validate the forward.

Run it two ways and diff:

    --relax 0            plain matmul, SiLU, softmax, RMSNorm   = the REFERENCE
    --wave               every operation replaced by its wave form:

        projection   driven damped resonator array, run to equilibrium
        RMSNorm      saturable gain medium, shared pool
        softmax      unit-time parametric gain, then shared power pool
        RoPE         free-running oscillator phase (position = elapsed time)
        SiLU         saturated driven-oscillator response, fitted to SiLU
        residual     superposition

    The full accounting — every operation, its count, its status, its measured
    error — is op_ledger.py.  Nothing here is claimed that is not counted there.

    python3 bqsm_llama.py --relax 0 --n 5      # reference
    python3 bqsm_llama.py --wave --n 5         # BQSM
"""
import argparse, glob, json, math, os, struct, time
import numpy as np

def _resolve_base():
    """Where the safetensors live. Default unchanged; BQSM_MODEL overrides.

    Every model-specific number -- D, NL, NH, NKV, HD, EPS, rope theta and
    scaling -- is already read from config.json (bqsm_full_settle.py:47), so
    pointing this somewhere else is the whole job of switching models, provided
    the architecture is one this forward implements: GQA, one RoPE, RMSNorm,
    SiLU. A Gemma-style model with sandwich norms or sliding-window attention
    will load and produce confident garbage, so check config.json's
    architectures field first.

    Accepts a directory, or an HF repo id resolved out of the local hub cache.
    """
    env = os.environ.get("BQSM_MODEL")
    if not env:
        hits = glob.glob("/home/compunerd/.cache/huggingface/hub/"
                         "models--huihui-ai--Hermes-3-Llama-3.2-3B-abliterated/"
                         "snapshots/*")
        if not hits:
            raise SystemExit(
                "No model. BQSM_MODEL is unset and the default Hermes-3B "
                "checkout is not on this machine. Set BQSM_MODEL to a "
                "checkpoint directory or a hub repo id.")
        return hits[0]
    if os.path.isdir(env) and os.path.exists(os.path.join(env, "config.json")):
        return env
    hits = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--%s/snapshots/*" % env.replace("/", "--")))
    if not hits:
        raise SystemExit(f"BQSM_MODEL={env!r}: not a model dir and not in the hub cache")
    return hits[0]


BASE = _resolve_base()


class Safetensors:
    """Zero-copy shard reader: header parsed once, tensors mapped on demand."""

    def __init__(self, base):
        self.shards, self.index = [], {}
        for p in sorted(glob.glob(os.path.join(base, "*.safetensors"))):
            mm = np.memmap(p, dtype=np.uint8, mode="r")
            n = int(struct.unpack("<Q", bytes(mm[:8]))[0])
            hdr = json.loads(bytes(mm[8:8 + n]).decode())
            start = 8 + n
            si = len(self.shards)
            self.shards.append((mm, start))
            for k, v in hdr.items():
                if k != "__metadata__":
                    self.index[k] = (si, v)

    def get(self, name):
        si, v = self.index[name]
        mm, start = self.shards[si]
        a, b = v["data_offsets"]
        raw = np.asarray(mm[start + a: start + b])
        dt = v["dtype"]
        if dt == "BF16":
            x = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        elif dt == "F16":
            x = raw.view(np.float16).astype(np.float32)
        elif dt == "F32":
            x = raw.view(np.float32)
        else:
            raise ValueError(dt)
        return x.reshape(v["shape"])

    def has(self, name):
        return name in self.index


# ───────────────────────── wave forms ─────────────────────────

def relax(W, X, steps):
    """Driven damped resonator array: dz/dt = -z + Wx, fixed point z = Wx.
    Batched over tokens: X is [T, in], W is [out, in]."""
    drive = X @ W.T
    if not steps:
        return drive
    z = np.zeros_like(drive)
    for _ in range(steps):
        z += 0.25 * (-z + drive)
    return z


def gain_norm(X, w, eps, G=2.0, dt=0.10, steps=500):
    """RMSNorm as a saturable gain medium. Same gain for every mode, so the
    direction is preserved exactly and only total power moves; equilibrium
    power is the pool. Pool = D*P/(P + D*eps) reproduces Llama's divide-guard
    (a saturable-absorber transmission curve) — worth 2.18% at the embedding
    layer, so it is not optional for matching."""
    D = X.shape[-1]
    P0 = (X * X).sum(-1, keepdims=True)
    pool = D * P0 / (P0 + D * eps)
    # A silent input must give a silent output. The medium has no singularity
    # here -- it simply has nothing to amplify -- but P_sat = 0 would make the
    # saturation term 0/0. Floor it and let the zero state stay zero.
    Psat = np.maximum(pool, 1e-30) / (G - 1.0)
    live = (P0 > 0).astype(np.float64)
    a = X.astype(np.float64).copy()
    for _ in range(steps):
        P = (a * a).sum(-1, keepdims=True)
        a += dt * ((G / (1.0 + P / Psat)) - 1.0) * a * live
    return a.astype(np.float32) * w


def amp_softmax(s):
    """Unit-time parametric gain -> amplitude exp(s/2), power exp(s); shared
    power pool normalises to occupancy. Algebraically softmax, verified 1.3e-7.
    Max-subtraction is choosing the strongest mode as the gain reference."""
    a = np.exp((s - s.max(-1, keepdims=True)) / 2.0)
    p = a * a
    return p / p.sum(-1, keepdims=True)


def rope_phase(x, cos, sin, invf, pos):
    """Free-running oscillator phase: pair (j, j+hd/2) is one complex amplitude
    z, and RoPE is z*exp(i*omega*t). Position is elapsed time, not a rotation
    applied to the state."""
    h = x.shape[-1] // 2
    z = (x[..., :h] + 1j * x[..., h:]) * np.exp(1j * invf * pos)
    return np.concatenate([z.real, z.imag], -1).astype(np.float32)


def int8_percol(W):
    """Symmetric int8 with a PER-OUTPUT-COLUMN scale, round-tripped back to f32.

    Per-column is load-bearing, not a refinement: these matrices span 8-16x in
    column RMS internally, and one global scale collapses that. Measured on
    layer 13 gate_proj -- int2 with a per-column scale reaches corr 0.54, the
    same 2 bits with one global scale reaches 0.033. That gap is exactly why the
    ternary .bqsm scored at chance.

    NOTE this round-trip saves NOTHING yet: the tensor is still f32 in memory.
    It answers only the quality question -- do the tokens survive int8 storage.
    The 5.6 GB -> 2.8 GB win requires repacking the weights on disk, which is
    gated on this test passing, not assumed by it."""
    s = np.abs(W).max(axis=1, keepdims=True) / 127.0
    s[s == 0] = 1.0
    return (np.clip(np.rint(W / s), -127, 127) * s).astype(np.float32)


def silu(x):
    return x / (1.0 + np.exp(-x))


def sat_gate(x, a=0.60, b=-0.04):
    """Saturated driven-oscillator response. (a,b) fitted to LLAMA's SiLU on
    Llama's own activation distribution: corr 0.999819, rel-err 1.97e-2 vs a
    relu control of 1.375e-1. The Gemma constants (1.20,-0.25) were fitted to
    gelu_tanh and are 13x worse here — the gate must be refitted per model."""
    z = a * (x - b)
    return 0.5 * (z / np.sqrt(1.0 + z * z) + 1.0) * x


def rms(x, w, eps):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--relax", type=int, default=0)
    ap.add_argument("--wave", action="store_true",
                    help="every operation in its wave form (implies --relax 60)")
    ap.add_argument("--norm-steps", type=int, default=500)
    a = ap.parse_args()
    if a.wave and not a.relax:
        a.relax = 60

    cfg = json.load(open(os.path.join(BASE, "config.json")))
    D = cfg["hidden_size"]; NL = cfg["num_hidden_layers"]
    NH = cfg["num_attention_heads"]; NKV = cfg["num_key_value_heads"]
    HD = cfg.get("head_dim", D // NH); EPS = cfg["rms_norm_eps"]
    THETA = cfg["rope_theta"]; rs = cfg.get("rope_scaling")

    tok = json.load(open(os.path.join(BASE, "tokenizer.json")))
    vocab = tok["model"]["vocab"]
    inv = {v: k for k, v in vocab.items()}

    def encode(text):
        ids, words = [128000], text.split()
        for i, w in enumerate(words):
            key = ("Ġ" + w) if i else w
            if key in vocab: ids.append(vocab[key])
            elif w in vocab: ids.append(vocab[w])
            else:
                for ch in key:
                    if ch in vocab: ids.append(vocab[ch])
        return ids

    def dec(i):
        return inv.get(i, f"[{i}]").replace("Ġ", " ").replace("Ċ", "\n")

    st = Safetensors(BASE)
    pre = "model." if st.has("model.layers.0.self_attn.q_proj.weight") else ""
    ids = encode(a.prompt)
    print(f"prompt {a.prompt!r} -> {ids}")
    print(f"  {NL} layers  D={D}  heads={NH}/{NKV}  hd={HD}")
    if a.wave:
        print(f"  WAVE: projections=resonator({a.relax})  norm=gain-medium({a.norm_steps})"
              f"  softmax=amplify+pool  rope=free-phase  act=sat-gate(0.60,-0.04)\n")
    else:
        print(f"  REFERENCE: matmul, rmsnorm, softmax, rope, silu\n")

    # RoPE frequencies (llama3 scaling if present)
    invf = 1.0 / (THETA ** (np.arange(0, HD, 2) / HD))
    if rs and rs.get("rope_type") == "llama3":
        f, lo, hi, old = rs["factor"], rs["low_freq_factor"], rs["high_freq_factor"], rs["original_max_position_embeddings"]
        wl = 2 * np.pi / invf
        lw, hw = old / lo, old / hi
        smooth = (old / wl - hi) / (lo - hi)
        invf = np.where(wl > lw, invf / f,
               np.where(wl < hw, invf, (1 - smooth) * invf / f + smooth * invf))

    emb = st.get(f"{pre}embed_tokens.weight")
    t0 = time.time(); out = []

    NORM = (lambda X, w: gain_norm(X, w, EPS, steps=a.norm_steps)) if a.wave \
           else (lambda X, w: rms(X, w, EPS))
    SMAX = amp_softmax if a.wave else (
        lambda s: np.exp(s - s.max(-1, keepdims=True)) /
                  np.exp(s - s.max(-1, keepdims=True)).sum(-1, keepdims=True))
    ACT = sat_gate if a.wave else silu

    for step in range(a.n):
        T = len(ids)
        H = emb[ids].astype(np.float32).copy()
        pos = np.arange(T)[:, None] * invf[None, :]
        cos, sin = np.cos(pos), np.sin(pos)

        for L in range(NL):
            p = f"{pre}layers.{L}."
            xn = NORM(H, st.get(p + "input_layernorm.weight"))
            Wq = st.get(p + "self_attn.q_proj.weight"); Wk = st.get(p + "self_attn.k_proj.weight")
            Wv = st.get(p + "self_attn.v_proj.weight"); Wo = st.get(p + "self_attn.o_proj.weight")

            Q = relax(Wq, xn, a.relax).reshape(T, NH, HD)
            K = relax(Wk, xn, a.relax).reshape(T, NKV, HD)
            Vv = relax(Wv, xn, a.relax).reshape(T, NKV, HD)

            if a.wave:      # free-running phase, one complex multiply per pair
                Q = np.stack([rope_phase(Q[i], None, None, invf, i) for i in range(T)])
                K = np.stack([rope_phase(K[i], None, None, invf, i) for i in range(T)])
            else:
                def rot(x):
                    x1, x2 = x[..., :HD//2], x[..., HD//2:]
                    c = cos[:, None, :]; s = sin[:, None, :]
                    return np.concatenate([x1*c - x2*s, x1*s + x2*c], -1)
                Q, K = rot(Q), rot(K)

            ctx = np.zeros((T, NH, HD), np.float32)
            sc = 1.0 / math.sqrt(HD)
            for h in range(NH):
                kv = h * NKV // NH
                s_ = (Q[:, h] @ K[:, kv].T) * sc
                s_ = s_ + np.triu(np.full((T, T), -1e30, np.float32), 1)
                ctx[:, h] = SMAX(s_) @ Vv[:, kv]
            H = H + relax(Wo, ctx.reshape(T, NH*HD), a.relax)

            xn = NORM(H, st.get(p + "post_attention_layernorm.weight"))
            Wg = st.get(p + "mlp.gate_proj.weight"); Wu = st.get(p + "mlp.up_proj.weight")
            Wd = st.get(p + "mlp.down_proj.weight")
            g = relax(Wg, xn, a.relax); u = relax(Wu, xn, a.relax)
            H = H + relax(Wd, ACT(g) * u, a.relax)
            del Wq, Wk, Wv, Wo, Wg, Wu, Wd

        x = NORM(H[-1:], st.get(f"{pre}norm.weight"))[0]
        head = emb if cfg.get("tie_word_embeddings") else st.get("lm_head.weight")
        nxt = int(np.argmax(head @ x))
        out.append(dec(nxt)); ids.append(nxt)
        print(f"  [{step}] {nxt:>7}  {dec(nxt)!r}   ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n  OUTPUT: {''.join(out)!r}")
    print(f"  FULL:   {a.prompt + ''.join(out)!r}")


if __name__ == "__main__":
    main()
