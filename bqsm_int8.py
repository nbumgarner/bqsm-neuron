#!/usr/bin/env python3
"""
bqsm_int8.py — the settle with the weights resident.

2.82 GB of int8 with per-column scales, held in RAM, widened to f32 inside the
AVX2 registers. That single fact deletes everything else: no prefetcher, no
pinning, no MADV_WILLNEED/DONTNEED, no LRU sequential-scan pathology, no disk
after startup, no 11.3 GB allocation that can OOM the machine.

Measured on the way here:
  bf16 5.64 GB does not fit in ~4 GB -> every settle re-reads it at 553 MB/s
  int8 2.82 GB fits                  -> every settle is RAM-bound
  int8 per-column, 6 real matrices   -> W rel err 0.0105, y cosine 0.999941
  int8 through all 28 layers         -> token 12366 ' Paris', correct

SRP readout: measured and REMOVED. It was built to replace a 12,211 ms f32
vocabulary scan; the bf16 kernel does that same scan in 53-86 ms, so SRP's
overhead (512-iteration projection, argpartition over 128,256, gathering 1024
rows) now makes it SLOWER -- 82-159 ms -- as well as approximate. It also broke
token 4: with 512-bit codes the Hamming distances sit in a narrow integer band
(199-225), so ties are enormous; the true token had 884 strictly closer but
1,135 tied-or-closer, and argpartition dropped it from k=1024. Note that
bqsm_srp.rank_of measures a stable-sort position while shortlist uses
argpartition, which breaks ties arbitrarily -- so rank_of understates the risk,
and the "#57 of 128,256" figure that justified k=1024 measured the wrong thing.

Exactness-preserving elsewhere: gain_norm is replaced by its closed-form fixed
point (the medium settles to P* = pool with direction intact, so it IS rms());
KV cache and last-position-only are exact by causality.

    python3 bqsm_int8.py --build          # quantise once  (~2.82 GB cache)
    python3 bqsm_int8.py --n 5 --verify   # settle, check the golden tokens
"""
import argparse, ctypes, json, math, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqsm_full_settle as FS
from bqsm_llama import Safetensors, BASE, sat_gate, amp_softmax, rope_phase
from bqsm_full_settle import D, NL, NH, NKV, HD, EPS

HERE = os.path.dirname(os.path.abspath(__file__))
# BQSM_BLOB pairs with BQSM_MODEL (bqsm_llama._resolve_base). Keep them in step:
# the blob is quantised from one specific checkpoint and the index records that
# checkpoint's shapes, so a mismatched pair loads without error and is wrong.
CACHE = os.environ.get("BQSM_BLOB") or os.path.join(HERE, "llama3b.int8")
IDX = CACHE + ".json"
PROJ = [("Wq", "self_attn.q_proj"), ("Wk", "self_attn.k_proj"),
        ("Wv", "self_attn.v_proj"), ("Wo", "self_attn.o_proj"),
        ("Wg", "mlp.gate_proj"), ("Wu", "mlp.up_proj"), ("Wd", "mlp.down_proj")]

def _load(soname, sym, argtypes):
    """The compiled kernel if there is one, otherwise numpy.

    On x86 and on aarch64 (built via arm/build.sh) this finds a real .so and
    nothing below runs. On a machine with no compiler -- a phone in Termux
    before `pkg install clang` -- it falls back to numpy, which is slower but
    numerically identical: verified elementwise against all four C kernels,
    max relative error 5.4e-07, i.e. FMA reassociation only.

    PHOX_KERNELS=numpy forces the fallback, which is how you A/B a suspected
    SIMD bug against a reference that has no shuffles in it.
    """
    if os.environ.get("PHOX_KERNELS") != "numpy":
        try:
            lib = ctypes.CDLL(os.path.join(HERE, soname))
            getattr(lib, sym).argtypes = argtypes
            return lib
        except OSError:
            pass
    from arm.kernels_numpy import _Shim
    return _Shim()


_lib = _load("libint8.so", "int8_gemv", [ctypes.c_void_p] * 4 + [ctypes.c_int] * 2)
_lgm = _load("libint8gemm.so", "int8_gemm", [ctypes.c_void_p] * 4 + [ctypes.c_int] * 3)
_lbf = _load("libbf16.so", "bf16_gemv", [ctypes.c_void_p] * 3 + [ctypes.c_int] * 2)


def bf16_view(st, name):
    """Zero-copy uint16 view; converting embed_tokens to f32 would cost 1.58 GB
    on top of the 2.82 GB of weights, which is the OOM line on this box."""
    si, v = st.index[name]
    mm, start = st.shards[si]
    a, b = v["data_offsets"]
    return np.asarray(mm[start + a: start + b]).view(np.uint16), tuple(v["shape"])


def bf16_row(raw, shape, i):
    r = raw[i * shape[1]:(i + 1) * shape[1]]
    out = np.zeros(shape[1], np.float32)
    out.view(np.uint16)[1::2] = r
    return out[None]


def bf16_rows(raw, shape, idx):
    """Decode only the shortlisted rows of the head. 1024 x 3072 = 12.6 MB,
    against 128,256 x 3072 = 1.58 GB for the dense scan."""
    nin = shape[1]
    g = raw.reshape(shape[0], nin)[idx]
    out = np.zeros((len(idx), nin), np.float32)
    out.view(np.uint16).reshape(len(idx), nin, 2)[..., 1] = g
    return out


class _Shape:
    """SRP only needs emb.shape when the codebook is already cached."""
    def __init__(self, shape): self.shape = shape


def bf16_logits(raw, shape, z):
    nout, nin = shape
    xc = np.ascontiguousarray(z.ravel(), np.float32)
    y = np.empty(nout, np.float32)
    _lbf.bf16_gemv(raw.ctypes.data, xc.ctypes.data, y.ctypes.data, nout, nin)
    return y


def build():
    """Stream the bf16 model once, quantise per output row, write int8 + scales."""
    st = Safetensors(BASE)
    idx, off = {}, 0
    t0 = time.time()
    with open(CACHE, "wb") as f:
        for L in range(NL):
            p = f"model.layers.{L}."
            for key, nm in PROJ:
                W = st.get(p + nm + ".weight").astype(np.float32)
                s = np.maximum(np.abs(W).max(1), 1e-30) / 127.0
                q = np.clip(np.rint(W / s[:, None]), -127, 127).astype(np.int8)
                f.write(q.tobytes()); f.write(s.astype(np.float32).tobytes())
                idx[f"{L}.{key}"] = [off, list(W.shape)]
                off += q.nbytes + s.nbytes
                del W, q, s
            for key, nm in (("w1", "input_layernorm"), ("w2", "post_attention_layernorm")):
                v = st.get(p + nm + ".weight").astype(np.float32)
                f.write(v.tobytes()); idx[f"{L}.{key}"] = [off, list(v.shape)]
                off += v.nbytes
            print(f"\r  layer {L+1}/{NL}  {off/1e9:.2f} GB", end="", flush=True)
    json.dump(idx, open(IDX, "w"))
    print(f"\n  built {CACHE}  {off/1e9:.2f} GB in {time.time()-t0:.0f}s")


class Engine:
    """Weights held in one memory-mapped 2.82 GB blob (reclaimable). Nothing streams."""

    def __init__(self, invf, blob=None):
        """blob=None loads the 2.82 GB cache. Pass an existing engine's .blob to
        share it: a second stream then costs only its own KV cache (224 KB per
        token), not another copy of the weights."""
        self.invf = invf
        if blob is None:
            # memory-map, not fromfile: the 2.82 GB stays file-backed and
            # RECLAIMABLE. np.fromfile made it anonymous RSS -- the first thing
            # the OOM killer takes (it killed this engine at 18:23 on the 7 GB
            # laptop). Mapped, the kernel evicts weight pages under pressure and
            # pages them back from disk on the next read, so the engine degrades
            # instead of dying. Steady-state speed is unchanged: the forward pass
            # touches every weight each token, so the hot set stays in page cache.
            blob = np.memmap(CACHE, dtype=np.uint8, mode="r")
        self.blob, self.idx = blob, json.load(open(IDX))
        self.base = blob.ctypes.data
        self.kv = [None] * NL
        self.ids = []          # the token sequence that produced the current KV

    def W(self, L, key):
        o, (nout, nin) = self.idx[f"{L}.{key}"]
        return self.base + o, self.base + o + nout * nin, nout, nin

    def gemv(self, L, key, x):
        wa, sa, nout, nin = self.W(L, key)
        xc = np.ascontiguousarray(x.ravel(), np.float32)
        y = np.empty(nout, np.float32)
        _lib.int8_gemv(ctypes.c_void_p(wa), ctypes.c_void_p(sa),
                       xc.ctypes.data, y.ctypes.data, nout, nin)
        return y.reshape(1, nout)

    def gemm(self, L, key, X):
        """Batched: one weight load reused across every row of X."""
        wa, sa, nout, nin = self.W(L, key)
        Xc = np.ascontiguousarray(X, np.float32)
        B = Xc.shape[0]
        Y = np.empty((B, nout), np.float32)
        _lgm.int8_gemm(ctypes.c_void_p(wa), ctypes.c_void_p(sa),
                       Xc.ctypes.data, Y.ctypes.data, nout, nin, B)
        return Y

    def truncate(self, n):
        """Drop cached K/V past position n. RoPE is baked in at absolute
        position, so a cache entry is only valid where it was computed --
        append-only reuse is exact, editing earlier text is not."""
        if n <= 0:
            self.kv = [None] * NL
            return
        for L in range(NL):
            if self.kv[L] is not None:
                K, V = self.kv[L]
                self.kv[L] = (K[:n], V[:n])

    def settle_batch(self, X, start, wnorm, inject=None, g=0.0, at=14):
        """Settle T positions at once. X is (T, D) for positions start..start+T-1.

        inject/g/at implement weak coupling from a second stream: after layer
        `at`, the residual takes `x += g * inject`. g is the whole dial -- small
        values colour the state, large ones overwrite it. The model was never
        trained to receive an injected vector, so the usable range is an
        empirical question (see inner_voice.py --sweep), not an assumption."""
        T = X.shape[0]
        x = np.ascontiguousarray(X, np.float32)
        sc = 1.0 / math.sqrt(HD)
        self.deltas = [] if getattr(self, "trace", False) else None
        for L in range(NL):
            x0_ = x
            xn1 = self.norm(x, self.vec(L, "w1"))
            k = self.gemm(L, "Wk", xn1).reshape(T, NKV, HD)
            v = self.gemm(L, "Wv", xn1).reshape(T, NKV, HD)
            k = np.stack([rope_phase(k[i], None, None, self.invf, start + i)
                          for i in range(T)])
            # KV stays f32. An fp16 cache halves the footprint but forces a cast
            # back on every layer of every token, and the cast is O(context):
            # measured 1% overhead at 64 tokens, 9% at 1k, 37% at 4k. It buys
            # reach at the cost of speed in exactly the region that needs reach.
            if self.kv[L] is None:
                self.kv[L] = (k, v)
            else:
                pk, pv = self.kv[L]
                self.kv[L] = (np.concatenate([pk, k]), np.concatenate([pv, v]))
            K, V = self.kv[L]
            q = self.gemm(L, "Wq", xn1).reshape(T, NH, HD)
            q = np.stack([rope_phase(q[i], None, None, self.invf, start + i)
                          for i in range(T)])
            ctx = np.empty((T, NH, HD), np.float32)
            for hh in range(NH):
                kv = hh * NKV // NH
                sm = (q[:, hh] @ K[:, kv].T) * sc          # (T, start+T)
                for i in range(T):                          # causal: pos start+i
                    sm[i, start + i + 1:] = -1e30
                ctx[:, hh] = amp_softmax(sm) @ V[:, kv]
            a = x + self.gemm(L, "Wo", ctx.reshape(T, NH * HD))
            xn2 = self.norm(a, self.vec(L, "w2"))
            h = sat_gate(self.gemm(L, "Wg", xn2)) * self.gemm(L, "Wu", xn2)
            if self.deltas is not None:
                self.deltas.append((float(np.linalg.norm(a - x0_)),
                                    float(np.linalg.norm(self.gemm(L, "Wd", h)))))
            x = a + self.gemm(L, "Wd", h)
            if inject is not None and L == at and g != 0.0:
                x = x + (g * inject[-1:] if inject.ndim == 2 else g * inject)
            if L == at:
                # layer-`at` residual = the space the injection lives in. Captured so
                # plasticity keys/values/recall all share ONE space (aligned injection).
                self._at = x[-1].astype(np.float32).copy()
        return self.norm(x, wnorm)

    def prefill(self, ids, embed, wnorm):
        """Reuse the longest common prefix of the cached sequence, settle the
        rest in one batched pass. Returns (last state, n_reused, n_settled)."""
        n = 0
        while n < len(self.ids) and n < len(ids) and self.ids[n] == ids[n]:
            n += 1
        n = min(n, len(ids) - 1)          # always settle at least the last token
        self.truncate(n)
        new = ids[n:]
        X = np.vstack([embed(t) for t in new])
        z = self.settle_batch(X, n, wnorm)
        self.ids = list(ids)
        return z, n, len(new)          # full (T_new, D): callers slice [-1:]

    def append(self, tok, embed, wnorm, inject=None, g=0.0, at=14):
        """One generated token onto the end."""
        z = self.settle_batch(embed(tok), len(self.ids), wnorm, inject, g, at)
        self.ids.append(tok)
        return z[-1:]

    def vec(self, L, key):
        o, shp = self.idx[f"{L}.{key}"]
        return self.blob[o:o + 4 * shp[0]].view(np.float32)

    def norm(self, X, w):
        """Closed-form fixed point of the saturable gain medium: P* = pool,
        direction preserved, so a* = X*sqrt(D/(P0 + D*eps)) -- exactly rms()."""
        n = X.shape[-1]
        P0 = (X.astype(np.float32) ** 2).sum(-1, keepdims=True)
        return (X * np.sqrt(n / (P0 + n * EPS), dtype=np.float32)) * w

    def settle(self, drive, tpos, wnorm, reset=False):
        if reset:
            self.kv = [None] * NL
        x = drive
        for L in range(NL):
            x0_ = x
            xn1 = self.norm(x, self.vec(L, "w1"))
            k = self.gemv(L, "Wk", xn1).reshape(1, NKV, HD)
            v = self.gemv(L, "Wv", xn1).reshape(1, NKV, HD)
            k = rope_phase(k[0], None, None, self.invf, tpos)[None]
            # KV stays f32. An fp16 cache halves the footprint but forces a cast
            # back on every layer of every token, and the cast is O(context):
            # measured 1% overhead at 64 tokens, 9% at 1k, 37% at 4k. It buys
            # reach at the cost of speed in exactly the region that needs reach.
            if self.kv[L] is None:
                self.kv[L] = (k, v)
            else:
                pk, pv = self.kv[L]
                self.kv[L] = (np.concatenate([pk, k]), np.concatenate([pv, v]))
            K, V = self.kv[L]
            q = self.gemv(L, "Wq", xn1).reshape(1, NH, HD)
            q = rope_phase(q[0], None, None, self.invf, tpos)[None]
            ctx = np.empty((1, NH, HD), np.float32)
            sc = 1.0 / math.sqrt(HD)
            for hh in range(NH):
                kv = hh * NKV // NH
                ctx[:, hh] = amp_softmax((q[:, hh] @ K[:, kv].T) * sc) @ V[:, kv]
            a = x + self.gemv(L, "Wo", ctx.reshape(1, NH * HD))
            xn2 = self.norm(a, self.vec(L, "w2"))
            h = sat_gate(self.gemv(L, "Wg", xn2)) * self.gemv(L, "Wu", xn2)
            x = a + self.gemv(L, "Wd", h)
        return self.norm(x, wnorm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--n", type=int, default=5, help="max new tokens (cap)")
    ap.add_argument("--until-stop", action="store_true",
                    help="generate until an EOS token instead of a fixed count")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()

    if a.build or not os.path.exists(CACHE):
        build()
        if a.build:
            return

    st = Safetensors(BASE)
    tok = json.load(open(os.path.join(BASE, "tokenizer.json")))
    vocab = tok["model"]["vocab"]; inv = {v: k for k, v in vocab.items()}
    def dec(i): return inv.get(i, f"[{i}]").replace("Ġ", " ").replace("Ċ", "\n")
    ids = [128000] + [vocab[("Ġ" + w) if i else w] for i, w in enumerate(a.prompt.split())]

    # generation_config is authoritative; config.json's scalar eos is stale here
    gp = os.path.join(BASE, "generation_config.json")
    e = json.load(open(gp))["eos_token_id"] if os.path.exists(gp) else FS.CFG["eos_token_id"]
    EOS = set(e if isinstance(e, list) else [e])

    wnorm = st.get("model.norm.weight")
    ename = "model.embed_tokens.weight" if FS.CFG.get("tie_word_embeddings") else "lm_head.weight"
    eraw, eshape = bf16_view(st, "model.embed_tokens.weight")
    hraw, hshape = bf16_view(st, ename)

    def readout(zz):
        """Dense bf16 scan. SRP was measured and removed -- see module docstring."""
        return int(np.argmax(bf16_logits(hraw, hshape, zz))), None

    t0 = time.time()
    eng = Engine(FS.make_invf())
    print(f"  loaded {eng.blob.nbytes/1e9:.2f} GB int8, resident, in {time.time()-t0:.1f}s")

    tp = time.time()
    for i, tk in enumerate(ids):
        z = eng.settle(bf16_row(eraw, eshape, tk), i, wnorm, reset=(i == 0))
    print(f"  prefill {len(ids)} positions: {time.time()-tp:.2f}s\n")

    out, t0, stopped = [], time.time(), None
    for step in range(a.n):
        tr = time.time()
        nxt, rank = readout(z[-1])
        t_read = time.time() - tr
        if nxt in EOS:
            stopped = nxt
            print(f"  [{step}] {nxt:>7}  <EOS {dec(nxt)!r}>  -- stopping", flush=True)
            break
        out.append(dec(nxt)); ids.append(nxt)
        tw = time.time()
        z = eng.settle(bf16_row(eraw, eshape, nxt), len(ids) - 1, wnorm)
        print(f"  [{step}] {nxt:>7}  {dec(nxt)!r}   settle {time.time()-tw:.3f}s"
              f"   readout {t_read*1000:6.1f}ms"
              f"{f'   hamming rank {rank}/1024' if rank is not None else ''}", flush=True)
    el, n = time.time() - t0, max(len(out), 1)
    print(f"\n  {len(out)} tokens in {el:.2f}s   ({el/n:.3f}s per settle, "
          f"{n/el:.2f} tok/s)")
    print(f"  {'stopped on EOS ' + str(stopped) if stopped else 'hit the --n cap'}")
    print(f"  OUTPUT: {''.join(out)!r}")
    print(f"  FULL:   {a.prompt + ''.join(out)!r}")
    if a.verify:
        g = json.load(open(os.path.join(HERE, "golden.json")))
        exp = g["entries"].get(f"{a.prompt}|{a.n}", {}).get("tokens")
        got = ids[-a.n:]
        print(f"  golden : {exp}\n  got    : {got}\n  "
              f"{'MATCH — calculation intact' if exp == got else '*** MISMATCH ***'}")


if __name__ == "__main__":
    main()
