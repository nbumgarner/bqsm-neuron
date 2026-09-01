#!/usr/bin/env python3
"""
bqsm_full_settle.py — the ENTIRE forward as one system, one settle.

Not 28 layers run in sequence. One state vector holding every intermediate the
model computes, one block-structured coupling operator A built from all the real
weights, and one equilibrium:

        z  =  F(z ; x)  =  phi( A z  +  B x )

The next token IS the equilibrium of that system. There is no forward pass in
this description -- there is a network, and it settles.

  311 blocks   critical path 227   7,606,272 oscillators for a 6-token context

Two schedules for the one fixed point, and the difference between them is the
whole point:

  GAUSS-SEIDEL  blocks updated in place, topological order. Lands in ONE sweep,
                because the coupling is a DAG and one ordered pass walks it.
                This schedule is exactly the conventional forward pass -- which
                is the honest relationship between the two processes.

  JACOBI        every block updates simultaneously from the previous state.
                Nothing is sequenced. This is what physical oscillators do, and
                it lands in `depth` sweeps because information crosses one block
                boundary per sweep.

  Same equilibrium. Different schedule. On a CPU, Gauss-Seidel is free and
  Jacobi costs `depth` times more, because a CPU fakes simultaneity by looping.
  On hardware where the blocks genuinely move at once, that factor is 1.

Convergence is EXACT AND FINITE, not asymptotic: a feedforward network is a DAG,
so the iteration is nilpotent -- it lands on the fixed point at depth rather
than approaching it. No solver, no tolerance, no damping.

    python3 bqsm_full_settle.py --n 5              # settle, emit tokens
    python3 bqsm_full_settle.py --jacobi 2         # prove both schedules agree
"""
import argparse, json, math, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bqsm_llama import (Safetensors, BASE, gain_norm, amp_softmax, rope_phase,
                        sat_gate, rms, silu, int8_percol)

CFG = json.load(open(os.path.join(BASE, "config.json")))
D   = CFG["hidden_size"]
FF  = CFG["intermediate_size"]
NL  = CFG["num_hidden_layers"]
NH  = CFG["num_attention_heads"]
NKV = CFG["num_key_value_heads"]
HD  = CFG.get("head_dim", D // NH)
EPS = CFG["rms_norm_eps"]

# One layer's blocks and what each one reads. This IS the sparsity pattern of A.
LAYER_BLOCKS = [("xn1", D), ("q", D), ("k", NKV * HD), ("v", NKV * HD),
                ("ctx", D), ("a", D), ("xn2", D), ("g", FF), ("u", FF),
                ("h", FF), ("y", D)]
LAYER_DEPS = {"xn1": ["<in>"], "q": ["xn1"], "k": ["xn1"], "v": ["xn1"],
              "ctx": ["q", "k", "v"], "a": ["ctx", "<in>"], "xn2": ["a"],
              "g": ["xn2"], "u": ["xn2"], "h": ["g", "u"], "y": ["a", "h"]}


def build_graph(n_layers=NL):
    """The full coupling DAG: embed -> 28 layers -> final norm -> logits."""
    deps = {"embed": []}
    order = ["embed"]
    for L in range(n_layers):
        src = "embed" if L == 0 else f"L{L-1}.y"
        for name, _ in LAYER_BLOCKS:
            key = f"L{L}.{name}"
            deps[key] = [src if d == "<in>" else f"L{L}.{d}" for d in LAYER_DEPS[name]]
            order.append(key)
    deps["norm"] = [f"L{n_layers-1}.y"]; order.append("norm")
    deps["logits"] = ["norm"];           order.append("logits")
    depth = {}
    for k in order:
        depth[k] = 1 + max([depth[d] for d in deps[k]], default=0)
    return order, deps, depth


class FullSystem:
    """The whole model as one coupled system. Weights are streamed per layer in
    Gauss-Seidel (the model never sits in RAM); Jacobi holds the layers it needs
    resident, which is why it is demonstrated on a few layers rather than 28."""

    def __init__(self, st, pre, T, invf, wave=True, n_layers=NL, resident=False,
                 int8=False):
        self.st, self.pre, self.T, self.invf = st, pre, T, invf
        self.wave, self.NLay, self.int8 = wave, n_layers, int8
        self.mask = np.triu(np.full((T, T), -1e30, np.float32), 1)
        self.order, self.deps, self.depth = build_graph(n_layers)
        self.W = {}
        if resident:
            for L in range(n_layers):
                self.W[L] = self._load(L)

    def _load(self, L):
        p = f"{self.pre}layers.{L}."
        g = self.st.get
        # Norm vectors stay bf16: they are 3072 elements against 45M in the
        # projections, so quantizing them buys no bytes and only adds error.
        q = int8_percol if self.int8 else (lambda W: W)
        return dict(w1=g(p + "input_layernorm.weight"), w2=g(p + "post_attention_layernorm.weight"),
                    Wq=q(g(p + "self_attn.q_proj.weight")), Wk=q(g(p + "self_attn.k_proj.weight")),
                    Wv=q(g(p + "self_attn.v_proj.weight")), Wo=q(g(p + "self_attn.o_proj.weight")),
                    Wg=q(g(p + "mlp.gate_proj.weight")), Wu=q(g(p + "mlp.up_proj.weight")),
                    Wd=q(g(p + "mlp.down_proj.weight")))

    # ---- phi: the nonlinearities that live INSIDE the fixed point ----
    def _norm(self, X, w):
        return gain_norm(X, w, EPS, steps=400) if self.wave else rms(X, w, EPS)

    def _act(self, x):
        return sat_gate(x) if self.wave else silu(x)

    def _smax(self, s):
        if self.wave:
            return amp_softmax(s)
        e = np.exp(s - s.max(-1, keepdims=True))
        return e / e.sum(-1, keepdims=True)

    def _attend(self, q, k, v):
        T = self.T
        Q = q.reshape(T, NH, HD); K = k.reshape(T, NKV, HD); V = v.reshape(T, NKV, HD)
        if self.wave:
            Q = np.stack([rope_phase(Q[i], None, None, self.invf, i) for i in range(T)])
            K = np.stack([rope_phase(K[i], None, None, self.invf, i) for i in range(T)])
        else:
            pos = np.arange(T)[:, None] * self.invf[None, :]
            c, s = np.cos(pos)[:, None, :], np.sin(pos)[:, None, :]
            def rot(X):
                x1, x2 = X[..., :HD//2], X[..., HD//2:]
                return np.concatenate([x1*c - x2*s, x1*s + x2*c], -1)
            Q, K = rot(Q), rot(K)
        out = np.zeros((T, NH, HD), np.float32)
        sc = 1.0 / math.sqrt(HD)
        for hh in range(NH):
            kv = hh * NKV // NH
            out[:, hh] = self._smax((Q[:, hh] @ K[:, kv].T) * sc + self.mask) @ V[:, kv]
        return out.reshape(T, NH * HD)

    def rule(self, key, z, drive, W):
        """One coupling rule. Reads only other blocks -- no control flow."""
        if key == "embed":  return drive
        if key == "norm":   return self._norm(z[f"L{self.NLay-1}.y"], self.wnorm)
        if key == "logits": return z["norm"] @ self.head.T
        L, nm = key.split("."); L = int(L[1:])
        src = z["embed"] if L == 0 else z[f"L{L-1}.y"]
        w = W[L]
        if nm == "xn1": return self._norm(src, w["w1"])
        if nm == "q":   return z[f"L{L}.xn1"] @ w["Wq"].T
        if nm == "k":   return z[f"L{L}.xn1"] @ w["Wk"].T
        if nm == "v":   return z[f"L{L}.xn1"] @ w["Wv"].T
        if nm == "ctx": return self._attend(z[f"L{L}.q"], z[f"L{L}.k"], z[f"L{L}.v"])
        if nm == "a":   return src + z[f"L{L}.ctx"] @ w["Wo"].T
        if nm == "xn2": return self._norm(z[f"L{L}.a"], w["w2"])
        if nm == "g":   return z[f"L{L}.xn2"] @ w["Wg"].T
        if nm == "u":   return z[f"L{L}.xn2"] @ w["Wu"].T
        if nm == "h":   return self._act(z[f"L{L}.g"]) * z[f"L{L}.u"]
        if nm == "y":   return z[f"L{L}.a"] + z[f"L{L}.h"] @ w["Wd"].T
        raise KeyError(key)

    def zeros(self, vsz):
        z = {"embed": np.zeros((self.T, D), np.float32),
             "norm": np.zeros((self.T, D), np.float32),
             "logits": np.zeros((self.T, vsz), np.float32)}
        for L in range(self.NLay):
            for nm, d in LAYER_BLOCKS:
                z[f"L{L}.{nm}"] = np.zeros((self.T, d), np.float32)
        return z

    def settle_gauss_seidel(self, drive, vsz, on_layer=None, skip_logits=False):
        """In-place, topological order. One sweep reaches equilibrium exactly.
        Streams weights so the model is never resident."""
        z = self.zeros(vsz)
        z["embed"] = drive
        for L in range(self.NLay):
            W = {L: self._load(L)}
            for nm, _ in LAYER_BLOCKS:
                key = f"L{L}.{nm}"
                z[key] = self.rule(key, z, drive, W)
            for nm, _ in LAYER_BLOCKS:          # release everything but the handoff
                if nm != "y":
                    z[f"L{L}.{nm}"] = None
            if L > 0:
                z[f"L{L-1}.y"] = None
            del W
            if on_layer:
                on_layer(L)
        z["norm"] = self.rule("norm", z, drive, None)
        if not skip_logits:
            z["logits"] = self.rule("logits", z, drive, None)
        return z

    def settle_jacobi(self, drive, vsz, sweeps):
        """Everything at once. Nothing sequenced."""
        z = self.zeros(vsz)
        hist = []
        for s in range(sweeps):
            nz = {k: self.rule(k, z, drive, self.W) for k in self.order}
            delta = math.sqrt(sum(float(np.sum((nz[k] - z[k]) ** 2)) for k in self.order) /
                              (sum(float(np.sum(nz[k] ** 2)) for k in self.order) + 1e-30))
            z = nz
            hist.append(delta)
        return z, hist


def make_invf():
    invf = 1.0 / (CFG["rope_theta"] ** (np.arange(0, HD, 2) / HD))
    rs = CFG.get("rope_scaling")
    if rs and rs.get("rope_type") == "llama3":
        f, lo, hi, old = (rs["factor"], rs["low_freq_factor"],
                          rs["high_freq_factor"], rs["original_max_position_embeddings"])
        wl = 2 * np.pi / invf
        sm = (old / wl - hi) / (lo - hi)
        invf = np.where(wl > old / lo, invf / f,
               np.where(wl < old / hi, invf, (1 - sm) * invf / f + sm * invf))
    return invf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--reference", action="store_true")
    ap.add_argument("--jacobi", type=int, default=0,
                    help="prove both schedules agree, on this many layers")
    ap.add_argument("--int8", action="store_true",
                    help="int8 per-column weights (quality test; no bytes saved yet)")
    ap.add_argument("--srp", action="store_true",
                    help="SRP popcount readout instead of the dense vocab scan")
    a = ap.parse_args()

    st = Safetensors(BASE)
    pre = "model."
    tok = json.load(open(os.path.join(BASE, "tokenizer.json")))
    vocab = tok["model"]["vocab"]; inv = {v: k for k, v in vocab.items()}

    def encode(text):
        ids, words = [128000], text.split()
        for i, w in enumerate(words):
            key = ("Ġ" + w) if i else w
            if key in vocab: ids.append(vocab[key])
            elif w in vocab: ids.append(vocab[w])
        return ids

    def dec(i):
        return inv.get(i, f"[{i}]").replace("Ġ", " ").replace("Ċ", "\n")

    emb = st.get(pre + "embed_tokens.weight")
    ids = encode(a.prompt)
    invf = make_invf()
    vsz = emb.shape[0]

    order, deps, depth = build_graph(NL)
    dmax = max(depth.values())
    nosc = (sum(d for _, d in LAYER_BLOCKS) * NL + 2 * D) * len(ids)
    print(f"the whole forward as ONE system  ({'wave' if not a.reference else 'reference'} phi)")
    print(f"  blocks {len(order)}   critical path {dmax}   "
          f"{nosc:,} oscillators for {len(ids)} tokens")
    print(f"  weights: {'int8 per-column (quality test)' if a.int8 else 'bf16'}"
          f"   readout: {'srp popcount (512b)' if a.srp else 'dense vocab scan'}")
    print(f"  equilibrium z = F(z;x) — the next token IS the fixed point\n")

    # ---------- both schedules agree ----------
    if a.jacobi:
        K = a.jacobi
        oK, _, dK = build_graph(K)
        dm = max(dK.values())
        print(f"  proving the two schedules reach ONE equilibrium ({K} layers, "
              f"critical path {dm}):\n")
        sysK = FullSystem(st, pre, len(ids), invf, wave=not a.reference,
                          n_layers=K, resident=True)
        sysK.wnorm = st.get(f"{pre}norm.weight"); sysK.head = emb
        drive = emb[ids].astype(np.float32).copy()
        gs = sysK.settle_gauss_seidel(drive, vsz)
        gsl = gs["logits"]
        zj, hist = sysK.settle_jacobi(drive, vsz, dm + 2)
        print(f"    {'sweep':>6}{'state change':>16}{'logit err vs gauss-seidel':>28}")
        print("    " + "-" * 50)
        zz = sysK.zeros(vsz)
        for s in range(1, dm + 3):
            zz = {k: sysK.rule(k, zz, drive, sysK.W) for k in sysK.order}
            e = float(np.linalg.norm(zz["logits"] - gsl) / (np.linalg.norm(gsl) + 1e-30))
            mark = "   <-- settled" if e == 0.0 else ""
            print(f"    {s:>6}{hist[s-1]:>16.3e}{e:>28.3e}{mark}")
            if e == 0.0:
                break
        print(f"\n    gauss-seidel reached the same point in 1 sweep.")
        print(f"    same equilibrium, two schedules — {dm}x apart on a CPU, 1x on hardware.\n")
        return

    # ---------- settle the real thing ----------
    sysm = FullSystem(st, pre, len(ids), invf, wave=not a.reference, int8=a.int8)
    sysm.wnorm = st.get(f"{pre}norm.weight")
    sysm.head = emb if CFG.get("tie_word_embeddings") else st.get("lm_head.weight")
    srp = None
    if a.srp:
        from bqsm_srp import SRP
        srp = SRP(sysm.head)
    t0 = time.time(); out = []
    for step in range(a.n):
        sysm.T = len(ids)
        sysm.mask = np.triu(np.full((sysm.T, sysm.T), -1e30, np.float32), 1)
        drive = emb[ids].astype(np.float32).copy()
        # With --srp the logits block is never materialised: the readout is a
        # Hamming search over 512-bit codes, so the equilibrium is read by
        # resonance rather than by scanning the whole vocabulary.
        z = sysm.settle_gauss_seidel(drive, vsz, skip_logits=srp is not None)
        nxt = (srp.shortlist(z["norm"][-1], sysm.head, k=1024) if srp
               else int(np.argmax(z["logits"][-1])))
        out.append(dec(nxt)); ids.append(nxt)
        print(f"  [{step}] {nxt:>7}  {dec(nxt)!r}   ({time.time()-t0:.0f}s)", flush=True)
    print(f"\n  OUTPUT: {''.join(out)!r}")
    print(f"  FULL:   {a.prompt + ''.join(out)!r}")


if __name__ == "__main__":
    main()
