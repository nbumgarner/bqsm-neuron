#!/usr/bin/env python3
"""
bqsm_srp.py — Sign Random Projection popcount readout, ported to the Python path.

Optional shortlist readout: instead of a dense 128256x3072 scan of the vocab head,
project the settled state to a compact sign code and shortlist candidate tokens.

Sparse SRP, 512 bits, each hyperplane sampling SRP_K=64 dims with random signs,
codes derived from the model's REAL embeddings. A fixed xorshift32 stream builds
identical planes across runs.

    bit b of token t  =  sign( sum_i sgn[b][i] * emb[t][ dim[b][i] ] )
    readout           =  argmin_t  popcount( q XOR code[t] )

ONE THING THIS CANNOT DO, stated up front. Hamming distance on sign bits ranks by
ANGLE. The logit argmax ranks by DOT PRODUCT, which is angle times magnitude:

    logit[t] = |e_t| * |x| * cos(theta_t)

SRP drops |e_t| entirely. Wherever the embedding row norms vary, the two
rankings can disagree, and no number of bits fixes that -- it is not a resolution
problem, it is the wrong quantity. The 3797x figure in the README came with
"exact argmax agreement" measured on Gemma probes; whether that holds here is a
question for measurement, which is what --selftest does.

    python3 bqsm_srp.py --selftest
"""
import argparse, json, os, struct, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bqsm_llama import Safetensors, BASE

BITS = 512
WORDS = BITS // 64
K = 64
CACHE = os.path.expanduser("~/models/llama3b-srp512.npz")


def _xorshift32(n, seed=0xC0FFEE01):
    """The C engine's RNG, so both build the same planes."""
    out = np.empty(n, np.uint32)
    s = np.uint32(seed)
    for i in range(n):
        s ^= np.uint32(s << np.uint32(13))
        s ^= np.uint32(s >> np.uint32(17))
        s ^= np.uint32(s << np.uint32(5))
        out[i] = s
    return out


class SRP:
    def __init__(self, emb, cache=CACHE, verbose=True):
        self.V, self.D = emb.shape
        if cache and os.path.exists(cache):
            z = np.load(cache)
            if int(z["V"]) == self.V and int(z["D"]) == self.D:
                self.dims, self.sgns, self.book = z["dims"], z["sgns"], z["book"]
                if verbose:
                    print(f"  srp codebook loaded from {cache}")
                return
        r = _xorshift32(BITS * K * 2)
        self.dims = (r[0::2] % np.uint32(self.D)).astype(np.int32).reshape(BITS, K)
        self.sgns = np.where((r[1::2] & np.uint32(1)) == 1, 1.0, -1.0).astype(np.float32).reshape(BITS, K)
        t0 = time.time()
        book = np.zeros((self.V, WORDS), np.uint64)
        for b in range(BITS):
            acc = emb[:, self.dims[b]] @ self.sgns[b]
            np.bitwise_or(book[:, b >> 6],
                          np.where(acc > 0, np.uint64(1) << np.uint64(b & 63), np.uint64(0)),
                          out=book[:, b >> 6])
        self.book = book
        if verbose:
            print(f"  srp codebook built in {time.time()-t0:.1f}s "
                  f"({self.V:,} x {BITS} bits = {book.nbytes/1e6:.1f} MB)")
        if cache:
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            np.savez(cache, dims=self.dims, sgns=self.sgns, book=self.book,
                     V=self.V, D=self.D)

    def project(self, x):
        a = (x[self.dims] * self.sgns).sum(1)           # [BITS]
        q = np.zeros(WORDS, np.uint64)
        for b in range(BITS):
            if a[b] > 0:
                q[b >> 6] |= np.uint64(1) << np.uint64(b & 63)
        return q

    def readout(self, x):
        """Pure Hamming argmin. NOT SAFE as a readout -- see shortlist()."""
        q = self.project(x)
        d = np.bitwise_count(self.book ^ q).sum(1)
        return int(np.argmin(d)), d

    def shortlist(self, x, head, k=1024):
        """SRP as a candidate generator, then an EXACT rescore of the shortlist.

        Hamming argmin alone is wrong here: a real post-28-layer state sits at
        distance ~216/512 from every embedding row (random is 256), so the top
        logit's margin is a few bits and 512-bit sign noise (~11 bits) swamps it.
        Measured: pure argmin picks the wrong token while the true one ranks #57.

        Rescoring the shortlist exactly makes the result exact whenever the true
        argmax is inside it -- k=1024 against an observed rank of 57 is a wide
        margin, and `rank_of` below is how you check rather than assume."""
        q = self.project(x)
        d = np.bitwise_count(self.book ^ q).sum(1)
        cand = np.argpartition(d, k)[:k]
        return int(cand[np.argmax(head[cand] @ x)])

    def rank_of(self, x, target):
        """Where the true argmax sits in the Hamming ordering. This is the number
        that decides whether k is big enough; it must be measured per model."""
        q = self.project(x)
        d = np.bitwise_count(self.book ^ q).sum(1)
        return int(np.where(np.argsort(d, kind="stable") == target)[0][0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--probes", type=int, default=64)
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(BASE, "config.json")))
    EPS = cfg["rms_norm_eps"]
    st = Safetensors(BASE)
    emb = st.get("model.embed_tokens.weight")
    wn = st.get("model.norm.weight")
    print(f"srp readout   vocab {emb.shape[0]:,}   D {emb.shape[1]}   "
          f"{BITS} bits, sparse K={K}\n")
    srp = SRP(emb)

    if not a.selftest:
        return

    # ---- REAL readout inputs. This test previously used normed embedding rows
    # as probes and scored 64/64, then the actual model emitted the wrong token
    # on the first try. That was SELF-RETRIEVAL: the probe WAS row t, so its code
    # nearly equalled the codebook entry for t (Hamming 30/512). It measured
    # nothing about the readout. Real post-28-layer states sit at ~216/512, where
    # random is 256. Probes must come from a real settle. ----
    from bqsm_full_settle import FullSystem, make_invf
    ids = [128000, 791, 6864, 315, 9822, 374]
    sysm = FullSystem(st, "model.", len(ids), make_invf(), wave=True)
    sysm.wnorm, sysm.head = wn, emb
    print("  running real settles to collect genuine readout inputs ...")
    X, seq = [], list(ids)
    for _ in range(max(1, a.probes)):
        sysm.T = len(seq)
        sysm.mask = np.triu(np.full((sysm.T, sysm.T), -1e30, np.float32), 1)
        z = sysm.settle_gauss_seidel(emb[seq].astype(np.float32).copy(),
                                     emb.shape[0], skip_logits=True)
        x = z["norm"][-1]
        X.append(x)
        seq.append(int(np.argmax(emb @ x)))
    X = np.stack(X)
    n = len(X)

    t0 = time.time(); dense = [int(np.argmax(emb @ x)) for x in X]
    t_dense = (time.time() - t0) / n
    t0 = time.time(); bare = [srp.readout(x)[0] for x in X]
    t_bare = (time.time() - t0) / n
    t0 = time.time(); short = [srp.shortlist(x, emb, k=1024) for x in X]
    t_short = (time.time() - t0) / n

    ranks = [srp.rank_of(x, d) for x, d in zip(X, dense)]
    ab = sum(int(g == d) for g, d in zip(bare, dense))
    as_ = sum(int(g == d) for g, d in zip(short, dense))

    print(f"\n  {'readout':<34}{'ms/token':>11}{'exact':>9}{'speedup':>10}")
    print("  " + "-" * 64)
    print(f"  {'dense scan (ground truth)':<34}{t_dense*1000:>11.1f}{f'{n}/{n}':>9}{'1.0x':>10}")
    print(f"  {'srp hamming argmin (bare)':<34}{t_bare*1000:>11.1f}"
          f"{f'{ab}/{n}':>9}{t_dense/t_bare:>9.1f}x")
    print(f"  {'srp shortlist k=1024 + rescore':<34}{t_short*1000:>11.1f}"
          f"{f'{as_}/{n}':>9}{t_dense/t_short:>9.1f}x")
    print(f"\n  rank of the true argmax in the hamming ordering: "
          f"max {max(ranks)}, median {int(np.median(ranks))}  (k=1024)")
    if max(ranks) >= 1024:
        print(f"  *** k IS TOO SMALL for this model -- raise it above {max(ranks)}")
    else:
        print(f"  headroom {1024/max(1,max(ranks)):.0f}x. Exact only while this holds;"
              f" it is a property of the model, not a guarantee.")


if __name__ == "__main__":
    main()
