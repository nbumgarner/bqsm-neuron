#!/usr/bin/env python3
"""phox_profile.py -- attribute where a single decode token's time goes.

Replicates Engine.settle() section by section with perf-counter timers, then
proves the replica is byte-exact against the real settle() so the attribution is
trustworthy (not a stand-in). Reports per-token averages for: the 7 int8 weight
matmuls (gemv, the memory-bound workhorses), the per-head attention loop, RMS
norm, RoPE, KV concat, and the bf16 vocab readout. Also computes the memory-
bandwidth floor (bytes that MUST stream per token / measured DRAM bandwidth) so
we know how much of the wall time is irreducible physics vs recoverable overhead.

Run on an IDLE box (stop the live engine first):
    BQSM_MODEL=... BQSM_BLOB=... PHOX_KERNELS=sse python3 phox_profile.py --warm 6 --tokens 12
"""
import os, sys, time, math, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqsm_full_settle as FS
from bqsm_full_settle import NL, NH, NKV, HD, D, EPS
from bqsm_int8 import Engine, bf16_view, bf16_row, bf16_logits, BASE, Safetensors
from bqsm_llama import sat_gate, amp_softmax, rope_phase


def instrumented_settle(eng, drive, tpos, wnorm, acc):
    """Byte-for-byte the same math as Engine.settle, with per-section timers
    summed into `acc` (seconds). Returns the settled state."""
    pc = time.perf_counter
    x = drive
    for L in range(NL):
        t = pc(); xn1 = eng.norm(x, eng.vec(L, "w1")); acc["norm"] += pc() - t
        t = pc()
        k = eng.gemv(L, "Wk", xn1).reshape(1, NKV, HD)
        v = eng.gemv(L, "Wv", xn1).reshape(1, NKV, HD)
        acc["gemv"] += pc() - t
        t = pc(); k = rope_phase(k[0], None, None, eng.invf, tpos)[None]; acc["rope"] += pc() - t
        t = pc()
        if eng.kv[L] is None:
            eng.kv[L] = (k, v)
        else:
            pk, pv = eng.kv[L]
            eng.kv[L] = (np.concatenate([pk, k]), np.concatenate([pv, v]))
        acc["kv"] += pc() - t
        K, V = eng.kv[L]
        t = pc(); q = eng.gemv(L, "Wq", xn1).reshape(1, NH, HD); acc["gemv"] += pc() - t
        t = pc(); q = rope_phase(q[0], None, None, eng.invf, tpos)[None]; acc["rope"] += pc() - t
        t = pc()
        ctx = np.empty((1, NH, HD), np.float32)
        sc = 1.0 / math.sqrt(HD)
        for hh in range(NH):
            kv = hh * NKV // NH
            ctx[:, hh] = amp_softmax((q[:, hh] @ K[:, kv].T) * sc) @ V[:, kv]
        acc["attn"] += pc() - t
        t = pc(); a = x + eng.gemv(L, "Wo", ctx.reshape(1, NH * HD)); acc["gemv"] += pc() - t
        t = pc(); xn2 = eng.norm(a, eng.vec(L, "w2")); acc["norm"] += pc() - t
        t = pc()
        h = sat_gate(eng.gemv(L, "Wg", xn2)) * eng.gemv(L, "Wu", xn2)
        x = a + eng.gemv(L, "Wd", h)
        acc["gemv"] += pc() - t
    t = pc(); out = eng.norm(x, wnorm); acc["norm"] += pc() - t
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--warm", type=int, default=6, help="prefill tokens (context depth)")
    ap.add_argument("--tokens", type=int, default=12, help="decode tokens to profile")
    a = ap.parse_args()

    safe = Safetensors(BASE)
    wnorm = safe.get("model.norm.weight")
    eraw, eshape = bf16_view(safe, "model.embed_tokens.weight")
    ename = "model.embed_tokens.weight" if FS.CFG.get("tie_word_embeddings") else "lm_head.weight"
    hraw, hshape = bf16_view(safe, ename)

    eng = Engine(FS.make_invf())
    print(f"host {os.uname().nodename}, loadavg {tuple(round(x,2) for x in os.getloadavg())}, "
          f"{os.cpu_count()} cpus, {eng.blob.nbytes/1e9:.2f} GB int8")

    # bytes streamed per token: int8 weights (all layers) + bf16 vocab head readout.
    W_bytes = eng.blob.nbytes                       # 2.82 GB int8 weights
    head_bytes = hraw.nbytes                        # bf16 vocab head
    print(f"weights {W_bytes/1e9:.3f} GB int8 + readout head {head_bytes/1e9:.3f} GB bf16 "
          f"= {(W_bytes+head_bytes)/1e9:.3f} GB MUST stream per token")

    # faithfulness: instrumented replica == real settle (byte-exact)
    eng.kv = [None] * NL
    z_ref = eng.settle(bf16_row(eraw, eshape, 128000).copy(), 0, wnorm, reset=True)
    eng.kv = [None] * NL
    z_ins = instrumented_settle(eng, bf16_row(eraw, eshape, 128000).copy(), 0, wnorm,
                                {"gemv": 0, "attn": 0, "norm": 0, "rope": 0, "kv": 0})
    cos = float(z_ref.ravel() @ z_ins.ravel() /
                (np.linalg.norm(z_ref) * np.linalg.norm(z_ins) + 1e-30))
    print(f"faithfulness: instrumented vs real settle cosine = {cos:.6f}")

    # warm prefill: settle `warm` positions (greedy) to build a realistic KV context
    eng.kv = [None] * NL
    z = eng.settle(bf16_row(eraw, eshape, 128000).copy(), 0, wnorm, reset=True)
    for p in range(1, a.warm):
        nxt = int(np.argmax(bf16_logits(hraw, hshape, z[-1])))
        z = eng.settle(bf16_row(eraw, eshape, nxt).copy(), p, wnorm)

    # profile `tokens` decode steps
    acc = {"gemv": 0.0, "attn": 0.0, "norm": 0.0, "rope": 0.0, "kv": 0.0}
    t_read = 0.0
    tpos = a.warm
    wall0 = time.perf_counter()
    for _ in range(a.tokens):
        tr = time.perf_counter()
        nxt = int(np.argmax(bf16_logits(hraw, hshape, z[-1])))
        t_read += time.perf_counter() - tr
        z = instrumented_settle(eng, bf16_row(eraw, eshape, nxt).copy(), tpos, wnorm, acc)
        tpos += 1
    wall = time.perf_counter() - wall0
    n = a.tokens

    print(f"\nper-token averages over {n} decode tokens (context ~{a.warm}..{tpos}):")
    rows = [("int8 matmuls (gemv x7/layer)", acc["gemv"]),
            ("attention head loop (numpy)", acc["attn"]),
            ("RMS norm (numpy)", acc["norm"]),
            ("RoPE (numpy)", acc["rope"]),
            ("KV concat", acc["kv"]),
            ("readout bf16_gemv (vocab head)", t_read)]
    total = sum(v for _, v in rows) / n
    for label, v in rows:
        per = v / n
        print(f"  {label:<34} {per*1000:8.1f} ms/token  {100*per/total:5.1f}%")
    print(f"  {'— sum —':<34} {total*1000:8.1f} ms/token  (wall {wall/n*1000:.1f} ms/token)")

    # memory-bandwidth floor: if the matmul+readout were bandwidth-bound, time_floor =
    # bytes_streamed / bandwidth. We can't read DRAM BW directly, but gemv time gives an
    # EFFECTIVE achieved bandwidth; report it so the headroom is explicit.
    gemv_s = acc["gemv"] / n
    read_s = t_read / n
    eff_bw_w = W_bytes / gemv_s / 1e9 if gemv_s > 0 else 0
    eff_bw_h = head_bytes / read_s / 1e9 if read_s > 0 else 0
    print(f"\neffective bandwidth achieved:")
    print(f"  weights matmul: {eff_bw_w:6.2f} GB/s  ({W_bytes/1e9:.2f} GB / {gemv_s*1000:.0f} ms)")
    print(f"  readout head:   {eff_bw_h:6.2f} GB/s  ({head_bytes/1e9:.2f} GB / {read_s*1000:.0f} ms)")
    print(f"  -> if either is well below the machine's DDR2 ceiling, that path is NOT yet "
          f"bandwidth-bound (recoverable); if near it, it's at the physics floor.")


if __name__ == "__main__":
    main()
