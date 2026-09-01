"""sparse+bulk state packer.

Two uses:
  * vector-level rate-distortion measurement (int8 vs entropy floor), and
  * KV-cache packing for save_state -- store the settled state at a chosen
    tolerance instead of flat float16.

The KV cache is spiky (a few massive-activation channels carry most of the
energy, the rest is ~Gaussian), so uniform float16 spends 16 bits everywhere.
Quantising per row to a tolerance-matched step turns the bulk into small
integers that DEFLATE (np.savez_compressed) then entropy-codes -- 1.68x smaller
than float16 at float16-equivalent fidelity (rel err ~1e-3), measured on real
Dolphin-8B projections. Loosen the tolerance for more, at the cost of resume
fidelity. Round-trips within tol by construction.
"""
import numpy as np


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def entropy(qi):
    _, c = np.unique(qi, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def int8_pack(x):
    s = max(np.abs(x).max(), 1e-30) / 127.0
    q = np.clip(np.rint(x / s), -127, 127)
    return q * s, 8.0, rmse(x, q * s), q


# ----- KV-cache packer used by save_state/load_state -----

def pack_kv(arr, tol=0.001):
    """arr: (T, H, HD) or (T, D) float. Per-row (per settled position) quantise
    to a step giving ~tol relative error. Returns a dict of arrays that
    np.savez_compressed then DEFLATEs. Round-trips within tol."""
    a = np.asarray(arr, np.float32)
    shape = a.shape
    flat = a.reshape(shape[0], -1)
    scale = np.maximum(np.abs(flat).max(1), 1e-30) * (tol / 2.0)
    q = np.rint(flat / scale[:, None])
    lo, hi = q.min(), q.max()
    dt = np.int16 if (hi < 32767 and lo > -32768) else np.int32
    return {"q": q.astype(dt), "scale": scale.astype(np.float32),
            "shape": np.array(shape, np.int32)}


def unpack_kv(rec):
    q = rec["q"].astype(np.float32)
    scale = rec["scale"]
    shape = tuple(int(s) for s in rec["shape"])
    return (q * scale[:, None]).reshape(shape)
