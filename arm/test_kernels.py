#!/usr/bin/env python3
"""test_kernels.py — prove the compiled kernels compute what they claim.

Run this on the target machine after arm/build.sh. It compares every kernel
against a plain numpy reference at the same precision and reports max absolute
and relative error. There is no tolerance hand-waving: f32 FMA reassociation is
the only legitimate source of disagreement, so the bar is tight.

A SIMD kernel that is subtly wrong -- a bad shuffle, a sign-extension that
should have been zero-extension, one lane transposed -- produces output that
looks plausible and is not. Cosine similarity hides exactly that class of bug,
so this checks elementwise error, not similarity.

    python3 arm/test_kernels.py
"""
import ctypes
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
rng = np.random.default_rng(0)


BUILT = 0


def load(name, sym, argtypes):
    global BUILT
    path = os.path.join(OUT, name)
    if not os.path.exists(path):
        return None
    lib = ctypes.CDLL(path)
    fn = getattr(lib, sym)
    fn.argtypes = argtypes
    BUILT += 1
    return fn


def report(tag, got, want):
    ae = np.abs(got - want).max()
    scale = max(np.abs(want).max(), 1e-30)
    re = ae / scale
    ok = re < 2e-6
    print(f"  {tag:<12} max|err| {ae:11.3e}   rel {re:9.2e}   {'OK' if ok else 'FAIL'}")
    return ok


def t_int8_gemv(nout=257, nin=3072):          # deliberately not a multiple of 8
    fn = load("libint8.so", "int8_gemv", [ctypes.c_void_p] * 4 + [ctypes.c_int] * 2)
    if fn is None:
        print("  int8_gemv    (not built)")
        return True
    W = rng.integers(-127, 128, (nout, nin), dtype=np.int8)
    s = (rng.random(nout, dtype=np.float32) * 0.01 + 1e-4).astype(np.float32)
    x = rng.standard_normal(nin, dtype=np.float32)
    y = np.zeros(nout, dtype=np.float32)
    fn(W.ctypes.data, s.ctypes.data, x.ctypes.data, y.ctypes.data, nout, nin)
    return report("int8_gemv", y, (W.astype(np.float32) @ x) * s)


def t_int8_gemm(nout=128, nin=3072, B=7):     # B not a multiple of BB=4
    fn = load("libint8gemm.so", "int8_gemm", [ctypes.c_void_p] * 4 + [ctypes.c_int] * 3)
    if fn is None:
        print("  int8_gemm    (not built)")
        return True
    W = rng.integers(-127, 128, (nout, nin), dtype=np.int8)
    s = (rng.random(nout, dtype=np.float32) * 0.01 + 1e-4).astype(np.float32)
    X = rng.standard_normal((B, nin), dtype=np.float32)
    Y = np.zeros((B, nout), dtype=np.float32)
    fn(W.ctypes.data, s.ctypes.data, X.ctypes.data, Y.ctypes.data, nout, nin, B)
    return report("int8_gemm", Y, (X @ W.astype(np.float32).T) * s)


def t_bf16_gemv(nout=193, nin=3072):
    fn = load("libbf16.so", "bf16_gemv", [ctypes.c_void_p] * 3 + [ctypes.c_int] * 2)
    if fn is None:
        print("  bf16_gemv    (not built)")
        return True
    Wf = rng.standard_normal((nout, nin), dtype=np.float32)
    W = (Wf.view(np.uint32) >> 16).astype(np.uint16)          # truncate to bf16
    ref = (W.astype(np.uint32) << 16).view(np.float32)        # exactly what the kernel sees
    x = rng.standard_normal(nin, dtype=np.float32)
    y = np.zeros(nout, dtype=np.float32)
    fn(W.ctypes.data, x.ctypes.data, y.ctypes.data, nout, nin)
    return report("bf16_gemv", y, ref @ x)


def t_int4_gemv(nout=96, nin=3072):
    fn = load("libint4.so", "int4_gemv", [ctypes.c_void_p] * 4 + [ctypes.c_int] * 2)
    if fn is None:
        print("  int4_gemv    (not built)")
        return True
    codes = rng.integers(0, 16, (nout, nin), dtype=np.uint8)
    lut = rng.standard_normal((nout, 16), dtype=np.float32)
    # PACKING: in each group of 8, byte k = w[k] low nibble | w[k+4] high nibble
    g = codes.reshape(nout, nin // 8, 8)
    packed = (g[:, :, :4] | (g[:, :, 4:] << 4)).reshape(nout, nin // 2)
    packed = np.ascontiguousarray(packed.astype(np.uint8))
    x = rng.standard_normal(nin, dtype=np.float32)
    y = np.zeros(nout, dtype=np.float32)
    fn(packed.ctypes.data, lut.ctypes.data, x.ctypes.data, y.ctypes.data, nout, nin)
    ref = lut[np.arange(nout)[:, None], codes.astype(np.intp)]
    return report("int4_gemv", y, ref @ x)


if __name__ == "__main__":
    import platform
    print(f"  machine {platform.machine()}   numpy {np.__version__}")
    ok = all([t_int8_gemv(), t_int8_gemm(), t_bf16_gemv(), t_int4_gemv()])
    # A skipped kernel is not a passing kernel. Reporting "all agree" when
    # nothing was built is the exact false-green this file exists to prevent.
    if BUILT == 0:
        print("\n  NOTHING WAS BUILT -- this is not a pass. Run arm/build.sh first.")
        sys.exit(2)
    print(f"\n  {BUILT}/4 kernels built. "
          + ("all built kernels agree with numpy" if ok else "MISMATCH -- do not ship"))
    sys.exit(0 if ok and BUILT == 4 else 1)
