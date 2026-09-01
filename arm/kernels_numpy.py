"""kernels_numpy.py — the kernels with no compiler at all.

A phone in Termux without `pkg install clang`, an aarch64 box with no build
tools, a fresh checkout: this makes the engine run anywhere numpy runs. It is
slower than NEON and much slower than AVX2, because the whole point of the C
kernels is widening int8 to f32 inside the registers so the expanded weights
never touch RAM -- numpy cannot do that, it must materialise. Expect roughly
3-5x the memory traffic on the int8 paths.

It exists so "does it run on my phone" and "is it fast on my phone" are separate
questions, answered in that order.

The shim mimics the ctypes surface exactly: the same call signature, taking raw
addresses, so bqsm_int8.py cannot tell the difference. Addresses arrive as
either int or ctypes.c_void_p, matching how the real loader is called.
"""
import ctypes

import numpy as np


def _addr(p):
    """ctypes.c_void_p or plain int -> int address."""
    return p.value if isinstance(p, ctypes.c_void_p) else int(p)


def _arr(p, dtype, shape):
    """View foreign memory as a numpy array without copying it."""
    n = int(np.prod(shape))
    buf = (ctypes.c_char * (n * np.dtype(dtype).itemsize)).from_address(_addr(p))
    return np.frombuffer(buf, dtype=dtype, count=n).reshape(shape)


class _Shim:
    """One object standing in for a CDLL handle."""

    @staticmethod
    def int8_gemv(W, scale, x, y, nout, nin):
        w = _arr(W, np.int8, (nout, nin))
        s = _arr(scale, np.float32, (nout,))
        xv = _arr(x, np.float32, (nin,))
        out = _arr(y, np.float32, (nout,))
        out[:] = (w @ xv.astype(np.float32)) * s

    @staticmethod
    def int8_gemm(W, scale, X, Y, nout, nin, B):
        w = _arr(W, np.int8, (nout, nin))
        s = _arr(scale, np.float32, (nout,))
        xv = _arr(X, np.float32, (B, nin))
        out = _arr(Y, np.float32, (B, nout))
        out[:] = (xv @ w.astype(np.float32).T) * s

    @staticmethod
    def bf16_gemv(W, x, y, nout, nin):
        raw = _arr(W, np.uint16, (nout, nin))
        wf = (raw.astype(np.uint32) << 16).view(np.float32)   # bf16 -> f32
        out = _arr(y, np.float32, (nout,))
        out[:] = wf @ _arr(x, np.float32, (nin,))

    @staticmethod
    def int4_gemv(W, lut, x, y, nout, nin):
        packed = _arr(W, np.uint8, (nout, nin // 2))
        tab = _arr(lut, np.float32, (nout, 16))
        g = packed.reshape(nout, nin // 8, 4)
        codes = np.concatenate([g & 0x0F, g >> 4], axis=2).reshape(nout, nin)
        wf = tab[np.arange(nout)[:, None], codes.astype(np.intp)]
        out = _arr(y, np.float32, (nout,))
        out[:] = wf @ _arr(x, np.float32, (nin,))


#: the three handles bqsm_int8.py expects, all backed by the same shim
lib_int8 = lib_int8gemm = lib_bf16 = lib_int4 = _Shim()
