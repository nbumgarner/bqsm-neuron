#!/usr/bin/env bash
# build.sh — compile the GEMV kernels for whatever machine this is.
#
# Emits the SAME .so names into bqsm_assist/ that the x86 build produces, so
# bqsm_int8.py needs no change and no dispatch logic: the loader just finds a
# library that happens to be NEON instead of AVX2.
#
#   ./arm/build.sh            build for this machine
#   ./arm/build.sh --check    build, then verify against numpy
set -eu

HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$(dirname "$HERE")"
ARCH="$(uname -m)"

case "$ARCH" in
    aarch64|arm64)  SRC="$HERE";        ISA="-march=armv8-a" ;;
    x86_64|amd64)   SRC="$OUT";         ISA="-mavx2 -mfma"   ;;
    *) echo "unsupported arch: $ARCH -- use the numpy fallback (PHOX_KERNELS=numpy)" >&2
       exit 1 ;;
esac

CC="${CC:-cc}"
command -v "$CC" >/dev/null || { echo "no C compiler ($CC). Termux: pkg install clang" >&2; exit 1; }

# OpenMP is optional. Termux needs `pkg install libomp`; without it the kernels
# still work, single-threaded, which on a phone is often what you want anyway
# because the big cores throttle before all of them are busy.
OMP="-fopenmp"
if ! echo 'int main(void){return 0;}' | $CC -fopenmp -x c - -o /dev/null 2>/dev/null; then
    echo "  note: no OpenMP, building single-threaded"
    OMP=""
fi

echo "  arch $ARCH   source $SRC   $ISA $OMP"
build() {  # build <source.c> <output.so>
    [ -f "$SRC/$1" ] || { echo "  skip $1 (not present for this arch)"; return 0; }
    $CC -O3 $ISA $OMP -shared -fPIC -o "$OUT/$2" "$SRC/$1"
    echo "  built $2"
}

build int8_gemv.c libint8.so
build int8_gemm.c libint8gemm.so
build bf16_gemv.c libbf16.so
build int4_gemv.c libint4.so

if [ "${1:-}" = "--check" ]; then
    echo
    python3 "$HERE/test_kernels.py"
fi
