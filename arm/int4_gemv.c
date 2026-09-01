/* int4_gemv.c (aarch64/NEON) — GEMV over 4-bit weights with a per-row 16-entry
 * Lloyd-Max level table. Two weights per byte, non-uniform levels, so
 * dequantisation is a table lookup rather than a multiply.
 *
 * AVX2 does the lookup with _mm256_permutevar8x32_ps twice (low and high halves
 * of the table) and blends on bit 3, because it can only gather 8 floats at a
 * time. NEON does it in ONE instruction and needs no blend: vqtbl4q_u8 indexes
 * a 64-byte table, and 16 floats IS 64 bytes, so the whole codebook is directly
 * addressable. The only work is turning a 4-bit code c into the four byte
 * indices {4c, 4c+1, 4c+2, 4c+3} that name that float's bytes:
 *
 *   4c        vshl_n_u8(code, 2)
 *   replicate vqtbl1q_u8 with {0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3}
 *   +{0,1,2,3} vaddq_u8
 *
 * This is the one kernel where NEON is architecturally nicer than AVX2 rather
 * than merely equivalent. It assumes little-endian byte order within the float,
 * which aarch64 is.
 *
 * PACKING (unchanged from the x86 kernel): within each group of 8 weights, byte
 * k holds w[k] in the low nibble and w[k+4] in the high nibble.
 *
 * W is packed 4-bit [nout, nin/2]; lut is f32[nout][16]; x, y are f32.
 * nin must be a multiple of 8.
 *
 * cc -O3 -march=armv8-a -fopenmp -shared -fPIC -o libint4.so int4_gemv.c
 */
#include <arm_neon.h>
#include <stddef.h>      /* size_t -- immintrin.h pulled this in, arm_neon.h does not */
#include <stdint.h>

void int4_gemv(const uint8_t *W, const float *lut, const float *x, float *y,
               int nout, int nin)
{
    const int stride = nin >> 1;                     /* bytes per row */
    static const uint8_t REP[16] = {0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3};
    static const uint8_t OFF[16] = {0,1,2,3, 0,1,2,3, 0,1,2,3, 0,1,2,3};
    const uint8x16_t rep = vld1q_u8(REP), off = vld1q_u8(OFF);

#pragma omp parallel for schedule(static)
    for (int o = 0; o < nout; ++o) {
        const uint8_t *w = W + (size_t)o * (size_t)stride;
        /* the 16-float codebook as a 64-byte vqtbl4 table */
        const uint8x16x4_t tab = vld1q_u8_x4((const uint8_t *)(lut + (size_t)o * 16));
        float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
        int i = 0;
        for (; i + 8 <= nin; i += 8) {
            /* 4 bytes hold 8 nibbles: low = w[0..3], high = w[4..7] */
            uint32_t packed;
            __builtin_memcpy(&packed, w + (i >> 1), 4);
            const uint8x8_t b = vreinterpret_u8_u32(vdup_n_u32(packed));
            const uint8x8_t lo4 = vand_u8(b, vdup_n_u8(0x0F));
            const uint8x8_t hi4 = vshr_n_u8(b, 4);

            /* code -> byte indices {4c, 4c+1, 4c+2, 4c+3} */
            uint8x16_t il = vaddq_u8(vqtbl1q_u8(vcombine_u8(vshl_n_u8(lo4, 2),
                                                            lo4), rep), off);
            uint8x16_t ih = vaddq_u8(vqtbl1q_u8(vcombine_u8(vshl_n_u8(hi4, 2),
                                                            hi4), rep), off);
            float32x4_t wl = vreinterpretq_f32_u8(vqtbl4q_u8(tab, il));
            float32x4_t wh = vreinterpretq_f32_u8(vqtbl4q_u8(tab, ih));

            a0 = vfmaq_f32(a0, wl, vld1q_f32(x + i));
            a1 = vfmaq_f32(a1, wh, vld1q_f32(x + i + 4));
        }
        y[o] = vaddvq_f32(vaddq_f32(a0, a1));
    }
}
