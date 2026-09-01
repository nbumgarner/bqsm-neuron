/* bf16_gemv.c (aarch64/NEON) — GEMV reading bf16 weights directly, expanded to
 * f32 inside the registers so the 16 zero bits never reach RAM.
 *
 * bf16 IS the top half of f32, so the widening is one instruction on NEON:
 *   vshll_n_u16(v, 16)   widen 4x uint16 -> 4x uint32 AND shift left 16
 * which is strictly tidier than the AVX2 pair (cvtepu16_epi32 then slli_epi32).
 * Reinterpreting the result as f32 is free.
 *
 * cc -O3 -march=armv8-a -fopenmp -shared -fPIC -o libbf16.so bf16_gemv.c
 */
#include <arm_neon.h>
#include <stddef.h>      /* size_t -- immintrin.h pulled this in, arm_neon.h does not */
#include <stdint.h>
#include <string.h>

/* y[nout] = W[nout,nin] @ x[nin] ; W bf16 row-major, x/y f32 */
void bf16_gemv(const uint16_t *W, const float *x, float *y, int nout, int nin)
{
#pragma omp parallel for schedule(static)
    for (int o = 0; o < nout; ++o) {
        const uint16_t *w = W + (size_t)o * (size_t)nin;
        float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
        float32x4_t a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);
        int i = 0;
        for (; i + 16 <= nin; i += 16) {
            uint16x8_t p0 = vld1q_u16(w + i);
            uint16x8_t p1 = vld1q_u16(w + i + 8);
            a0 = vfmaq_f32(a0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(p0), 16)),
                           vld1q_f32(x + i));
            a1 = vfmaq_f32(a1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(p0), 16)),
                           vld1q_f32(x + i + 4));
            a2 = vfmaq_f32(a2, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(p1), 16)),
                           vld1q_f32(x + i + 8));
            a3 = vfmaq_f32(a3, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(p1), 16)),
                           vld1q_f32(x + i + 12));
        }
        for (; i + 8 <= nin; i += 8) {
            uint16x8_t p = vld1q_u16(w + i);
            a0 = vfmaq_f32(a0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(p), 16)),
                           vld1q_f32(x + i));
            a1 = vfmaq_f32(a1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(p), 16)),
                           vld1q_f32(x + i + 4));
        }
        float32x4_t acc = vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3));
        float s = vaddvq_f32(acc);
        for (; i < nin; ++i) {                                /* tail */
            uint32_t u = (uint32_t)w[i] << 16;
            float wv;
            memcpy(&wv, &u, 4);
            s += wv * x[i];
        }
        y[o] = s;
    }
}
