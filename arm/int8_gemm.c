/* int8_gemm.c (aarch64/NEON) — batched GEMV: Y[b,:] = scale * (W @ X[b,:]).
 *
 * The point is weight REUSE, and it survives the port intact: each 8-wide chunk
 * of weights is widened to f32 once in a register and reused against B input
 * rows before it is discarded. Same bytes off the bus, B times the work.
 *
 * NEON carries 128-bit vectors, so the widened chunk is two float32x4_t rather
 * than one __m256. That doubles the register pressure of the reused value, so
 * BB stays at 4: 4 rows x 2 halves = 8 accumulators, plus 2 weight halves and
 * 2 activation halves = 12 of the 32 available v-registers. Comfortable.
 *
 * W int8 [nout, nin] row-major, scale f32[nout], X f32[B, nin], Y f32[B, nout].
 * nin must be a multiple of 8 (3072, 8192, 1024 all are).
 *
 * cc -O3 -march=armv8-a -fopenmp -shared -fPIC -o libint8gemm.so int8_gemm.c
 */
#include <arm_neon.h>
#include <stddef.h>      /* size_t -- immintrin.h pulled this in, arm_neon.h does not */
#include <stdint.h>

#define BB 4

void int8_gemm(const int8_t *W, const float *scale, const float *X, float *Y,
               int nout, int nin, int B)
{
#pragma omp parallel for schedule(static)
    for (int o = 0; o < nout; ++o) {
        const int8_t *w = W + (size_t)o * (size_t)nin;
        const float s = scale[o];
        for (int b0 = 0; b0 < B; b0 += BB) {
            const int nb = (B - b0 < BB) ? (B - b0) : BB;
            float32x4_t lo0 = vdupq_n_f32(0.0f), hi0 = vdupq_n_f32(0.0f);
            float32x4_t lo1 = vdupq_n_f32(0.0f), hi1 = vdupq_n_f32(0.0f);
            float32x4_t lo2 = vdupq_n_f32(0.0f), hi2 = vdupq_n_f32(0.0f);
            float32x4_t lo3 = vdupq_n_f32(0.0f), hi3 = vdupq_n_f32(0.0f);
            const float *x0 = X + (size_t)(b0 + 0) * nin;
            const float *x1 = X + (size_t)(b0 + (nb > 1 ? 1 : 0)) * nin;
            const float *x2 = X + (size_t)(b0 + (nb > 2 ? 2 : 0)) * nin;
            const float *x3 = X + (size_t)(b0 + (nb > 3 ? 3 : 0)) * nin;
            for (int i = 0; i < nin; i += 8) {
                /* ONE widen, reused nb times -- this is the whole optimisation */
                int16x8_t e = vmovl_s8(vld1_s8(w + i));
                float32x4_t wl = vcvtq_f32_s32(vmovl_s16(vget_low_s16(e)));
                float32x4_t wh = vcvtq_f32_s32(vmovl_s16(vget_high_s16(e)));
                lo0 = vfmaq_f32(lo0, wl, vld1q_f32(x0 + i));
                hi0 = vfmaq_f32(hi0, wh, vld1q_f32(x0 + i + 4));
                if (nb > 1) {
                    lo1 = vfmaq_f32(lo1, wl, vld1q_f32(x1 + i));
                    hi1 = vfmaq_f32(hi1, wh, vld1q_f32(x1 + i + 4));
                }
                if (nb > 2) {
                    lo2 = vfmaq_f32(lo2, wl, vld1q_f32(x2 + i));
                    hi2 = vfmaq_f32(hi2, wh, vld1q_f32(x2 + i + 4));
                }
                if (nb > 3) {
                    lo3 = vfmaq_f32(lo3, wl, vld1q_f32(x3 + i));
                    hi3 = vfmaq_f32(hi3, wh, vld1q_f32(x3 + i + 4));
                }
            }
            const float32x4_t acc[4] = {vaddq_f32(lo0, hi0), vaddq_f32(lo1, hi1),
                                        vaddq_f32(lo2, hi2), vaddq_f32(lo3, hi3)};
            for (int k = 0; k < nb; ++k)
                Y[(size_t)(b0 + k) * nout + o] = vaddvq_f32(acc[k]) * s;
        }
    }
}
