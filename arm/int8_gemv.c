/* int8_gemv.c (aarch64/NEON) — GEMV over int8 weights with a per-row scale,
 * widened to f32 inside the registers. Line-for-line the same algorithm as the
 * AVX2 kernel; only the widening ladder differs.
 *
 *   y[o] = scale[o] * sum_i  W[o,i] * x[i]
 *
 * AVX2 gets from int8 to f32 in one step (_mm256_cvtepi8_epi32). NEON has no
 * such instruction, so it climbs: s8 -> s16 (vmovl_s8) -> s32 (vmovl_s16) ->
 * f32 (vcvtq_f32_s32). Four 128-bit accumulators cover the same 16 weights per
 * iteration that AVX2 does with two 256-bit ones, and hide FMA latency the
 * same way.
 *
 * Deliberately NOT using SDOT (vdotq_s32): x is f32, not int8, so there is no
 * integer dot product to be had here without quantising the activations too.
 * That is a different experiment and it would change the numerics.
 *
 * cc -O3 -march=armv8-a -fopenmp -shared -fPIC -o libint8.so int8_gemv.c
 */
#include <arm_neon.h>
#include <stddef.h>      /* size_t -- immintrin.h pulled this in, arm_neon.h does not */
#include <stdint.h>

void int8_gemv(const int8_t *W, const float *scale, const float *x, float *y,
               int nout, int nin)
{
#pragma omp parallel for schedule(static)
    for (int o = 0; o < nout; ++o) {
        const int8_t *w = W + (size_t)o * (size_t)nin;
        float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
        float32x4_t a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);
        int i = 0;
        for (; i + 16 <= nin; i += 16) {
            int8x16_t b  = vld1q_s8(w + i);                  /* 16 int8 */
            int16x8_t lo = vmovl_s8(vget_low_s8(b));         /* w[0..7]  */
            int16x8_t hi = vmovl_s8(vget_high_s8(b));        /* w[8..15] */
            a0 = vfmaq_f32(a0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))),
                           vld1q_f32(x + i));
            a1 = vfmaq_f32(a1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))),
                           vld1q_f32(x + i + 4));
            a2 = vfmaq_f32(a2, vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))),
                           vld1q_f32(x + i + 8));
            a3 = vfmaq_f32(a3, vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))),
                           vld1q_f32(x + i + 12));
        }
        for (; i + 8 <= nin; i += 8) {
            int16x8_t e = vmovl_s8(vld1_s8(w + i));
            a0 = vfmaq_f32(a0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(e))),
                           vld1q_f32(x + i));
            a1 = vfmaq_f32(a1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(e))),
                           vld1q_f32(x + i + 4));
        }
        float32x4_t acc = vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3));
        float s = vaddvq_f32(acc);                            /* aarch64 only */
        for (; i < nin; ++i)
            s += (float)w[i] * x[i];
        y[o] = s * scale[o];
    }
}
