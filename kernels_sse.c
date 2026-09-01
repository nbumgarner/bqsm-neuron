/* SSE4.1 + OpenMP int8 kernels for the 2008 dual-Xeon (no AVX).
   Keeps weights int8, widens in-register (pmovsxbw/pmovsxwd) -> avoids the numpy
   fallback's 4x float32 blowup. y[o] = scale[o] * sum_i W[o][i]*x[i]. */
#include <stdint.h>
#include <smmintrin.h>

static inline float dot_i8_f32(const int8_t* w, const float* x, int nin){
    __m128 acc = _mm_setzero_ps();
    int i=0;
    for(; i+16<=nin; i+=16){
        __m128i w8 = _mm_loadu_si128((const __m128i*)(w+i));
        __m128i lo = _mm_cvtepi8_epi16(w8);
        __m128i hi = _mm_cvtepi8_epi16(_mm_srli_si128(w8,8));
        __m128 w0 = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(lo));
        __m128 w1 = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_srli_si128(lo,8)));
        __m128 w2 = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(hi));
        __m128 w3 = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_srli_si128(hi,8)));
        acc = _mm_add_ps(acc, _mm_mul_ps(w0, _mm_loadu_ps(x+i)));
        acc = _mm_add_ps(acc, _mm_mul_ps(w1, _mm_loadu_ps(x+i+4)));
        acc = _mm_add_ps(acc, _mm_mul_ps(w2, _mm_loadu_ps(x+i+8)));
        acc = _mm_add_ps(acc, _mm_mul_ps(w3, _mm_loadu_ps(x+i+12)));
    }
    __m128 t = _mm_hadd_ps(acc, acc); t = _mm_hadd_ps(t, t);
    float s = _mm_cvtss_f32(t);
    for(; i<nin; i++) s += (float)w[i]*x[i];
    return s;
}
void int8_gemv(const int8_t* W, const float* scale, const float* x, float* y, int nout, int nin){
    #pragma omp parallel for schedule(static)
    for(int o=0;o<nout;o++) y[o] = dot_i8_f32(W+(long)o*nin, x, nin) * scale[o];
}
void int8_gemm(const int8_t* W, const float* scale, const float* X, float* Y, int nout, int nin, int B){
    #pragma omp parallel for schedule(static)
    for(int o=0;o<nout;o++){
        const int8_t* w = W+(long)o*nin;
        for(int b=0;b<B;b++) Y[(long)b*nout+o] = dot_i8_f32(w, X+(long)b*nin, nin) * scale[o];
    }
}
