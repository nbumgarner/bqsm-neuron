/* SSE4.1 + OpenMP bf16 kernels for the readout — gemv (single) + gemm (batched).
   bf16 -> f32 is a 16-bit left shift (bf16 bits into the high half of float32).
   Built as libbf16_mt.so so the live server's libbf16.so is never touched. */
#include <stdint.h>
#include <smmintrin.h>
static inline float dot_bf16_f32(const uint16_t* w, const float* x, int nin){
    __m128 acc=_mm_setzero_ps(); __m128i z=_mm_setzero_si128(); int i=0;
    for(; i+8<=nin; i+=8){
        __m128i w16=_mm_loadu_si128((const __m128i*)(w+i));
        __m128 w0=_mm_castsi128_ps(_mm_unpacklo_epi16(z,w16));
        __m128 w1=_mm_castsi128_ps(_mm_unpackhi_epi16(z,w16));
        acc=_mm_add_ps(acc,_mm_mul_ps(w0,_mm_loadu_ps(x+i)));
        acc=_mm_add_ps(acc,_mm_mul_ps(w1,_mm_loadu_ps(x+i+4)));
    }
    __m128 t=_mm_hadd_ps(acc,acc); t=_mm_hadd_ps(t,t);
    float s=_mm_cvtss_f32(t);
    for(; i<nin; i++){ uint32_t b=((uint32_t)w[i])<<16; float f; __builtin_memcpy(&f,&b,4); s+=f*x[i]; }
    return s;
}
void bf16_gemv(const uint16_t* W, const float* x, float* y, int nout, int nin){
    #pragma omp parallel for schedule(static)
    for(int o=0;o<nout;o++) y[o]=dot_bf16_f32(W+(long)o*nin, x, nin);
}
/* Batched: one vocab-head row load reused across all B stream states. */
void bf16_gemm(const uint16_t* W, const float* X, float* Y, int nout, int nin, int B){
    #pragma omp parallel for schedule(static)
    for(int o=0;o<nout;o++){
        const uint16_t* w = W+(long)o*nin;
        for(int b=0;b<B;b++) Y[(long)b*nout+o] = dot_bf16_f32(w, X+(long)b*nin, nin);
    }
}
