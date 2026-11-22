/*
 * internal.cuh — shared kernel infrastructure of libfiver_cuda (not public).
 */
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "fiver/compat.h"
#include "fiver/format.h"

namespace fiver {

#define FIVER_FULL_MASK 0xFFFFFFFFu

/* Activation codes for the optional per-weight act_bits (float GEMV only). */
#define FIVER_ACT_IDENTITY 0u
#define FIVER_ACT_SQRT     1u
#define FIVER_ACT_LOG      2u
#define FIVER_ACT_SOFTSIGN 3u

/*
 * Device-resident packed matrix (rows × cols).  Row r starts at word r·ld;
 * ld = fiver_row_pitch(cols) is a multiple of 16 words, words beyond wpr
 * hold the zero code.  act_bits (optional): 2 bits per weight at index
 * (r·ld + w)·6 + i, 16 weights per uint32.
 */
struct Matrix {
    uint32_t*       words     = nullptr;
    float*          row_scale = nullptr;
    const uint32_t* act_bits  = nullptr;
    uint32_t        rows = 0, cols = 0, wpr = 0, ld = 0;
    int             grid = FIVER_GRID_CENTERED;
};

/* Grow-only device scratch buffers, one per slot. */
struct Workspace {
    enum Slot { XT, PART, XQ, XQ3, SX, SXQ, XQ4, DYT, NUM_SLOTS };
    void*  ptr[NUM_SLOTS]  = {};
    size_t size[NUM_SLOTS] = {};
    void* get(Slot s, size_t bytes);
    void  release();
};

namespace dev {
static __constant__ float lut[FIVER_NUM_GRIDS][32]  = FIVER_LUT_INIT;
static __constant__ int   e[FIVER_NUM_GRIDS][32]    = FIVER_E_INIT;
static __constant__ float escale[FIVER_NUM_GRIDS]   = FIVER_ESCALE_INIT;
static __constant__ float inv_escale[FIVER_NUM_GRIDS] = { 1.0f / 127.0f, 1.0f / 127.0f, 1.0f / 64.0f };

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(FIVER_FULL_MASK, v, off);
    return v;
}

__device__ __forceinline__ int warp_sum_i(int v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(FIVER_FULL_MASK, v, off);
    return v;
}

/* Odd, increasing activations on the decoded unscaled weight. */
__host__ __device__ __forceinline__ float apply_act(float x, uint32_t code) {
    switch (code & 3u) {
        case FIVER_ACT_SQRT:     return copysignf(sqrtf(fabsf(x)), x);
        case FIVER_ACT_LOG:      return copysignf(log1pf(fabsf(x)), x);
        case FIVER_ACT_SOFTSIGN: return x / (1.0f + fabsf(x));
        default:                 return x;
    }
}

__device__ __forceinline__ uint32_t act_code(const uint32_t* act, size_t idx) {
    return (__ldg(&act[idx >> 4]) >> ((idx & 15u) * 2u)) & 3u;
}
}  // namespace dev

/* ── forward, float ───────────────────────────────────────────────────── */
enum GemvVariant { GEMV_V1 = 0, GEMV_V2 = 1, GEMV_V2_MAGIC_BITS = 2 };

/* Y[b, r] = row_scale[r] · Σ_c W[r, c] · X[b, c] (+ bias[r]).  X: batch × cols. */
void gemv_float(const Matrix& m, const float* x, float* y, uint32_t batch, const float* bias,
                GemvVariant variant, cudaStream_t s);

/* Tiled FP32 GEMM (sm_61 SIMT), deterministic split-K.  splits = 0: automatic. */
void gemm_float(const Matrix& m, const float* x, float* y, uint32_t batch, const float* bias,
                Workspace& ws, int num_sms, uint32_t splits, cudaStream_t s);

/* ── forward, int8 ────────────────────────────────────────────────────── */
/* Tiled __dp4a GEMM (sm_61+), deterministic integer split-K. */
void gemm_dp4a(const Matrix& m, const float* x, float* y, uint32_t batch, const float* bias,
               Workspace& ws, int num_sms, uint32_t splits, cudaStream_t s);

/* Warp-per-row __dp4a GEMV (good for batch ≤ 8). */
void gemv_dp4a(const Matrix& m, const float* x, float* y, uint32_t batch, const float* bias,
               Workspace& ws, cudaStream_t s);

/* INT8 Tensor-Core GEMM via WMMA (sm_75+).  Not validated on hardware yet. */
bool gemm_tc(const Matrix& m, const float* x, float* y, uint32_t batch, const float* bias,
             Workspace& ws, cudaStream_t s);

/* ── backward ─────────────────────────────────────────────────────────── */
/* dX = Wᵀ · (row_scale ∘ dY), tiled, fixed reduction order (no atomics). */
void dgrad(const Matrix& m, const float* dy, float* dx, uint32_t batch, Workspace& ws,
           int num_sms, uint32_t splits, cudaStream_t s);

/* Fused G = dYᵀ·X and STE update k ← clamp(k − sign(lr·G)); bias −= lr·Σ_b dY. */
void wgrad_ste(const Matrix& m, const float* x, const float* dy, float* bias, uint32_t batch,
               float lr, cudaStream_t s);

/* Previous kernels, kept as references for the benchmark. */
void dgrad_atomic_legacy(const Matrix& m, const float* dy, float* dx, uint32_t batch, cudaStream_t s);
void ste_update_legacy(const Matrix& m, const float* x, const float* dy, float* bias, uint32_t batch,
                       float lr, cudaStream_t s);

}  // namespace fiver
