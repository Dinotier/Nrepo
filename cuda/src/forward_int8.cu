/*
 * forward_int8.cu — INT8 forward paths.
 *
 *   v(k) ≈ offset + e(k)/E,   x ≈ sx · xq  (per input row, sx = max|x|/127)
 *   y[b,r] = row_scale[r] · sx[b] / E · ( Σ_c e[r,c]·xq[b,c] + offset·E·Σ_c xq[b,c] )
 *   CENTERED/ANCHORED: E = 127 (ANCHORED: offset 1); MAGIC: E = 64, exact e.
 *
 * gemm_dp4a (new, batch > 8): same tiling as gemm_float — CTA 128 rows × 32
 *   batch, K step 48 = 12 groups of 4 codes packed into one int32, so each
 *   inner step is 32 __dp4a = 128 MACs per thread.  x is quantised and
 *   transposed into xq4[kgroup][npad] in one pass.  Split-K partials are
 *   int32 and summed exactly in a fixed order.
 * gemv_dp4a (warp per row) and gemm_tc (WMMA, sm_75+) are the previous
 *   kernels on the padded row pitch.  gemm_tc is not validated on hardware.
 *
 * int32 range: |e·xq| ≤ 127², so cols ≤ 65536 never overflows (checked by
 * the API).  Per-weight act_bits are not supported here.
 */
#include "internal.cuh"
#if !defined(FIVER_EMU)
#include <mma.h>
#endif

namespace fiver {

uint32_t choose_splits(uint32_t base_ctas, uint32_t k_steps, int num_sms);

namespace {

constexpr uint32_t BM = 128, BN = 32, BKW = 8, BG = BKW * 6 / 4;   /* 12 int32 groups per step */

__device__ __forceinline__ int pack4(int a, int b, int c, int d) {
    return (a & 0xFF) | ((b & 0xFF) << 8) | ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
}

/* Row-wise max|x| reduction for a 256-thread block. */
__device__ float block_absmax(const float* xr, uint32_t cols, bool valid) {
    __shared__ float red[8];
    float m = 0.0f;
    if (valid)
        for (uint32_t c = threadIdx.x; c < cols; c += blockDim.x) m = fmaxf(m, fabsf(xr[c]));
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_down_sync(FIVER_FULL_MASK, m, off));
    if ((threadIdx.x & 31u) == 0) red[threadIdx.x >> 5] = m;
    __syncthreads();
    float r = 0.0f;
    for (uint32_t i = 0; i < (blockDim.x >> 5); i++) r = fmaxf(r, red[i]);
    __syncthreads();
    return r;
}

__device__ int block_sum_i(int v) {
    __shared__ int red[8];
    v = dev::warp_sum_i(v);
    if ((threadIdx.x & 31u) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    int r = 0;
    for (uint32_t i = 0; i < (blockDim.x >> 5); i++) r += red[i];
    __syncthreads();
    return r;
}

/* quantise + transpose: xq4[g * npad + n] = bytes xq[n][4g .. 4g+3]; one block per n < npad */
__global__ void k_quant_transpose(const float* __restrict__ X, int* __restrict__ xq4,
                                  float* __restrict__ sx, int* __restrict__ sxq,
                                  uint32_t cols, uint32_t batch, uint32_t npad, uint32_t groups)
{
    const uint32_t n  = blockIdx.x;
    const bool     ok = n < batch;
    const float*   xr = X + (size_t)n * cols;
    const float    m  = block_absmax(xr, cols, ok);
    const float    inv = m > 0.0f ? 127.0f / m : 0.0f;
    int sum = 0;
    for (uint32_t g = threadIdx.x; g < groups; g += blockDim.x) {
        int q[4];
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const uint32_t c = 4u * g + (uint32_t)j;
            q[j] = (ok && c < cols) ? max(-127, min(127, __float2int_rn(xr[c] * inv))) : 0;
            sum += q[j];
        }
        xq4[(size_t)g * npad + n] = pack4(q[0], q[1], q[2], q[3]);
    }
    sum = block_sum_i(sum);
    if (threadIdx.x == 0) { sx[n] = m / 127.0f; sxq[n] = sum; }
}

__global__ void __launch_bounds__(128)
k_gemm_dp4a(const uint32_t* __restrict__ W, const float* __restrict__ row_scale,
            const int* __restrict__ xq4, const float* __restrict__ sx, const int* __restrict__ sxq,
            float* __restrict__ y, int* __restrict__ part, const float* __restrict__ bias,
            uint32_t rows, uint32_t ld, uint32_t batch, uint32_t npad,
            uint32_t steps_per_split, int grid, int write_partial)
{
    __shared__ __align__(16) int As[BG][BM];
    __shared__ __align__(16) int Bs[BG][BN];

    const uint32_t t    = threadIdx.x;
    const uint32_t lane = t & 31u, warp = t >> 5;
    const uint32_t row0 = blockIdx.x * BM;
    const uint32_t n0   = blockIdx.y * BN;
    const uint32_t ks0  = blockIdx.z * steps_per_split;
    const uint32_t ks1  = min(ld / BKW, ks0 + steps_per_split);
    const int      elut = dev::e[grid][lane];
    const uint32_t pad  = fiver_pad_word(grid);

    int acc[4][8];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 8; j++) acc[i][j] = 0;

    uint4 wa, wb;
    int   xb[3];
    auto load = [&](uint32_t ks) {
        const uint32_t r = row0 + t;
        if (r < rows) {
            const uint4* p = reinterpret_cast<const uint4*>(W + (size_t)r * ld + ks * BKW);
            wa = __ldg(p);
            wb = __ldg(p + 1);
        } else {
            wa = make_uint4(pad, pad, pad, pad);
            wb = wa;
        }
#pragma unroll
        for (int p = 0; p < 3; p++) {
            const uint32_t e = (uint32_t)p * 128u + t, g = e >> 5, n = e & 31u;
            xb[p] = __ldg(&xq4[(size_t)(ks * BG + g) * npad + n0 + n]);
        }
    };
    auto store = [&]() {
        const uint32_t words[8] = { wa.x, wa.y, wa.z, wa.w, wb.x, wb.y, wb.z, wb.w };
        int ev[48];
#pragma unroll
        for (int w = 0; w < 8; w++)
#pragma unroll
            for (int i = 0; i < 6; i++)
                ev[w * 6 + i] = __shfl_sync(FIVER_FULL_MASK, elut, (int)fiver_k_at(words[w], i));
#pragma unroll
        for (int g = 0; g < (int)BG; g++)
            As[g][t] = pack4(ev[4 * g], ev[4 * g + 1], ev[4 * g + 2], ev[4 * g + 3]);
#pragma unroll
        for (int p = 0; p < 3; p++) {
            const uint32_t e = (uint32_t)p * 128u + t, g = e >> 5, n = e & 31u;
            Bs[g][n] = xb[p];
        }
    };

    if (ks0 < ks1) {
        load(ks0);
        store();
        __syncthreads();
    }
    for (uint32_t ks = ks0; ks < ks1; ks++) {
        const bool more = ks + 1 < ks1;
        if (more) load(ks + 1);
#pragma unroll
        for (uint32_t g = 0; g < BG; g++) {
            const int4 a  = *reinterpret_cast<const int4*>(&As[g][4u * lane]);
            const int4 bl = *reinterpret_cast<const int4*>(&Bs[g][8u * warp]);
            const int4 bh = *reinterpret_cast<const int4*>(&Bs[g][8u * warp + 4u]);
            const int av[4] = { a.x, a.y, a.z, a.w };
            const int bv[8] = { bl.x, bl.y, bl.z, bl.w, bh.x, bh.y, bh.z, bh.w };
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 8; j++) acc[i][j] = __dp4a(av[i], bv[j], acc[i][j]);
        }
        __syncthreads();
        if (more) {
            store();
            __syncthreads();
        }
    }

    const float off = grid == FIVER_GRID_ANCHORED ? dev::escale[grid] : 0.0f;
    const float ie  = dev::inv_escale[grid];
    const size_t pbase = (size_t)blockIdx.z * npad * rows;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const uint32_t r = row0 + 4u * lane + (uint32_t)i;
        if (r >= rows) continue;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const uint32_t n = n0 + 8u * warp + (uint32_t)j;
            if (write_partial) {
                part[pbase + (size_t)n * rows + r] = acc[i][j];
            } else if (n < batch) {
                y[(size_t)n * rows + r] = ((float)acc[i][j] + off * (float)sxq[n]) * row_scale[r] * ie * sx[n]
                                          + (bias ? bias[r] : 0.0f);
            }
        }
    }
}

__global__ void k_reduce_partials_i32(const int* __restrict__ part, float* __restrict__ y,
                                      const float* __restrict__ row_scale, const float* __restrict__ sx,
                                      const int* __restrict__ sxq, const float* __restrict__ bias,
                                      uint32_t rows, uint32_t batch, uint32_t npad, uint32_t splits, int grid)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)batch * rows) return;
    const uint32_t r = (uint32_t)(i % rows), n = (uint32_t)(i / rows);
    int sum = 0;
    for (uint32_t s = 0; s < splits; s++) sum += part[(size_t)s * npad * rows + i];
    const float off = grid == FIVER_GRID_ANCHORED ? dev::escale[grid] : 0.0f;
    y[i] = ((float)sum + off * (float)sxq[n]) * row_scale[r] * dev::inv_escale[grid] * sx[n]
           + (bias ? bias[r] : 0.0f);
}

/* ── previous kernels on the padded pitch ─────────────────────────────── */

/* Per-row int8 quantisation into xq[batch_pad × ldq] (row-major) + sx, Σxq. */
__global__ void k_quantize_rows(const float* __restrict__ X, int8_t* __restrict__ xq,
                                float* __restrict__ sx, int* __restrict__ sxq,
                                uint32_t cols, uint32_t ldq, uint32_t batch)
{
    const uint32_t b  = blockIdx.x;
    const bool     ok = b < batch;
    const float*   xr = X + (size_t)b * cols;
    const float    m  = block_absmax(xr, cols, ok);
    const float    inv = m > 0.0f ? 127.0f / m : 0.0f;
    int sum = 0;
    for (uint32_t c = threadIdx.x; c < ldq; c += blockDim.x) {
        int q = 0;
        if (ok && c < cols) q = max(-127, min(127, __float2int_rn(xr[c] * inv)));
        xq[(size_t)b * ldq + c] = (int8_t)q;
        sum += q;
    }
    sum = block_sum_i(sum);
    if (threadIdx.x == 0) { sx[b] = m / 127.0f; sxq[b] = sum; }
}

__global__ void k_pack_triplets(const int8_t* __restrict__ xq, int* __restrict__ xq3,
                                uint32_t ldq, uint32_t triplets, uint32_t batch)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)batch * triplets) return;
    const uint32_t b = (uint32_t)(i / triplets), t = (uint32_t)(i % triplets);
    const int8_t*  p = xq + (size_t)b * ldq + 3u * t;
    xq3[i] = pack4(p[0], p[1], p[2], 0);
}

template <int BT>
__global__ void k_gemv_dp4a(const uint32_t* __restrict__ W, const float* __restrict__ row_scale,
                            const int* __restrict__ xq3, const float* __restrict__ sx,
                            const int* __restrict__ sxq, float* __restrict__ Y,
                            const float* __restrict__ bias, uint32_t rows, uint32_t ld,
                            uint32_t batch, int grid)
{
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row  = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const uint32_t b0   = blockIdx.y * (uint32_t)BT;
    const int      elut = dev::e[grid][lane];
    if (row >= rows) return;

    const uint32_t T = 2u * ld;
    int acc[BT];
#pragma unroll
    for (int b = 0; b < BT; b++) acc[b] = 0;

    for (uint32_t base = 0; base < ld; base += 32u) {
        const uint32_t wi   = base + lane;
        const bool     ok   = wi < ld;
        const uint32_t word = ok ? __ldg(&W[(size_t)row * ld + wi]) : 0u;
        int e[6];
#pragma unroll
        for (int i = 0; i < 6; i++) e[i] = __shfl_sync(FIVER_FULL_MASK, elut, (int)fiver_k_at(word, i));
        if (!ok) continue;
        const int E0 = pack4(e[0], e[1], e[2], 0), E1 = pack4(e[3], e[4], e[5], 0);
#pragma unroll
        for (int b = 0; b < BT; b++) {
            if (b0 + b >= batch) break;
            const int2 xv = __ldg(reinterpret_cast<const int2*>(xq3 + (size_t)(b0 + b) * T) + wi);
            acc[b] = __dp4a(E1, xv.y, __dp4a(E0, xv.x, acc[b]));
        }
    }
#pragma unroll
    for (int b = 0; b < BT; b++) acc[b] = dev::warp_sum_i(acc[b]);
    if (lane == 0) {
        const float f   = row_scale[row] * dev::inv_escale[grid];
        const float off = grid == FIVER_GRID_ANCHORED ? dev::escale[grid] : 0.0f;
#pragma unroll
        for (int b = 0; b < BT; b++)
            if (b0 + b < batch)
                Y[(size_t)(b0 + b) * rows + row] = ((float)acc[b] + off * (float)sxq[b0 + b]) * f * sx[b0 + b]
                                                   + (bias ? bias[row] : 0.0f);
    }
}

constexpr uint32_t TC_TILE = 16, TC_KW = 16;   /* 16 words = 96 columns per chunk */

__global__ void k_gemm_tc(const uint32_t* __restrict__ W, const float* __restrict__ row_scale,
                          const int8_t* __restrict__ xq, const float* __restrict__ sx,
                          const int* __restrict__ sxq, float* __restrict__ Y,
                          const float* __restrict__ bias, uint32_t rows, uint32_t ld,
                          uint32_t batch, uint32_t ldq, int grid)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    using namespace nvcuda;
    constexpr uint32_t TC_KC = TC_KW * 6u;
    __shared__ __align__(32) signed char As[TC_KC * TC_TILE];   /* col-major [c][r] */
    __shared__ __align__(32) signed char Bs[TC_KC * TC_TILE];   /* row-major [c][b] */
    __shared__ __align__(32) int         Cs[TC_TILE * TC_TILE];

    const uint32_t lane = threadIdx.x;
    const uint32_t row0 = blockIdx.x * TC_TILE, b0 = blockIdx.y * TC_TILE;
    const int      elut = dev::e[grid][lane];
    const uint32_t pad  = fiver_pad_word(grid);

    wmma::fragment<wmma::accumulator, 16, 16, 16, int> acc;
    wmma::fill_fragment(acc, 0);
    for (uint32_t kw = 0; kw < ld; kw += TC_KW) {
        for (uint32_t j = 0; j < 8u; j++) {
            const uint32_t idx = j * 32u + lane, r = idx & 15u, wc = idx >> 4;
            const uint32_t row = row0 + r;
            const uint32_t word = row < rows ? __ldg(&W[(size_t)row * ld + kw + wc]) : pad;
#pragma unroll
            for (int i = 0; i < 6; i++)
                As[(wc * 6u + (uint32_t)i) * TC_TILE + r] =
                    (signed char)__shfl_sync(FIVER_FULL_MASK, elut, (int)fiver_k_at(word, i));
        }
        for (uint32_t j = 0; j < 12u; j++) {
            const uint32_t idx = j * 32u + lane, b = idx & 15u, q = idx >> 4;
            const int v = __ldg(reinterpret_cast<const int*>(xq + (size_t)(b0 + b) * ldq + kw * 6u) + q);
#pragma unroll
            for (int t = 0; t < 4; t++) Bs[(q * 4u + (uint32_t)t) * TC_TILE + b] = (signed char)(v >> (8 * t));
        }
        __syncwarp();
#pragma unroll
        for (int kk = 0; kk < (int)(TC_KC / 16u); kk++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::col_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::row_major> bf;
            wmma::load_matrix_sync(a, As + kk * 256, TC_TILE);
            wmma::load_matrix_sync(bf, Bs + kk * 256, TC_TILE);
            wmma::mma_sync(acc, a, bf, acc);
        }
        __syncwarp();
    }
    wmma::store_matrix_sync(Cs, acc, TC_TILE, wmma::mem_row_major);
    __syncwarp();
    const float off = grid == FIVER_GRID_ANCHORED ? dev::escale[grid] : 0.0f;
    for (uint32_t idx = lane; idx < TC_TILE * TC_TILE; idx += 32u) {
        const uint32_t r = idx >> 4, b = idx & 15u;
        if (row0 + r < rows && b0 + b < batch)
            Y[(size_t)(b0 + b) * rows + row0 + r] =
                ((float)Cs[idx] + off * (float)sxq[b0 + b]) * row_scale[row0 + r] * dev::inv_escale[grid]
                * sx[b0 + b] + (bias ? bias[row0 + r] : 0.0f);
    }
#endif
}

}  // namespace

void gemm_dp4a(const Matrix& m, const float* x, float* y, uint32_t batch, const float* bias,
               Workspace& ws, int num_sms, uint32_t splits, cudaStream_t s)
{
    const uint32_t npad   = (batch + BN - 1u) / BN * BN;
    const uint32_t groups = m.ld * 6u / 4u;
    const uint32_t steps  = m.ld / BKW;
    const uint32_t tiles  = ((m.rows + BM - 1u) / BM) * (npad / BN);
    if (splits == 0) splits = choose_splits(tiles, steps, num_sms);
    if (splits > steps) splits = steps;
    const uint32_t per = (steps + splits - 1u) / splits;
    splits = (steps + per - 1u) / per;

    int*   xq4 = static_cast<int*>(ws.get(Workspace::XQ4, (size_t)groups * npad * sizeof(int)));
    float* sx  = static_cast<float*>(ws.get(Workspace::SX, npad * sizeof(float)));
    int*   sxq = static_cast<int*>(ws.get(Workspace::SXQ, npad * sizeof(int)));
    FIVER_LAUNCH(k_quant_transpose, dim3(npad), dim3(256), s, x, xq4, sx, sxq, m.cols, batch, npad, groups);

    const dim3 grid((m.rows + BM - 1u) / BM, npad / BN, splits);
    if (splits == 1) {
        FIVER_LAUNCH(k_gemm_dp4a, grid, dim3(128), s, m.words, m.row_scale, xq4, sx, sxq, y,
                     (int*)nullptr, bias, m.rows, m.ld, batch, npad, per, m.grid, 0);
        return;
    }
    int* part = static_cast<int*>(ws.get(Workspace::PART, (size_t)splits * npad * m.rows * sizeof(int)));
    FIVER_LAUNCH(k_gemm_dp4a, grid, dim3(128), s, m.words, m.row_scale, xq4, sx, sxq, y, part, bias,
                 m.rows, m.ld, batch, npad, per, m.grid, 1);
    const size_t n = (size_t)batch * m.rows;
    FIVER_LAUNCH(k_reduce_partials_i32, dim3((unsigned)((n + 255u) / 256u)), dim3(256), s,
                 part, y, m.row_scale, sx, sxq, bias, m.rows, batch, npad, splits, m.grid);
}

/* Quantise rows for the previous kernels: xq [batch_pad16 × ld·6], sx, Σxq. */
static void quantize_rows(const Matrix& m, const float* x, uint32_t batch, Workspace& ws,
                          int8_t** xq, float** sx, int** sxq, uint32_t* ldq, uint32_t* batch_pad,
                          cudaStream_t s)
{
    *ldq       = m.ld * 6u;
    *batch_pad = (batch + 15u) / 16u * 16u;
    *xq  = static_cast<int8_t*>(ws.get(Workspace::XQ, (size_t)*batch_pad * *ldq));
    *sx  = static_cast<float*>(ws.get(Workspace::SX, *batch_pad * sizeof(float)));
    *sxq = static_cast<int*>(ws.get(Workspace::SXQ, *batch_pad * sizeof(int)));
    FIVER_LAUNCH(k_quantize_rows, dim3(*batch_pad), dim3(256), s, x, *xq, *sx, *sxq, m.cols, *ldq, batch);
}

void gemv_dp4a(const Matrix& m, const float* x, float* y, uint32_t batch, const float* bias,
               Workspace& ws, cudaStream_t s)
{
    int8_t* xq; float* sx; int* sxq; uint32_t ldq, bpad;
    quantize_rows(m, x, batch, ws, &xq, &sx, &sxq, &ldq, &bpad, s);
    const uint32_t triplets = 2u * m.ld;
    int* xq3 = static_cast<int*>(ws.get(Workspace::XQ3, (size_t)batch * triplets * sizeof(int)));
    const size_t n3 = (size_t)batch * triplets;
    FIVER_LAUNCH(k_pack_triplets, dim3((unsigned)((n3 + 255u) / 256u)), dim3(256), s, xq, xq3, ldq, triplets, batch);
    constexpr int BT = 4;
    const dim3 grid((m.rows + 3u) / 4u, (batch + BT - 1u) / BT);
    auto k = &k_gemv_dp4a<BT>;
    FIVER_LAUNCH(k, grid, dim3(128), s, m.words, m.row_scale, xq3, sx, sxq, y, bias, m.rows, m.ld, batch, m.grid);
}

bool gemm_tc(const Matrix& m, const float* x, float* y, uint32_t batch, const float* bias,
             Workspace& ws, cudaStream_t s)
{
#if defined(FIVER_EMU)
    (void)m; (void)x; (void)y; (void)batch; (void)bias; (void)ws; (void)s;
    return false;
#else
    int dev_id = 0;
    cudaDeviceProp p;
    if (cudaGetDevice(&dev_id) != cudaSuccess || cudaGetDeviceProperties(&p, dev_id) != cudaSuccess
        || p.major * 10 + p.minor < 75)
        return false;
    int8_t* xq; float* sx; int* sxq; uint32_t ldq, bpad;
    quantize_rows(m, x, batch, ws, &xq, &sx, &sxq, &ldq, &bpad, s);
    const dim3 grid((m.rows + TC_TILE - 1u) / TC_TILE, bpad / TC_TILE);
    FIVER_LAUNCH(k_gemm_tc, grid, dim3(32), s, m.words, m.row_scale, xq, sx, sxq, y, bias, m.rows, m.ld,
                 batch, ldq, m.grid);
    return true;
#endif
}

}  // namespace fiver
