/*
 * forward_gemm.cu — tiled FP32 GEMM for batch > 8 (sm_61 SIMT, runs everywhere).
 *
 *   Y[b, r] = row_scale[r] · Σ_c v(k[r, c]) · X[b, c] (+ bias[r])
 *
 * As a GEMM: C (rows × batch) = A (rows × K, decoded W) · B (K × batch, Xᵀ).
 *
 * CTA tile 128 rows × 32 batch, K step 48 columns (8 words), 128 threads.
 *   decode:  thread t owns tile row t: 2 × uint4 loads (8 words), 48 shuffles,
 *            stores As[k][t] — consecutive lanes, consecutive banks.
 *   compute: warp j owns batch columns 8j..8j+7, lane l rows 4l..4l+3:
 *            per k one float4 of A (lane-contiguous) and two float4 of B that
 *            are identical for the whole warp (shared-memory broadcast), then
 *            32 FMAs.  On Pascal this is ~6 shared-memory wavefronts against
 *            8 FMA issue cycles per warp and k: FMA-bound by design.
 *   pipeline: the next tile's words and x are prefetched into registers while
 *            the current tile is computed (one shared buffer, two barriers).
 *
 * X is first transposed and zero-padded to xt[kpad × npad] (kpad = ld·6,
 * npad = multiple of 32), so the inner loops have no bounds checks.
 *
 * Split-K over the K steps keeps all SMs busy for few row tiles (EmojiNet:
 * 512 × 10000).  Partials go to a buffer and are summed in a fixed order by
 * a second kernel: results are deterministic, no atomics.
 */
#include "internal.cuh"

namespace fiver {
namespace {

constexpr uint32_t BM = 128, BN = 32, BKW = 8, BK = BKW * 6;

__global__ void k_transpose_pad(const float* __restrict__ x, float* __restrict__ xt,
                                uint32_t batch, uint32_t cols, uint32_t npad, uint32_t kpad)
{
    __shared__ float tile[32][33];
    const uint32_t c0 = blockIdx.x * 32u, b0 = blockIdx.y * 32u;
    for (uint32_t j = threadIdx.y; j < 32u; j += 8u) {
        const uint32_t b = b0 + j, c = c0 + threadIdx.x;
        tile[j][threadIdx.x] = (b < batch && c < cols) ? __ldg(&x[(size_t)b * cols + c]) : 0.0f;
    }
    __syncthreads();
    for (uint32_t j = threadIdx.y; j < 32u; j += 8u) {
        const uint32_t c = c0 + j, b = b0 + threadIdx.x;
        if (c < kpad && b < npad) xt[(size_t)c * npad + b] = tile[threadIdx.x][j];
    }
}

__global__ void __launch_bounds__(128)
k_gemm_f32(const uint32_t* __restrict__ W, const float* __restrict__ row_scale,
           const float* __restrict__ xt, float* __restrict__ out, const float* __restrict__ bias,
           uint32_t rows, uint32_t ld, uint32_t batch, uint32_t npad,
           uint32_t steps_per_split, int grid, int write_partial)
{
    __shared__ __align__(16) float As[BK][BM];
    __shared__ __align__(16) float Bs[BK][BN];

    const uint32_t t    = threadIdx.x;
    const uint32_t lane = t & 31u, warp = t >> 5;
    const uint32_t row0 = blockIdx.x * BM;
    const uint32_t n0   = blockIdx.y * BN;
    const uint32_t ks0  = blockIdx.z * steps_per_split;
    const uint32_t ks1  = min(ld / BKW, ks0 + steps_per_split);
    const float    lut  = dev::lut[grid][lane];
    const uint32_t pad  = fiver_pad_word(grid);

    float acc[4][8];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 8; j++) acc[i][j] = 0.0f;

    uint4  wa, wb;
    float4 xb[3];
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
            const uint32_t e = (uint32_t)p * 128u + t, k = e >> 3, n4 = e & 7u;
            xb[p] = __ldg(reinterpret_cast<const float4*>(xt + (size_t)(ks * BK + k) * npad + n0 + 4u * n4));
        }
    };
    auto store = [&]() {
        const uint32_t words[8] = { wa.x, wa.y, wa.z, wa.w, wb.x, wb.y, wb.z, wb.w };
#pragma unroll
        for (int w = 0; w < 8; w++)
#pragma unroll
            for (int i = 0; i < 6; i++)
                As[w * 6 + i][t] = __shfl_sync(FIVER_FULL_MASK, lut, (int)fiver_k_at(words[w], i));
#pragma unroll
        for (int p = 0; p < 3; p++) {
            const uint32_t e = (uint32_t)p * 128u + t, k = e >> 3, n4 = e & 7u;
            *reinterpret_cast<float4*>(&Bs[k][4u * n4]) = xb[p];
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
#pragma unroll 8
        for (uint32_t k = 0; k < BK; k++) {
            const float4 a  = *reinterpret_cast<const float4*>(&As[k][4u * lane]);
            const float4 bl = *reinterpret_cast<const float4*>(&Bs[k][8u * warp]);
            const float4 bh = *reinterpret_cast<const float4*>(&Bs[k][8u * warp + 4u]);
            const float av[4] = { a.x, a.y, a.z, a.w };
            const float bv[8] = { bl.x, bl.y, bl.z, bl.w, bh.x, bh.y, bh.z, bh.w };
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 8; j++) acc[i][j] = fmaf(av[i], bv[j], acc[i][j]);
        }
        __syncthreads();
        if (more) {
            store();
            __syncthreads();
        }
    }

    const size_t part = (size_t)blockIdx.z * npad * rows;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const uint32_t r = row0 + 4u * lane + (uint32_t)i;
        if (r >= rows) continue;
        const float s  = write_partial ? 1.0f : row_scale[r];
        const float bb = (!write_partial && bias) ? bias[r] : 0.0f;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const uint32_t n = n0 + 8u * warp + (uint32_t)j;
            if (write_partial) out[part + (size_t)n * rows + r] = acc[i][j];
            else if (n < batch) out[(size_t)n * rows + r] = acc[i][j] * s + bb;
        }
    }
}

__global__ void k_reduce_partials_f32(const float* __restrict__ part, float* __restrict__ y,
                                      const float* __restrict__ row_scale, const float* __restrict__ bias,
                                      uint32_t rows, uint32_t batch, uint32_t npad, uint32_t splits)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)batch * rows) return;
    const uint32_t r = (uint32_t)(i % rows);
    float sum = 0.0f;
    for (uint32_t s = 0; s < splits; s++) sum += part[(size_t)s * npad * rows + i];
    y[i] = sum * row_scale[r] + (bias ? bias[r] : 0.0f);
}

}  // namespace

uint32_t choose_splits(uint32_t base_ctas, uint32_t k_steps, int num_sms)
{
    uint32_t s = 1;
    const uint32_t target = 2u * (uint32_t)(num_sms > 0 ? num_sms : 1);
    while (base_ctas * s < target && s * 2u <= k_steps && s < 16u) s *= 2u;
    return s;
}

void transpose_pad(const float* x, float* xt, uint32_t batch, uint32_t cols, uint32_t npad,
                   uint32_t kpad, cudaStream_t s)
{
    const dim3 grid((kpad + 31u) / 32u, (npad + 31u) / 32u);
    FIVER_LAUNCH(k_transpose_pad, grid, dim3(32, 8), s, x, xt, batch, cols, npad, kpad);
}

void gemm_float(const Matrix& m, const float* x, float* y, uint32_t batch, const float* bias,
                Workspace& ws, int num_sms, uint32_t splits, cudaStream_t s)
{
    const uint32_t npad  = (batch + BN - 1u) / BN * BN;
    const uint32_t kpad  = m.ld * 6u;
    const uint32_t steps = m.ld / BKW;
    const uint32_t tiles = ((m.rows + BM - 1u) / BM) * (npad / BN);
    if (splits == 0) splits = choose_splits(tiles, steps, num_sms);
    if (splits > steps) splits = steps;
    const uint32_t per = (steps + splits - 1u) / splits;
    splits = (steps + per - 1u) / per;

    float* xt = static_cast<float*>(ws.get(Workspace::XT, (size_t)kpad * npad * sizeof(float)));
    transpose_pad(x, xt, batch, m.cols, npad, kpad, s);

    const dim3 grid((m.rows + BM - 1u) / BM, npad / BN, splits);
    if (splits == 1) {
        FIVER_LAUNCH(k_gemm_f32, grid, dim3(128), s, m.words, m.row_scale, xt, y, bias,
                     m.rows, m.ld, batch, npad, per, m.grid, 0);
        return;
    }
    float* part = static_cast<float*>(ws.get(Workspace::PART, (size_t)splits * npad * m.rows * sizeof(float)));
    FIVER_LAUNCH(k_gemm_f32, grid, dim3(128), s, m.words, m.row_scale, xt, part, bias,
                 m.rows, m.ld, batch, npad, per, m.grid, 1);
    const size_t n = (size_t)batch * m.rows;
    FIVER_LAUNCH(k_reduce_partials_f32, dim3((unsigned)((n + 255u) / 256u)), dim3(256), s,
                 part, y, m.row_scale, bias, m.rows, batch, npad, splits);
}

}  // namespace fiver
