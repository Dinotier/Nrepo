/*
 * backward.cu — fused backward on packed words, deterministic.
 *
 * For one minibatch (dY already contains the activation derivative):
 *
 *   dgrad:     dX[b,c] = Σ_r row_scale[r] · v(k[r,c]) · dY[b,r]
 *              uses the weights before the update.
 *   wgrad_ste: G[r,c] = Σ_b dY[b,r] · X[b,c]   (registers only, never stored)
 *              k ← clamp(k − sign(lr · G), 0, 31);  bias[r] −= lr · Σ_b dY[b,r]
 *
 * All grids increase in k and row_scale ≥ 0, so sign(∂L/∂k) = sign(∂L/∂W).
 * Matrices with per-weight act_bits are refused by the API.
 *
 * dgrad tile: 192 columns (32 words) × 32 batch, rows in chunks of 32.
 *   Lane l owns word l (6 columns), warp j batch rows 8j..8j+7.  The decoded
 *   weight chunk lives in shared memory with a row stride of 194 floats:
 *   the float2 reads Ws[r][6l..6l+5] then hit 32 distinct banks.  The
 *   (row_scale·dY) chunk is read as warp-wide float4 broadcasts.  Per row:
 *   3 + 2 shared loads for 48 FMAs.  Split over row chunks with a fixed-order
 *   reduction — no atomics, bit-identical results run to run.
 *
 * wgrad_ste tile: 64 rows × 96 columns (16 words), batch in chunks of 32.
 *   Thread (warp j, lane l) owns word l % 16 of rows 8g..8g+7 with
 *   g = l/16 + 2j: it accumulates exactly the 48 gradients of the codes it
 *   rewrites, so the in-place update needs no synchronisation.  X is reused
 *   for 64 rows and dY for 96 columns per shared-memory load.
 */
#include "internal.cuh"

namespace fiver {

uint32_t choose_splits(uint32_t base_ctas, uint32_t k_steps, int num_sms);

namespace {

constexpr uint32_t DG_BW = 32, DG_BN = 32, DG_BR = 32, DG_WS = DG_BW * 6 + 2;

__global__ void __launch_bounds__(128)
k_dgrad(const uint32_t* __restrict__ W, const float* __restrict__ row_scale,
        const float* __restrict__ dY, float* __restrict__ out,
        uint32_t rows, uint32_t cols, uint32_t ld, uint32_t batch, uint32_t npad,
        uint32_t chunks_per_split, int grid, int write_partial)
{
    __shared__ __align__(16) float Ws[DG_BR][DG_WS];
    __shared__ __align__(16) float Ds[DG_BR][DG_BN + 4];

    const uint32_t t    = threadIdx.x;
    const uint32_t lane = t & 31u, warp = t >> 5;
    const uint32_t w0   = blockIdx.x * DG_BW;
    const uint32_t n0   = blockIdx.y * DG_BN;
    const uint32_t nch  = (rows + DG_BR - 1u) / DG_BR;
    const uint32_t ch0  = blockIdx.z * chunks_per_split;
    const uint32_t ch1  = min(nch, ch0 + chunks_per_split);
    const float    lut  = dev::lut[grid][lane];
    const uint32_t pad  = fiver_pad_word(grid);

    float acc[8][6];
#pragma unroll
    for (int q = 0; q < 8; q++)
#pragma unroll
        for (int i = 0; i < 6; i++) acc[q][i] = 0.0f;

    for (uint32_t ch = ch0; ch < ch1; ch++) {
        const uint32_t r0 = ch * DG_BR;
        __syncthreads();
        {   /* decode: thread t → chunk row t/4, words (t%4)·8 .. +7 of the tile */
            const uint32_t rr = t >> 2, q = t & 3u, r = r0 + rr, wq = w0 + q * 8u;
            uint4 wa = make_uint4(pad, pad, pad, pad), wb = wa;
            if (r < rows && wq + 8u <= ld) {
                const uint4* p = reinterpret_cast<const uint4*>(W + (size_t)r * ld + wq);
                wa = __ldg(p);
                wb = __ldg(p + 1);
            }
            const uint32_t words[8] = { wa.x, wa.y, wa.z, wa.w, wb.x, wb.y, wb.z, wb.w };
#pragma unroll
            for (int w = 0; w < 8; w++)
#pragma unroll
                for (int i = 0; i < 6; i++)
                    Ws[rr][q * 48u + (uint32_t)(w * 6 + i)] =
                        __shfl_sync(FIVER_FULL_MASK, lut, (int)fiver_k_at(words[w], i));
        }
#pragma unroll
        for (int p = 0; p < 8; p++) {   /* Ds[rr][n] = row_scale · dY, zero outside */
            const uint32_t e = (uint32_t)p * 128u + t, rr = e & 31u, n = e >> 5;
            const uint32_t r = r0 + rr, b = n0 + n;
            Ds[rr][n] = (r < rows && b < batch) ? row_scale[r] * __ldg(&dY[(size_t)b * rows + r]) : 0.0f;
        }
        __syncthreads();
        const uint32_t rmax = min(DG_BR, rows - r0);
        for (uint32_t rr = 0; rr < rmax; rr++) {
            const float2 a  = *reinterpret_cast<const float2*>(&Ws[rr][6u * lane]);
            const float2 b  = *reinterpret_cast<const float2*>(&Ws[rr][6u * lane + 2u]);
            const float2 c  = *reinterpret_cast<const float2*>(&Ws[rr][6u * lane + 4u]);
            const float4 dl = *reinterpret_cast<const float4*>(&Ds[rr][8u * warp]);
            const float4 dh = *reinterpret_cast<const float4*>(&Ds[rr][8u * warp + 4u]);
            const float wv[6] = { a.x, a.y, b.x, b.y, c.x, c.y };
            const float dv[8] = { dl.x, dl.y, dl.z, dl.w, dh.x, dh.y, dh.z, dh.w };
#pragma unroll
            for (int q = 0; q < 8; q++)
#pragma unroll
                for (int i = 0; i < 6; i++) acc[q][i] = fmaf(dv[q], wv[i], acc[q][i]);
        }
    }

    const size_t pbase = (size_t)blockIdx.z * npad * cols;
#pragma unroll
    for (int q = 0; q < 8; q++) {
        const uint32_t n = n0 + 8u * warp + (uint32_t)q;
#pragma unroll
        for (int i = 0; i < 6; i++) {
            const uint32_t c = (w0 + lane) * 6u + (uint32_t)i;
            if (c >= cols) continue;
            if (write_partial) out[pbase + (size_t)n * cols + c] = acc[q][i];
            else if (n < batch) out[(size_t)n * cols + c] = acc[q][i];
        }
    }
}

__global__ void k_reduce_dgrad(const float* __restrict__ part, float* __restrict__ dx,
                               uint32_t cols, uint32_t batch, uint32_t npad, uint32_t splits)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)batch * cols) return;
    float s = 0.0f;
    for (uint32_t z = 0; z < splits; z++) s += part[(size_t)z * npad * cols + i];
    dx[i] = s;
}

constexpr uint32_t WG_BR = 64, WG_BW = 16, WG_BC = WG_BW * 6, WG_BB = 32;

__global__ void __launch_bounds__(128)
k_wgrad_ste(uint32_t* __restrict__ W, const float* __restrict__ X, const float* __restrict__ dY,
            float* __restrict__ bias, uint32_t rows, uint32_t cols, uint32_t wpr, uint32_t ld,
            uint32_t batch, float lr)
{
    __shared__ __align__(16) float Xs[WG_BB][WG_BC];
    __shared__ __align__(16) float Ds[WG_BB][WG_BR];

    const uint32_t t    = threadIdx.x;
    const uint32_t lane = t & 31u, warp = t >> 5;
    const uint32_t w    = lane & 15u;
    const uint32_t g    = (lane >> 4) + 2u * warp;
    const uint32_t r0   = blockIdx.y * WG_BR;
    const uint32_t wt0  = blockIdx.x * WG_BW;
    const uint32_t c0   = wt0 * 6u;
    const bool     do_bias = bias != nullptr && blockIdx.x == 0;

    float acc[8][6], bsum[8];
#pragma unroll
    for (int q = 0; q < 8; q++) {
        bsum[q] = 0.0f;
#pragma unroll
        for (int i = 0; i < 6; i++) acc[q][i] = 0.0f;
    }

    for (uint32_t b0 = 0; b0 < batch; b0 += WG_BB) {
        __syncthreads();
#pragma unroll 4
        for (int p = 0; p < (int)(WG_BB * WG_BC / 128u); p++) {
            const uint32_t e = (uint32_t)p * 128u + t, bb = e / WG_BC, cc = e % WG_BC;
            const uint32_t b = b0 + bb, c = c0 + cc;
            Xs[bb][cc] = (b < batch && c < cols) ? __ldg(&X[(size_t)b * cols + c]) : 0.0f;
        }
#pragma unroll 4
        for (int p = 0; p < (int)(WG_BB * WG_BR / 128u); p++) {
            const uint32_t e = (uint32_t)p * 128u + t, bb = e / WG_BR, rr = e % WG_BR;
            const uint32_t b = b0 + bb, r = r0 + rr;
            Ds[bb][rr] = (b < batch && r < rows) ? __ldg(&dY[(size_t)b * rows + r]) : 0.0f;
        }
        __syncthreads();
        const uint32_t bmax = min(WG_BB, batch - b0);
        for (uint32_t bb = 0; bb < bmax; bb++) {
            const float2 a  = *reinterpret_cast<const float2*>(&Xs[bb][6u * w]);
            const float2 b  = *reinterpret_cast<const float2*>(&Xs[bb][6u * w + 2u]);
            const float2 c  = *reinterpret_cast<const float2*>(&Xs[bb][6u * w + 4u]);
            const float4 dl = *reinterpret_cast<const float4*>(&Ds[bb][8u * g]);
            const float4 dh = *reinterpret_cast<const float4*>(&Ds[bb][8u * g + 4u]);
            const float xv[6] = { a.x, a.y, b.x, b.y, c.x, c.y };
            const float dv[8] = { dl.x, dl.y, dl.z, dl.w, dh.x, dh.y, dh.z, dh.w };
#pragma unroll
            for (int q = 0; q < 8; q++) {
#pragma unroll
                for (int i = 0; i < 6; i++) acc[q][i] = fmaf(dv[q], xv[i], acc[q][i]);
                bsum[q] += dv[q];
            }
        }
    }

    const uint32_t wi = wt0 + w;
#pragma unroll
    for (int q = 0; q < 8; q++) {
        const uint32_t r = r0 + 8u * g + (uint32_t)q;
        if (r >= rows) continue;
        if (wi < wpr) {
            uint32_t* wp   = &W[(size_t)r * ld + wi];
            const uint32_t word = *wp;
            uint32_t out = word;
#pragma unroll
            for (int i = 0; i < 6; i++) {
                if (wi * 6u + (uint32_t)i >= cols) break;          /* padding keeps its code */
                const uint32_t sh = 5u * (uint32_t)i;
                const uint32_t k  = (word >> sh) & 0x1Fu;
                const float    d  = acc[q][i] * lr;
                const uint32_t kn = d > 0.0f ? (k > 0u ? k - 1u : 0u)
                                  : d < 0.0f ? (k < FIVER_K_MAX ? k + 1u : FIVER_K_MAX) : k;
                out = (out & ~(0x1Fu << sh)) | (kn << sh);
            }
            if (out != word) *wp = out;
        }
        if (do_bias && w == 0) bias[r] -= lr * bsum[q];
    }
}

/* ── previous kernels (benchmark references) ─────────────────────────── */
template <int BT>
__global__ void k_dgrad_atomic(const uint32_t* __restrict__ W, const float* __restrict__ row_scale,
                               const float* __restrict__ dY, float* __restrict__ dX,
                               uint32_t rows, uint32_t cols, uint32_t ld, uint32_t batch,
                               uint32_t rows_per_slice, int grid)
{
    const uint32_t wi    = blockIdx.x * blockDim.x + threadIdx.x;
    const bool     valid = wi < ld;
    const uint32_t r0    = blockIdx.y * rows_per_slice;
    const uint32_t r1    = min(rows, r0 + rows_per_slice);
    const uint32_t b0    = blockIdx.z * (uint32_t)BT;
    const float    lut   = dev::lut[grid][threadIdx.x & 31u];

    float acc[BT][6];
#pragma unroll
    for (int b = 0; b < BT; b++)
#pragma unroll
        for (int i = 0; i < 6; i++) acc[b][i] = 0.0f;

    for (uint32_t r = r0; r < r1; r++) {
        const uint32_t word = valid ? __ldg(&W[(size_t)r * ld + wi]) : 0u;
        float w[6];
#pragma unroll
        for (int i = 0; i < 6; i++) w[i] = __shfl_sync(FIVER_FULL_MASK, lut, (int)fiver_k_at(word, i));
        if (!valid) continue;
        const float s = row_scale[r];
#pragma unroll
        for (int b = 0; b < BT; b++) {
            if (b0 + b >= batch) break;
            const float gy = s * __ldg(&dY[(size_t)(b0 + b) * rows + r]);
#pragma unroll
            for (int i = 0; i < 6; i++) acc[b][i] = fmaf(w[i], gy, acc[b][i]);
        }
    }
    if (!valid) return;
#pragma unroll
    for (int b = 0; b < BT; b++) {
        if (b0 + b >= batch) break;
#pragma unroll
        for (int i = 0; i < 6; i++) {
            const uint32_t c = wi * 6u + (uint32_t)i;
            if (c < cols) atomicAdd(&dX[(size_t)(b0 + b) * cols + c], acc[b][i]);
        }
    }
}

constexpr int LEG_RT = 4;

__global__ void k_ste_update_legacy(uint32_t* __restrict__ W, const float* __restrict__ X,
                                    const float* __restrict__ dY, float* __restrict__ bias,
                                    uint32_t rows, uint32_t cols, uint32_t wpr, uint32_t ld,
                                    uint32_t batch, float lr)
{
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row0 = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * LEG_RT;
    if (row0 >= rows) return;
    for (uint32_t wi = lane; wi < wpr; wi += 32u) {
        float g[LEG_RT][6];
#pragma unroll
        for (int q = 0; q < LEG_RT; q++)
#pragma unroll
            for (int i = 0; i < 6; i++) g[q][i] = 0.0f;
        const uint32_t c0 = wi * 6u;
        for (uint32_t b = 0; b < batch; b++) {
            const float* xr = X + (size_t)b * cols;
            float xv[6];
#pragma unroll
            for (int i = 0; i < 6; i++) xv[i] = (c0 + i < cols) ? __ldg(&xr[c0 + i]) : 0.0f;
#pragma unroll
            for (int q = 0; q < LEG_RT; q++) {
                if (row0 + q >= rows) break;
                const float gy = __ldg(&dY[(size_t)b * rows + row0 + q]);
#pragma unroll
                for (int i = 0; i < 6; i++) g[q][i] = fmaf(gy, xv[i], g[q][i]);
            }
        }
#pragma unroll
        for (int q = 0; q < LEG_RT; q++) {
            if (row0 + q >= rows) break;
            uint32_t* wp = &W[(size_t)(row0 + q) * ld + wi];
            const uint32_t word = *wp;
            uint32_t out = word;
#pragma unroll
            for (int i = 0; i < 6; i++) {
                if (c0 + i >= cols) break;
                const uint32_t sh = 5u * (uint32_t)i, k = (word >> sh) & 0x1Fu;
                const float d = g[q][i] * lr;
                const uint32_t kn = d > 0.0f ? (k > 0u ? k - 1u : 0u)
                                  : d < 0.0f ? (k < FIVER_K_MAX ? k + 1u : FIVER_K_MAX) : k;
                out = (out & ~(0x1Fu << sh)) | (kn << sh);
            }
            if (out != word) *wp = out;
        }
    }
    if (bias != nullptr && lane < (uint32_t)LEG_RT && row0 + lane < rows) {
        float gs = 0.0f;
        for (uint32_t b = 0; b < batch; b++) gs += dY[(size_t)b * rows + row0 + lane];
        bias[row0 + lane] -= lr * gs;
    }
}

}  // namespace

void dgrad(const Matrix& m, const float* dy, float* dx, uint32_t batch, Workspace& ws,
           int num_sms, uint32_t splits, cudaStream_t s)
{
    const uint32_t npad  = (batch + DG_BN - 1u) / DG_BN * DG_BN;
    const uint32_t nch   = (m.rows + DG_BR - 1u) / DG_BR;
    const uint32_t ctile = (m.ld + DG_BW - 1u) / DG_BW;
    if (splits == 0) splits = choose_splits(ctile * (npad / DG_BN), nch, num_sms);
    if (splits > nch) splits = nch;
    const uint32_t per = (nch + splits - 1u) / splits;
    splits = (nch + per - 1u) / per;

    const dim3 grid(ctile, npad / DG_BN, splits);
    if (splits == 1) {
        FIVER_LAUNCH(k_dgrad, grid, dim3(128), s, m.words, m.row_scale, dy, dx, m.rows, m.cols, m.ld,
                     batch, npad, per, m.grid, 0);
        return;
    }
    float* part = static_cast<float*>(ws.get(Workspace::PART, (size_t)splits * npad * m.cols * sizeof(float)));
    FIVER_LAUNCH(k_dgrad, grid, dim3(128), s, m.words, m.row_scale, dy, part, m.rows, m.cols, m.ld,
                 batch, npad, per, m.grid, 1);
    const size_t n = (size_t)batch * m.cols;
    FIVER_LAUNCH(k_reduce_dgrad, dim3((unsigned)((n + 255u) / 256u)), dim3(256), s, part, dx, m.cols,
                 batch, npad, splits);
}

void wgrad_ste(const Matrix& m, const float* x, const float* dy, float* bias, uint32_t batch,
               float lr, cudaStream_t s)
{
    const dim3 grid((m.wpr + WG_BW - 1u) / WG_BW, (m.rows + WG_BR - 1u) / WG_BR);
    FIVER_LAUNCH(k_wgrad_ste, grid, dim3(128), s, m.words, x, dy, bias, m.rows, m.cols, m.wpr, m.ld,
                 batch, lr);
}

void dgrad_atomic_legacy(const Matrix& m, const float* dy, float* dx, uint32_t batch, cudaStream_t s)
{
    constexpr int BT = 4;
    const uint32_t rows_per_slice = 64u;
    cudaMemsetAsync(dx, 0, (size_t)batch * m.cols * sizeof(float), s);
    const dim3 grid((m.ld + 127u) / 128u, (m.rows + rows_per_slice - 1u) / rows_per_slice, (batch + BT - 1u) / BT);
    auto k = &k_dgrad_atomic<BT>;
    FIVER_LAUNCH(k, grid, dim3(128), s, m.words, m.row_scale, dy, dx, m.rows, m.cols, m.ld, batch,
                 rows_per_slice, m.grid);
}

void ste_update_legacy(const Matrix& m, const float* x, const float* dy, float* bias, uint32_t batch,
                       float lr, cudaStream_t s)
{
    const uint32_t rows_per_block = 4u * LEG_RT;
    FIVER_LAUNCH(k_ste_update_legacy, dim3((m.rows + rows_per_block - 1u) / rows_per_block), dim3(128), s,
                 m.words, x, dy, bias, m.rows, m.cols, m.wpr, m.ld, batch, lr);
}

}  // namespace fiver
