/*
 * forward_gemv.cu — float GEMV paths (batch ≤ 8).
 *
 * V1 (reference): one warp per output row, x read from global per lane.
 *   Per word: 1 weight load, 6 x loads, 6 shuffles — load/shuffle issue
 *   bound at roughly the DRAM rate on sm_61 (estimate).
 *
 * V2: one warp handles RPW rows; the CTA (4 warps) stages a chunk of x in
 *   shared memory, transposed per word ("6 planes"), so lane l reads
 *   xs[b][i][word l] conflict-free and reuses it for all RPW rows.
 *   Per word and row: 1 weight load, 6 shuffles, 6·BT FMAs; x loads are
 *   amortised over RPW rows.  Lane ↔ word interleaving keeps every weight
 *   load a contiguous 128-byte warp access.
 *   MAGIC_BITS decodes with the shift/subtract formula instead of the LUT
 *   (benchmark comparison only).  ACT applies per-weight act_bits.
 */
#include "internal.cuh"

namespace fiver {
namespace {

constexpr int GEMV_WARPS = 4;

/* ── V1 ───────────────────────────────────────────────────────────────── */
template <int BT>
__global__ void k_gemv_v1(const uint32_t* __restrict__ W, const float* __restrict__ row_scale,
                          const uint32_t* __restrict__ act, const float* __restrict__ X,
                          float* __restrict__ Y, const float* __restrict__ bias,
                          uint32_t rows, uint32_t cols, uint32_t ld, uint32_t batch, int grid)
{
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row  = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const uint32_t b0   = blockIdx.y * (uint32_t)BT;
    const float    lut  = dev::lut[grid][lane];
    if (row >= rows) return;                                   /* uniform per warp */

    float acc[BT];
#pragma unroll
    for (int b = 0; b < BT; b++) acc[b] = 0.0f;

    for (uint32_t base = 0; base < ld; base += 32u) {
        const uint32_t wi   = base + lane;
        const bool     ok   = wi < ld;
        const uint32_t word = ok ? __ldg(&W[(size_t)row * ld + wi]) : 0u;
        float w[6];
#pragma unroll
        for (int i = 0; i < 6; i++) w[i] = __shfl_sync(FIVER_FULL_MASK, lut, (int)fiver_k_at(word, i));
        if (!ok) continue;
        if (act) {
#pragma unroll
            for (int i = 0; i < 6; i++)
                w[i] = dev::apply_act(w[i], dev::act_code(act, ((size_t)row * ld + wi) * 6u + i));
        }
        const uint32_t c0 = wi * 6u;
#pragma unroll
        for (int b = 0; b < BT; b++) {
            if (b0 + b >= batch) break;
            const float* xr = X + (size_t)(b0 + b) * cols;
#pragma unroll
            for (int i = 0; i < 6; i++)
                if (c0 + i < cols) acc[b] = fmaf(w[i], __ldg(&xr[c0 + i]), acc[b]);
        }
    }
#pragma unroll
    for (int b = 0; b < BT; b++) acc[b] = dev::warp_sum(acc[b]);
    if (lane == 0) {
        const float s  = row_scale[row];
        const float bb = bias ? bias[row] : 0.0f;
#pragma unroll
        for (int b = 0; b < BT; b++)
            if (b0 + b < batch) Y[(size_t)(b0 + b) * rows + row] = acc[b] * s + bb;
    }
}

/* ── V2 ───────────────────────────────────────────────────────────────── */
constexpr uint32_t V2_KW = 128;          /* words per x chunk (4 per lane) = 768 columns */
constexpr uint32_t V2_PS = V2_KW + 4;    /* plane stride: 2-way at most on the fill stores */

template <int RPW, int BT, bool MAGIC_BITS, bool ACT>
__global__ void k_gemv_v2(const uint32_t* __restrict__ W, const float* __restrict__ row_scale,
                          const uint32_t* __restrict__ act, const float* __restrict__ X,
                          float* __restrict__ Y, const float* __restrict__ bias,
                          uint32_t rows, uint32_t cols, uint32_t ld, uint32_t batch, int grid)
{
    __shared__ float xs[BT][6][V2_PS];

    const uint32_t tid   = threadIdx.x;
    const uint32_t lane  = tid & 31u;
    const uint32_t row0  = (blockIdx.x * GEMV_WARPS + (tid >> 5)) * (uint32_t)RPW;
    const uint32_t b0    = blockIdx.y * (uint32_t)BT;
    const uint32_t pad   = fiver_pad_word(grid);
    const float    lut   = dev::lut[grid][lane];

    float acc[RPW][BT];
#pragma unroll
    for (int q = 0; q < RPW; q++)
#pragma unroll
        for (int b = 0; b < BT; b++) acc[q][b] = 0.0f;

    for (uint32_t w0 = 0; w0 < ld; w0 += V2_KW) {
        /* stage x[b0 .. b0+BT) × columns of this chunk, zero beyond cols / batch */
        __syncthreads();
        for (uint32_t idx = tid; idx < (uint32_t)BT * V2_KW * 6u; idx += blockDim.x) {
            const uint32_t b  = idx / (V2_KW * 6u);
            const uint32_t cc = idx % (V2_KW * 6u);
            const uint32_t c  = w0 * 6u + cc;
            float v = 0.0f;
            if (b0 + b < batch && c < cols) v = __ldg(&X[(size_t)(b0 + b) * cols + c]);
            xs[b][cc % 6u][cc / 6u] = v;
        }
        __syncthreads();

#pragma unroll
        for (int j = 0; j < (int)(V2_KW / 32u); j++) {
            const uint32_t wl = lane + 32u * (uint32_t)j;
            const uint32_t w  = w0 + wl;
            float xv[BT][6];
#pragma unroll
            for (int b = 0; b < BT; b++)
#pragma unroll
                for (int i = 0; i < 6; i++) xv[b][i] = xs[b][i][wl];

#pragma unroll
            for (int q = 0; q < RPW; q++) {
                const uint32_t row = row0 + (uint32_t)q;
                if (row >= rows) break;                        /* uniform per warp */
                const uint32_t word = w < ld ? __ldg(&W[(size_t)row * ld + w]) : pad;
                float v[6];
#pragma unroll
                for (int i = 0; i < 6; i++) {
                    const uint32_t k = fiver_k_at(word, i);
                    v[i] = MAGIC_BITS ? fiver_magic_value(k) : __shfl_sync(FIVER_FULL_MASK, lut, (int)k);
                }
                if (ACT && w < ld) {
#pragma unroll
                    for (int i = 0; i < 6; i++)
                        v[i] = dev::apply_act(v[i], dev::act_code(act, ((size_t)row * ld + w) * 6u + i));
                }
#pragma unroll
                for (int b = 0; b < BT; b++)
#pragma unroll
                    for (int i = 0; i < 6; i++) acc[q][b] = fmaf(v[i], xv[b][i], acc[q][b]);
            }
        }
    }

#pragma unroll
    for (int q = 0; q < RPW; q++) {
        const uint32_t row = row0 + (uint32_t)q;
        float s = 0.0f, bb = 0.0f;
        if (row < rows) { s = row_scale[row]; bb = bias ? bias[row] : 0.0f; }
#pragma unroll
        for (int b = 0; b < BT; b++) {
            const float t = dev::warp_sum(acc[q][b]);
            if (lane == 0 && row < rows && b0 + b < batch) Y[(size_t)(b0 + b) * rows + row] = t * s + bb;
        }
    }
}

template <int RPW, int BT>
void launch_v2(const Matrix& m, const float* x, float* y, uint32_t batch, const float* bias,
               bool magic_bits, cudaStream_t s)
{
    const dim3 grid((m.rows + GEMV_WARPS * RPW - 1) / (GEMV_WARPS * RPW), (batch + BT - 1) / BT);
    const dim3 block(32 * GEMV_WARPS);
    if (m.act_bits) {
        auto k = &k_gemv_v2<RPW, BT, false, true>;
        FIVER_LAUNCH(k, grid, block, s, m.words, m.row_scale, m.act_bits, x, y, bias,
                     m.rows, m.cols, m.ld, batch, m.grid);
    } else if (magic_bits && m.grid == FIVER_GRID_MAGIC) {
        auto k = &k_gemv_v2<RPW, BT, true, false>;
        FIVER_LAUNCH(k, grid, block, s, m.words, m.row_scale, m.act_bits, x, y, bias,
                     m.rows, m.cols, m.ld, batch, m.grid);
    } else {
        auto k = &k_gemv_v2<RPW, BT, false, false>;
        FIVER_LAUNCH(k, grid, block, s, m.words, m.row_scale, m.act_bits, x, y, bias,
                     m.rows, m.cols, m.ld, batch, m.grid);
    }
}

}  // namespace

void gemv_float(const Matrix& m, const float* x, float* y, uint32_t batch, const float* bias,
                GemvVariant variant, cudaStream_t s)
{
    if (variant == GEMV_V1) {
        constexpr int BT = 4;
        const dim3 grid((m.rows + GEMV_WARPS - 1) / GEMV_WARPS, (batch + BT - 1) / BT);
        auto k = &k_gemv_v1<BT>;
        FIVER_LAUNCH(k, grid, dim3(32 * GEMV_WARPS), s, m.words, m.row_scale, m.act_bits, x, y,
                     bias, m.rows, m.cols, m.ld, batch, m.grid);
        return;
    }
    const bool bits = variant == GEMV_V2_MAGIC_BITS;
    if (batch <= 4) launch_v2<4, 4>(m, x, y, batch, bias, bits, s);
    else            launch_v2<2, 8>(m, x, y, batch, bias, bits, s);
}

}  // namespace fiver
