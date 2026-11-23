/*
 * prmt_lut.cu — int8 table decodes on the GPU: shuffle + shift/or versus
 * shuffle + PRMT (__byte_perm), producing packed int8x4 operands for __dp4a.
 *
 * A 32-bit shuffle moves one word, so a lane can hold 4 int8 table entries:
 * 32 lanes × 4 bytes = 128 entries (7 bit) behind one shuffle, two registers
 * per lane give 256 (8 bit).  PRMT then picks the right byte out of the
 * fetched word(s) and packs bytes of several codes into one register — the
 * selection and the packing in the same instruction.
 *
 *   e5_shift / e5_prmt   Fiver words (6 codes/word, 32-entry e table) → two
 *                        triplet operands E0, E1 as used by gemv_dp4a
 *   e7_shift / e7_prmt   byte codes < 128 → one int8x4 per 4 codes
 *   e8_shift / e8_prmt   byte codes < 256 → one int8x4 per 4 codes
 *
 * Correctness is checked against a host table lookup (GPU or CPU emulator);
 * instruction counts come from the sm_61 SASS (see docs/ARCHITEKTUR_CUDA.md).
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "fiver/compat.h"
#include "fiver/format.h"

#define FULL 0xFFFFFFFFu

static __constant__ int8_t T8[256];

__device__ __forceinline__ unsigned lane_word(unsigned base) {   /* bytes T8[base .. base+3] */
    return (unsigned)(uint8_t)T8[base] | ((unsigned)(uint8_t)T8[base + 1] << 8) |
           ((unsigned)(uint8_t)T8[base + 2] << 16) | ((unsigned)(uint8_t)T8[base + 3] << 24);
}

/* ── 5 bit, Fiver words → E0 = (e0,e1,e2,·), E1 = (e3,e4,e5,·) ───────── */
__global__ void e5_shift(const uint32_t* __restrict__ W, int2* __restrict__ out, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int elut = T8[threadIdx.x & 31u];
    const uint32_t w = i < n ? W[i] : 0u;
    int e[6];
#pragma unroll
    for (int j = 0; j < 6; j++) e[j] = __shfl_sync(FULL, elut, (int)fiver_k_at(w, j));
    if (i < n) out[i] = make_int2((e[0] & 0xFF) | ((e[1] & 0xFF) << 8) | ((e[2] & 0xFF) << 16),
                                  (e[3] & 0xFF) | ((e[4] & 0xFF) << 8) | ((e[5] & 0xFF) << 16));
}

__global__ void e5_prmt(const uint32_t* __restrict__ W, int2* __restrict__ out, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned t4 = lane_word(4u * (threadIdx.x & 7u));        /* lane L%8 holds e[4L..4L+3] */
    const uint32_t w = i < n ? W[i] : 0u;
    unsigned v[6], s[6];
#pragma unroll
    for (int j = 0; j < 6; j++) {
        const uint32_t k = fiver_k_at(w, j);
        v[j] = __shfl_sync(FULL, t4, (int)(k >> 2));
        s[j] = k & 3u;
    }
    const unsigned E0 = __byte_perm(__byte_perm(v[0], v[1], s[0] | ((s[1] + 4u) << 4)), v[2], 0x10u | ((s[2] + 4u) << 8));
    const unsigned E1 = __byte_perm(__byte_perm(v[3], v[4], s[3] | ((s[4] + 4u) << 4)), v[5], 0x10u | ((s[5] + 4u) << 8));
    if (i < n) out[i] = make_int2((int)E0, (int)E1);
}

/* ── 7 bit, byte codes → int8x4 ─────────────────────────────────────── */
__global__ void e7_shift(const uint32_t* __restrict__ C, unsigned* __restrict__ out, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned t4 = lane_word(4u * (threadIdx.x & 31u));
    const uint32_t c = i < n ? C[i] : 0u;
    unsigned r = 0;
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const unsigned k = (c >> (8 * j)) & 127u;
        const unsigned v = __shfl_sync(FULL, t4, (int)(k >> 2));
        r |= ((v >> (8u * (k & 3u))) & 0xFFu) << (8 * j);
    }
    if (i < n) out[i] = r;
}

__global__ void e7_prmt(const uint32_t* __restrict__ C, unsigned* __restrict__ out, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned t4 = lane_word(4u * (threadIdx.x & 31u));
    const uint32_t c = i < n ? C[i] : 0u;
    unsigned v[4], s[4];
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const unsigned k = (c >> (8 * j)) & 127u;
        v[j] = __shfl_sync(FULL, t4, (int)(k >> 2));
        s[j] = k & 3u;
    }
    const unsigned p01 = __byte_perm(v[0], v[1], s[0] | ((s[1] + 4u) << 4));
    const unsigned p23 = __byte_perm(v[2], v[3], s[2] | ((s[3] + 4u) << 4));
    if (i < n) out[i] = __byte_perm(p01, p23, 0x5410u);
}

/* ── 8 bit, byte codes → int8x4 (two table words per lane) ──────────── */
__global__ void e8_shift(const uint32_t* __restrict__ C, unsigned* __restrict__ out, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned lo = lane_word(4u * (threadIdx.x & 31u)), hi = lane_word(128u + 4u * (threadIdx.x & 31u));
    const uint32_t c = i < n ? C[i] : 0u;
    unsigned r = 0;
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const unsigned k = (c >> (8 * j)) & 255u;
        const unsigned a = __shfl_sync(FULL, lo, (int)((k >> 2) & 31u));
        const unsigned b = __shfl_sync(FULL, hi, (int)((k >> 2) & 31u));
        r |= (((k & 128u) ? b : a) >> (8u * (k & 3u)) & 0xFFu) << (8 * j);
    }
    if (i < n) out[i] = r;
}

__global__ void e8_prmt(const uint32_t* __restrict__ C, unsigned* __restrict__ out, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned lo = lane_word(4u * (threadIdx.x & 31u)), hi = lane_word(128u + 4u * (threadIdx.x & 31u));
    const uint32_t c = i < n ? C[i] : 0u;
    unsigned b[4];
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const unsigned k = (c >> (8 * j)) & 255u;
        const unsigned a = __shfl_sync(FULL, lo, (int)((k >> 2) & 31u));
        const unsigned h = __shfl_sync(FULL, hi, (int)((k >> 2) & 31u));
        b[j] = __byte_perm(a, h, (k & 3u) | ((k >> 5) & 4u));    /* byte k&3 of a (k<128) or h */
    }
    if (i < n) out[i] = __byte_perm(__byte_perm(b[0], b[1], 0x40u), __byte_perm(b[2], b[3], 0x40u), 0x5410u);
}

/* ── driver ───────────────────────────────────────────────────────────── */
int main() {
    int8_t tab[256];
    srand(3);
    for (int i = 0; i < 256; i++) tab[i] = (int8_t)(rand() & 0xFF);
    for (int i = 0; i < 32; i++) tab[i] = (int8_t)fiver_e_host[FIVER_GRID_CENTERED][i];
#ifdef FIVER_EMU
    memcpy(T8, tab, sizeof(tab));
#else
    cudaMemcpyToSymbol(T8, tab, sizeof(tab));
#endif
    const uint32_t n = 1u << 14;
    std::vector<uint32_t> words(n), c7(n), c8(n);
    for (uint32_t i = 0; i < n; i++) {
        uint32_t w = 0;
        for (int j = 0; j < 6; j++) w |= (uint32_t)(rand() & 31) << (5 * j);
        words[i] = w | ((uint32_t)(rand() & 3) << 30);
        c7[i] = (uint32_t)rand() & 0x7F7F7F7Fu;
        c8[i] = (uint32_t)rand() | ((uint32_t)rand() << 16);
    }
    uint32_t *dW, *d7, *d8; unsigned* dout; int2* dout2;
    cudaMalloc(&dW, n * 4); cudaMalloc(&d7, n * 4); cudaMalloc(&d8, n * 4);
    cudaMalloc(&dout, n * 4); cudaMalloc(&dout2, n * 8);
    cudaMemcpy(dW, words.data(), n * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d7, c7.data(), n * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d8, c8.data(), n * 4, cudaMemcpyHostToDevice);
    const dim3 grid(n / 128u), block(128);
    int fails = 0;

    for (int v = 0; v < 2; v++) {
        if (v == 0) FIVER_LAUNCH(e5_shift, grid, block, 0, dW, dout2, n);
        else        FIVER_LAUNCH(e5_prmt, grid, block, 0, dW, dout2, n);
        cudaDeviceSynchronize();
        std::vector<int2> o(n);
        cudaMemcpy(o.data(), dout2, n * 8, cudaMemcpyDeviceToHost);
        uint32_t bad = 0;
        for (uint32_t i = 0; i < n; i++)
            for (int j = 0; j < 6; j++) {
                const int reg = j < 3 ? o[i].x : o[i].y;
                bad += (int8_t)(reg >> (8 * (j % 3))) != tab[fiver_k_at(words[i], j)];
            }
        fails += bad != 0;
        printf("  %-9s 5-bit Fiver words -> dp4a triplets  %s\n", v ? "e5_prmt" : "e5_shift", bad ? "FAIL" : "ok");
    }
    struct { const char* name; void (*k)(const uint32_t*, unsigned*, uint32_t); uint32_t* in; const std::vector<uint32_t>* h; int bits; } K[] = {
        { "e7_shift", e7_shift, d7, &c7, 7 }, { "e7_prmt", e7_prmt, d7, &c7, 7 },
        { "e8_shift", e8_shift, d8, &c8, 8 }, { "e8_prmt", e8_prmt, d8, &c8, 8 } };
    for (auto& k : K) {
        auto fn = k.k;
        FIVER_LAUNCH(fn, grid, block, 0, k.in, dout, n);
        cudaDeviceSynchronize();
        std::vector<unsigned> o(n);
        cudaMemcpy(o.data(), dout, n * 4, cudaMemcpyDeviceToHost);
        uint32_t bad = 0;
        for (uint32_t i = 0; i < n; i++)
            for (int j = 0; j < 4; j++)
                bad += (int8_t)(o[i] >> (8 * j)) != tab[((*k.h)[i] >> (8 * j)) & ((1u << k.bits) - 1u)];
        fails += bad != 0;
        printf("  %-9s %d-bit byte codes -> int8x4         %s\n", k.name, k.bits, bad ? "FAIL" : "ok");
    }
    printf("%s\n", fails ? "PRMT TEST FAILED" : "PRMT TEST PASSED");
    return fails != 0;
}
