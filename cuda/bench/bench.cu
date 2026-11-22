/*
 * bench.cu — performance of all paths, reported against the hardware limits.
 *
 *   --bench-gemv   batch 1/4/8, 4096 × 4096:   GEMV v1 / v2 / v2 magic bits / dp4a / dense FP32
 *   --bench-gemm   batch 16…128 + EmojiNet layer shapes: tiled FP32 / tiled dp4a / TC / GEMV v2
 *   --bench-bwd    dgrad tiled vs atomic, wgrad+STE tiled vs previous kernel
 *
 * "% DRAM" uses the packed weight bytes only, "% peak" the FP32 (or 4× for
 * __dp4a) arithmetic peak from the device properties.  Numbers vary with
 * clocks; run a few times.
 */
#include <cmath>
#include <cstdio>
#include <vector>

#include "../src/internal.cuh"

using namespace fiver;

namespace {

struct Peak { double dram_gbs, fp32_tflops; };

Peak device_peak() {
    int d = 0;
    cudaDeviceProp p;
    cudaGetDevice(&d);
    cudaGetDeviceProperties(&p, d);
    const int sm = p.major * 10 + p.minor;
    const int cores = (sm == 61 || sm == 86 || sm == 89) ? 128 : (sm >= 70 ? 64 : 128);
    int clock_khz = 0, mem_khz = 0, bus = 0;
#ifndef FIVER_EMU
    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, d);
    cudaDeviceGetAttribute(&mem_khz, cudaDevAttrMemoryClockRate, d);
    cudaDeviceGetAttribute(&bus, cudaDevAttrGlobalMemoryBusWidth, d);
#endif
    return { 2.0 * mem_khz * 1e3 * (bus / 8) / 1e9, (double)p.multiProcessorCount * cores * clock_khz * 1e3 * 2 / 1e12 };
}

template <class F>
double time_ms(F f, int it) {
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    f();
    cudaDeviceSynchronize();
    cudaEventRecord(a);
    for (int i = 0; i < it; i++) f();
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms / it;
}

uint32_t rs = 0x9E3779B9u;
uint32_t rnd() { rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5; return rs; }

struct Bench {
    Matrix m;
    float *x = nullptr, *y = nullptr, *wf = nullptr, *dy = nullptr, *dx = nullptr;
    uint32_t batch = 0;
    Bench(int grid, uint32_t rows, uint32_t cols, uint32_t batch_, bool dense) : batch(batch_) {
        m.rows = rows; m.cols = cols; m.wpr = fiver_words_per_row(cols); m.ld = fiver_row_pitch(cols); m.grid = grid;
        std::vector<uint32_t> w((size_t)rows * m.ld, fiver_pad_word(grid));
        for (uint32_t r = 0; r < rows; r++)
            for (uint32_t i = 0; i < m.wpr; i++) {
                uint8_t k[6];
                for (int j = 0; j < 6; j++) k[j] = (i * 6 + j < cols) ? (uint8_t)(rnd() & 31u) : (uint8_t)fiver_zero_code(grid);
                w[(size_t)r * m.ld + i] = fiver_pack_word(k);
            }
        std::vector<float> s(rows, 0.05f), xv((size_t)batch * cols), dyv((size_t)batch * rows);
        for (auto& v : xv)  v = (float)(rnd() & 0xFFFF) / 32768.0f - 1.0f;
        for (auto& v : dyv) v = (float)(rnd() & 0xFFFF) / 32768.0f - 1.0f;
        cudaMalloc(&m.words, w.size() * 4);     cudaMemcpy(m.words, w.data(), w.size() * 4, cudaMemcpyHostToDevice);
        cudaMalloc(&m.row_scale, rows * 4);     cudaMemcpy(m.row_scale, s.data(), rows * 4, cudaMemcpyHostToDevice);
        cudaMalloc(&x, xv.size() * 4);          cudaMemcpy(x, xv.data(), xv.size() * 4, cudaMemcpyHostToDevice);
        cudaMalloc(&dy, dyv.size() * 4);        cudaMemcpy(dy, dyv.data(), dyv.size() * 4, cudaMemcpyHostToDevice);
        cudaMalloc(&y, (size_t)batch * rows * 4);
        cudaMalloc(&dx, (size_t)batch * cols * 4);
        if (dense) {
            std::vector<float> wfv((size_t)rows * cols);
            for (uint32_t r = 0; r < rows; r++)
                for (uint32_t c = 0; c < cols; c++)
                    wfv[(size_t)r * cols + c] = 0.05f * fiver_lut_host[grid][fiver_k_at(w[(size_t)r * m.ld + c / 6], (int)(c % 6))];
            cudaMalloc(&wf, wfv.size() * 4);
            cudaMemcpy(wf, wfv.data(), wfv.size() * 4, cudaMemcpyHostToDevice);
        }
    }
    ~Bench() { cudaFree(m.words); cudaFree(m.row_scale); cudaFree(x); cudaFree(y); cudaFree(dy); cudaFree(dx); if (wf) cudaFree(wf); }
    double packed_bytes() const { return (double)m.rows * m.ld * 4 + m.rows * 4; }
    double flops() const { return 2.0 * m.rows * m.cols * batch; }
};

/* dense FP32 reference: warp per row, 4 batch rows per pass */
__global__ void k_dense_gemv(const float* __restrict__ W, const float* __restrict__ X, float* __restrict__ Y,
                             uint32_t rows, uint32_t cols, uint32_t batch) {
    const uint32_t lane = threadIdx.x & 31u, row = blockIdx.x * 4u + (threadIdx.x >> 5), b0 = blockIdx.y * 4u;
    if (row >= rows) return;
    float acc[4] = { 0, 0, 0, 0 };
    for (uint32_t c = lane; c < cols; c += 32u) {
        const float w = __ldg(&W[(size_t)row * cols + c]);
        for (int b = 0; b < 4; b++)
            if (b0 + b < batch) acc[b] = fmaf(w, __ldg(&X[(size_t)(b0 + b) * cols + c]), acc[b]);
    }
    for (int b = 0; b < 4; b++) {
        const float t = dev::warp_sum(acc[b]);
        if (lane == 0 && b0 + b < batch) Y[(size_t)(b0 + b) * rows + row] = t;
    }
}

void line(const char* name, double ms, double bytes, double flops, const Peak& pk, double peak_factor = 1.0) {
    const double gbs = bytes / (ms * 1e6), tf = flops / (ms * 1e9);
    printf("  %-24s %9.4f ms  %7.1f GB/s (%5.1f%% DRAM)  %7.1f GFLOP/s (%5.1f%% peak)\n", name, ms, gbs,
           pk.dram_gbs > 0 ? 100 * gbs / pk.dram_gbs : 0.0, tf, pk.fp32_tflops > 0 ? 100 * tf / (1e3 * pk.fp32_tflops * peak_factor) : 0.0);
}

}  // namespace

void bench_gemv() {
    const Peak pk = device_peak();
    for (uint32_t batch : { 1u, 4u, 8u }) {
        Bench b(FIVER_GRID_CENTERED, 4096, 4096, batch, true);
        Bench bm(FIVER_GRID_MAGIC, 4096, 4096, batch, false);
        Workspace ws;
        printf("GEMV 4096x4096 batch=%u\n", batch);
        const double dense_bytes = 4096.0 * 4096 * 4;
        line("dense FP32 weights", time_ms([&] { FIVER_LAUNCH(k_dense_gemv, dim3(1024, (batch + 3) / 4), dim3(128), 0, b.wf, b.x, b.y, 4096u, 4096u, batch); }, 50), dense_bytes, b.flops(), pk);
        line("Fiver v1", time_ms([&] { gemv_float(b.m, b.x, b.y, batch, nullptr, GEMV_V1, 0); }, 50), b.packed_bytes(), b.flops(), pk);
        line("Fiver v2", time_ms([&] { gemv_float(b.m, b.x, b.y, batch, nullptr, GEMV_V2, 0); }, 50), b.packed_bytes(), b.flops(), pk);
        line("magic v2 LUT", time_ms([&] { gemv_float(bm.m, bm.x, bm.y, batch, nullptr, GEMV_V2, 0); }, 50), bm.packed_bytes(), bm.flops(), pk);
        line("magic v2 bit formula", time_ms([&] { gemv_float(bm.m, bm.x, bm.y, batch, nullptr, GEMV_V2_MAGIC_BITS, 0); }, 50), bm.packed_bytes(), bm.flops(), pk);
        line("Fiver dp4a gemv", time_ms([&] { gemv_dp4a(b.m, b.x, b.y, batch, nullptr, ws, 0); }, 50), b.packed_bytes(), b.flops(), pk, 4.0);
        ws.release();
        printf("\n");
    }
}

void bench_gemm() {
    const Peak pk = device_peak();
    int d = 0; cudaDeviceProp p; cudaGetDevice(&d); cudaGetDeviceProperties(&p, d);
    struct S { uint32_t rows, cols, batch; const char* what; } shapes[] = {
        {4096, 4096, 16, ""}, {4096, 4096, 32, ""}, {4096, 4096, 64, ""}, {4096, 4096, 128, ""},
        {512, 10000, 32, "EmojiNet img/txt"}, {256, 1024, 32, "EmojiNet hidden"}, {576, 256, 32, "EmojiNet output"} };
    for (auto sh : shapes) {
        Bench b(FIVER_GRID_CENTERED, sh.rows, sh.cols, sh.batch, false);
        Workspace ws;
        printf("GEMM %ux%u batch=%u %s\n", sh.rows, sh.cols, sh.batch, sh.what);
        line("Fiver v2 gemv (tiles of 8)", time_ms([&] { gemv_float(b.m, b.x, b.y, sh.batch, nullptr, GEMV_V2, 0); }, 20), b.packed_bytes(), b.flops(), pk);
        line("tiled FP32", time_ms([&] { gemm_float(b.m, b.x, b.y, sh.batch, nullptr, ws, p.multiProcessorCount, 0, 0); }, 20), b.packed_bytes(), b.flops(), pk);
        line("tiled dp4a", time_ms([&] { gemm_dp4a(b.m, b.x, b.y, sh.batch, nullptr, ws, p.multiProcessorCount, 0, 0); }, 20), b.packed_bytes(), b.flops(), pk, 4.0);
        if (gemm_tc(b.m, b.x, b.y, sh.batch, nullptr, ws, 0))
            line("TC (WMMA, unvalidated)", time_ms([&] { gemm_tc(b.m, b.x, b.y, sh.batch, nullptr, ws, 0); }, 20), b.packed_bytes(), b.flops(), pk);
        ws.release();
        printf("\n");
    }
}

void bench_bwd() {
    const Peak pk = device_peak();
    int d = 0; cudaDeviceProp p; cudaGetDevice(&d); cudaGetDeviceProperties(&p, d);
    struct S { uint32_t rows, cols, batch; const char* what; } shapes[] = {
        {4096, 4096, 32, ""}, {4096, 4096, 128, ""}, {512, 10000, 32, "EmojiNet img/txt"},
        {256, 1024, 32, "EmojiNet hidden"}, {576, 256, 32, "EmojiNet output"} };
    for (auto sh : shapes) {
        Bench b(FIVER_GRID_CENTERED, sh.rows, sh.cols, sh.batch, false);
        Workspace ws;
        printf("Backward %ux%u batch=%u %s\n", sh.rows, sh.cols, sh.batch, sh.what);
        line("dgrad tiled", time_ms([&] { dgrad(b.m, b.dy, b.dx, sh.batch, ws, p.multiProcessorCount, 0, 0); }, 20), b.packed_bytes(), b.flops(), pk);
        line("dgrad atomic (previous)", time_ms([&] { dgrad_atomic_legacy(b.m, b.dy, b.dx, sh.batch, 0); }, 20), b.packed_bytes(), b.flops(), pk);
        line("wgrad+STE tiled", time_ms([&] { wgrad_ste(b.m, b.x, b.dy, nullptr, sh.batch, 0.01f, 0); }, 20), b.packed_bytes(), b.flops(), pk);
        line("wgrad+STE previous", time_ms([&] { ste_update_legacy(b.m, b.x, b.dy, nullptr, sh.batch, 0.01f, 0); }, 20), b.packed_bytes(), b.flops(), pk);
        ws.release();
        printf("\n");
    }
}
