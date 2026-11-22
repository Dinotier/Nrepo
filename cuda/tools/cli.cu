/*
 * compact_weights_test — command-line front end (called by the C# controller).
 *
 *   --bench-a      raw Quake inverse sqrt (0x5F375A86, no Newton step)
 *   --bench-b      __frsqrt_rn
 *   --test         correctness tests of libfiver_cuda
 *   --bench-gemv / --bench-gemm / --bench-bwd   kernel benchmarks
 *   (no flag: everything)
 *
 * Built with -DFIVER_EMU the same binary runs on the CPU emulator; the
 * benchmarks are skipped there because emulated timings mean nothing.
 */
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "fiver/compat.h"
#include "fiver/format.h"

int  run_all_tests();
void bench_gemv();
void bench_gemm();
void bench_bwd();

#define CW_QUAKE_CONST       0x5F375A86U
#define CW_SQRT2_HIGH        1.5f         /* error +6.07% */
#define CW_SQRT2_LOW         1.35f        /* error -4.54% */

namespace {

#define CW_CUDA_CHECK(call)                                                              \
    do {                                                                                 \
        cudaError_t e_ = (call);                                                         \
        if (e_ != cudaSuccess) {                                                         \
            fprintf(stderr, "[CW ERROR] %s:%d %s: %s\n", __FILE__, __LINE__, #call,      \
                    cudaGetErrorString(e_));                                             \
            exit(EXIT_FAILURE);                                                          \
        }                                                                                \
    } while (0)

/* Raw Quake inverse sqrt — one shift + one subtraction, no Newton-Raphson.
 * Raw error vs exact rsqrt: up to 3.44 % (0.175 % would need one NR step). */
__host__ __device__ __forceinline__ float cw_quake_rsqrt(float x) {
    uint32_t i;
    memcpy(&i, &x, sizeof(i));
    i = CW_QUAKE_CONST - (i >> 1);
    float y;
    memcpy(&y, &i, sizeof(y));
    return y;
}

__global__ void bench_kernel_quake(const float* __restrict__ in, float* __restrict__ out, uint32_t n) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = cw_quake_rsqrt(in[idx]);
}

__global__ void bench_kernel_hw(const float* __restrict__ in, float* __restrict__ out, uint32_t n) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = __frsqrt_rn(in[idx]);
}

void run_rsqrt_benchmark(const char* label, bool quake, int iterations) {
    const uint32_t N = 1u << 20;
    float *d_in, *d_out;
    float* h_in  = (float*)malloc(N * sizeof(float));
    float* h_out = (float*)malloc(N * sizeof(float));
    float* h_ref = (float*)malloc(N * sizeof(float));
    for (uint32_t i = 0; i < N; i++) {                 /* 2^(m/4): the Fiver delta denominators */
        h_in[i]  = powf(2.0f, (float)(i & 15u) * 0.25f);
        h_ref[i] = 1.0f / sqrtf(h_in[i]);
    }
    CW_CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
    CW_CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
    CW_CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));
    const dim3 grid((N + 127u) / 128u), block(128);
    auto launch = [&] {
        if (quake) FIVER_LAUNCH(bench_kernel_quake, grid, block, 0, d_in, d_out, N);
        else       FIVER_LAUNCH(bench_kernel_hw, grid, block, 0, d_in, d_out, N);
    };
    launch();
    CW_CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t t0, t1;
    CW_CUDA_CHECK(cudaEventCreate(&t0));
    CW_CUDA_CHECK(cudaEventCreate(&t1));
    CW_CUDA_CHECK(cudaEventRecord(t0));
    for (int it = 0; it < iterations; it++) launch();
    CW_CUDA_CHECK(cudaEventRecord(t1));
    CW_CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0.0f;
    CW_CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    CW_CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
    double max_err = 0.0, sum_err = 0.0;
    for (uint32_t i = 0; i < N; i++) {
        const double rel = fabs((double)h_out[i] - h_ref[i]) / (h_ref[i] + 1e-12);
        if (rel > max_err) max_err = rel;
        sum_err += rel;
    }
    printf("%-22s | %8.3f ns/op | max_err %7.4f%% | avg_err %7.4f%%\n", label,
           (ms * 1e6f) / ((float)iterations * (float)N), max_err * 100.0, sum_err / N * 100.0);
    const float s2 = sqrtf(2.0f);
    printf("  [sqrt2 approx]  HIGH=%.4f err=+%.2f%%  LOW=%.4f err=-%.2f%%\n", CW_SQRT2_HIGH,
           fabsf(CW_SQRT2_HIGH - s2) / s2 * 100.0f, CW_SQRT2_LOW, fabsf(CW_SQRT2_LOW - s2) / s2 * 100.0f);
    cudaFree(d_in); cudaFree(d_out);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    free(h_in); free(h_out); free(h_ref);
}

void print_device_info() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) {
        fprintf(stderr, "[CW WARN]  no CUDA device available\n");
        return;
    }
    cudaDeviceProp p;
    CW_CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
    const int sm = p.major * 10 + p.minor;
    printf("[device]  %s  (sm_%d)  %.0f MB  %d SMs\n", p.name, sm, (double)p.totalGlobalMem / (1024.0 * 1024.0),
           p.multiProcessorCount);
    printf("[device]  int8: __dp4a %s   INT8 Tensor Cores %s\n\n", sm >= 61 ? "yes" : "no",
           sm >= 75 ? "yes (path not validated yet)" : "no (needs sm_75+)");
}

}  // namespace

int main(int argc, char** argv) {
    printf("compact_weights — Fiver 5-bit (centered / anchored / magic grids)\n");
    printf("==================================================================\n\n");
    print_device_info();

    bool a = false, b = false, t = false, g = false, m = false, w = false;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--bench-a"))    a = true;
        if (!strcmp(argv[i], "--bench-b"))    b = true;
        if (!strcmp(argv[i], "--test"))       t = true;
        if (!strcmp(argv[i], "--bench-gemv")) g = true;
        if (!strcmp(argv[i], "--bench-gemm")) m = true;
        if (!strcmp(argv[i], "--bench-bwd"))  w = true;
    }
    if (!a && !b && !t && !g && !m && !w) a = b = t = g = m = w = true;

#ifdef FIVER_EMU
    if (a || b || g || m || w) printf("[emulator] benchmarks skipped — emulated timings are meaningless\n\n");
    a = b = g = m = w = false;
#endif
    if (a || b) {
        printf("%-22s | %-16s | %-18s | %-18s\n", "METHOD", "TIME_NS/OP", "MAX_REL_ERROR", "AVG_REL_ERROR");
        printf("--------------------------------------------------------------------------------\n");
        if (a) run_rsqrt_benchmark("Quake-raw (A)", true, 10);
        if (b) run_rsqrt_benchmark("__frsqrt_rn (B)", false, 10);
        printf("\n");
    }
    if (g) bench_gemv();
    if (m) bench_gemm();
    if (w) bench_bwd();
    int ok = 1;
    if (t) ok = run_all_tests();
    return ok ? 0 : 1;
}
