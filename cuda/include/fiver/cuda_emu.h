/*
 * cuda_emu.h — CPU emulation of the CUDA subset used by the Fiver kernels.
 *
 * For tests on machines without a GPU: compile the same kernel sources as
 * C++ with -DFIVER_EMU.  Every CUDA thread is an OS thread; blocks run one
 * after another, so `__shared__` maps to `static` (one copy per kernel,
 * reused by each block in turn).  __syncthreads / __syncwarp are barriers
 * that exited threads leave (as on the GPU), warp shuffles go through a
 * per-warp exchange buffer.  A barrier that is not reached by every live
 * thread — a divergent __shfl_sync or __syncthreads — aborts with a message
 * after a timeout instead of hanging silently.
 *
 * Not emulated: WMMA / MMA (sm_75+ paths compile to empty bodies here),
 * timing, memory spaces, races between blocks.
 */
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __constant__
#define __shared__ static
#define __align__(n) __attribute__((aligned(n)))
#define __launch_bounds__(...)

using std::max;
using std::min;

struct dim3 {
    unsigned x, y, z;
    constexpr dim3(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1) : x(x_), y(y_), z(z_) {}
};
struct alignas(8)  int2   { int x, y; };
struct alignas(16) int4   { int x, y, z, w; };
struct alignas(16) uint4  { unsigned x, y, z, w; };
struct alignas(8)  float2 { float x, y; };
struct alignas(16) float4 { float x, y, z, w; };

namespace cw_emu {

class Barrier {
public:
    explicit Barrier(int n) : expected_(n), count_(n) {}
    void arrive_and_wait(const char* what) {
        std::unique_lock<std::mutex> l(m_);
        const unsigned gen = gen_;
        if (--count_ == 0) { next_phase(); return; }
        if (!cv_.wait_for(l, std::chrono::seconds(60), [&] { return gen != gen_; })) {
            fprintf(stderr, "[cw_emu] %s never completed — divergent %s?\n", what, what);
            std::abort();
        }
    }
    void arrive_and_drop() {
        std::lock_guard<std::mutex> l(m_);
        --expected_;
        if (--count_ == 0) next_phase();
    }
private:
    void next_phase() { ++gen_; count_ = expected_; cv_.notify_all(); }
    std::mutex m_;
    std::condition_variable cv_;
    int expected_, count_;
    unsigned gen_ = 0;
};

struct BlockCtx {
    std::unique_ptr<Barrier>              block;
    std::vector<std::unique_ptr<Barrier>> warps;
    std::vector<uint32_t>                 slots;   /* one exchange slot per thread */
};

inline thread_local dim3      tl_thread;
inline thread_local dim3      tl_block;
inline thread_local BlockCtx* tl_ctx = nullptr;
inline dim3                   g_grid, g_block;

template <class F>
void launch(const char* name, F&& body, dim3 grid, dim3 block) {
    (void)name;
    g_grid = grid; g_block = block;
    const unsigned nthreads = block.x * block.y * block.z;
    const unsigned nwarps   = (nthreads + 31u) / 32u;
    for (unsigned bz = 0; bz < grid.z; bz++)
        for (unsigned by = 0; by < grid.y; by++)
            for (unsigned bx = 0; bx < grid.x; bx++) {
                BlockCtx ctx;
                ctx.block = std::make_unique<Barrier>((int)nthreads);
                for (unsigned w = 0; w < nwarps; w++)
                    ctx.warps.push_back(std::make_unique<Barrier>(
                        (int)std::min(32u, nthreads - 32u * w)));
                ctx.slots.assign(nwarps * 32u, 0u);
                std::vector<std::thread> threads;
                threads.reserve(nthreads);
                for (unsigned t = 0; t < nthreads; t++)
                    threads.emplace_back([&, t] {
                        tl_thread = dim3(t % block.x, (t / block.x) % block.y, t / (block.x * block.y));
                        tl_block  = dim3(bx, by, bz);
                        tl_ctx    = &ctx;
                        body();
                        ctx.warps[t / 32u]->arrive_and_drop();
                        ctx.block->arrive_and_drop();
                    });
                for (auto& th : threads) th.join();
            }
}

inline unsigned linear_tid() {
    return tl_thread.x + g_block.x * (tl_thread.y + g_block.y * tl_thread.z);
}

}  // namespace cw_emu

#define threadIdx (::cw_emu::tl_thread)
#define blockIdx  (::cw_emu::tl_block)
#define blockDim  (::cw_emu::g_block)
#define gridDim   (::cw_emu::g_grid)

inline void __syncthreads() { cw_emu::tl_ctx->block->arrive_and_wait("__syncthreads"); }
inline void __syncwarp(unsigned = 0xFFFFFFFFu) {
    cw_emu::tl_ctx->warps[cw_emu::linear_tid() / 32u]->arrive_and_wait("__syncwarp");
}

namespace cw_emu {
template <class T>
T shfl_from(T v, int (*src_of)(int lane, int arg), int arg) {
    static_assert(sizeof(T) == 4, "only 32-bit shuffles are emulated");
    const unsigned tid  = linear_tid();
    const int      lane = (int)(tid & 31u);
    uint32_t* slots = &tl_ctx->slots[(tid / 32u) * 32u];
    Barrier&  bar   = *tl_ctx->warps[tid / 32u];
    std::memcpy(&slots[lane], &v, 4);
    bar.arrive_and_wait("__shfl_sync");
    const int src = src_of(lane, arg);
    uint32_t r = (src >= 0 && src < 32) ? slots[src] : slots[lane];
    bar.arrive_and_wait("__shfl_sync");
    T out;
    std::memcpy(&out, &r, 4);
    return out;
}
inline int src_idx(int, int a)       { return a & 31; }
inline int src_down(int lane, int a) { return lane + a; }
inline int src_xor(int lane, int a)  { return lane ^ a; }
}  // namespace cw_emu

template <class T> T __shfl_sync(unsigned, T v, int src)          { return cw_emu::shfl_from(v, cw_emu::src_idx, src); }
template <class T> T __shfl_down_sync(unsigned, T v, unsigned d)  { return cw_emu::shfl_from(v, cw_emu::src_down, (int)d); }
template <class T> T __shfl_xor_sync(unsigned, T v, int m)        { return cw_emu::shfl_from(v, cw_emu::src_xor, m); }

template <class T> inline T __ldg(const T* p) { return *p; }

inline float    __int_as_float(int i)        { float f; std::memcpy(&f, &i, 4); return f; }
inline int      __float_as_int(float f)      { int i; std::memcpy(&i, &f, 4); return i; }
inline float    __uint_as_float(unsigned u)  { float f; std::memcpy(&f, &u, 4); return f; }
inline unsigned __float_as_uint(float f)     { unsigned u; std::memcpy(&u, &f, 4); return u; }
inline int      __float2int_rn(float x)      { return (int)std::nearbyint(x); }
inline float    __fsqrt_rn(float x)          { return std::sqrt(x); }
inline float    __frsqrt_rn(float x)         { return 1.0f / std::sqrt(x); }
inline float    __fdividef(float a, float b) { return a / b; }

inline int __dp4a(int a, int b, int c) {
    for (int i = 0; i < 4; i++) c += (int)(int8_t)(a >> (8 * i)) * (int)(int8_t)(b >> (8 * i));
    return c;
}

inline float atomicAdd(float* p, float v) { return std::atomic_ref<float>(*p).fetch_add(v); }
inline int   atomicAdd(int* p, int v)     { return std::atomic_ref<int>(*p).fetch_add(v); }

inline int2   make_int2(int x, int y)                         { return {x, y}; }
inline uint4  make_uint4(unsigned x, unsigned y, unsigned z, unsigned w) { return {x, y, z, w}; }
inline float4 make_float4(float x, float y, float z, float w) { return {x, y, z, w}; }

/* ── runtime API ─────────────────────────────────────────────────────────── */
typedef int   cudaError_t;
typedef void* cudaStream_t;
typedef struct { std::chrono::steady_clock::time_point t; } cudaEvent_impl;
typedef cudaEvent_impl* cudaEvent_t;
enum { cudaSuccess = 0, cudaErrorMemoryAllocation = 2 };
enum cudaMemcpyKind { cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice,
                      cudaMemcpyHostToHost, cudaMemcpyDefault };

struct cudaDeviceProp {
    char   name[256];
    int    major, minor, multiProcessorCount, memoryClockRate, memoryBusWidth;
    size_t totalGlobalMem;
};

inline const char* cudaGetErrorString(cudaError_t e) { return e == cudaSuccess ? "no error" : "emulator error"; }
inline cudaError_t cudaGetLastError()       { return cudaSuccess; }
inline cudaError_t cudaPeekAtLastError()    { return cudaSuccess; }
inline cudaError_t cudaDeviceSynchronize()  { return cudaSuccess; }
inline cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaStreamCreate(cudaStream_t* s) { *s = nullptr; return cudaSuccess; }
inline cudaError_t cudaStreamDestroy(cudaStream_t)     { return cudaSuccess; }
inline cudaError_t cudaSetDevice(int)       { return cudaSuccess; }
inline cudaError_t cudaGetDevice(int* d)    { *d = 0; return cudaSuccess; }
inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* p, int) {
    std::memset(p, 0, sizeof(*p));
    std::snprintf(p->name, sizeof(p->name), "CPU emulator (behaves as sm_61)");
    p->major = 6; p->minor = 1; p->multiProcessorCount = 4;
    p->memoryClockRate = 1; p->memoryBusWidth = 8; p->totalGlobalMem = (size_t)1 << 32;
    return cudaSuccess;
}
inline cudaError_t cudaMalloc(void** p, size_t n) {
    *p = std::aligned_alloc(256, (n + 255) / 256 * 256 + 256);
    return *p ? cudaSuccess : cudaErrorMemoryAllocation;
}
template <class T> cudaError_t cudaMalloc(T** p, size_t n) { return cudaMalloc((void**)p, n); }
inline cudaError_t cudaFree(void* p) { std::free(p); return cudaSuccess; }
inline cudaError_t cudaMemcpy(void* d, const void* s, size_t n, cudaMemcpyKind) { std::memcpy(d, s, n); return cudaSuccess; }
inline cudaError_t cudaMemcpyAsync(void* d, const void* s, size_t n, cudaMemcpyKind, cudaStream_t = nullptr) {
    std::memcpy(d, s, n); return cudaSuccess;
}
inline cudaError_t cudaMemset(void* d, int v, size_t n) { std::memset(d, v, n); return cudaSuccess; }
inline cudaError_t cudaMemsetAsync(void* d, int v, size_t n, cudaStream_t = nullptr) { std::memset(d, v, n); return cudaSuccess; }
inline cudaError_t cudaEventCreate(cudaEvent_t* e)     { *e = new cudaEvent_impl; return cudaSuccess; }
inline cudaError_t cudaEventDestroy(cudaEvent_t e)     { delete e; return cudaSuccess; }
inline cudaError_t cudaEventRecord(cudaEvent_t e, cudaStream_t = nullptr) { e->t = std::chrono::steady_clock::now(); return cudaSuccess; }
inline cudaError_t cudaEventSynchronize(cudaEvent_t)   { return cudaSuccess; }
inline cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t a, cudaEvent_t b) {
    *ms = std::chrono::duration<float, std::milli>(b->t - a->t).count(); return cudaSuccess;
}
