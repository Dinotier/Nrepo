/*
 * tests.cu — correctness tests for libfiver_cuda (GPU or CPU emulator).
 *
 * Every path × every grid.  Float paths against a double-precision host
 * reference, INT8 paths exactly against the integer sum recomputed from the
 * device-quantised x, tiled kernels for bit-identical repeat runs, the STE
 * update code by code, and the C ABI including its error cases.
 */
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../src/internal.cuh"
#include "fiver/fiver.h"

using namespace fiver;

namespace {

int g_failures = 0;

#define CHECK_CUDA(call)                                                                \
    do {                                                                                \
        cudaError_t e_ = (call);                                                        \
        if (e_ != cudaSuccess) {                                                        \
            fprintf(stderr, "[test] %s:%d %s: %s\n", __FILE__, __LINE__, #call,         \
                    cudaGetErrorString(e_));                                            \
            exit(EXIT_FAILURE);                                                         \
        }                                                                               \
    } while (0)

void report(bool ok, const char* fmt, ...) __attribute__((format(printf, 2, 3)));
void report(bool ok, const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    printf("  ");
    vprintf(fmt, ap);
    va_end(ap);
    printf("  %s\n", ok ? "ok" : "FAIL");
    if (!ok) g_failures++;
}

const char* gname(int g) { return g == FIVER_GRID_ANCHORED ? "anch" : g == FIVER_GRID_MAGIC ? "magc" : "cent"; }

uint32_t rng_state = 0x2545F491u;
uint32_t rnd() { rng_state ^= rng_state << 13; rng_state ^= rng_state >> 17; rng_state ^= rng_state << 5; return rng_state; }
float rndf(float lo, float hi) { return lo + (hi - lo) * (float)(rnd() & 0xFFFFFFu) / (float)0x1000000; }

int num_sms() {
    int d = 0;
    cudaDeviceProp p;
    CHECK_CUDA(cudaGetDevice(&d));
    CHECK_CUDA(cudaGetDeviceProperties(&p, d));
    return p.multiProcessorCount;
}

/* Host copy of a matrix: external words (row stride wpr) plus device-layout act bits. */
struct Host {
    int grid; uint32_t rows, cols, wpr, ld;
    std::vector<uint32_t> words, act;
    std::vector<float> scale;

    float value(uint32_t r, uint32_t c) const {
        return fiver_value_ref(grid, fiver_k_at(words[(size_t)r * wpr + c / 6], (int)(c % 6)));
    }
    float weight(uint32_t r, uint32_t c) const {   /* activated, unscaled */
        float w = value(r, c);
        if (!act.empty()) {
            const size_t idx = ((size_t)r * ld + c / 6) * 6 + c % 6;
            w = dev::apply_act(w, (act[idx >> 4] >> ((idx & 15u) * 2u)) & 3u);
        }
        return w;
    }
};

Host make_host(int grid, uint32_t rows, uint32_t cols, bool with_act) {
    Host h{grid, rows, cols, fiver_words_per_row(cols), fiver_row_pitch(cols), {}, {}, {}};
    h.words.resize((size_t)rows * h.wpr);
    h.scale.resize(rows);
    for (uint32_t r = 0; r < rows; r++) {
        h.scale[r] = rndf(0.01f, 0.1f);
        for (uint32_t w = 0; w < h.wpr; w++) {
            uint8_t k[6];
            for (int i = 0; i < 6; i++)
                k[i] = (w * 6 + (uint32_t)i < cols) ? (uint8_t)(rnd() & 31u) : (uint8_t)fiver_zero_code(grid);
            h.words[(size_t)r * h.wpr + w] = fiver_pack_word(k);
        }
    }
    if (with_act) {
        h.act.resize(((size_t)rows * h.ld * 6 + 15) / 16);
        for (auto& a : h.act) a = rnd();
    }
    return h;
}

struct Dev {
    Matrix m;
    uint32_t* act = nullptr;
};

Dev upload(const Host& h) {
    Dev d;
    d.m.rows = h.rows; d.m.cols = h.cols; d.m.wpr = h.wpr; d.m.ld = h.ld; d.m.grid = h.grid;
    std::vector<uint32_t> pitched((size_t)h.rows * h.ld, fiver_pad_word(h.grid));
    for (uint32_t r = 0; r < h.rows; r++)
        memcpy(&pitched[(size_t)r * h.ld], &h.words[(size_t)r * h.wpr], h.wpr * 4);
    CHECK_CUDA(cudaMalloc(&d.m.words, pitched.size() * 4));
    CHECK_CUDA(cudaMalloc(&d.m.row_scale, h.rows * 4));
    CHECK_CUDA(cudaMemcpy(d.m.words, pitched.data(), pitched.size() * 4, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d.m.row_scale, h.scale.data(), h.rows * 4, cudaMemcpyHostToDevice));
    if (!h.act.empty()) {
        CHECK_CUDA(cudaMalloc(&d.act, h.act.size() * 4));
        CHECK_CUDA(cudaMemcpy(d.act, h.act.data(), h.act.size() * 4, cudaMemcpyHostToDevice));
        d.m.act_bits = d.act;
    }
    return d;
}

void release(Dev& d) {
    cudaFree(d.m.words); cudaFree(d.m.row_scale);
    if (d.act) cudaFree(d.act);
}

std::vector<uint32_t> download_words(const Dev& d) {
    std::vector<uint32_t> pitched((size_t)d.m.rows * d.m.ld), out((size_t)d.m.rows * d.m.wpr);
    CHECK_CUDA(cudaMemcpy(pitched.data(), d.m.words, pitched.size() * 4, cudaMemcpyDeviceToHost));
    for (uint32_t r = 0; r < d.m.rows; r++)
        memcpy(&out[(size_t)r * d.m.wpr], &pitched[(size_t)r * d.m.ld], d.m.wpr * 4);
    return out;
}

float* to_dev(const std::vector<float>& v) {
    float* p;
    CHECK_CUDA(cudaMalloc(&p, v.size() * 4 + 4));
    CHECK_CUDA(cudaMemcpy(p, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
    return p;
}

std::vector<float> to_host(const float* p, size_t n) {
    std::vector<float> v(n);
    CHECK_CUDA(cudaMemcpy(v.data(), p, n * 4, cudaMemcpyDeviceToHost));
    return v;
}

std::vector<float> rand_vec(size_t n, float lo, float hi) {
    std::vector<float> v(n);
    for (auto& x : v) x = rndf(lo, hi);
    return v;
}

/* max over outputs of |y − ref| / Σ|terms| (relative to the magnitude of the dot product) */
double forward_err(const Host& h, const std::vector<float>& x, const std::vector<float>& y,
                   const std::vector<float>& bias, uint32_t batch) {
    double worst = 0.0;
    for (uint32_t b = 0; b < batch; b++)
        for (uint32_t r = 0; r < h.rows; r++) {
            double ref = 0.0, mag = 0.0;
            for (uint32_t c = 0; c < h.cols; c++) {
                const double t = (double)h.weight(r, c) * x[(size_t)b * h.cols + c];
                ref += t; mag += fabs(t);
            }
            ref = ref * h.scale[r] + (bias.empty() ? 0.0 : bias[r]);
            mag = mag * h.scale[r] + 1e-12;
            worst = fmax(worst, fabs(y[(size_t)b * h.rows + r] - ref) / mag);
        }
    return worst;
}

/* ── 1. tables ─────────────────────────────────────────────────────────── */
__global__ void k_probe(float* v, int* e, float* bits, int grid) {
    const int lane = (int)(threadIdx.x & 31u);
    const float lut = dev::lut[grid][lane];
    const int   el  = dev::e[grid][lane];
    v[lane]    = __shfl_sync(FIVER_FULL_MASK, lut, lane);
    e[lane]    = __shfl_sync(FIVER_FULL_MASK, el, lane);
    bits[lane] = fiver_magic_value((uint32_t)lane);
}

void test_tables() {
    printf("[test] value / int8 tables vs formula:\n");
    float *dv, *db; int* de;
    CHECK_CUDA(cudaMalloc(&dv, 128)); CHECK_CUDA(cudaMalloc(&db, 128)); CHECK_CUDA(cudaMalloc(&de, 128));
    for (int g = 0; g < FIVER_NUM_GRIDS; g++) {
        FIVER_LAUNCH(k_probe, dim3(1), dim3(32), 0, dv, de, db, g);
        CHECK_CUDA(cudaDeviceSynchronize());
        float v[32], bits[32]; int e[32];
        CHECK_CUDA(cudaMemcpy(v, dv, 128, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(bits, db, 128, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(e, de, 128, cudaMemcpyDeviceToHost));
        int bad = 0;
        for (uint32_t k = 0; k < 32; k++) {
            if (fabsf(v[k] - fiver_value_ref(g, k)) > 1e-6f) bad++;
            if (v[k] != fiver_lut_host[g][k] || e[k] != fiver_e_host[g][k]) bad++;
            if (e[k] != fiver_e_ref(g, k)) bad++;
            if (k && !(v[k] > v[k - 1] && e[k] > e[k - 1])) bad++;
            if (g == FIVER_GRID_MAGIC && (bits[k] != v[k] || (float)e[k] / 64.0f != v[k])) bad++;
        }
        if (v[fiver_zero_code(g)] != 0.0f) bad++;
        report(bad == 0, "grid=%s  32 values, int8 codes, monotone, zero code%s", gname(g),
               g == FIVER_GRID_MAGIC ? ", formula == table, int8 exact" : "");
    }
    cudaFree(dv); cudaFree(db); cudaFree(de);
}

/* ── 2. float forward ─────────────────────────────────────────────────── */
void test_float_forward() {
    printf("[test] float forward (GEMV v1/v2, tiled GEMM) vs double reference:\n");
    struct S { uint32_t rows, cols, batch; } shapes[] = {
        {37, 1001, 5}, {64, 6, 1}, {3, 7, 4}, {130, 1030, 9}, {17, 97, 33}, {200, 777, 70} };
    Workspace ws;
    const int sms = num_sms();
    for (int g = 0; g < FIVER_NUM_GRIDS; g++)
        for (auto sh : shapes) {
            for (int act = 0; act <= 1; act++) {
                Host h = make_host(g, sh.rows, sh.cols, act);
                Dev  d = upload(h);
                auto x = rand_vec((size_t)sh.batch * sh.cols, -1.0f, 1.0f);
                auto bias = rand_vec(sh.rows, -0.5f, 0.5f);
                float* dx = to_dev(x); float* db = to_dev(bias);
                float* dy; CHECK_CUDA(cudaMalloc(&dy, (size_t)sh.batch * sh.rows * 4));
                std::vector<GemvVariant> vars = { GEMV_V1, GEMV_V2 };
                if (g == FIVER_GRID_MAGIC) vars.push_back(GEMV_V2_MAGIC_BITS);
                for (auto v : vars) {
                    gemv_float(d.m, dx, dy, sh.batch, v == GEMV_V1 ? nullptr : db, v, 0);
                    CHECK_CUDA(cudaDeviceSynchronize());
                    const double e = forward_err(h, x, to_host(dy, (size_t)sh.batch * sh.rows),
                                                 v == GEMV_V1 ? std::vector<float>() : bias, sh.batch);
                    report(e < 1e-5, "gemv %-7s grid=%s %4ux%-5u b=%-3u act=%d  err=%.1e",
                           v == GEMV_V1 ? "v1" : v == GEMV_V2 ? "v2" : "v2-bits", gname(g),
                           sh.rows, sh.cols, sh.batch, act, e);
                }
                if (!act) {
                    for (uint32_t splits : { 0u, 1u, 3u }) {
                        gemm_float(d.m, dx, dy, sh.batch, db, ws, sms, splits, 0);
                        CHECK_CUDA(cudaDeviceSynchronize());
                        auto y1 = to_host(dy, (size_t)sh.batch * sh.rows);
                        gemm_float(d.m, dx, dy, sh.batch, db, ws, sms, splits, 0);
                        CHECK_CUDA(cudaDeviceSynchronize());
                        auto y2 = to_host(dy, (size_t)sh.batch * sh.rows);
                        const double e = forward_err(h, x, y1, bias, sh.batch);
                        const bool same = memcmp(y1.data(), y2.data(), y1.size() * 4) == 0;
                        report(e < 1e-5 && same, "gemm f32 grid=%s %4ux%-5u b=%-3u splits=%u  err=%.1e  repeat bit-identical=%s",
                               gname(g), sh.rows, sh.cols, sh.batch, splits, e, same ? "yes" : "NO");
                    }
                }
                cudaFree(dx); cudaFree(db); cudaFree(dy); release(d);
            }
        }
    ws.release();
}

/* ── 3. INT8 forward ──────────────────────────────────────────────────── */
/* exact integer reference from device xq (row-major, ldq) */
double int8_exact_err(const Host& h, const std::vector<int8_t>& xq, uint32_t ldq, const std::vector<float>& sx,
                      const std::vector<float>& y, const std::vector<float>& bias, uint32_t batch,
                      const std::vector<float>& x, double* qerr) {
    const long long off = h.grid == FIVER_GRID_ANCHORED ? (long long)fiver_escale_host[h.grid] : 0;
    double worst = 0.0, maxq = 0.0, maxref = 0.0;
    for (uint32_t b = 0; b < batch; b++) {
        long long sxq = 0;
        for (uint32_t c = 0; c < h.cols; c++) sxq += xq[(size_t)b * ldq + c];
        for (uint32_t r = 0; r < h.rows; r++) {
            long long acc = off * sxq;
            double yf = 0.0;
            for (uint32_t c = 0; c < h.cols; c++) {
                const uint32_t k = fiver_k_at(h.words[(size_t)r * h.wpr + c / 6], (int)(c % 6));
                acc += (long long)fiver_e_ref(h.grid, k) * xq[(size_t)b * ldq + c];
                yf  += (double)fiver_value_ref(h.grid, k) * x[(size_t)b * h.cols + c];
            }
            const double unit = (double)h.scale[r] * sx[b] / fiver_escale_host[h.grid];
            const double bb   = bias.empty() ? 0.0 : bias[r];
            const double yref = (double)acc * unit + bb;
            const double yv   = y[(size_t)b * h.rows + r];
            worst = fmax(worst, fabs(yv - yref) / (1e-6 * fabs(yref) + 0.25 * unit + 1e-6 * fabs(bb) + 1e-30));
            yf = yf * h.scale[r] + bb;
            maxq = fmax(maxq, fabs(yv - yf));
            maxref = fmax(maxref, fabs(yf));
        }
    }
    *qerr = maxq / (maxref + 1e-30);
    return worst;
}

void test_int8_forward() {
    printf("[test] INT8 forward (dp4a GEMV, tiled dp4a GEMM, TC) vs exact integer reference:\n");
    struct S { uint32_t rows, cols, batch; } shapes[] = {
        {37, 1001, 5}, {64, 6, 1}, {3, 7, 4}, {130, 1030, 9}, {17, 97, 33}, {200, 777, 70} };
    Workspace ws;
    const int sms = num_sms();
    for (int g = 0; g < FIVER_NUM_GRIDS; g++)
        for (auto sh : shapes) {
            Host h = make_host(g, sh.rows, sh.cols, false);
            Dev  d = upload(h);
            auto x = rand_vec((size_t)sh.batch * sh.cols, -1.0f, 1.0f);
            for (uint32_t c = 0; c < sh.cols; c++) x[c] = 0.0f;            /* row 0: sx = 0 */
            auto bias = rand_vec(sh.rows, -0.5f, 0.5f);
            float* dx = to_dev(x); float* db = to_dev(bias);
            float* dy; CHECK_CUDA(cudaMalloc(&dy, (size_t)sh.batch * sh.rows * 4));
            const uint32_t ldq = h.ld * 6;

            /* GEMV path: xq row-major in the XQ slot */
            gemv_dp4a(d.m, dx, dy, sh.batch, db, ws, 0);
            CHECK_CUDA(cudaDeviceSynchronize());
            {
                std::vector<int8_t> xq((size_t)sh.batch * ldq);
                CHECK_CUDA(cudaMemcpy(xq.data(), ws.ptr[Workspace::XQ], xq.size(), cudaMemcpyDeviceToHost));
                auto sx = to_host((const float*)ws.ptr[Workspace::SX], sh.batch);
                double q;
                const double e = int8_exact_err(h, xq, ldq, sx, to_host(dy, (size_t)sh.batch * sh.rows), bias, sh.batch, x, &q);
                report(e <= 1.0, "dp4a gemv  grid=%s %4ux%-5u b=%-3u  int-exact=%s  quant_err=%.1e",
                       gname(g), sh.rows, sh.cols, sh.batch, e <= 1.0 ? "yes" : "NO", q);
            }
            /* tiled GEMM: xq4 transposed in the XQ4 slot */
            for (uint32_t splits : { 0u, 3u }) {
                gemm_dp4a(d.m, dx, dy, sh.batch, db, ws, sms, splits, 0);
                CHECK_CUDA(cudaDeviceSynchronize());
                auto y1 = to_host(dy, (size_t)sh.batch * sh.rows);
                gemm_dp4a(d.m, dx, dy, sh.batch, db, ws, sms, splits, 0);
                CHECK_CUDA(cudaDeviceSynchronize());
                auto y2 = to_host(dy, (size_t)sh.batch * sh.rows);
                const uint32_t npad = (sh.batch + 31) / 32 * 32, groups = ldq / 4;
                std::vector<int> xq4((size_t)groups * npad);
                CHECK_CUDA(cudaMemcpy(xq4.data(), ws.ptr[Workspace::XQ4], xq4.size() * 4, cudaMemcpyDeviceToHost));
                std::vector<int8_t> xq((size_t)sh.batch * ldq);
                for (uint32_t n = 0; n < sh.batch; n++)
                    for (uint32_t c = 0; c < ldq; c++)
                        xq[(size_t)n * ldq + c] = (int8_t)(xq4[(size_t)(c / 4) * npad + n] >> (8 * (c % 4)));
                auto sx = to_host((const float*)ws.ptr[Workspace::SX], sh.batch);
                double q;
                const double e = int8_exact_err(h, xq, ldq, sx, y1, bias, sh.batch, x, &q);
                const bool same = memcmp(y1.data(), y2.data(), y1.size() * 4) == 0;
                report(e <= 1.0 && same, "dp4a gemm  grid=%s %4ux%-5u b=%-3u splits=%u  int-exact=%s  repeat bit-identical=%s  quant_err=%.1e",
                       gname(g), sh.rows, sh.cols, sh.batch, splits, e <= 1.0 ? "yes" : "NO", same ? "yes" : "NO", q);
            }
            if (gemm_tc(d.m, dx, dy, sh.batch, db, ws, 0)) {
                CHECK_CUDA(cudaDeviceSynchronize());
                std::vector<int8_t> xq((size_t)((sh.batch + 15) / 16 * 16) * ldq);
                CHECK_CUDA(cudaMemcpy(xq.data(), ws.ptr[Workspace::XQ], xq.size(), cudaMemcpyDeviceToHost));
                auto sx = to_host((const float*)ws.ptr[Workspace::SX], sh.batch);
                double q;
                const double e = int8_exact_err(h, xq, ldq, sx, to_host(dy, (size_t)sh.batch * sh.rows), bias, sh.batch, x, &q);
                report(e <= 1.0, "tc         grid=%s %4ux%-5u b=%-3u  int-exact=%s", gname(g), sh.rows, sh.cols,
                       sh.batch, e <= 1.0 ? "yes" : "NO");
            }
            cudaFree(dx); cudaFree(db); cudaFree(dy); release(d);
        }
    ws.release();
}

/* ── 4. backward ──────────────────────────────────────────────────────── */
void test_backward() {
    printf("[test] backward: tiled dX (deterministic) and fused wgrad + STE vs references:\n");
    struct S { uint32_t rows, cols, batch; float lr; } shapes[] = {
        {37, 1001, 5, 0.05f}, {3, 7, 4, -0.05f}, {130, 1030, 33, 0.05f}, {9, 6, 1, 0.05f},
        {70, 400, 70, 0.02f}, {100, 1200, 40, 0.05f} };
    Workspace ws;
    const int sms = num_sms();
    for (int g = 0; g < FIVER_NUM_GRIDS; g++)
        for (auto sh : shapes) {
            Host h = make_host(g, sh.rows, sh.cols, false);
            auto x  = rand_vec((size_t)sh.batch * sh.cols, -1.0f, 1.0f);
            auto dy = rand_vec((size_t)sh.batch * sh.rows, -1.0f, 1.0f);
            for (uint32_t b = 0; b < sh.batch; b++) dy[(size_t)b * sh.rows] = 0.0f;   /* row 0: G = 0 */
            auto bias = rand_vec(sh.rows, 0.0f, 1.0f);

            /* dX */
            Dev d = upload(h);
            float* ddy = to_dev(dy);
            float* ddx; CHECK_CUDA(cudaMalloc(&ddx, (size_t)sh.batch * sh.cols * 4));
            double worst = 0.0;
            bool det = true;
            std::vector<float> first;
            for (uint32_t splits : { 0u, 1u, 2u }) {
                for (int rep = 0; rep < 2; rep++) {
                    dgrad(d.m, ddy, ddx, sh.batch, ws, sms, splits, 0);
                    CHECK_CUDA(cudaDeviceSynchronize());
                    auto gx = to_host(ddx, (size_t)sh.batch * sh.cols);
                    if (rep == 0) first = gx;
                    else det = det && memcmp(first.data(), gx.data(), gx.size() * 4) == 0;
                }
                for (uint32_t b = 0; b < sh.batch; b++)
                    for (uint32_t c = 0; c < sh.cols; c++) {
                        double ref = 0.0, mag = 0.0;
                        for (uint32_t r = 0; r < sh.rows; r++) {
                            const double t = (double)h.scale[r] * h.value(r, c) * dy[(size_t)b * sh.rows + r];
                            ref += t; mag += fabs(t);
                        }
                        worst = fmax(worst, fabs(first[(size_t)b * sh.cols + c] - ref) / (mag + 1e-30));
                    }
            }
            dgrad_atomic_legacy(d.m, ddy, ddx, sh.batch, 0);
            CHECK_CUDA(cudaDeviceSynchronize());
            auto legacy = to_host(ddx, (size_t)sh.batch * sh.cols);
            double vs_legacy = 0.0;
            for (size_t i = 0; i < legacy.size(); i++)
                vs_legacy = fmax(vs_legacy, fabs(legacy[i] - first[i]) / (fabs(first[i]) + 1e-3));
            report(worst < 1e-5 && det, "dgrad      grid=%s %4ux%-5u b=%-3u  err=%.1e  repeat bit-identical=%s  vs atomic=%.1e",
                   gname(g), sh.rows, sh.cols, sh.batch, worst, det ? "yes" : "NO", vs_legacy);

            /* fused wgrad + STE (and legacy for comparison) */
            Dev d2 = upload(h);
            float* ddx2 = to_dev(x);
            float* dbias = to_dev(bias);
            wgrad_ste(d.m, ddx2, ddy, dbias, sh.batch, sh.lr, 0);
            ste_update_legacy(d2.m, ddx2, ddy, nullptr, sh.batch, sh.lr, 0);
            CHECK_CUDA(cudaDeviceSynchronize());
            auto nw = download_words(d), nw_leg = download_words(d2);
            auto nb = to_host(dbias, sh.rows);
            uint32_t wrong = 0, ambig = 0, moved = 0, frozen = 0, legacy_diff = 0;
            for (uint32_t r = 0; r < sh.rows; r++)
                for (uint32_t w = 0; w < h.wpr; w++) {
                    const uint32_t ow = h.words[(size_t)r * h.wpr + w], nwv = nw[(size_t)r * h.wpr + w];
                    if ((ow >> 30) != (nwv >> 30)) frozen++;
                    for (int i = 0; i < 6; i++) {
                        const uint32_t c = w * 6 + (uint32_t)i;
                        const uint32_t ko = fiver_k_at(ow, i), kn = fiver_k_at(nwv, i);
                        if (c >= sh.cols) { if (kn != ko) frozen++; continue; }
                        double G = 0.0, mag = 0.0;
                        for (uint32_t b = 0; b < sh.batch; b++) {
                            const double t = (double)dy[(size_t)b * sh.rows + r] * x[(size_t)b * sh.cols + c];
                            G += t; mag += fabs(t);
                        }
                        const double dd = G * sh.lr;
                        uint32_t ke = ko;
                        if (dd > 0 && ko > 0) ke = ko - 1;
                        if (dd < 0 && ko < 31) ke = ko + 1;
                        if (kn != ko) moved++;
                        const bool amb = fabs(G) <= 1e-5 * mag;
                        if (kn != ke) { if (amb) ambig++; else wrong++; }
                        if (kn != fiver_k_at(nw_leg[(size_t)r * h.wpr + w], i) && !amb) legacy_diff++;
                    }
                }
            double berr = 0.0;
            for (uint32_t r = 0; r < sh.rows; r++) {
                double gs = 0.0;
                for (uint32_t b = 0; b < sh.batch; b++) gs += dy[(size_t)b * sh.rows + r];
                berr = fmax(berr, fabs(nb[r] - (bias[r] - (double)sh.lr * gs)));
            }
            report(wrong == 0 && frozen == 0 && berr < 1e-5 && legacy_diff == 0,
                   "wgrad+ste  grid=%s %4ux%-5u b=%-3u lr=%+.2f  moved=%u wrong=%u ambiguous=%u pad/reserved=%u  vs legacy=%u  bias_err=%.1e",
                   gname(g), sh.rows, sh.cols, sh.batch, sh.lr, moved, wrong, ambig, frozen, legacy_diff, berr);

            cudaFree(ddy); cudaFree(ddx); cudaFree(ddx2); cudaFree(dbias);
            release(d); release(d2);
        }
    ws.release();
}

/* ── 5. C ABI ─────────────────────────────────────────────────────────── */
void test_abi() {
    printf("[test] C ABI:\n");
    fiver_ctx* ctx = nullptr;
    report(fiver_ctx_create(0, &ctx) == FIVER_OK && ctx, "fiver_ctx_create (abi %d, emulator %d)",
           fiver_abi_version(), fiver_is_emulator());
    for (int g = 0; g < FIVER_NUM_GRIDS; g++) {
        const uint32_t rows = 45, cols = 301, batch = 20;
        Host h = make_host(g, rows, cols, false);
        fiver_matrix* m = nullptr;
        bool ok = fiver_matrix_create(ctx, h.words.data(), h.scale.data(), rows, cols, g, &m) == FIVER_OK;
        std::vector<uint32_t> back(h.words.size());
        ok = ok && fiver_matrix_read(ctx, m, back.data()) == FIVER_OK && back == h.words;
        auto x = rand_vec((size_t)batch * cols, -1.0f, 1.0f), bias = rand_vec(rows, -0.5f, 0.5f);
        std::vector<float> y((size_t)batch * rows), y8(y.size());
        ok = ok && fiver_forward(ctx, m, x.data(), y.data(), batch, bias.data(), FIVER_PATH_FLOAT) == FIVER_OK;
        const double ef = forward_err(h, x, y, bias, batch);
        ok = ok && fiver_forward(ctx, m, x.data(), y8.data(), batch, bias.data(), FIVER_PATH_INT8) == FIVER_OK;
        double num = 0, den = 0;
        for (size_t i = 0; i < y.size(); i++) { num = fmax(num, fabs(y8[i] - y[i])); den = fmax(den, fabs(y[i])); }
        auto dy = rand_vec((size_t)batch * rows, -1.0f, 1.0f);
        std::vector<float> dx((size_t)batch * cols);
        ok = ok && fiver_dgrad(ctx, m, dy.data(), dx.data(), batch) == FIVER_OK;
        std::vector<float> b2 = bias;
        ok = ok && fiver_ste_update(ctx, m, x.data(), dy.data(), batch, 0.05f, b2.data()) == FIVER_OK;
        ok = ok && fiver_matrix_read(ctx, m, back.data()) == FIVER_OK && back != h.words && b2 != bias;
        report(ok && ef < 1e-5 && num / den < 0.05, "grid=%s create/read/forward(float, int8)/dgrad/ste_update  float err=%.1e  int8 vs float=%.1e",
               gname(g), ef, num / den);
        fiver_matrix_destroy(m);
    }
    {
        fiver_matrix* m = nullptr;
        std::vector<uint32_t> w(fiver_words_per_row(70000), fiver_pad_word(0));
        float s = 1.0f;
        bool ok = fiver_matrix_create(ctx, w.data(), &s, 1, 70000, 0, &m) == FIVER_OK;
        std::vector<float> x(70000, 0.5f), y(1);
        ok = ok && fiver_forward(ctx, m, x.data(), y.data(), 1, nullptr, FIVER_PATH_INT8) == FIVER_ERR_UNSUPPORTED;
        const std::string msg = fiver_ctx_error(ctx);
        ok = ok && fiver_forward(ctx, m, x.data(), y.data(), 1, nullptr, FIVER_PATH_FLOAT) == FIVER_OK;
        report(ok && !msg.empty(), "cols > 65536: INT8 refused (\"%s\"), float works", msg.c_str());
        fiver_matrix_destroy(m);
    }
    {
        fiver_matrix* m = nullptr;
        const bool rejected = fiver_matrix_create(ctx, nullptr, nullptr, 0, 0, 0, &m) == FIVER_ERR_ARG && !m;
        const std::string msg = fiver_ctx_error(ctx);
        report(rejected && !msg.empty(), "invalid arguments rejected (\"%s\")", msg.c_str());
    }
    fiver_ctx_destroy(ctx);
}

}  // namespace

int run_all_tests() {
    g_failures = 0;
    test_tables();
    test_float_forward();
    test_int8_forward();
    test_backward();
    test_abi();
    printf("TEST %s (%d failure%s)\n", g_failures ? "FAILED" : "PASSED", g_failures, g_failures == 1 ? "" : "s");
    return g_failures == 0;
}
