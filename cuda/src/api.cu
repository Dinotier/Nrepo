/*
 * api.cu — C ABI, contexts, matrix handles and path dispatch.
 */
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "fiver/fiver.h"
#include "internal.cuh"

namespace fiver {

void* Workspace::get(Slot s, size_t bytes) {
    if (bytes == 0) bytes = 1;
    if (size[s] < bytes) {
        if (ptr[s]) cudaFree(ptr[s]);
        ptr[s] = nullptr;
        size[s] = 0;
        if (cudaMalloc(&ptr[s], bytes) != cudaSuccess) return nullptr;
        size[s] = bytes;
    }
    return ptr[s];
}

void Workspace::release() {
    for (int i = 0; i < NUM_SLOTS; i++) {
        if (ptr[i]) cudaFree(ptr[i]);
        ptr[i] = nullptr;
        size[i] = 0;
    }
}

}  // namespace fiver

struct fiver_matrix {
    fiver::Matrix m;
};

struct fiver_ctx {
    int                 device  = 0;
    int                 sm      = 0;
    int                 num_sms = 1;
    char                name[256] = {};
    cudaStream_t        stream  = nullptr;
    fiver::Workspace    ws;
    void*               io[4]   = {};
    size_t              io_size[4] = {};
    std::string         err;

    void* io_buf(int i, size_t bytes) {
        if (io_size[i] < bytes) {
            if (io[i]) cudaFree(io[i]);
            io[i] = nullptr;
            io_size[i] = 0;
            if (cudaMalloc(&io[i], bytes) != cudaSuccess) return nullptr;
            io_size[i] = bytes;
        }
        return io[i];
    }
};

namespace {

fiver_status fail(fiver_ctx* ctx, fiver_status st, const std::string& msg) {
    if (ctx) ctx->err = msg;
    return st;
}

fiver_status check_cuda(fiver_ctx* ctx, const char* where) {
    cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) e = cudaStreamSynchronize(ctx->stream);
    if (e != cudaSuccess) return fail(ctx, FIVER_ERR_CUDA, std::string(where) + ": " + cudaGetErrorString(e));
    ctx->err.clear();
    return FIVER_OK;
}

#define FIVER_TRY_ALLOC(ptr, ctx) \
    do { if (!(ptr)) return fail((ctx), FIVER_ERR_ALLOC, "device allocation failed"); } while (0)

}  // namespace

extern "C" {

int fiver_abi_version(void) { return FIVER_ABI_VERSION; }

int fiver_is_emulator(void) {
#ifdef FIVER_EMU
    return 1;
#else
    return 0;
#endif
}

fiver_status fiver_ctx_create(int device, fiver_ctx** out) {
    if (!out) return FIVER_ERR_ARG;
    *out = nullptr;
    if (cudaSetDevice(device) != cudaSuccess) return FIVER_ERR_CUDA;
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, device) != cudaSuccess) return FIVER_ERR_CUDA;
    fiver_ctx* ctx = new fiver_ctx;
    ctx->device  = device;
    ctx->sm      = p.major * 10 + p.minor;
    ctx->num_sms = p.multiProcessorCount;
    snprintf(ctx->name, sizeof(ctx->name), "%s", p.name);
    *out = ctx;
    return FIVER_OK;
}

void fiver_ctx_destroy(fiver_ctx* ctx) {
    if (!ctx) return;
    ctx->ws.release();
    for (int i = 0; i < 4; i++) if (ctx->io[i]) cudaFree(ctx->io[i]);
    delete ctx;
}

const char* fiver_ctx_error(const fiver_ctx* ctx) { return ctx ? ctx->err.c_str() : "no context"; }

fiver_status fiver_ctx_device(fiver_ctx* ctx, char* name, int name_len, int* sm, int* num_sms) {
    if (!ctx) return FIVER_ERR_ARG;
    if (name && name_len > 0) snprintf(name, (size_t)name_len, "%s", ctx->name);
    if (sm) *sm = ctx->sm;
    if (num_sms) *num_sms = ctx->num_sms;
    return FIVER_OK;
}

fiver_status fiver_matrix_create(fiver_ctx* ctx, const uint32_t* words, const float* row_scale,
                                 uint32_t rows, uint32_t cols, int grid, fiver_matrix** out) {
    if (!ctx || !words || !row_scale || !out || rows == 0 || cols == 0 || grid < 0 || grid >= FIVER_NUM_GRIDS)
        return fail(ctx, FIVER_ERR_ARG, "fiver_matrix_create: invalid argument");
    fiver::Matrix m;
    m.rows = rows; m.cols = cols; m.grid = grid;
    m.wpr  = fiver_words_per_row(cols);
    m.ld   = fiver_row_pitch(cols);
    std::vector<uint32_t> host((size_t)rows * m.ld, fiver_pad_word(grid));
    for (uint32_t r = 0; r < rows; r++)
        std::memcpy(&host[(size_t)r * m.ld], &words[(size_t)r * m.wpr], m.wpr * sizeof(uint32_t));
    if (cudaMalloc(&m.words, host.size() * sizeof(uint32_t)) != cudaSuccess)
        return fail(ctx, FIVER_ERR_ALLOC, "device allocation failed");
    if (cudaMalloc(&m.row_scale, rows * sizeof(float)) != cudaSuccess) {
        cudaFree(m.words);
        return fail(ctx, FIVER_ERR_ALLOC, "device allocation failed");
    }
    cudaMemcpy(m.words, host.data(), host.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(m.row_scale, row_scale, rows * sizeof(float), cudaMemcpyHostToDevice);
    fiver_status st = check_cuda(ctx, "fiver_matrix_create");
    if (st != FIVER_OK) { cudaFree(m.words); cudaFree(m.row_scale); return st; }
    *out = new fiver_matrix{ m };
    return FIVER_OK;
}

void fiver_matrix_destroy(fiver_matrix* mh) {
    if (!mh) return;
    cudaFree(mh->m.words);
    cudaFree(mh->m.row_scale);
    delete mh;
}

fiver_status fiver_matrix_read(fiver_ctx* ctx, const fiver_matrix* mh, uint32_t* words) {
    if (!ctx || !mh || !words) return fail(ctx, FIVER_ERR_ARG, "fiver_matrix_read: invalid argument");
    const fiver::Matrix& m = mh->m;
    std::vector<uint32_t> host((size_t)m.rows * m.ld);
    cudaMemcpy(host.data(), m.words, host.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    for (uint32_t r = 0; r < m.rows; r++)
        std::memcpy(&words[(size_t)r * m.wpr], &host[(size_t)r * m.ld], m.wpr * sizeof(uint32_t));
    return check_cuda(ctx, "fiver_matrix_read");
}

fiver_status fiver_forward(fiver_ctx* ctx, const fiver_matrix* mh, const float* x, float* y,
                           uint32_t batch, const float* bias, int path) {
    if (!ctx || !mh || !x || !y || batch == 0) return fail(ctx, FIVER_ERR_ARG, "fiver_forward: invalid argument");
    const fiver::Matrix& m = mh->m;
    const bool int8 = path == FIVER_PATH_INT8 || path == FIVER_PATH_INT8_TC;
    if (int8 && m.act_bits) return fail(ctx, FIVER_ERR_UNSUPPORTED, "act_bits: float GEMV only");
    if (int8 && m.cols > 65536u) return fail(ctx, FIVER_ERR_UNSUPPORTED, "INT8 path: cols > 65536");

    float* dx = static_cast<float*>(ctx->io_buf(0, (size_t)batch * m.cols * sizeof(float)));
    float* dy = static_cast<float*>(ctx->io_buf(1, (size_t)batch * m.rows * sizeof(float)));
    float* db = bias ? static_cast<float*>(ctx->io_buf(2, m.rows * sizeof(float))) : nullptr;
    FIVER_TRY_ALLOC(dx, ctx);
    FIVER_TRY_ALLOC(dy, ctx);
    if (bias) FIVER_TRY_ALLOC(db, ctx);
    cudaMemcpyAsync(dx, x, (size_t)batch * m.cols * sizeof(float), cudaMemcpyHostToDevice, ctx->stream);
    if (bias) cudaMemcpyAsync(db, bias, m.rows * sizeof(float), cudaMemcpyHostToDevice, ctx->stream);

    if (path == FIVER_PATH_INT8_TC) {
        if (!fiver::gemm_tc(m, dx, dy, batch, db, ctx->ws, ctx->stream))
            return fail(ctx, FIVER_ERR_UNSUPPORTED, "INT8 Tensor Cores need sm_75+");
    } else if (int8) {
        if (batch <= 8) fiver::gemv_dp4a(m, dx, dy, batch, db, ctx->ws, ctx->stream);
        else            fiver::gemm_dp4a(m, dx, dy, batch, db, ctx->ws, ctx->num_sms, 0, ctx->stream);
    } else if (m.act_bits || batch <= 8) {
        fiver::gemv_float(m, dx, dy, batch, db, fiver::GEMV_V2, ctx->stream);
    } else {
        fiver::gemm_float(m, dx, dy, batch, db, ctx->ws, ctx->num_sms, 0, ctx->stream);
    }
    cudaMemcpyAsync(y, dy, (size_t)batch * m.rows * sizeof(float), cudaMemcpyDeviceToHost, ctx->stream);
    return check_cuda(ctx, "fiver_forward");
}

fiver_status fiver_dgrad(fiver_ctx* ctx, const fiver_matrix* mh, const float* dy, float* dx,
                         uint32_t batch) {
    if (!ctx || !mh || !dy || !dx || batch == 0) return fail(ctx, FIVER_ERR_ARG, "fiver_dgrad: invalid argument");
    const fiver::Matrix& m = mh->m;
    if (m.act_bits) return fail(ctx, FIVER_ERR_UNSUPPORTED, "act_bits: float GEMV only");
    float* d_dy = static_cast<float*>(ctx->io_buf(1, (size_t)batch * m.rows * sizeof(float)));
    float* d_dx = static_cast<float*>(ctx->io_buf(0, (size_t)batch * m.cols * sizeof(float)));
    FIVER_TRY_ALLOC(d_dy, ctx);
    FIVER_TRY_ALLOC(d_dx, ctx);
    cudaMemcpyAsync(d_dy, dy, (size_t)batch * m.rows * sizeof(float), cudaMemcpyHostToDevice, ctx->stream);
    fiver::dgrad(m, d_dy, d_dx, batch, ctx->ws, ctx->num_sms, 0, ctx->stream);
    cudaMemcpyAsync(dx, d_dx, (size_t)batch * m.cols * sizeof(float), cudaMemcpyDeviceToHost, ctx->stream);
    return check_cuda(ctx, "fiver_dgrad");
}

fiver_status fiver_ste_update(fiver_ctx* ctx, fiver_matrix* mh, const float* x, const float* dy,
                              uint32_t batch, float lr, float* bias) {
    if (!ctx || !mh || !x || !dy || batch == 0) return fail(ctx, FIVER_ERR_ARG, "fiver_ste_update: invalid argument");
    const fiver::Matrix& m = mh->m;
    if (m.act_bits) return fail(ctx, FIVER_ERR_UNSUPPORTED, "act_bits: float GEMV only");
    float* d_x  = static_cast<float*>(ctx->io_buf(0, (size_t)batch * m.cols * sizeof(float)));
    float* d_dy = static_cast<float*>(ctx->io_buf(1, (size_t)batch * m.rows * sizeof(float)));
    float* d_b  = bias ? static_cast<float*>(ctx->io_buf(2, m.rows * sizeof(float))) : nullptr;
    FIVER_TRY_ALLOC(d_x, ctx);
    FIVER_TRY_ALLOC(d_dy, ctx);
    if (bias) FIVER_TRY_ALLOC(d_b, ctx);
    cudaMemcpyAsync(d_x, x, (size_t)batch * m.cols * sizeof(float), cudaMemcpyHostToDevice, ctx->stream);
    cudaMemcpyAsync(d_dy, dy, (size_t)batch * m.rows * sizeof(float), cudaMemcpyHostToDevice, ctx->stream);
    if (bias) cudaMemcpyAsync(d_b, bias, m.rows * sizeof(float), cudaMemcpyHostToDevice, ctx->stream);
    fiver::wgrad_ste(m, d_x, d_dy, d_b, batch, lr, ctx->stream);
    if (bias) cudaMemcpyAsync(bias, d_b, m.rows * sizeof(float), cudaMemcpyDeviceToHost, ctx->stream);
    return check_cuda(ctx, "fiver_ste_update");
}

}  // extern "C"
