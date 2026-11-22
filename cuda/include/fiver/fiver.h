/*
 * fiver/fiver.h — C ABI of libfiver_cuda (loaded from Python via ctypes).
 *
 * All x / y / dy / dx / bias pointers are host pointers (row-major float32):
 * the library copies them through its own device buffers.  Matrices stay on
 * the device between calls.  Every function returns FIVER_OK or an error;
 * fiver_ctx_error() gives the message.  A context is not thread-safe.
 *
 * Shapes: W is rows × cols (out × in).  x: batch × cols, y: batch × rows,
 * dy: batch × rows, dx: batch × cols, bias: rows.
 * words use the external layout of fiver/format.h (row stride ceil(cols/6)).
 */
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#define FIVER_API __declspec(dllexport)
#else
#define FIVER_API __attribute__((visibility("default")))
#endif

#define FIVER_ABI_VERSION 1

typedef struct fiver_ctx    fiver_ctx;
typedef struct fiver_matrix fiver_matrix;

typedef enum {
    FIVER_OK              = 0,
    FIVER_ERR_ARG         = 1,
    FIVER_ERR_CUDA        = 2,
    FIVER_ERR_UNSUPPORTED = 3,
    FIVER_ERR_ALLOC       = 4
} fiver_status;

enum {
    FIVER_PATH_AUTO    = 0,   /* currently = FLOAT                                    */
    FIVER_PATH_FLOAT   = 1,   /* GEMV v2 for batch ≤ 8, tiled FP32 GEMM above           */
    FIVER_PATH_INT8    = 2,   /* __dp4a GEMV for batch ≤ 8, tiled __dp4a GEMM above     */
    FIVER_PATH_INT8_TC = 3    /* WMMA Tensor Cores (sm_75+), not validated on hardware */
};

FIVER_API int          fiver_abi_version(void);
FIVER_API int          fiver_is_emulator(void);

FIVER_API fiver_status fiver_ctx_create(int device, fiver_ctx** out);
FIVER_API void         fiver_ctx_destroy(fiver_ctx* ctx);
FIVER_API const char*  fiver_ctx_error(const fiver_ctx* ctx);
FIVER_API fiver_status fiver_ctx_device(fiver_ctx* ctx, char* name, int name_len, int* sm, int* num_sms);

FIVER_API fiver_status fiver_matrix_create(fiver_ctx* ctx, const uint32_t* words, const float* row_scale,
                                           uint32_t rows, uint32_t cols, int grid, fiver_matrix** out);
FIVER_API void         fiver_matrix_destroy(fiver_matrix* m);
FIVER_API fiver_status fiver_matrix_read(fiver_ctx* ctx, const fiver_matrix* m, uint32_t* words);

/* y = row_scale ∘ (W · x) + bias   (bias may be NULL) */
FIVER_API fiver_status fiver_forward(fiver_ctx* ctx, const fiver_matrix* m, const float* x, float* y,
                                     uint32_t batch, const float* bias, int path);

/* dx = Wᵀ · (row_scale ∘ dy) with the current weights; deterministic */
FIVER_API fiver_status fiver_dgrad(fiver_ctx* ctx, const fiver_matrix* m, const float* dy, float* dx,
                                   uint32_t batch);

/* k ← clamp(k − sign(lr · dyᵀx), 0, 31) in place; bias (in/out, may be NULL) −= lr · Σ_b dy */
FIVER_API fiver_status fiver_ste_update(fiver_ctx* ctx, fiver_matrix* m, const float* x, const float* dy,
                                        uint32_t batch, float lr, float* bias);

#ifdef __cplusplus
}
#endif
