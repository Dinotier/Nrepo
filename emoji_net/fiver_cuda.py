"""
fiver_cuda.py — ctypes binding to libfiver_cuda (C ABI: cuda/include/fiver/fiver.h).

Library search order:
  1. $FIVER_CUDA_LIB
  2. <repo>/build/cuda/libfiver_cuda.so       (CUDA build)
  3. <repo>/build-emu/cuda/libfiver_cuda.so   (CPU emulator build, tests only)

All arrays are float32 / uint8 numpy arrays; the library copies them through
its own device buffers, matrices stay on the device between calls.
"""
import ctypes
import os
from pathlib import Path

import numpy as np

GRID_IDS = {"centered": 0, "anchored": 1, "magic": 2}
PATH_IDS = {"auto": 0, "float": 1, "int8": 2, "int8_tc": 3}
ZERO_CODE = {"centered": 16, "anchored": 0, "magic": 16}
ABI_VERSION = 1

_REPO = Path(__file__).resolve().parent.parent
_lib = None


class FiverCudaError(RuntimeError):
    pass


def _candidates():
    env = os.environ.get("FIVER_CUDA_LIB")
    if env:
        yield Path(env)
    yield _REPO / "build" / "cuda" / "libfiver_cuda.so"
    yield _REPO / "build-emu" / "cuda" / "libfiver_cuda.so"


def load(path=None):
    """Load the library once; raises FiverCudaError if it is missing or incompatible."""
    global _lib
    if _lib is not None:
        return _lib
    paths = [Path(path)] if path else list(_candidates())
    for p in paths:
        if p.is_file():
            lib = ctypes.CDLL(str(p))
            break
    else:
        raise FiverCudaError("libfiver_cuda not found (tried: " + ", ".join(map(str, paths)) + ")")

    c_u32p = ctypes.POINTER(ctypes.c_uint32)
    c_f32p = ctypes.POINTER(ctypes.c_float)
    vp = ctypes.c_void_p
    lib.fiver_abi_version.restype = ctypes.c_int
    lib.fiver_is_emulator.restype = ctypes.c_int
    lib.fiver_ctx_create.argtypes = [ctypes.c_int, ctypes.POINTER(vp)]
    lib.fiver_ctx_destroy.argtypes = [vp]
    lib.fiver_ctx_error.argtypes = [vp]
    lib.fiver_ctx_error.restype = ctypes.c_char_p
    lib.fiver_ctx_device.argtypes = [vp, ctypes.c_char_p, ctypes.c_int,
                                     ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
    lib.fiver_matrix_create.argtypes = [vp, c_u32p, c_f32p, ctypes.c_uint32, ctypes.c_uint32,
                                        ctypes.c_int, ctypes.POINTER(vp)]
    lib.fiver_matrix_destroy.argtypes = [vp]
    lib.fiver_matrix_read.argtypes = [vp, vp, c_u32p]
    lib.fiver_forward.argtypes = [vp, vp, c_f32p, c_f32p, ctypes.c_uint32, c_f32p, ctypes.c_int]
    lib.fiver_dgrad.argtypes = [vp, vp, c_f32p, c_f32p, ctypes.c_uint32]
    lib.fiver_ste_update.argtypes = [vp, vp, c_f32p, c_f32p, ctypes.c_uint32, ctypes.c_float, c_f32p]
    for f in ("fiver_ctx_create", "fiver_ctx_device", "fiver_matrix_create", "fiver_matrix_read",
              "fiver_forward", "fiver_dgrad", "fiver_ste_update"):
        getattr(lib, f).restype = ctypes.c_int
    if lib.fiver_abi_version() != ABI_VERSION:
        raise FiverCudaError(f"ABI mismatch: library {lib.fiver_abi_version()}, binding {ABI_VERSION}")
    _lib = lib
    return lib


def available():
    try:
        load()
        return True
    except (FiverCudaError, OSError):
        return False


def _f32(a):
    return np.ascontiguousarray(a, dtype=np.float32)


def _ptr(a, ctype):
    return a.ctypes.data_as(ctypes.POINTER(ctype)) if a is not None else None


def pack_codes(k, grid):
    """uint8 codes (rows × cols) → external words (rows × ceil(cols/6)) uint32."""
    k = np.asarray(k, dtype=np.uint32)
    rows, cols = k.shape
    pad = (-cols) % 6
    if pad:
        k = np.pad(k, ((0, 0), (0, pad)), constant_values=ZERO_CODE[grid])
    k6 = k.reshape(rows, -1, 6)
    words = np.zeros(k6.shape[:2], dtype=np.uint32)
    for i in range(6):
        words |= (k6[..., i] & 31) << (5 * i)
    return np.ascontiguousarray(words)


def unpack_codes(words, cols):
    words = np.asarray(words, dtype=np.uint32)
    k = np.stack([((words >> (5 * i)) & 31).astype(np.uint8) for i in range(6)], axis=-1)
    return np.ascontiguousarray(k.reshape(words.shape[0], -1)[:, :cols])


class Context:
    def __init__(self, device=0):
        self.lib = load()
        self._h = ctypes.c_void_p()
        self._check(self.lib.fiver_ctx_create(device, ctypes.byref(self._h)), "fiver_ctx_create")
        name = ctypes.create_string_buffer(256)
        sm, nsm = ctypes.c_int(), ctypes.c_int()
        self.lib.fiver_ctx_device(self._h, name, 256, ctypes.byref(sm), ctypes.byref(nsm))
        self.device_name, self.sm, self.num_sms = name.value.decode(), sm.value, nsm.value
        self.emulator = bool(self.lib.fiver_is_emulator())

    def _check(self, status, what):
        if status != 0:
            msg = self.lib.fiver_ctx_error(self._h).decode() if self._h else ""
            raise FiverCudaError(f"{what} failed (status {status}): {msg}")

    def close(self):
        if self._h:
            self.lib.fiver_ctx_destroy(self._h)
            self._h = ctypes.c_void_p()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def matrix(self, k, row_scale, grid="centered"):
        return Matrix(self, k, row_scale, grid)

    def forward(self, m, x, bias=None, path="float"):
        x = _f32(x)
        batch = x.shape[0]
        y = np.empty((batch, m.rows), dtype=np.float32)
        b = _f32(bias) if bias is not None else None
        self._check(self.lib.fiver_forward(self._h, m._h, _ptr(x, ctypes.c_float), _ptr(y, ctypes.c_float),
                                           batch, _ptr(b, ctypes.c_float), PATH_IDS[path]), "fiver_forward")
        return y

    def dgrad(self, m, dy):
        dy = _f32(dy)
        dx = np.empty((dy.shape[0], m.cols), dtype=np.float32)
        self._check(self.lib.fiver_dgrad(self._h, m._h, _ptr(dy, ctypes.c_float), _ptr(dx, ctypes.c_float),
                                         dy.shape[0]), "fiver_dgrad")
        return dx

    def ste_update(self, m, x, dy, lr, bias=None):
        """In-place code update; bias (float32 array) is updated in place if given."""
        x, dy = _f32(x), _f32(dy)
        if bias is not None and (bias.dtype != np.float32 or not bias.flags.c_contiguous):
            raise ValueError("bias must be a contiguous float32 array (updated in place)")
        self._check(self.lib.fiver_ste_update(self._h, m._h, _ptr(x, ctypes.c_float), _ptr(dy, ctypes.c_float),
                                              x.shape[0], float(lr), _ptr(bias, ctypes.c_float)),
                    "fiver_ste_update")
        m._host_cache = None


class Matrix:
    def __init__(self, ctx, k, row_scale, grid="centered"):
        k = np.asarray(k, dtype=np.uint8)
        self.ctx, self.rows, self.cols, self.grid = ctx, int(k.shape[0]), int(k.shape[1]), grid
        words = pack_codes(k, grid)
        scale = _f32(np.broadcast_to(row_scale, (self.rows,)))
        self._h = ctypes.c_void_p()
        ctx._check(ctx.lib.fiver_matrix_create(ctx._h, _ptr(words, ctypes.c_uint32), _ptr(scale, ctypes.c_float),
                                               self.rows, self.cols, GRID_IDS[grid], ctypes.byref(self._h)),
                   "fiver_matrix_create")
        self._host_cache = np.array(k, copy=True)

    def codes(self):
        """Current codes (rows × cols uint8), downloaded after updates."""
        if self._host_cache is None:
            words = np.empty((self.rows, -(-self.cols // 6)), dtype=np.uint32)
            self.ctx._check(self.ctx.lib.fiver_matrix_read(self.ctx._h, self._h, _ptr(words, ctypes.c_uint32)),
                            "fiver_matrix_read")
            self._host_cache = unpack_codes(words, self.cols)
        return self._host_cache

    def close(self):
        if self._h:
            self.ctx.lib.fiver_matrix_destroy(self._h)
            self._h = ctypes.c_void_p()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


_default_ctx = None


def default_context():
    global _default_ctx
    if _default_ctx is None:
        _default_ctx = Context(0)
    return _default_ctx
