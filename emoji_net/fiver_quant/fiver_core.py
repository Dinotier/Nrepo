"""
fiver_core.py — Core Fiver quantization/dequantization for external weight tensors.

Extends fiver.py with:
  - float_to_fiver(): quantize arbitrary float tensors → Fiver k-values + scale + sign
  - fiver_reconstruct(): inverse
  - per-tensor scale to handle any value range (not just [0,2])

Fiver encoding: k ∈ [0..31], w(k) = 1 ± 2^(-k/4) ∈ [0.0 .. 2.0]
With per-tensor scale s: stored_value = s * w(k) * sign_factor
"""
import sys
import numpy as np

# Pre-compute all 32 Fiver values once
_DELTA_LUT = np.array([2.0 ** (-m / 4.0) for m in range(16)], dtype=np.float64)
_K_TO_W = np.empty(32, dtype=np.float64)
for _k in range(32):
    _dir = (_k >> 4) & 1
    _mag = (31 - _k) if _dir else _k
    _K_TO_W[_k] = 1.0 + (_DELTA_LUT[_mag] if _dir else -_DELTA_LUT[_mag])

FIVER_VALUES = _K_TO_W.astype(np.float32)  # public, shape (32,)


def float_to_fiver(
    x: np.ndarray,
    use_sign: bool = True,
) -> dict:
    """
    Quantize a float32/float16 tensor to Fiver 5-bit representation.

    Parameters
    ----------
    x         : input weight tensor, any shape
    use_sign  : if True, handle negative values via sign_bits;
                if False, clamp negatives to 0.0 (unsigned-only layers)

    Returns dict with keys:
      'k_data'    : uint8 ndarray, same shape as x, values in [0..31]
      'scale'     : float32 scalar — multiply Fiver values by this to recover x
      'sign_bits' : uint8 ndarray (0=pos, 1=neg), same shape, or None if use_sign=False
      'shape'     : original shape tuple
      'dtype'     : original dtype
    """
    orig_shape = x.shape
    orig_dtype = x.dtype
    x = x.astype(np.float64).ravel()

    if use_sign:
        sign_bits = (x < 0).astype(np.uint8)
        x_abs = np.abs(x)
    else:
        sign_bits = None
        x_abs = np.clip(x, 0.0, np.inf)

    # Per-tensor scale: map max |x| → 2.0 (top of Fiver range)
    x_max = x_abs.max()
    if x_max < 1e-12:
        # Zero or near-zero tensor — return all k=15 (w≈0.926, near 1.0 center)
        k_data = np.full(x.shape, 15, dtype=np.uint8)
        return {
            'k_data': k_data.reshape(orig_shape),
            'scale': np.float32(0.0),
            'sign_bits': (sign_bits.reshape(orig_shape) if sign_bits is not None else None),
            'shape': orig_shape,
            'dtype': orig_dtype,
        }

    scale = np.float32(x_max / 2.0)
    x_norm = np.clip(x_abs / float(scale), 0.0, 2.0)  # normalized to [0, 2]

    # Vectorized nearest-neighbor search in FIVER_VALUES
    # dists: (N, 32) — find argmin along axis=1
    dists = np.abs(x_norm[:, np.newaxis] - FIVER_VALUES[np.newaxis, :].astype(np.float64))
    k_data = np.argmin(dists, axis=1).astype(np.uint8)

    return {
        'k_data':    k_data.reshape(orig_shape),
        'scale':     scale,
        'sign_bits': (sign_bits.reshape(orig_shape) if sign_bits is not None else None),
        'shape':     orig_shape,
        'dtype':     orig_dtype,
    }


def fiver_reconstruct(q: dict) -> np.ndarray:
    """
    Reconstruct float32 tensor from Fiver quantization dict (output of float_to_fiver).
    """
    k   = q['k_data'].astype(np.uint8).ravel()
    w   = FIVER_VALUES[k].astype(np.float64)
    w  *= float(q['scale'])
    if q['sign_bits'] is not None:
        sign_factor = np.where(q['sign_bits'].ravel(), -1.0, 1.0)
        w *= sign_factor
    return w.reshape(q['shape']).astype(np.float32)


def quantization_error(original: np.ndarray, q: dict) -> dict:
    """Compute reconstruction error metrics between original and quantized tensor."""
    recon = fiver_reconstruct(q)
    orig_f = original.astype(np.float32)
    diff   = orig_f - recon

    rel_err = np.abs(diff) / (np.abs(orig_f) + 1e-8)

    # KL divergence between normalized histograms (32 bins)
    bins = 32
    orig_hist, edges = np.histogram(orig_f.ravel(), bins=bins, density=True)
    rec_hist,  _     = np.histogram(recon.ravel(),  bins=edges, density=True)
    eps = 1e-10
    orig_hist = orig_hist + eps
    rec_hist  = rec_hist  + eps
    orig_hist /= orig_hist.sum()
    rec_hist  /= rec_hist.sum()
    kl = float(np.sum(orig_hist * np.log(orig_hist / rec_hist)))

    return {
        'l2_norm':       float(np.sqrt((diff**2).mean())),
        'max_abs_err':   float(np.abs(diff).max()),
        'mean_rel_err':  float(rel_err.mean()),
        'kl_divergence': kl,
        'bits_original': original.astype(np.float32).nbytes * 8,
        'bits_fiver':    q['k_data'].size * 5,  # 5 bits per k value
        'compression':   original.astype(np.float32).nbytes / max(q['k_data'].size * 5 / 8, 1),
    }


def memory_summary(name: str, shape: tuple, dtype, scale: float) -> str:
    """One-line memory comparison string."""
    n      = int(np.prod(shape))
    fp32   = n * 4
    fp16   = n * 2
    fiver5 = n * 5 // 8
    return (f"  {name:40s}  shape={str(shape):20s}"
            f"  FP32={fp32/1e6:.1f}MB  FP16={fp16/1e6:.1f}MB"
            f"  Fiver5={fiver5/1e6:.1f}MB  scale={scale:.4f}")
