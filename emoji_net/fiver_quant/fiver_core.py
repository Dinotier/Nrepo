"""
fiver_core.py — Fiver 5-bit quantisation for external weight tensors.

Code k ∈ [0..31].  Two value grids (identical to compact_weights.cu):

  centered (default, zero-centred weights — every LLM weight matrix):
      v(k) = −2^(−k/4) for k < 16,   v(16) = 0,   v(k) = +2^(−(31−k)/4) for k > 16
      Sign is the MSB of k; no sign bits.  Dense around 0 like the weights.
  anchored (original Fiver, positive values clustered near 1, e.g. RMSNorm γ):
      w(k) = 1 − 2^(−k/4) for k < 16,   1 + 2^(−(31−k)/4) for k ≥ 16

Tensor value:  x[r, c] ≈ scale[r] · v(k[r, c])   — one float scale per row
(row = all leading dimensions, columns = last dimension; 1-D = one row).

Packed storage: 6 codes per little-endian uint32 (bits 0-29, 30-31 reserved),
rows padded to a multiple of 6 with the grid's zero code (16 / 0).
5.333 bits/weight + 32 bits per row.
"""
import sys

import numpy as np

_DELTA = 2.0 ** (-np.arange(16) / 4.0)

ANCHORED = np.r_[1.0 - _DELTA, (1.0 + _DELTA)[::-1]]       # ascending in k
CENTERED = ANCHORED - 1.0
CENTERED[16] = 0.0

GRIDS     = {"centered": CENTERED, "anchored": ANCHORED}
ZERO_CODE = {"centered": 16, "anchored": 0}
WEIGHTS_PER_WORD = 6


def candidate_grids(x: np.ndarray) -> list[str]:
    """Grids worth trying: anchored only for tensors clustered at a positive value."""
    x = np.asarray(x, dtype=np.float64)
    return ["centered", "anchored"] if x.size and x.mean() > x.std() else ["centered"]


def _as_rows(a: np.ndarray) -> np.ndarray:
    return a.reshape(1, -1) if a.ndim <= 1 else a.reshape(-1, a.shape[-1])


def _nearest_codes(xn: np.ndarray, V: np.ndarray) -> np.ndarray:
    mid = (V[1:] + V[:-1]) / 2.0
    return np.searchsorted(mid, xn).astype(np.uint8)


def float_to_fiver(x: np.ndarray, grid: str = "auto", n_scales: int = 48,
                   chunk_rows: int = 4096, quiet: bool = False) -> dict:
    """
    Quantise a float tensor to Fiver codes with one MSE-optimal scale per row.

    The scale is searched over n_scales geometric steps between 2 % and 100 %
    of max|row| / max|v| (choosing the max alone wastes most codes on
    outliers — measured 3–10× worse on SmolLM-135M).

    grid="auto" quantises with every candidate grid and keeps the one with the
    lower error (anchored clips the few negatives such tensors may contain).

    Returns dict: k_data (uint8, x.shape), scale (float32, rows),
                  grid (str), shape (tuple).
    """
    if grid == "auto":
        qs = [float_to_fiver(x, g, n_scales, chunk_rows, quiet=True) for g in candidate_grids(x)]
        return min(qs, key=lambda q: quantization_error(x, q)["rel_rmse"])
    if grid not in GRIDS:
        raise ValueError(f"grid must be 'auto', 'centered' or 'anchored', got {grid!r}")
    V = GRIDS[grid]
    shape = tuple(x.shape)
    X = _as_rows(np.asarray(x, dtype=np.float64))
    if grid == "anchored" and X.size and X.min() < 0:
        if not quiet:
            print(f"[fiver_core] WARNING: anchored grid clips {int((X < 0).sum())} negative values to 0",
                  file=sys.stderr)
        X = np.maximum(X, 0.0)

    rows = X.shape[0]
    k_out = np.empty(X.shape, dtype=np.uint8)
    s_out = np.zeros(rows, dtype=np.float32)
    t_grid = np.geomspace(0.02, 1.0, n_scales)

    for r0 in range(0, rows, chunk_rows):
        Xc   = X[r0:r0 + chunk_rows]
        base = np.abs(Xc).max(axis=1) / np.abs(V).max()
        best_e = np.full(len(Xc), np.inf)
        best_k = np.full(Xc.shape, ZERO_CODE[grid], dtype=np.uint8)
        best_s = np.zeros(len(Xc))
        for t in t_grid:
            s  = base * t
            xn = np.divide(Xc, s[:, None], out=np.zeros_like(Xc), where=s[:, None] > 0)
            k  = _nearest_codes(xn, V)
            e  = ((Xc - V[k] * s[:, None]) ** 2).sum(axis=1)
            better = (e < best_e) & (s > 0)
            best_e[better] = e[better]
            best_k[better] = k[better]
            best_s[better] = s[better]
        k_out[r0:r0 + chunk_rows] = best_k
        s_out[r0:r0 + chunk_rows] = best_s

    return {"k_data": k_out.reshape(shape), "scale": s_out, "grid": grid, "shape": shape}


def fiver_reconstruct(q: dict) -> np.ndarray:
    """Inverse of float_to_fiver → float32 tensor."""
    V = GRIDS[str(q["grid"])]
    k = _as_rows(np.asarray(q["k_data"]))
    return (V[k] * np.asarray(q["scale"], dtype=np.float64)[:, None]).reshape(q["shape"]).astype(np.float32)


def bits_per_weight(shape: tuple) -> float:
    cols = int(shape[-1]) if len(shape) else 1
    rows = int(np.prod(shape)) // max(cols, 1)
    wpr  = -(-cols // WEIGHTS_PER_WORD)
    return (rows * wpr * 32 + rows * 32) / max(rows * cols, 1)


def quantization_error(original: np.ndarray, q: dict) -> dict:
    """Reconstruction error metrics between original and quantised tensor."""
    o = np.asarray(original, dtype=np.float64)
    r = fiver_reconstruct(q).astype(np.float64)
    diff = o - r
    rel  = float(np.sqrt((diff ** 2).sum() / max((o ** 2).sum(), 1e-30)))
    bpw  = bits_per_weight(q["shape"])
    return {
        "rel_rmse":     rel,
        "sqnr_db":      float(-20 * np.log10(max(rel, 1e-12))),
        "max_abs_err":  float(np.abs(diff).max()) if diff.size else 0.0,
        "bpw":          bpw,
        "compression":  32.0 / bpw,
    }


def memory_summary(name: str, shape: tuple, grid: str) -> str:
    """One-line memory comparison string."""
    n   = int(np.prod(shape))
    bpw = bits_per_weight(shape)
    return (f"  {name:48s}  shape={str(tuple(shape)):18s} grid={grid:8s}"
            f"  FP32={n * 4 / 1e6:8.2f}MB  FP16={n * 2 / 1e6:8.2f}MB"
            f"  Fiver={n * bpw / 8 / 1e6:7.2f}MB ({bpw:.3f} bpw)")


def pack_fiver(k_data: np.ndarray, grid: str) -> bytes:
    """Row-padded little-endian uint32 words, 6 codes each (k_i at bits 5i)."""
    k = _as_rows(np.asarray(k_data)).astype(np.uint32)
    pad = (-k.shape[1]) % WEIGHTS_PER_WORD
    if pad:
        k = np.pad(k, ((0, 0), (0, pad)), constant_values=ZERO_CODE[grid])
    k6 = k.reshape(k.shape[0], -1, WEIGHTS_PER_WORD)
    words = np.zeros(k6.shape[:2], dtype=np.uint32)
    for i in range(WEIGHTS_PER_WORD):
        words |= (k6[..., i] & 0x1F) << (5 * i)
    return words.astype("<u4").tobytes()


def unpack_fiver(buf: bytes, shape: tuple) -> np.ndarray:
    """Inverse of pack_fiver → uint8 codes with `shape`."""
    cols = int(shape[-1]) if len(shape) else 1
    rows = int(np.prod(shape)) // max(cols, 1)
    wpr  = -(-cols // WEIGHTS_PER_WORD)
    words = np.frombuffer(buf, dtype="<u4", count=rows * wpr).reshape(rows, wpr)
    k = np.stack([((words >> (5 * i)) & 0x1F).astype(np.uint8) for i in range(WEIGHTS_PER_WORD)],
                 axis=-1)
    return k.reshape(rows, wpr * WEIGHTS_PER_WORD)[:, :cols].reshape(shape)


def save_npz(path, q: dict) -> None:
    np.savez_compressed(path, k_data=q["k_data"], scale=np.asarray(q["scale"], np.float32),
                        grid=np.array(q["grid"]), shape=np.array(q["shape"], dtype=np.int64))


def load_npz(path) -> dict:
    d = np.load(str(path), allow_pickle=False)
    shape = tuple(int(s) for s in d["shape"]) if "shape" in d else d["k_data"].shape
    return {"k_data": d["k_data"].reshape(shape), "scale": d["scale"],
            "grid": str(d["grid"]), "shape": shape}
