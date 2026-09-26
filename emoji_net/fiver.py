"""
fiver.py — Symmetric 5-bit Fiver weight encoding.

Encoding: k in [0..31], MSB (bit4) encodes direction, bits[3:0] = magnitude.
  k in [ 0..15] (bit4=0): w = 1 - 2^(-k/4)       range [0.0  .. 0.926]
  k in [16..31] (bit4=1): w = 1 + 2^(-(31-k)/4)   range [1.074 .. 2.0 ]

Values are monotonically increasing with k.
Distribution is sigmoid-shaped: fine resolution near 1.0, coarse at extremes.
No separate sign/formula arrays — direction is implicit in the MSB.

STE update rule: k += -sign(grad), saturating at [0, 31].
One integer increment/decrement per weight per step.
"""
import numpy as np

# 16-entry LUT: delta_lut[m] = 2^(-m/4) for m = 0..15
DELTA_LUT = np.array([2.0**(-m / 4.0) for m in range(16)], dtype=np.float32)


def fiver_to_float(k: np.ndarray) -> np.ndarray:
    """Materialize Fiver k-values (uint8, [0..31]) to float32 weights."""
    k   = np.asarray(k, dtype=np.uint8)
    dir_ = (k >> 4) & 1                      # MSB: 0=below 1.0, 1=above 1.0
    mag  = np.where(dir_, 31 - k, k) & 0xF   # magnitude index 0..15
    delta = DELTA_LUT[mag]
    return (1.0 + np.where(dir_, +delta, -delta)).astype(np.float32)


class FiverLayer:
    """Dense layer with Fiver-quantized weights and float32 biases.

    All weight gradient accumulation uses the STE (Straight-Through Estimator):
    gradients flow through materialize() as if it were the identity function.
    The discrete update is: k -= sign(grad_k) (saturating).
    """

    def __init__(self, in_dim: int, out_dim: int, act: str = 'identity',
                 use_sign: bool = False, seed: int = 0):
        self.in_dim   = in_dim
        self.out_dim  = out_dim
        self.act      = act
        self.use_sign = use_sign

        rng = np.random.default_rng(seed)
        # Initialize k near center (w close to 1.0) for stable early training
        self.k_data   = rng.integers(12, 20, size=(out_dim, in_dim), dtype=np.uint8)
        self.sign_bits = (np.zeros((out_dim, in_dim), dtype=np.uint8)
                          if use_sign else None)
        self.bias = np.zeros(out_dim, dtype=np.float32)

        self._grad_k    = np.zeros((out_dim, in_dim), dtype=np.float32)
        self._grad_bias = np.zeros(out_dim, dtype=np.float32)

    # ── Forward ──────────────────────────────────────────────────────────

    def materialize(self) -> np.ndarray:
        """Return float32 weight matrix (out_dim × in_dim)."""
        w = fiver_to_float(self.k_data)
        if self.use_sign and self.sign_bits is not None:
            w = w * np.where(self.sign_bits, -1.0, 1.0)
        return w

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (..., in_dim) → (..., out_dim)"""
        W   = self.materialize()                   # (out_dim, in_dim)
        out = x @ W.T + self.bias                  # (..., out_dim)
        return _apply_act(out, self.act)

    # ── Backward ─────────────────────────────────────────────────────────

    def backward(self, x: np.ndarray, grad_out: np.ndarray) -> np.ndarray:
        """Accumulate gradients via STE. Returns grad_x."""
        pre_act = x @ self.materialize().T + self.bias
        g       = _act_grad(grad_out, self.act, pre_act)  # through activation

        # Flatten batch dimensions for gradient accumulation
        x_flat  = x.reshape(-1, self.in_dim)
        g_flat  = g.reshape(-1, self.out_dim)

        self._grad_k    += g_flat.T @ x_flat         # (out_dim, in_dim)
        self._grad_bias += g_flat.sum(axis=0)

        return g @ self.materialize()                 # grad_x: (..., in_dim)

    # ── Parameter update ─────────────────────────────────────────────────

    def step(self, lr: float):
        """STE discrete update: k -= sign(accumulated_grad), saturating [0,31]."""
        delta_k = np.sign(self._grad_k * lr).astype(np.int16)
        self.k_data = np.clip(
            self.k_data.astype(np.int16) - delta_k, 0, 31
        ).astype(np.uint8)
        self.bias -= lr * self._grad_bias
        self._grad_k[:]    = 0.0
        self._grad_bias[:] = 0.0

    @property
    def num_params(self) -> int:
        return int(self.k_data.size + self.bias.size)


# ── Activation functions ──────────────────────────────────────────────────

def _apply_act(x: np.ndarray, act: str) -> np.ndarray:
    if act == 'softsign': return x / (1.0 + np.abs(x))
    if act == 'relu':     return np.maximum(0.0, x)
    if act == 'gelu':
        return x * 0.5 * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x**3)))
    return x  # identity


def _act_grad(grad: np.ndarray, act: str, pre: np.ndarray) -> np.ndarray:
    if act == 'softsign':
        return grad / (1.0 + np.abs(pre))**2
    if act == 'relu':
        return grad * (pre > 0.0).astype(np.float32)
    if act == 'gelu':
        t   = np.tanh(0.7978845608028654 * (pre + 0.044715 * pre**3))
        dt  = 1.0 - t**2
        d_inner = 0.7978845608028654 * (1.0 + 3.0 * 0.044715 * pre**2)
        return grad * (0.5 * (1.0 + t) + 0.5 * pre * dt * d_inner)
    return grad  # identity
