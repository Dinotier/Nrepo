"""
fiver.py — Fiver 5-bit weight encoding for training (EmojiNet).

Code k ∈ [0..31], two value grids (identical to compact_weights.cu):

  centered (default):  v(k) = −2^(−k/4) for k < 16,  v(16) = 0,  +2^(−(31−k)/4) for k > 16
                       signed weights, sign = MSB of k, no sign bits
  anchored (original): w(k) = 1 − 2^(−k/4) for k < 16,  1 + 2^(−(31−k)/4) for k ≥ 16
                       positive weights in [0, 2], dense around 1
  magic:               like centered, magnitude = float_bits(0x3F800000 − (m << 21))
                       (shift + subtract, no table; exact in fp16 and int8)

Both grids are strictly increasing in k.  Effective weight W = scale · v(k)
with a fixed per-layer scale (centered/magic: 2/√in_dim, anchored: 1).

STE update rule: k −= sign(lr · ∂L/∂W), saturating at [0, 31].
One integer increment/decrement per weight per step; on the centered grid a
weight changes sign by walking through code 16.
"""
import numpy as np

_DELTA = 2.0 ** (-np.arange(16) / 4.0)
ANCHORED = np.r_[1.0 - _DELTA, (1.0 + _DELTA)[::-1]].astype(np.float32)
CENTERED = (ANCHORED - 1.0).astype(np.float32)
CENTERED[16] = 0.0
_m = np.arange(16, dtype=np.uint32)
_MAG = (np.uint32(0x3F800000) - (_m << 21)).view(np.float32)
MAGIC = np.r_[-_MAG, 0.0, _MAG[:15][::-1]].astype(np.float32)
GRIDS = {"centered": CENTERED, "anchored": ANCHORED, "magic": MAGIC}


def fiver_to_float(k: np.ndarray, grid: str = "centered") -> np.ndarray:
    """Materialize Fiver codes (uint8, [0..31]) to float32 values of the grid."""
    return GRIDS[grid][np.asarray(k, dtype=np.uint8)]


class FiverLayer:
    """Dense layer with Fiver-quantized weights and float32 biases.

    Gradients flow through materialize() as if it were the identity (STE).
    The discrete update is k −= sign(grad) (saturating); since every grid is
    increasing in k and scale > 0, sign(∂L/∂k) = sign(∂L/∂W).
    """

    def __init__(self, in_dim: int, out_dim: int, act: str = 'identity',
                 grid: str = 'centered', seed: int = 0):
        if grid not in GRIDS:
            raise ValueError(f"grid must be one of {sorted(GRIDS)}, got {grid!r}")
        self.in_dim  = in_dim
        self.out_dim = out_dim
        self.act     = act
        self.grid    = grid
        self.scale   = np.float32(1.0 if grid == 'anchored' else 2.0 / np.sqrt(in_dim))

        rng = np.random.default_rng(seed)
        # Codes 12..20: centered → small weights around 0, anchored → around 1.0
        self.k_data = rng.integers(12, 21, size=(out_dim, in_dim), dtype=np.uint8)
        self.bias   = np.zeros(out_dim, dtype=np.float32)

        self._grad_k    = np.zeros((out_dim, in_dim), dtype=np.float32)
        self._grad_bias = np.zeros(out_dim, dtype=np.float32)

    # ── Forward ──────────────────────────────────────────────────────────

    def materialize(self) -> np.ndarray:
        """Return float32 weight matrix (out_dim × in_dim)."""
        return self.scale * fiver_to_float(self.k_data, self.grid)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (..., in_dim) → (..., out_dim)"""
        out = x @ self.materialize().T + self.bias
        return _apply_act(out, self.act)

    # ── Backward ─────────────────────────────────────────────────────────

    def backward(self, x: np.ndarray, grad_out: np.ndarray) -> np.ndarray:
        """Accumulate gradients via STE. Returns grad_x."""
        W       = self.materialize()
        pre_act = x @ W.T + self.bias
        g       = _act_grad(grad_out, self.act, pre_act)

        x_flat = x.reshape(-1, self.in_dim)
        g_flat = g.reshape(-1, self.out_dim)

        self._grad_k    += g_flat.T @ x_flat         # ∂L/∂W up to the positive scale
        self._grad_bias += g_flat.sum(axis=0)

        return g @ W                                  # grad_x: (..., in_dim)

    # ── Parameter update ─────────────────────────────────────────────────

    def step(self, lr: float):
        """STE discrete update: k −= sign(lr · grad), saturating [0, 31].
        Same rule as cw_fiver_ste_update in compact_weights.cu."""
        delta_k = np.sign(self._grad_k * lr).astype(np.int16)
        self.k_data = np.clip(self.k_data.astype(np.int16) - delta_k, 0, 31).astype(np.uint8)
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
