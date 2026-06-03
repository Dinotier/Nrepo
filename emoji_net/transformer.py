"""
transformer.py — LLM mathematical operations, all custom (no nn.* or torch.*).

All weight matrices use FiverLayer (quantized).
All activations and intermediate computations use float32 numpy arrays.

Implements:
  scaled_dot_attention(Q, K, V, mask)
  multi_head_attention(x, W_Q, W_K, W_V, W_O, num_heads, mask)
  LayerNorm
  FFN (two FiverLayers with GELU expand + identity contract)
  softmax(x, axis)
  binary_cross_entropy(logits, targets)
"""
import numpy as np
from fiver import FiverLayer


# ═══════════════════════════════════════════════════════════════════════════
#  SCALED DOT-PRODUCT ATTENTION
# ═══════════════════════════════════════════════════════════════════════════

def scaled_dot_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Scaled dot-product attention.

    Q : (batch, heads, seq_q, d_k)   — query vectors
    K : (batch, heads, seq_k, d_k)   — key vectors
    V : (batch, heads, seq_k, d_v)   — value vectors
    mask : (batch, 1, seq_q, seq_k) bool, True = position to mask (→ -inf)

    Returns
    -------
    context      : (batch, heads, seq_q, d_v)
    attn_weights : (batch, heads, seq_q, seq_k)  — probability distribution

    Note: Q, K, V are full-precision float32 activations produced by the Fiver
    projection layers W_Q, W_K, W_V.  The attention matmuls themselves carry no
    Fiver weights — only W_Q/K/V/O (the projection matrices) are quantized.
    """
    d_k = Q.shape[-1]

    # ── 1. Score matrix: QKᵀ / √d_k ──────────────────────────────────────
    #    Shape: (batch, heads, seq_q, seq_k)
    scores = np.matmul(Q, K.swapaxes(-2, -1)) / np.sqrt(float(d_k))

    # ── 2. Optional causal / padding mask ────────────────────────────────
    #    Masked positions receive -1e9 before softmax → probability ≈ 0.
    if mask is not None:
        scores = np.where(mask, -1e9, scores)

    # ── 3. Numerically stable softmax over key dimension ─────────────────
    #    Subtract max before exp to prevent overflow (log-sum-exp trick).
    scores_stable   = scores - scores.max(axis=-1, keepdims=True)
    exp_scores      = np.exp(scores_stable)
    attn_weights    = exp_scores / (exp_scores.sum(axis=-1, keepdims=True) + 1e-9)

    # ── 4. Context vector: weighted sum over values ───────────────────────
    #    Shape: (batch, heads, seq_q, d_v)
    context = np.matmul(attn_weights, V)

    return context, attn_weights


# ═══════════════════════════════════════════════════════════════════════════
#  MULTI-HEAD ATTENTION
# ═══════════════════════════════════════════════════════════════════════════

def multi_head_attention(
    x: np.ndarray,
    W_Q: FiverLayer,
    W_K: FiverLayer,
    W_V: FiverLayer,
    W_O: FiverLayer,
    num_heads: int,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Multi-head self-attention with Fiver-quantized projection weights.

    x         : (batch, seq, d_model)
    W_Q/K/V/O : FiverLayer objects, each (d_model → d_model)
    num_heads  : number of attention heads (d_model must be divisible)

    Forward pass:
      1. Project x with Fiver weights W_Q, W_K, W_V → Q, K, V  (float)
      2. Reshape to (batch, heads, seq, d_head)
      3. scaled_dot_attention (pure float, no Fiver)
      4. Concatenate heads, project with Fiver W_O
    """
    B, T, d = x.shape
    d_h     = d // num_heads
    x_flat  = x.reshape(B * T, d)

    # Fiver-quantized projections → float activations
    Q = W_Q.forward(x_flat).reshape(B, T, num_heads, d_h).transpose(0, 2, 1, 3)
    K = W_K.forward(x_flat).reshape(B, T, num_heads, d_h).transpose(0, 2, 1, 3)
    V = W_V.forward(x_flat).reshape(B, T, num_heads, d_h).transpose(0, 2, 1, 3)

    # Pure-float scaled dot-product attention
    ctx, _  = scaled_dot_attention(Q, K, V, mask)

    # Merge heads and project with Fiver W_O
    ctx_cat = ctx.transpose(0, 2, 1, 3).reshape(B * T, d)
    return W_O.forward(ctx_cat).reshape(B, T, d)


# ═══════════════════════════════════════════════════════════════════════════
#  LAYER NORMALIZATION
# ═══════════════════════════════════════════════════════════════════════════

class LayerNorm:
    """Layer normalization with full-precision learnable γ (scale) and β (shift).

    Parameters γ and β are kept in float32 — they are few, stable, and
    critical for training convergence.  Not Fiver-quantized.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        self.dim   = dim
        self.eps   = eps
        self.gamma = np.ones(dim,  dtype=np.float32)
        self.beta  = np.zeros(dim, dtype=np.float32)
        self._dgamma = np.zeros_like(self.gamma)
        self._dbeta  = np.zeros_like(self.beta)
        self._cache: dict = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (..., dim) → normalized (..., dim)"""
        mu    = x.mean(axis=-1, keepdims=True)
        var   = x.var(axis=-1,  keepdims=True)
        x_hat = (x - mu) / np.sqrt(var + self.eps)
        self._cache = {'x_hat': x_hat, 'var': var}
        return self.gamma * x_hat + self.beta

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        x_hat = self._cache['x_hat']
        N     = x_hat.shape[-1]
        axes  = tuple(range(grad_out.ndim - 1))
        self._dgamma += (grad_out * x_hat).sum(axis=axes)
        self._dbeta  += grad_out.sum(axis=axes)
        dx_hat = grad_out * self.gamma
        std    = np.sqrt(self._cache['var'] + self.eps)
        return (dx_hat
                - dx_hat.mean(axis=-1, keepdims=True)
                - x_hat * (dx_hat * x_hat).mean(axis=-1, keepdims=True)) / std

    def step(self, lr: float):
        self.gamma -= lr * self._dgamma
        self.beta  -= lr * self._dbeta
        self._dgamma[:] = 0.0
        self._dbeta[:]  = 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  FEED-FORWARD NETWORK (FFN / MLP)
# ═══════════════════════════════════════════════════════════════════════════

class FFN:
    """Two-layer FFN: d_model → 4·d_model (GELU) → d_model.

    Both weight matrices are Fiver-quantized FiverLayers.
    Bias vectors are full-precision float32.
    """

    def __init__(self, d_model: int):
        self.W1 = FiverLayer(d_model,     4 * d_model, act='gelu')
        self.W2 = FiverLayer(4 * d_model, d_model,     act='identity')

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.W2.forward(self.W1.forward(x))

    def backward(self, x: np.ndarray, grad_out: np.ndarray) -> np.ndarray:
        h      = self.W1.forward(x)
        grad_h = self.W2.backward(h, grad_out)
        return self.W1.backward(x, grad_h)

    def step(self, lr: float):
        self.W1.step(lr)
        self.W2.step(lr)

    @property
    def num_params(self) -> int:
        return self.W1.num_params + self.W2.num_params


# ═══════════════════════════════════════════════════════════════════════════
#  LOSS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / (e.sum(axis=axis, keepdims=True) + 1e-9)


def binary_cross_entropy(
    logits: np.ndarray,
    targets: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Binary cross-entropy for multi-label classification.

    logits  : (batch, n_classes) — raw scores before sigmoid
    targets : (batch, n_classes) — binary float labels in {0, 1}

    Returns (loss_scalar, grad_logits).
    """
    p    = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    eps  = 1e-7
    loss = -(targets * np.log(p + eps) + (1.0 - targets) * np.log(1.0 - p + eps))
    grad = ((p - targets) / max(logits.shape[0], 1)).astype(np.float32)
    return float(loss.mean()), grad
