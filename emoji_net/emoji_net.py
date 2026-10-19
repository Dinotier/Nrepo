"""
emoji_net.py — Dual-input emoji categorization network.

Input A : 100×100 emoji image  (grayscale, normalized 0..1)  → 10 000 floats
Input B : 100×100 BoW matrix   (100-word vocab × 100 features) → 10 000 floats
Output  : (n_categories × max_elements) multi-label logits

Architecture:
  imgStream : FiverLayer(10000 → 512, softsign)  + LayerNorm
  txtStream : FiverLayer(10000 → 512, softsign)  + LayerNorm
  concat    : 1024
  hidden    : FiverLayer(1024  → 256, softsign)  + LayerNorm
  output    : FiverLayer(256   → n_cat×max_elem)
  reshape   : (n_cat, max_elem) → multi-label BCE

All Fiver-quantized weights. No nn.*, no torch.*, pure numpy.
"""
import numpy as np
from fiver import FiverLayer
from transformer import LayerNorm, binary_cross_entropy


class EmojiNet:

    def __init__(self, n_cat: int = 9, max_elem: int = 64):
        self.n_cat    = n_cat
        self.max_elem = max_elem
        out_dim       = n_cat * max_elem

        self.img_layer = FiverLayer(10_000, 512,     act='softsign', seed=1)
        self.txt_layer = FiverLayer(10_000, 512,     act='softsign', seed=2)
        self.hidden    = FiverLayer(1_024,  256,     act='softsign', seed=3)
        self.output    = FiverLayer(256,    out_dim, act='identity', seed=4)

        self.ln_img = LayerNorm(512)
        self.ln_txt = LayerNorm(512)
        self.ln_hid = LayerNorm(256)

    # ── Forward ──────────────────────────────────────────────────────────

    def forward(self, img: np.ndarray, txt: np.ndarray) -> np.ndarray:
        """
        img : (batch, 10000)
        txt : (batch, 10000)
        returns logits : (batch, n_cat, max_elem)
        """
        h_img  = self.ln_img.forward(self.img_layer.forward(img))   # (B, 512)
        h_txt  = self.ln_txt.forward(self.txt_layer.forward(txt))   # (B, 512)
        h      = np.concatenate([h_img, h_txt], axis=-1)             # (B, 1024)
        h      = self.ln_hid.forward(self.hidden.forward(h))         # (B, 256)
        logits = self.output.forward(h)                               # (B, out_dim)
        return logits.reshape(-1, self.n_cat, self.max_elem)

    # ── Loss ─────────────────────────────────────────────────────────────

    def loss(self, img: np.ndarray, txt: np.ndarray,
             labels: np.ndarray) -> tuple[float, np.ndarray]:
        """
        labels : (batch, n_cat, max_elem) binary float
        Returns (loss_scalar, grad_logits same shape as labels)
        """
        logits = self.forward(img, txt)
        B      = logits.shape[0]
        loss_v, grad = binary_cross_entropy(
            logits.reshape(B, -1),
            labels.reshape(B, -1),
        )
        return loss_v, grad.reshape(B, self.n_cat, self.max_elem)

    # ── Backward ─────────────────────────────────────────────────────────

    def backward(self, img: np.ndarray, txt: np.ndarray,
                 grad_logits: np.ndarray) -> None:
        """Backpropagate grad_logits through all layers, accumulate Fiver grads."""
        B   = img.shape[0]
        g   = grad_logits.reshape(B, -1)                    # (B, out_dim)

        # Recompute activations needed for backward passes
        h_img = self.ln_img.forward(self.img_layer.forward(img))
        h_txt = self.ln_txt.forward(self.txt_layer.forward(txt))
        h_cat = np.concatenate([h_img, h_txt], axis=-1)
        h_hid = self.ln_hid.forward(self.hidden.forward(h_cat))

        # Output → hidden
        g = self.output.backward(h_hid, g)
        g = self.ln_hid.backward(g)
        g = self.hidden.backward(h_cat, g)

        # Split gradient into image and text streams
        g_img = self.ln_img.backward(g[:, :512])
        g_txt = self.ln_txt.backward(g[:, 512:])
        self.img_layer.backward(img, g_img)
        self.txt_layer.backward(txt, g_txt)

    # ── Update ───────────────────────────────────────────────────────────

    def step(self, lr: float) -> None:
        for layer in (self.img_layer, self.txt_layer, self.hidden, self.output):
            layer.step(lr)
        for ln in (self.ln_img, self.ln_txt, self.ln_hid):
            ln.step(lr)

    @property
    def num_params(self) -> int:
        return sum(l.num_params for l in
                   (self.img_layer, self.txt_layer, self.hidden, self.output))
