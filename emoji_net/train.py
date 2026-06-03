"""
train.py — EmojiNet training loop with Fiver-quantized weights.

Called directly:
  python train.py [--backend cpu|cuda|mps] [--epochs N] [--lr F] [--batch-size N]

Called via NetworkController.cs:
  pyTorch_nv(params)   → python train.py --backend cuda   (CUDA)
  pyTorch_i86(params)  → python train.py --backend cpu    (CPU / default numpy)
  pyTorch_risc(params) → python train.py --backend mps    (ARM / FPGA)

The --backend flag is forwarded from C# and logged; the actual numpy computation
is backend-agnostic (CPU only for this implementation). Backend selection is the
hook point for future PyTorch / CUDA acceleration.
"""
import argparse
import sys
import time
import numpy as np
from pathlib import Path

from data_loader import load_openmoji, build_vocabulary, build_label_maps, EmojiDataset
from emoji_net import EmojiNet


def train(args: argparse.Namespace) -> None:
    print(f"[train] backend={args.backend}  epochs={args.epochs}"
          f"  lr={args.lr}  batch_size={args.batch_size}", flush=True)

    # ── Data loading ──────────────────────────────────────────────────────
    records = load_openmoji(args.data_dir)
    if not records:
        print("[train] no records found — check data download", file=sys.stderr)
        sys.exit(1)

    vocab                        = build_vocabulary(records)
    groups, subgroups, gmap, smap = build_label_maps(records)
    n_cat    = min(len(groups),    9)
    max_elem = min(len(subgroups), 64)

    dataset = EmojiDataset(records, vocab, gmap, smap, n_cat, max_elem)
    print(f"[train] {len(dataset)} samples  |  {n_cat} categories"
          f"  |  {max_elem} elements/category", flush=True)

    # ── Model ─────────────────────────────────────────────────────────────
    model = EmojiNet(n_cat=n_cat, max_elem=max_elem)
    print(f"[train] parameters: {model.num_params:,}", flush=True)

    rng     = np.random.default_rng(seed=0)
    indices = np.arange(len(dataset))

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(indices)
        epoch_loss = 0.0
        n_batches  = 0
        t0         = time.time()

        for start in range(0, len(indices) - args.batch_size + 1, args.batch_size):
            batch_idx              = indices[start : start + args.batch_size].tolist()
            imgs, txts, labels     = dataset.batch(batch_idx)

            loss, grad_logits      = model.loss(imgs, txts, labels)
            model.backward(imgs, txts, grad_logits)
            model.step(args.lr)

            epoch_loss += loss
            n_batches  += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed  = time.time() - t0
        print(f"[epoch {epoch:3d}/{args.epochs}]  loss={avg_loss:.5f}"
              f"  time={elapsed:.1f}s  batches={n_batches}", flush=True)

    # ── Save weights ───────────────────────────────────────────────────────
    ckpt_dir = Path(args.data_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for name, layer in (("img",  model.img_layer),
                        ("txt",  model.txt_layer),
                        ("hid",  model.hidden),
                        ("out",  model.output)):
        np.save(ckpt_dir / f"k_{name}.npy",    layer.k_data)
        np.save(ckpt_dir / f"bias_{name}.npy", layer.bias)

    print(f"[train] weights saved to {ckpt_dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EmojiNet")
    parser.add_argument("--backend",    default="cpu",
                        choices=["cpu", "cuda", "mps"],
                        help="compute backend (cpu=numpy, cuda/mps=future)")
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--lr",         type=float, default=0.01)
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--data-dir",   default="data")
    train(parser.parse_args())
