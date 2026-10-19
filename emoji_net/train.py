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
import signal
import sys
import time
import numpy as np
from pathlib import Path

from data_loader import load_openmoji, build_vocabulary, build_label_maps, EmojiDataset
from emoji_net import EmojiNet


def _save_checkpoint(model: EmojiNet, ckpt_dir: Path, tag: str = "") -> None:
    """Save all Fiver k_data and biases to ckpt_dir."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    for name, layer in (("img", model.img_layer), ("txt", model.txt_layer),
                        ("hid", model.hidden),    ("out", model.output)):
        np.save(ckpt_dir / f"k_{name}{suffix}.npy",    layer.k_data)
        np.save(ckpt_dir / f"bias_{name}{suffix}.npy", layer.bias)
    print(f"[train] checkpoint saved → {ckpt_dir}  (tag={tag or 'final'})",
          flush=True)


def _k_stats(model: EmojiNet) -> str:
    """One-line summary of k_data distribution across all layers."""
    all_k = np.concatenate([
        model.img_layer.k_data.flatten(),
        model.txt_layer.k_data.flatten(),
        model.hidden.k_data.flatten(),
        model.output.k_data.flatten(),
    ])
    sat_lo = int((all_k == 0).sum())
    sat_hi = int((all_k == 31).sum())
    return (f"k_mean={all_k.mean():.1f}  k_std={all_k.std():.1f}"
            f"  sat_lo={sat_lo}  sat_hi={sat_hi}")


def train(args: argparse.Namespace) -> None:
    print(f"[train] backend={args.backend}  epochs={args.epochs}"
          f"  lr={args.lr}  batch_size={args.batch_size}", flush=True)

    # ── Data loading ──────────────────────────────────────────────────────
    records = load_openmoji(args.data_dir)
    if not records:
        print("[train] ERROR: no records — check data download", file=sys.stderr)
        sys.exit(1)

    vocab                         = build_vocabulary(records)
    groups, subgroups, gmap, smap = build_label_maps(records)
    n_cat    = min(len(groups),    9)
    max_elem = min(len(subgroups), 64)

    dataset = EmojiDataset(records, vocab, gmap, smap, n_cat, max_elem)
    print(f"[train] {len(dataset)} samples  |  {n_cat} categories"
          f"  |  {max_elem} elements/category", flush=True)

    # ── Model ─────────────────────────────────────────────────────────────
    model    = EmojiNet(n_cat=n_cat, max_elem=max_elem)
    ckpt_dir = Path(args.data_dir) / "checkpoints"
    print(f"[train] parameters: {model.num_params:,}", flush=True)

    # ── Ctrl+C handler: save checkpoint before exiting ────────────────────
    interrupted = False
    def _on_interrupt(sig, frame):
        nonlocal interrupted
        interrupted = True
        print("\n[train] interrupted — saving checkpoint...", flush=True)
        _save_checkpoint(model, ckpt_dir, tag="interrupted")
        sys.exit(0)
    signal.signal(signal.SIGINT, _on_interrupt)

    rng     = np.random.default_rng(seed=0)
    indices = np.arange(len(dataset))
    t_start = time.time()

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(indices)
        epoch_loss = 0.0
        n_batches  = 0
        n_errors   = 0
        t0         = time.time()

        for start in range(0, len(indices) - args.batch_size + 1, args.batch_size):
            batch_idx = indices[start : start + args.batch_size].tolist()
            try:
                imgs, txts, labels = dataset.batch(batch_idx)
                loss, grad_logits  = model.loss(imgs, txts, labels)

                if not np.isfinite(loss):
                    print(f"[train] WARNING: non-finite loss={loss} at batch "
                          f"{n_batches} epoch {epoch} — skipping update",
                          file=sys.stderr, flush=True)
                    n_errors += 1
                    continue

                model.backward(imgs, txts, grad_logits)
                model.step(args.lr)
                epoch_loss += loss
                n_batches  += 1

            except Exception as exc:
                print(f"[train] ERROR in batch {n_batches}: {exc}",
                      file=sys.stderr, flush=True)
                n_errors += 1
                if n_errors > 10:
                    print("[train] too many errors — aborting", file=sys.stderr)
                    _save_checkpoint(model, ckpt_dir, tag=f"error_epoch{epoch}")
                    sys.exit(1)

        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed  = time.time() - t0
        total_elapsed = time.time() - t_start
        epochs_left   = args.epochs - epoch
        eta_s = (total_elapsed / epoch) * epochs_left if epoch > 0 else 0

        err_str = f"  errors={n_errors}" if n_errors else ""
        print(f"[epoch {epoch:3d}/{args.epochs}]  loss={avg_loss:.5f}"
              f"  time={elapsed:.1f}s  batches={n_batches}"
              f"  ETA={eta_s:.0f}s{err_str}", flush=True)

        # Weight distribution stats every 5 epochs
        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"  [{_k_stats(model)}]", flush=True)

        # Periodic checkpoint every 10 epochs
        if epoch % 10 == 0:
            _save_checkpoint(model, ckpt_dir, tag=f"epoch{epoch}")

    # ── Final save ────────────────────────────────────────────────────────
    _save_checkpoint(model, ckpt_dir)


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
