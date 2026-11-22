"""
quant_rwkv.py — Fiver 5-bit quantization for RWKV time-decay W vectors.

RWKV architecture uses a learned per-channel exponential time decay:
    state_t = exp(-exp(W)) * state_{t-1} + K_t * V_t

W itself is a signed real number (the decay rate exp(-exp(W)) is what lies
in (0, 1)), so it is quantised on the centered grid like any other signed
tensor; --grid auto picks anchored only for positive tensors clustered near
a non-zero value.  These are per-channel vectors: tiny compared with the
weight matrices, so the memory saving here is small.

Target tensors (matched by substring):
  - time_decay      RWKV-4/5: shape (n_head,) or (n_embd,)
  - time_mix_*      channel-mixing gating scalars, shape (1, 1, d_model)
  - att.time_decay  attention time-decay (same math, different path in RWKV-5/6)
  - ffn.time_mix_*  feed-forward mixing scalars

Usage:
  python quant_rwkv.py --model RWKV/rwkv-4-169m --dry-run
  python quant_rwkv.py --model ./rwkv_ckpt.pth   --out ./rwkv_fiver/
  python quant_rwkv.py --model ./rwkv5.safetensors --layers decay,mix --out ./out/
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

from fiver_core import bits_per_weight, float_to_fiver, memory_summary, quantization_error, save_npz

# Layer substrings for each logical group
DECAY_KEYS = ["time_decay", "att.time_decay", "time_decay_w1", "time_decay_w2"]
MIX_KEYS   = ["time_mix_k", "time_mix_v", "time_mix_r", "time_mix_g", "time_mix_w",
               "ffn.time_mix", "att.time_mix"]
BONUS_KEYS = ["time_first", "time_faaaa"]   # RWKV-5/6 bonus / shift terms


class _ShapeOnly:
    """Stand-in for a tensor in --dry-run: carries only the shape."""
    def __init__(self, shape):
        self.shape = tuple(shape)


def _load_rwkv_weights(model_path: str, dry_run: bool) -> dict[str, np.ndarray]:
    """
    Load RWKV checkpoint weights.

    Supports:
      1. Single .safetensors file
      2. Directory of .safetensors shards
      3. Single .pth / .pt PyTorch checkpoint
      4. HuggingFace hub model ID (uses transformers AutoModel)
    """
    print(f"[quant_rwkv] loading weights from: {model_path}", flush=True)
    p = Path(model_path)

    # ── safetensors ──────────────────────────────────────────────────────────
    try:
        from safetensors import safe_open
        import glob as _glob

        if p.is_file() and p.suffix == ".safetensors":
            st_files = [str(p)]
        else:
            st_files = sorted(_glob.glob(str(p / "*.safetensors")))

        if st_files:
            weights: dict[str, np.ndarray] = {}
            for f in st_files:
                with safe_open(f, framework="np", device="cpu") as st:
                    for key in st.keys():
                        weights[key] = (_ShapeOnly(st.get_slice(key).get_shape()) if dry_run
                                        else st.get_tensor(key))
            print(f"[quant_rwkv] loaded {len(weights)} tensors via safetensors", flush=True)
            return weights
    except ImportError:
        pass
    except Exception as exc:
        print(f"[quant_rwkv] safetensors attempt failed ({exc}), trying next loader…",
              flush=True)

    # ── PyTorch .pth/.pt ─────────────────────────────────────────────────────
    if p.is_file() and p.suffix in {".pth", ".pt"}:
        try:
            import torch
            ckpt = torch.load(str(p), map_location="cpu")
            # RWKV checkpoints may wrap state dict under a key
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]
            elif isinstance(ckpt, dict) and "model" in ckpt:
                ckpt = ckpt["model"]
            weights = {}
            for k, v in ckpt.items():
                if hasattr(v, "detach"):
                    weights[k] = (_ShapeOnly(v.shape) if dry_run
                                  else v.detach().float().numpy())
            print(f"[quant_rwkv] loaded {len(weights)} tensors via torch.load", flush=True)
            return weights
        except ImportError:
            print("[quant_rwkv] torch not available for .pth loading", file=sys.stderr)
        except Exception as exc:
            print(f"[quant_rwkv] torch.load failed: {exc}", file=sys.stderr)

    # ── HuggingFace transformers (hub or local dir) ───────────────────────────
    try:
        from transformers import AutoModelForCausalLM
        import torch
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,   # RWKV requires this on HF hub
            low_cpu_mem_usage=True,
        )
        weights = {k: v.detach().numpy() for k, v in model.state_dict().items()}
        print(f"[quant_rwkv] loaded {len(weights)} tensors via transformers", flush=True)
        del model
        return weights
    except ImportError:
        print("[quant_rwkv] ERROR: install transformers or safetensors:\n"
              "  pip install transformers safetensors", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"[quant_rwkv] ERROR loading model: {exc}", file=sys.stderr)
        sys.exit(1)


def _select_layers(weights: dict, layer_spec: str) -> list[str]:
    """Return weight keys matching comma-separated spec: decay | mix | bonus | all."""
    specs = [s.strip().lower() for s in layer_spec.split(",")]
    selected: list[str] = []
    for key in weights:
        key_l = key.lower()
        for spec in specs:
            if spec == "decay"  and any(k in key_l for k in DECAY_KEYS): selected.append(key); break
            if spec == "mix"    and any(k in key_l for k in MIX_KEYS):   selected.append(key); break
            if spec == "bonus"  and any(k in key_l for k in BONUS_KEYS): selected.append(key); break
            if spec == "all":                                              selected.append(key); break
    return selected


def _rwkv_decay_analysis(tensor: np.ndarray) -> dict:
    """
    Extra statistics specific to RWKV time-decay vectors.
    W values are stored as log(-log(decay_rate)) so exp(-exp(W)) ∈ (0,1).
    """
    flat = tensor.ravel().astype(np.float64)
    decay_rates = np.exp(-np.exp(flat))   # the actual per-step decay in [0,1)
    return {
        'w_min':          float(flat.min()),
        'w_max':          float(flat.max()),
        'decay_min':      float(decay_rates.min()),
        'decay_max':      float(decay_rates.max()),
        'decay_mean':     float(decay_rates.mean()),
        'near_zero_frac': float((np.abs(flat) < 0.1).mean()),
    }


def quantize_rwkv(args: argparse.Namespace) -> None:
    t0 = time.time()
    print(f"[quant_rwkv] model={args.model}  layers={args.layers}"
          f"  dry_run={args.dry_run}  out={args.out}", flush=True)

    weights  = _load_rwkv_weights(args.model, args.dry_run)
    selected = _select_layers(weights, args.layers)

    if not selected:
        print(f"[quant_rwkv] WARNING: no layers matched spec '{args.layers}'",
              file=sys.stderr)
        print("[quant_rwkv] available keys (first 30):")
        for k in list(weights.keys())[:30]:
            print(f"  {k}")
        return

    print(f"\n[quant_rwkv] quantizing {len(selected)} tensors\n")
    total_orig_mb  = 0.0
    total_fiver_mb = 0.0
    results: dict = {}

    for key in selected:
        shape = tuple(weights[key].shape)
        n     = int(np.prod(shape))
        total_orig_mb  += n * 4 / 1e6
        total_fiver_mb += n * bits_per_weight(shape) / 8 / 1e6

        if args.dry_run:
            print(memory_summary(key, shape, "auto"))
            continue

        tensor = np.asarray(weights[key], dtype=np.float32)
        q      = float_to_fiver(tensor, grid=args.grid)
        errs   = quantization_error(tensor, q)
        print(memory_summary(key, shape, q["grid"]))
        print(f"    rel_rmse={errs['rel_rmse']:.4f}  SQNR={errs['sqnr_db']:.1f} dB"
              f"  max_err={errs['max_abs_err']:.5f}")

        if any(k in key.lower() for k in DECAY_KEYS) and tensor.size >= 4:
            da = _rwkv_decay_analysis(tensor)
            print(f"    decay_rate ∈ [{da['decay_min']:.4f}, {da['decay_max']:.4f}]"
                  f"  mean={da['decay_mean']:.4f}  near_zero={da['near_zero_frac']:.2%}")
        results[key] = q

    print(f"\n[quant_rwkv] total: {total_orig_mb:.3f} MB → {total_fiver_mb:.3f} MB"
          f"  ({total_orig_mb / max(total_fiver_mb, 1e-6):.1f}× compression)")

    if args.dry_run or not results:
        print("[quant_rwkv] dry run complete — no files written")
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, q in results.items():
        save_npz(out_dir / f"{key.replace('/', '__').replace('.', '_')}.npz", q)

    manifest = out_dir / "manifest.txt"
    with open(manifest, "w") as f:
        f.write("# Fiver-quantized RWKV weights\n")
        f.write(f"# source: {args.model}\n")
        f.write(f"# layers: {args.layers}\n\n")
        for key in results:
            safe = key.replace("/", "__").replace(".", "_")
            f.write(f"{key}\t{safe}.npz\n")

    print(f"[quant_rwkv] saved {len(results)} tensors to {out_dir}")
    print(f"[quant_rwkv] total time: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fiver quantization for RWKV time-decay / mix layers")
    ap.add_argument("--model",   required=True,
                    help="HuggingFace model ID, local dir, or .pth/.safetensors file")
    ap.add_argument("--layers",  default="decay,mix",
                    help="Comma-separated: decay, mix, bonus, all  (default: decay,mix)")
    ap.add_argument("--out",     default="./rwkv_fiver",
                    help="Output directory for quantized .npz files")
    ap.add_argument("--grid",    default="auto", choices=["auto", "centered", "anchored", "magic"],
                    help="value grid (auto: anchored for positive near-1 tensors, else centered)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print memory savings without loading full weights")
    quantize_rwkv(ap.parse_args())
