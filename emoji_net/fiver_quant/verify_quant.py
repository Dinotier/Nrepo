"""
verify_quant.py — Validate Fiver quantization quality against original weights.

Compares each .npz produced by quant_llama.py / quant_rwkv.py against the
source model and reports:
  - Per-layer: L2, max_abs_err, mean_rel_err, KL divergence, compression ratio
  - Per-layer: histogram of k_data distribution (shows if Fiver range is well used)
  - Global summary table sorted by worst relative error
  - Perplexity estimate delta (token-level logit shift proxy, no forward pass needed)

No inference required: comparison is purely weight-space.  For true perplexity
delta you need a calibration dataset and a forward pass; that path is noted in
the output but skipped here to keep this tool dependency-free.

Usage:
  python verify_quant.py --manifest ./llama_fiver/manifest.txt \
                         --model meta-llama/Meta-Llama-3.1-8B
  python verify_quant.py --manifest ./rwkv_fiver/manifest.txt  \
                         --model RWKV/rwkv-4-169m --top 5
  python verify_quant.py --npz layer.npz --original-npy layer_orig.npy
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

from fiver_core import fiver_reconstruct, quantization_error, FIVER_VALUES


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_originals(model_path: str) -> dict[str, np.ndarray]:
    """Load all weight tensors from model (same logic as quant_llama._load_model_weights)."""
    import glob as _glob
    p = Path(model_path)

    try:
        from safetensors import safe_open
        if p.is_file() and p.suffix == ".safetensors":
            st_files = [str(p)]
        else:
            st_files = sorted(_glob.glob(str(p / "*.safetensors")))
        if st_files:
            weights: dict[str, np.ndarray] = {}
            for f in st_files:
                with safe_open(f, framework="np", device="cpu") as st:
                    for key in st.keys():
                        weights[key] = st.get_tensor(key)
            return weights
    except (ImportError, Exception):
        pass

    if p.is_file() and p.suffix in {".pth", ".pt"}:
        try:
            import torch
            ckpt = torch.load(str(p), map_location="cpu")
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]
            return {k: v.detach().float().numpy() for k, v in ckpt.items()
                    if hasattr(v, "detach")}
        except Exception:
            pass

    try:
        from transformers import AutoModelForCausalLM
        import torch
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float32, device_map="cpu",
            trust_remote_code=True, low_cpu_mem_usage=True,
        )
        weights = {k: v.detach().numpy() for k, v in model.state_dict().items()}
        del model
        return weights
    except ImportError:
        print("[verify_quant] ERROR: install transformers or safetensors", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"[verify_quant] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


def _load_manifest(manifest_path: Path) -> list[tuple[str, Path]]:
    entries: list[tuple[str, Path]] = []
    base = manifest_path.parent
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) == 2:
                entries.append((parts[0], base / parts[1]))
    return entries


def _k_histogram(k_data: np.ndarray, bins: int = 8) -> str:
    """ASCII histogram of k_data usage across [0..31]."""
    counts, _ = np.histogram(k_data.ravel(), bins=32, range=(0, 32))
    # Condense into `bins` groups
    group_size = 32 // bins
    groups = [counts[i*group_size:(i+1)*group_size].sum() for i in range(bins)]
    max_g  = max(groups) or 1
    bar_w  = 8
    bar    = "".join("█" * int(g / max_g * bar_w) or "▏" for g in groups)
    lo_pct = counts[:16].sum() / max(counts.sum(), 1)
    hi_pct = counts[16:].sum() / max(counts.sum(), 1)
    return f"k-hist[low={lo_pct:.0%}|high={hi_pct:.0%}]: [{bar}]"


def _logit_shift_proxy(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Proxy for perplexity delta without a forward pass.
    Uses mean absolute weight difference as fraction of original weight norm.
    True perplexity delta requires calibration data + full forward pass.
    """
    diff_norm = float(np.abs(original - reconstructed).mean())
    orig_norm = float(np.abs(original).mean()) + 1e-12
    return diff_norm / orig_norm


def _verify_single(name: str, orig: np.ndarray, npz_path: Path) -> dict | None:
    """Load one .npz, reconstruct, compare against original. Returns metrics dict."""
    try:
        d = np.load(str(npz_path), allow_pickle=False)
    except Exception as exc:
        print(f"  [SKIP] {name}: cannot load {npz_path}: {exc}", file=sys.stderr)
        return None

    q = {
        'k_data':    d['k_data'],
        'scale':     float(d['scale'][0]) if d['scale'].ndim > 0 else float(d['scale']),
        'sign_bits': d['sign_bits'] if 'sign_bits' in d else None,
        'shape':     d['k_data'].shape,
        'dtype':     np.float32,
    }

    orig_f32 = orig.astype(np.float32)
    recon    = fiver_reconstruct(q)

    if orig_f32.shape != recon.shape:
        # Shape mismatch — try flattening both to compare
        if orig_f32.size == recon.size:
            recon = recon.reshape(orig_f32.shape)
        else:
            print(f"  [SKIP] {name}: shape mismatch orig={orig_f32.shape} recon={recon.shape}",
                  file=sys.stderr)
            return None

    errs = quantization_error(orig_f32, q)

    return {
        'name':         name,
        'shape':        orig_f32.shape,
        'n':            orig_f32.size,
        'l2':           errs['l2_norm'],
        'max_err':      errs['max_abs_err'],
        'rel_err':      errs['mean_rel_err'],
        'kl':           errs['kl_divergence'],
        'compression':  errs['compression'],
        'logit_proxy':  _logit_shift_proxy(orig_f32, recon),
        'k_hist':       _k_histogram(q['k_data']),
        'orig_mb':      orig_f32.nbytes / 1e6,
        'fiver_mb':     q['k_data'].size * 5 / 8 / 1e6,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def verify(args: argparse.Namespace) -> None:
    t0 = time.time()

    # ── Build work list ───────────────────────────────────────────────────────
    if args.npz and args.original_npy:
        # Single-tensor mode
        orig = np.load(args.original_npy)
        entries = [(Path(args.original_npy).stem, Path(args.npz[0]))]
        originals = {Path(args.original_npy).stem: orig}
    elif args.manifest:
        entries   = _load_manifest(Path(args.manifest))
        if not args.model:
            print("[verify_quant] ERROR: --model required with --manifest", file=sys.stderr)
            sys.exit(1)
        print(f"[verify_quant] loading original weights from {args.model}…", flush=True)
        originals = _load_originals(args.model)
        print(f"[verify_quant] {len(originals)} original tensors loaded, "
              f"{len(entries)} quantized tensors in manifest\n")
    else:
        print("[verify_quant] ERROR: provide (--manifest --model) or (--npz --original-npy)",
              file=sys.stderr)
        sys.exit(1)

    # ── Verify each tensor ────────────────────────────────────────────────────
    all_results: list[dict] = []
    skipped = 0

    for orig_name, npz_path in entries:
        if orig_name not in originals:
            print(f"  [SKIP] {orig_name}: not found in original weights", file=sys.stderr)
            skipped += 1
            continue

        result = _verify_single(orig_name, originals[orig_name], npz_path)
        if result is None:
            skipped += 1
            continue

        all_results.append(result)
        r = result
        print(f"  {r['name'][:50]:50s}  shape={str(r['shape']):20s}")
        print(f"    L2={r['l2']:.5f}  max_err={r['max_err']:.5f}"
              f"  rel_err={r['rel_err']:.4f}  KL={r['kl']:.6f}"
              f"  compression={r['compression']:.1f}×  proxy={r['logit_proxy']:.4f}")
        print(f"    {r['k_hist']}")

    if not all_results:
        print("[verify_quant] no results — nothing to summarize")
        return

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "─" * 90)
    print(f"{'SUMMARY':^90}")
    print("─" * 90)

    # Sort by worst relative error
    sorted_r = sorted(all_results, key=lambda r: r['rel_err'], reverse=True)
    top_n    = args.top if args.top > 0 else len(sorted_r)
    print(f"\nWorst {top_n} layers by mean relative error:\n")
    header = f"{'layer name':50s}  {'rel_err':>8}  {'KL':>10}  {'compress':>8}  {'proxy':>8}"
    print(header)
    print("-" * len(header))
    for r in sorted_r[:top_n]:
        print(f"  {r['name'][:48]:48s}  {r['rel_err']:8.4f}  {r['kl']:10.6f}"
              f"  {r['compression']:8.1f}×  {r['logit_proxy']:8.4f}")

    # Aggregate statistics
    total_orig_mb  = sum(r['orig_mb']  for r in all_results)
    total_fiver_mb = sum(r['fiver_mb'] for r in all_results)
    mean_rel       = np.mean([r['rel_err']      for r in all_results])
    mean_kl        = np.mean([r['kl']           for r in all_results])
    mean_proxy     = np.mean([r['logit_proxy']  for r in all_results])
    worst_rel      = max(r['rel_err'] for r in all_results)
    worst_l2       = max(r['l2']      for r in all_results)

    print(f"\nAggregate ({len(all_results)} tensors, {skipped} skipped):")
    print(f"  original size : {total_orig_mb:.1f} MB")
    print(f"  fiver size    : {total_fiver_mb:.1f} MB")
    print(f"  compression   : {total_orig_mb / max(total_fiver_mb, 1e-6):.2f}×")
    print(f"  mean rel err  : {mean_rel:.4f}")
    print(f"  worst rel err : {worst_rel:.4f}")
    print(f"  worst L2      : {worst_l2:.5f}")
    print(f"  mean KL div   : {mean_kl:.6f}")
    print(f"  mean logit proxy: {mean_proxy:.4f}  (weight-space proxy; ≈0 ideal)")
    print()
    print("NOTE: True perplexity delta requires a calibration corpus + forward pass.")
    print("      Run with --model on the dequantized weights for full PPL evaluation.")
    print(f"\n[verify_quant] done in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Verify Fiver quantization quality")

    src = ap.add_argument_group("input sources (choose one mode)")
    src.add_argument("--manifest",      metavar="MANIFEST_TXT",
                     help="manifest.txt from quant_llama / quant_rwkv")
    src.add_argument("--model",         metavar="MODEL_PATH",
                     help="Original model path/ID (required with --manifest)")
    src.add_argument("--npz",           nargs=1, metavar="FILE.npz",
                     help="Single .npz file (use with --original-npy)")
    src.add_argument("--original-npy",  metavar="FILE.npy",
                     help="Original weight as .npy (use with --npz)")

    ap.add_argument("--top", type=int, default=10,
                    help="Number of worst-error layers to highlight (0=all, default=10)")
    verify(ap.parse_args())
