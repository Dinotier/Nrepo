"""
quant_llama.py — Fiver 5-bit quantization for LLaMA / Mistral embedding and RMSNorm layers.

Target layers (unsigned / near-1.0, highest memory impact):
  1. model.embed_tokens.weight    shape: (vocab_size, d_model)  e.g. 128K×4096 = 524M params
  2. model.norm.weight            RMSNorm gain γ, shape: (d_model,)   naturally near 1.0
  3. model.layers.*.input_layernorm.weight    per-layer RMSNorm
  4. model.layers.*.post_attention_layernorm.weight

Usage:
  python quant_llama.py --model meta-llama/Meta-Llama-3.1-8B --dry-run
  python quant_llama.py --model mistralai/Mistral-Nemo-Instruct-2407 --out ./llama_fiver/
  python quant_llama.py --model ./local_llama_dir --layers embed,norm --out ./out/
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

from fiver_core import float_to_fiver, fiver_reconstruct, quantization_error, memory_summary

# Names of target layer types (matched by substring)
EMBED_KEYS   = ["embed_tokens.weight", "lm_head.weight"]
NORM_KEYS    = ["layernorm.weight", "layer_norm.weight", "norm.weight",
                "input_layernorm.weight", "post_attention_layernorm.weight",
                "final_layernorm.weight"]
PROJ_KEYS    = ["o_proj.weight", "down_proj.weight"]  # output projections (post-activation)


def _load_model_weights(model_path: str, dry_run: bool) -> dict[str, np.ndarray]:
    """
    Load model weights from a HuggingFace model (local or hub).
    Returns dict: tensor_name → numpy float32 array.
    Falls back to safetensors → PyTorch → error.
    """
    print(f"[quant_llama] loading weights from: {model_path}", flush=True)

    # Try safetensors first (faster, no pickling)
    try:
        from safetensors import safe_open
        import os, glob
        pattern = str(Path(model_path) / "*.safetensors")
        files   = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"no .safetensors files in {model_path}")
        weights = {}
        for f in files:
            with safe_open(f, framework="np", device="cpu") as st:
                for key in st.keys():
                    if dry_run:
                        # Only load shapes, not data
                        weights[key] = st.get_slice(key)[:0]  # shape probe
                    else:
                        weights[key] = st.get_tensor(key)
        print(f"[quant_llama] loaded {len(weights)} tensors via safetensors", flush=True)
        return weights
    except (ImportError, FileNotFoundError) as e:
        print(f"[quant_llama] safetensors unavailable ({e}), trying transformers...",
              flush=True)

    # Try HuggingFace transformers
    try:
        from transformers import AutoModelForCausalLM
        import torch
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        weights = {k: v.detach().numpy() for k, v in model.state_dict().items()}
        print(f"[quant_llama] loaded {len(weights)} tensors via transformers", flush=True)
        del model
        return weights
    except ImportError:
        print("[quant_llama] ERROR: install transformers or safetensors:\n"
              "  pip install transformers safetensors", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"[quant_llama] ERROR loading model: {exc}", file=sys.stderr)
        sys.exit(1)


def _select_layers(weights: dict, layer_spec: str) -> list[str]:
    """Return list of weight keys matching --layers specification."""
    specs = [s.strip().lower() for s in layer_spec.split(",")]
    selected = []
    for key in weights:
        key_l = key.lower()
        for spec in specs:
            if spec == "embed"  and any(k in key_l for k in EMBED_KEYS):  selected.append(key); break
            if spec == "norm"   and any(k in key_l for k in NORM_KEYS):   selected.append(key); break
            if spec == "proj"   and any(k in key_l for k in PROJ_KEYS):   selected.append(key); break
            if spec == "all":                                               selected.append(key); break
    return selected


def quantize_llama(args: argparse.Namespace) -> None:
    t0 = time.time()
    print(f"[quant_llama] model={args.model}  layers={args.layers}"
          f"  dry_run={args.dry_run}  out={args.out}", flush=True)

    weights  = _load_model_weights(args.model, args.dry_run)
    selected = _select_layers(weights, args.layers)

    if not selected:
        print(f"[quant_llama] WARNING: no layers matched spec '{args.layers}'",
              file=sys.stderr)
        print("[quant_llama] available keys (first 20):")
        for k in list(weights.keys())[:20]:
            print(f"  {k}")
        return

    print(f"\n[quant_llama] quantizing {len(selected)} tensors\n")
    total_orig_mb  = 0.0
    total_fiver_mb = 0.0
    results = {}

    for key in selected:
        tensor = weights[key].astype(np.float32)

        if args.dry_run:
            n = int(np.prod(tensor.shape)) if tensor.size > 0 else 0
            fp32_mb  = n * 4 / 1e6
            fiver_mb = n * 5 / 8 / 1e6
            print(f"  [DRY] {key:50s}  shape={str(tensor.shape):20s}"
                  f"  FP32={fp32_mb:.1f}MB → Fiver={fiver_mb:.1f}MB"
                  f"  saving={fp32_mb - fiver_mb:.1f}MB")
            total_orig_mb  += fp32_mb
            total_fiver_mb += fiver_mb
            continue

        # Detect layer type — unsigned layers skip sign bits for efficiency
        key_l      = key.lower()
        is_norm    = any(k in key_l for k in NORM_KEYS)
        is_embed   = any(k in key_l for k in EMBED_KEYS)
        use_sign   = not is_norm   # RMSNorm γ is always positive, skip sign

        q    = float_to_fiver(tensor, use_sign=use_sign)
        errs = quantization_error(tensor, q)

        print(memory_summary(key, tensor.shape, tensor.dtype, float(q['scale'])))
        print(f"    L2={errs['l2_norm']:.5f}  max_err={errs['max_abs_err']:.5f}"
              f"  rel_err={errs['mean_rel_err']:.4f}  KL={errs['kl_divergence']:.6f}"
              f"  compression={errs['compression']:.1f}×")

        total_orig_mb  += tensor.nbytes / 1e6
        total_fiver_mb += q['k_data'].size * 5 / 8 / 1e6
        results[key]    = q

    print(f"\n[quant_llama] total: {total_orig_mb:.1f} MB → {total_fiver_mb:.1f} MB"
          f"  ({total_orig_mb / max(total_fiver_mb, 0.001):.1f}× compression)")

    if args.dry_run or not results:
        print("[quant_llama] dry run complete — no files written")
        return

    # Save
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, q in results.items():
        safe_name = key.replace("/", "__").replace(".", "_")
        save_path = out_dir / f"{safe_name}.npz"
        save_dict = {
            'k_data': q['k_data'],
            'scale':  np.array([q['scale']], dtype=np.float32),
        }
        if q['sign_bits'] is not None:
            save_dict['sign_bits'] = q['sign_bits']
        np.savez_compressed(save_path, **save_dict)

    manifest = out_dir / "manifest.txt"
    with open(manifest, "w") as f:
        f.write("# Fiver-quantized LLaMA weights\n")
        f.write(f"# source: {args.model}\n")
        f.write(f"# layers: {args.layers}\n\n")
        for key in results:
            safe = key.replace("/", "__").replace(".", "_")
            f.write(f"{key}\t{safe}.npz\n")

    print(f"[quant_llama] saved {len(results)} tensors to {out_dir}")
    print(f"[quant_llama] total time: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fiver quantization for LLaMA/Mistral")
    ap.add_argument("--model",   required=True,
                    help="HuggingFace model ID or local path")
    ap.add_argument("--layers",  default="embed,norm",
                    help="Comma-separated: embed, norm, proj, all  (default: embed,norm)")
    ap.add_argument("--out",     default="./llama_fiver",
                    help="Output directory for quantized .npz files")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print memory savings without loading full weights")
    quantize_llama(ap.parse_args())
