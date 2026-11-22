"""
analyze_weights.py — How well do real weight tensors fit the Fiver grids?

For every selected tensor:
  1. Distribution: mean/σ, share of negatives, kurtosis (Gauss = 3,
     Laplace = 6), share within σ/2, largest outlier in σ.
  2. Quantisation error (relative RMS ‖x − q‖ / ‖x‖), always with one
     MSE-optimal scale per row, for
        Fiver centered  5 bit   v = ∓2^(−m/4), code 16 = 0
        Fiver anchored  5 bit   w = 1 ∓ 2^(−m/4)  (only for tensors with mean > σ)
        Fiver magic     5 bit   |v| = float_bits(0x3F800000 − (m << 21))
        uniform int5    5 bit   −15..15
     and with --extended also the 6-bit variants (Fiver ±2^(−m/4), m < 32; int6).
  3. Which grid --grid auto would pick, and how the codes of that grid are used.

Measured on SmolLM-135M (LLaMA architecture), per-row scale:
  projection matrices  ≈ Gauss (kurtosis 4–5), centered Fiver 6.1–6.9 %, int5 6.2–7.0 %
  shared sign per triplet (old format) destroys them: 100 % error
  RMSNorm γ (positive, mean/σ 3–5): anchored grid

Usage:
  python analyze_weights.py --model ./SmolLM-135M --layers proj --top 8
  python analyze_weights.py --model meta-llama/Meta-Llama-3.1-8B --layers embed,norm,proj
  python analyze_weights.py --npy ./my_tensor.npy --extended
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from fiver_core import candidate_grids, float_to_fiver, fiver_reconstruct

LAYER_GROUPS = {
    "embed": ["embed_tokens.weight", "lm_head.weight"],
    "norm":  ["layernorm.weight", "layer_norm.weight", "norm.weight"],
    "proj":  ["q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
              "gate_proj.weight", "up_proj.weight", "down_proj.weight", "out_proj.weight"],
    "decay": ["time_decay"],
    "mix":   ["time_mix"],
}

_D32 = 2.0 ** (-np.arange(32) / 4.0)
EXTRA_GRIDS = {
    "uniform int5":      (np.arange(-15.0, 16.0), 5),
    "Fiver ±2^-m/4 6b":  (np.r_[-_D32, _D32[::-1]], 6),
    "uniform int6":      (np.arange(-31.0, 32.0), 6),
}


def _load_weights(model_path: str) -> dict:
    import glob as _g
    p = Path(model_path)
    try:
        from safetensors.numpy import load_file
        files = ([str(p)] if p.suffix == ".safetensors" else sorted(_g.glob(str(p / "*.safetensors"))))
        if files:
            w = {}
            for f in files:
                w.update(load_file(f))
            return w
    except ImportError:
        pass
    if p.is_file() and p.suffix in {".pth", ".pt"}:
        import torch
        ckpt = torch.load(str(p), map_location="cpu")
        ckpt = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        return {k: v.detach().float().numpy() for k, v in ckpt.items() if hasattr(v, "detach")}
    try:
        from transformers import AutoModelForCausalLM
        import torch
        m = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32,
                                                 device_map="cpu", low_cpu_mem_usage=True)
        return {k: v.detach().numpy() for k, v in m.state_dict().items()}
    except ImportError:
        print("[analyze] ERROR: install safetensors (or transformers + torch)", file=sys.stderr)
        sys.exit(1)


def _select(weights: dict, spec: str) -> list[str]:
    specs = [s.strip().lower() for s in spec.split(",")]
    return [k for k in weights
            if "all" in specs or any(g in k.lower() for s in specs for g in LAYER_GROUPS.get(s, []))]


def _rows(x: np.ndarray) -> np.ndarray:
    return x.reshape(1, -1) if x.ndim <= 1 else x.reshape(-1, x.shape[-1])


def _rel(x: np.ndarray, q: np.ndarray) -> float:
    return float(np.sqrt(((x - q) ** 2).sum() / max((x ** 2).sum(), 1e-30)))


def _best_row_quant(X: np.ndarray, V: np.ndarray, n_scales: int = 48) -> np.ndarray:
    """Nearest-value quantisation onto the signed grid V with one MSE-optimal scale per row."""
    mid  = (V[1:] + V[:-1]) / 2
    base = np.abs(X).max(1, keepdims=True) / np.abs(V).max()
    best_e, best_q = np.full(X.shape[0], np.inf), np.zeros_like(X)
    for t in np.geomspace(0.02, 1.0, n_scales):
        s = base * t
        q = V[np.searchsorted(mid, np.divide(X, s, out=np.zeros_like(X), where=s > 0))] * s
        e = ((X - q) ** 2).sum(1)
        b = e < best_e
        best_e[b], best_q[b] = e[b], q[b]
    return best_q


def analyze_tensor(name: str, x: np.ndarray, extended: bool) -> dict:
    X = _rows(np.asarray(x, dtype=np.float64))
    f = X.ravel()
    m, s = f.mean(), f.std() + 1e-30
    res = {
        "name": name, "n": f.size,
        "mean/std": m / s, "neg": float((f < 0).mean()),
        "kurt": float((((f - m) / s) ** 4).mean()),
        "within": float((np.abs(f - m) < s / 2).mean()),
        "max/std": float(np.abs(f - m).max() / s),
        "err": {},
    }
    best = None
    for grid in candidate_grids(X):
        q = float_to_fiver(X, grid=grid, quiet=True)
        e = res["err"][f"Fiver {grid}"] = _rel(X, fiver_reconstruct(q))
        if best is None or e < best[0]:
            best = (e, grid, q)
    q = float_to_fiver(X, grid="magic", quiet=True)
    res["err"]["Fiver magic"] = _rel(X, fiver_reconstruct(q))
    res["auto"] = best[1]
    res["k_usage"] = np.bincount(best[2]["k_data"].ravel(), minlength=32) / f.size
    for label, (V, bits) in EXTRA_GRIDS.items():
        if bits == 5 or extended:
            res["err"][label] = _rel(X, _best_row_quant(X, V))
    return res


def _print(r: dict) -> None:
    print("─" * 92)
    print(f"  {r['name']}   n={r['n']:,}")
    print(f"  mean/σ={r['mean/std']:+.3f}  negative={r['neg']:.0%}  kurtosis={r['kurt']:.1f}"
          f" (Gauss 3)  |x−μ|<σ/2: {r['within']:.0%} (Gauss 38%)  max={r['max/std']:.1f}σ")
    best = min(r["err"], key=r["err"].get)
    for label, e in r["err"].items():
        mark = "  ← best" if label == best else ""
        print(f"    {label:20s} rel err {e:7.4f}{mark}")
    u = r.get("k_usage")
    if u is not None:
        print(f"  --grid auto → {r['auto']}; code usage: k0-3 {u[:4].sum():.0%}  k4-11 {u[4:12].sum():.0%}"
              f"  k12-19 {u[12:20].sum():.0%}  k20-27 {u[20:28].sum():.0%}  k28-31 {u[28:].sum():.0%}")


def main(args: argparse.Namespace) -> None:
    if args.npy:
        weights = {Path(args.npy).stem: np.load(args.npy)}
        keys = list(weights)
    else:
        print(f"[analyze] loading {args.model} …", flush=True)
        weights = _load_weights(args.model)
        keys = _select(weights, args.layers)
        if not keys:
            print(f"[analyze] no tensors matched '{args.layers}'. First keys:")
            for k in list(weights)[:20]:
                print("  ", k)
            sys.exit(1)
        if args.top > 0:
            keys = keys[:args.top]

    rng = np.random.default_rng(0)
    results = []
    for k in keys:
        x = np.asarray(weights[k], dtype=np.float32)
        if x.ndim == 2 and x.shape[0] > args.max_rows:
            x = x[np.sort(rng.choice(x.shape[0], args.max_rows, replace=False))]
        r = analyze_tensor(k, x, args.extended)
        _print(r)
        results.append(r)

    print("═" * 92)
    labels = sorted({l for r in results for l in r["err"]})
    print(f"  mean rel err over {len(results)} tensors (per-row scale):")
    for l in labels:
        vals = [r["err"][l] for r in results if l in r["err"]]
        print(f"    {l:20s} {np.mean(vals):7.4f}   ({len(vals)} tensors)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Weight distributions vs Fiver grids")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--model", metavar="PATH_OR_ID")
    src.add_argument("--npy",   metavar="FILE.npy")
    ap.add_argument("--layers", default="proj,embed,norm", help="embed,norm,proj,decay,mix,all")
    ap.add_argument("--top", type=int, default=0, help="only the first N matching tensors (0 = all)")
    ap.add_argument("--max-rows", type=int, default=4096,
                    help="row subsample for very large tensors such as embeddings")
    ap.add_argument("--extended", action="store_true", help="also compare 6-bit grids")
    main(ap.parse_args())
