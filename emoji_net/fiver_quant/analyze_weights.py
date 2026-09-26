"""
analyze_weights.py — Analyse der tatsächlichen Gewichtsverteilungen in LLMs.

Untersucht für jeden ausgewählten Layer:
  1. Vorzeichen-Statistik
       - Anteil negativer Gewichte, VZ-Entropie
       - Sign-Kohärenz in Gruppen von 3 und 6 (für Fiver-Word-Encoding)
       - Top-4 Muster in 6er-Gruppen → informiert 2-Bit-VZ-Schema
  2. Betragsverteilung
       - Normiert auf [0,2] (Fiver-Bereich nach per-tensor Scale)
       - Anteil in unterer (w<1, k∈[0..15]) vs. oberer Hälfte (w>1, k∈[16..31])
  3. Fiver-Passung
       - Histogramm der gewählten k-Werte (ASCII)
       - Quantisierungsfehler, KL-Divergenz vs. uniform

2-Bit-VZ-Schema für 6er-Gruppen (uint32_t, 32/6 = 5.333 bpw):
  ┌────┬────┬──────────────────────────────────────────────────────┐
  │ S1 │ S0 │ Bedeutung                                            │
  ├────┼────┼──────────────────────────────────────────────────────┤
  │  0 │  0 │ alle 6 positiv                                       │
  │  1 │  1 │ alle 6 negativ                                       │
  │  0 │  1 │ k0..k2 positiv, k3..k5 negativ                      │
  │  1 │  0 │ k0..k2 negativ, k3..k5 positiv                      │
  └────┴────┴──────────────────────────────────────────────────────┘

Usage:
  python analyze_weights.py --model meta-llama/Meta-Llama-3.1-8B --layers embed,norm
  python analyze_weights.py --model ./rwkv_ckpt.pth --layers all --top 10
  python analyze_weights.py --npy ./my_tensor.npy
"""
import argparse
import sys
from pathlib import Path

import numpy as np

# ── Fiver LUT (inline, no circular import) ───────────────────────────────────
_DELTA  = np.array([2.0 ** (-m / 4.0) for m in range(16)], dtype=np.float64)
_FIVER  = np.empty(32, dtype=np.float64)
for _k in range(32):
    _d = (_k >> 4) & 1
    _m = (31 - _k) if _d else _k
    _FIVER[_k] = 1.0 + (_DELTA[_m] if _d else -_DELTA[_m])

LAYER_GROUPS = {
    "embed": ["embed_tokens.weight", "lm_head.weight"],
    "norm":  ["layernorm.weight", "layer_norm.weight", "norm.weight",
              "input_layernorm.weight", "post_attention_layernorm.weight",
              "final_layernorm.weight"],
    "proj":  ["o_proj.weight", "down_proj.weight", "out_proj.weight"],
    "decay": ["time_decay", "att.time_decay"],
    "mix":   ["time_mix_k", "time_mix_v", "time_mix_r", "time_mix_g"],
}


# ── Weight loading ────────────────────────────────────────────────────────────

def _load_weights(model_path: str) -> dict[str, np.ndarray]:
    import glob as _g
    p = Path(model_path)
    try:
        from safetensors import safe_open
        files = ([str(p)] if p.suffix == ".safetensors"
                 else sorted(_g.glob(str(p / "*.safetensors"))))
        if files:
            w: dict[str, np.ndarray] = {}
            for f in files:
                with safe_open(f, framework="np", device="cpu") as st:
                    for k in st.keys():
                        w[k] = st.get_tensor(k)
            return w
    except (ImportError, Exception):
        pass

    if p.is_file() and p.suffix in {".pth", ".pt"}:
        try:
            import torch
            ckpt = torch.load(str(p), map_location="cpu")
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]
            return {k: v.detach().float().numpy()
                    for k, v in ckpt.items() if hasattr(v, "detach")}
        except Exception:
            pass

    try:
        from transformers import AutoModelForCausalLM
        import torch
        m = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float32, device_map="cpu",
            trust_remote_code=True, low_cpu_mem_usage=True)
        w = {k: v.detach().numpy() for k, v in m.state_dict().items()}
        del m
        return w
    except ImportError:
        print("[analyze] ERROR: install transformers or safetensors", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"[analyze] ERROR: {exc}", file=sys.stderr); sys.exit(1)


def _select_layers(weights: dict, spec: str) -> list[str]:
    specs = [s.strip().lower() for s in spec.split(",")]
    out = []
    for key in weights:
        kl = key.lower()
        for sp in specs:
            if sp == "all":
                out.append(key); break
            if any(g in kl for g in LAYER_GROUPS.get(sp, [])):
                out.append(key); break
    return out


# ── Analysis ──────────────────────────────────────────────────────────────────

def _sign_analysis(flat: np.ndarray) -> dict:
    n       = len(flat)
    neg     = (flat < 0).astype(np.uint8)
    p_neg   = float(neg.mean())
    p       = np.clip(p_neg, 1e-12, 1 - 1e-12)
    entropy = float(-p * np.log2(p) - (1 - p) * np.log2(1 - p))

    def _grp(g: int) -> dict:
        pad = (-n) % g
        s = np.concatenate([neg, np.zeros(pad, np.uint8)]) if pad else neg
        m = s.reshape(-1, g)
        ng = len(m)
        ap = int((m.sum(1) == 0).sum()) / ng
        an = int((m.sum(1) == g).sum()) / ng
        return {"all_pos": ap, "all_neg": an, "mixed": 1-ap-an,
                "same_sign": ap + an}

    g3 = _grp(3)
    g6 = _grp(6)

    # 2-Bit-Schema: majority per half-sextet
    pad = (-n) % 6
    s6  = np.concatenate([neg, np.zeros(pad, np.uint8)]) if pad else neg
    m6  = s6.reshape(-1, 6)
    ng6 = len(m6)
    s1  = (m6[:, :3].sum(1) >= 2)   # first-half neg
    s0  = (m6[:,  3:].sum(1) >= 2)  # last-half  neg
    code = (s1.astype(np.uint8) << 1) | s0.astype(np.uint8)
    pats = {
        "00 (all +)":   int((code == 0).sum()) / ng6,
        "11 (all −)":   int((code == 3).sum()) / ng6,
        "10 (−−− +++)": int((code == 2).sum()) / ng6,
        "01 (+++ −−−)": int((code == 1).sum()) / ng6,
    }
    exact = (float((m6.sum(1) == 0).mean())        # rein+
             + float((m6.sum(1) == 6).mean())       # rein-
             + float(((m6[:,:3].sum(1)==0)&(m6[:,3:].sum(1)==3)).mean())
             + float(((m6[:,:3].sum(1)==3)&(m6[:,3:].sum(1)==0)).mean()))

    return {"p_neg": p_neg, "entropy": entropy, "group3": g3, "group6": g6,
            "pattern_2bit": pats, "exact_coverage": exact}


def _mag_analysis(flat: np.ndarray) -> dict:
    a   = np.abs(flat)
    n   = len(flat)
    pct = np.percentile(a, [0, 10, 25, 50, 75, 90, 99, 100]).tolist()

    xmax  = float(a.max())
    scale = xmax / 2.0 if xmax > 1e-12 else 1.0
    norm  = np.clip(a / scale, 0.0, 2.0)

    frac_lo   = float((norm < 1.0).mean())
    frac_hi   = float((norm >= 1.0).mean())
    frac_near = float(((norm > 0.85) & (norm < 1.15)).mean())

    dists  = np.abs(norm[:, None] - _FIVER[None, :])
    k_data = np.argmin(dists, 1).astype(np.uint8)
    k_hist = np.histogram(k_data, bins=32, range=(0, 32))[0]
    k_pct  = (k_hist / max(k_hist.sum(), 1)).tolist()

    recon  = _FIVER[k_data] * scale
    diff   = a - recon
    l2     = float(np.sqrt((diff**2).mean()))
    rel    = float((np.abs(diff) / (a + 1e-8)).mean())

    orig_h = np.histogram(norm, bins=32, range=(0, 2), density=True)[0]
    eps    = 1e-10
    op = orig_h + eps;  op /= op.sum()
    up = np.ones(32) / 32 + eps;  up /= up.sum()
    kl = float(np.sum(op * np.log(op / up)))

    return {"scale": scale, "pct_raw": pct, "frac_lo": frac_lo,
            "frac_hi": frac_hi, "frac_near": frac_near,
            "k_hist": k_pct, "k_mean": float(k_data.mean()),
            "k_std": float(k_data.std()), "l2": l2, "rel_err": rel, "kl": kl}


def _bar(vals: list[float]) -> str:
    b = "▁▂▃▄▅▆▇█"
    return "".join(b[max(0, min(7, int(v * 7 + .5)))] for v in vals)


def _print_report(name: str, sa: dict, ma: dict) -> None:
    print("─" * 78)
    print(f"  {name}")
    print(f"  n={int(len(ma['pct_raw']))}  scale={ma['scale']:.5f}")
    print()
    print(f"  [VZ]  neg={sa['p_neg']:.1%}  pos={1-sa['p_neg']:.1%}"
          f"  Entropie={sa['entropy']:.3f}")
    g3, g6 = sa["group3"], sa["group6"]
    print(f"  3er: all+={g3['all_pos']:.1%}  all-={g3['all_neg']:.1%}"
          f"  gemischt={g3['mixed']:.1%}  same={g3['same_sign']:.1%}")
    print(f"  6er: all+={g6['all_pos']:.1%}  all-={g6['all_neg']:.1%}"
          f"  gemischt={g6['mixed']:.1%}  same={g6['same_sign']:.1%}")
    print()
    print("  2-Bit-Schema (6er-Gruppen, Mehrheit pro Hälfte):")
    for code, frac in sa["pattern_2bit"].items():
        bar = "█" * max(1, int(frac * 24))
        print(f"    {code:20s}  {frac:5.1%}  {bar}")
    print(f"  exakte Abdeckung ohne Mehrheitsentscheid: {sa['exact_coverage']:.1%}")
    print()
    p = ma["pct_raw"]
    print(f"  [|w|]  p50={p[3]:.4f}  p90={p[5]:.4f}  max={p[7]:.4f}")
    print(f"  normiert [0,2]: w<1={ma['frac_lo']:.1%}  w≥1={ma['frac_hi']:.1%}"
          f"  |w−1|<0.15={ma['frac_near']:.1%}")
    print()
    print(f"  k-Verteilung (k=0..15 | k=16..31):  k̄={ma['k_mean']:.1f}  σ={ma['k_std']:.1f}")
    print(f"  {_bar(ma['k_hist'][:16])}|{_bar(ma['k_hist'][16:])}")
    print(f"  ← w<1 (untere Hälfte) →← w>1 (obere Hälfte) →")
    print()
    print(f"  [Fehler]  L2={ma['l2']:.5f}  rel={ma['rel_err']:.4f}"
          f"  KL(uniform)={ma['kl']:.4f}")
    print()


def _recommend(results: list[dict]) -> None:
    print("═" * 78)
    print("  EMPFEHLUNG VZ-SCHEMA")
    print("═" * 78)
    avg = lambda key: float(np.mean([r[key] for r in results]))
    a_pneg   = avg("p_neg")
    a_ent    = avg("entropy")
    a_s3     = avg("same3")
    a_s6     = avg("same6")
    a_exact  = avg("exact")
    a_near1  = avg("frac_near")

    print(f"  ∅ neg={a_pneg:.1%}  Entropie={a_ent:.3f}  same-3={a_s3:.1%}"
          f"  same-6={a_s6:.1%}  exakt={a_exact:.1%}  |w−1|<0.15={a_near1:.1%}")
    print()

    if a_pneg < 0.05:
        rec = "use_sign=False (rein positiver Layer, kein VZ-Bit nötig)"
    elif a_ent < 0.25:
        rec = "1-Bit shared / 3er-Gruppe (uint16, kaum Entropieverlust)"
    elif a_s6 > 0.75:
        rec = "1-Bit shared / 6er-Gruppe oder 2-Bit / 6er — beide gleich gut"
    elif a_exact > 0.55:
        rec = "2-Bit / 6er-Gruppe (uint32) — viele Gruppen passen exakt ins Schema"
    else:
        rec = "Per-Element VZ (separates sign_bits-Array) oder 2-Bit-Näherung"

    print(f"  → {rec}")
    print()
    print("  Format-Vergleich  (alle: 5.333 bpw = 32/6 = 16/3, ~6.0× vs FP32)")
    print("  ────────────────────────────────────────────────────────────────")
    print("  Format              Word  Werte  VZ-Bits  Muster  Anmerkung")
    print("  unsigned (no sign)  u16   3      0        1       nur pos. Layer")
    print("  1-Bit / 3 (u16)     u16   3      1        2       Mehrheit Triplet")
    print("  2-Bit / 6 (u32)     u32   6      2        4       Mehrheit Hälfte")
    print("  1-Bit / 1 (exact)   u8    1      1        2       exakt, 6.0 bpw")


# ── Main ──────────────────────────────────────────────────────────────────────

def analyze(args: argparse.Namespace) -> None:
    if args.npy:
        tensor   = np.load(args.npy).astype(np.float32)
        name     = Path(args.npy).stem
        weights  = {name: tensor}
        selected = [name]
    else:
        if not args.model:
            print("[analyze] ERROR: --model or --npy required", file=sys.stderr)
            sys.exit(1)
        print(f"[analyze] loading {args.model}…", flush=True)
        weights  = _load_weights(args.model)
        selected = _select_layers(weights, args.layers)
        if not selected:
            print(f"[analyze] no layers matched '{args.layers}'. First 20 keys:")
            for k in list(weights.keys())[:20]:
                print(f"  {k}")
            sys.exit(1)
        if args.top > 0:
            selected = selected[:args.top]

    print(f"[analyze] {len(selected)} layer(s)\n")
    agg: list[dict] = []
    for key in selected:
        flat = weights[key].astype(np.float32).ravel().astype(np.float64)
        sa   = _sign_analysis(flat)
        ma   = _mag_analysis(flat)
        _print_report(key, sa, ma)
        agg.append({
            "p_neg":     sa["p_neg"],
            "entropy":   sa["entropy"],
            "same3":     sa["group3"]["same_sign"],
            "same6":     sa["group6"]["same_sign"],
            "exact":     sa["exact_coverage"],
            "frac_near": ma["frac_near"],
        })

    _recommend(agg)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Analyse LLM-Gewichtsverteilungen für Fiver VZ-Schema-Entscheidung")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--model", metavar="PATH_OR_ID")
    src.add_argument("--npy",   metavar="FILE.npy")
    ap.add_argument("--layers", default="embed,norm",
                    help="embed,norm,proj,decay,mix,all  (default: embed,norm)")
    ap.add_argument("--top", type=int, default=0,
                    help="Nur erste N Layer (0=alle)")
    analyze(ap.parse_args())
