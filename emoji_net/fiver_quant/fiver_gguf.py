"""
fiver_gguf.py — Convert Fiver-quantised .npz files to a GGUF-compatible layout.

Storage per tensor (identical to FiverMatrix in compact_weights.cu):
═══════════════════════════════════════════════════════════════════════════════

  One uint32 word = 6 codes of one row, k_i at bits 5i, bits 30-31 reserved:

    ┌────┬────┬───────┬───────┬───────┬───────┬───────┬───────┐
    │ r  │ r  │  k5   │  k4   │  k3   │  k2   │  k1   │  k0   │
    │ 31 │ 30 │ 29-25 │ 24-20 │ 19-15 │ 14-10 │  9-5  │  4-0  │
    └────┴────┴───────┴───────┴───────┴───────┴───────┴───────┘

  Rows (last dimension = columns) are padded to a multiple of 6 with the
  grid's zero code (16 centered, 0 anchored).  Row r starts at word
  r · ceil(cols/6).  Value: x[r, c] = scale[r] · v_grid(k).

  Tensor data block (32-byte aligned):
    ┌──────────────────────────────────────────────────────────┐
    │  rows × ceil(cols/6) × uint32   (little-endian words)    │
    │  rows × float32                 (row scales)             │
    └──────────────────────────────────────────────────────────┘
  5.333 bits/weight + 32 bits per row.

GGUF type ids (not in llama.cpp; future patches can detect them):
  0x0105  Q5_FIVER_C   centered grid   v(k) = ∓2^(−m/4), code 16 = 0
  0x0106  Q5_FIVER_A   anchored grid   w(k) = 1 ∓ 2^(−m/4)

Usage:
  python fiver_gguf.py --manifest ./llama_fiver/manifest.txt --out ./llama_fiver.gguf
  python fiver_gguf.py --npz layer1.npz layer2.npz           --out ./custom.gguf
"""
import argparse
import struct
import sys
import time
from pathlib import Path

import numpy as np

from fiver_core import bits_per_weight, load_npz, pack_fiver

GGUF_MAGIC     = 0x46554747        # "GGUF" little-endian
GGUF_VERSION   = 3
GGUF_ALIGNMENT = 32

GGUF_TYPE_UINT32 = 4
GGUF_TYPE_STRING = 8

GGML_TYPE_Q5_FIVER = {"centered": 0x0105, "anchored": 0x0106}


def _gguf_str(s: str) -> bytes:
    enc = s.encode("utf-8")
    return struct.pack("<Q", len(enc)) + enc


def _gguf_kv_uint32(key: str, value: int) -> bytes:
    return _gguf_str(key) + struct.pack("<I", GGUF_TYPE_UINT32) + struct.pack("<I", value)


def _gguf_kv_string(key: str, value: str) -> bytes:
    return _gguf_str(key) + struct.pack("<I", GGUF_TYPE_STRING) + _gguf_str(value)


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


def tensor_block(q: dict) -> bytes:
    """Packed words followed by float32 row scales."""
    return pack_fiver(q["k_data"], q["grid"]) + np.asarray(q["scale"], "<f4").tobytes()


def convert_to_gguf(args: argparse.Namespace) -> None:
    t0 = time.time()

    entries: list[tuple[str, Path]] = []
    if args.manifest:
        entries = _load_manifest(Path(args.manifest))
        print(f"[fiver_gguf] manifest: {args.manifest} → {len(entries)} tensors")
    else:
        for npz_path in args.npz:
            p = Path(npz_path)
            entries.append((p.stem.replace("__", "/").replace("_", "."), p))
        print(f"[fiver_gguf] {len(entries)} NPZ files specified directly")

    if not entries:
        print("[fiver_gguf] ERROR: no tensors to convert", file=sys.stderr)
        sys.exit(1)

    tensors: list[dict] = []
    for name, npz_path in entries:
        try:
            q = load_npz(npz_path)
        except Exception as exc:
            print(f"[fiver_gguf] WARNING: skipping {npz_path}: {exc}", file=sys.stderr)
            continue
        data = tensor_block(q)
        n = int(np.prod(q["shape"]))
        print(f"  {name[:50]:50s}  grid={q['grid']:8s}  n={n:>10,}"
              f"  bpw={bits_per_weight(q['shape']):.3f}  {len(data) / 1024:9.1f} KB"
              f"  ({n * 4 / max(len(data), 1):.1f}× vs FP32)")
        tensors.append({"name": name, "shape": q["shape"], "grid": q["grid"], "data": data})

    if not tensors:
        print("[fiver_gguf] ERROR: nothing to write", file=sys.stderr)
        sys.exit(1)

    source_label = args.manifest or args.npz[0]
    kv  = _gguf_kv_string("general.architecture", "fiver_quant")
    kv += _gguf_kv_string("general.quantization", "Q5_FIVER")
    kv += _gguf_kv_uint32("general.alignment",    GGUF_ALIGNMENT)
    kv += _gguf_kv_string("general.source",       str(source_label))
    kv += _gguf_kv_string("fiver.packing",
                          "uint32 word = 6 codes at bits 5i, bits 30-31 reserved; rows padded "
                          "to 6 with the zero code; followed by float32 row scales")
    kv += _gguf_kv_string("fiver.grid.centered",
                          "type 0x0105: v=-2^(-k/4) k<16, v(16)=0, v=+2^(-(31-k)/4) k>16")
    kv += _gguf_kv_string("fiver.grid.anchored",
                          "type 0x0106: w=1-2^(-k/4) k<16, w=1+2^(-(31-k)/4) k>=16")
    n_kv = 7

    def _align(n: int) -> int:
        return (n + GGUF_ALIGNMENT - 1) // GGUF_ALIGNMENT * GGUF_ALIGNMENT

    offsets, cursor = [], 0
    for t in tensors:
        offsets.append(cursor)
        cursor = _align(cursor + len(t["data"]))

    info = b""
    for t, off in zip(tensors, offsets):
        info += _gguf_str(t["name"])
        info += struct.pack("<I", len(t["shape"]))
        for dim in reversed(t["shape"]):            # GGUF lists the fastest dimension first
            info += struct.pack("<Q", dim)
        info += struct.pack("<I", GGML_TYPE_Q5_FIVER[t["grid"]])
        info += struct.pack("<Q", off)

    header  = struct.pack("<I", GGUF_MAGIC) + struct.pack("<I", GGUF_VERSION)
    header += struct.pack("<Q", len(tensors)) + struct.pack("<Q", n_kv)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(header + kv + info)
        f.write(b"\x00" * (_align(f.tell()) - f.tell()))
        data_start = f.tell()
        for t, off in zip(tensors, offsets):
            f.write(b"\x00" * (data_start + off - f.tell()))
            f.write(t["data"])

    print(f"\n[fiver_gguf] wrote {out_path}  ({out_path.stat().st_size / 1e6:.2f} MB, "
          f"{len(tensors)} tensors)  in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Convert Fiver .npz bundles to GGUF (Q5_FIVER)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", metavar="MANIFEST_TXT",
                     help="manifest.txt written by quant_llama.py / quant_rwkv.py")
    src.add_argument("--npz", nargs="+", metavar="FILE.npz",
                     help="One or more .npz files to include directly")
    ap.add_argument("--out", default="./fiver_quant.gguf",
                    help="Output .gguf file path (default: ./fiver_quant.gguf)")
    convert_to_gguf(ap.parse_args())
