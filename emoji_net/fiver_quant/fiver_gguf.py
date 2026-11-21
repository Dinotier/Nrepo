"""
fiver_gguf.py — Convert Fiver-quantized .npz files to a GGUF-compatible layout.

Storage format: Q5_FIVER — one uint32 word = 6 weights + 2 sign bits
═══════════════════════════════════════════════════════════════════════════════

S_lo is the shared sign of k0..k2, S_hi of k3..k5 (majority vote per triplet).
Both layouts carry identical information; only the bit positions differ.
They match the CUDA kernel cw_fiver_gemv in compact_weights.cu exactly.

  --pack u16 (default): two little-endian uint16 triplets [S|k2|k1|k0]
    ┌──────┬─────┬─────┬─────┬──────┬─────┬─────┬─────┐
    │S_hi  │ k5  │ k4  │ k3  │S_lo  │ k2  │ k1  │ k0  │
    │ 31   │30-26│25-21│20-16│ 15   │14-10│ 9-5 │ 4-0 │
    └──────┴─────┴─────┴─────┴──────┴─────┴─────┴─────┘

  --pack u32: sextet with 2-bit sign code on top
    ┌──────┬──────┬─────┬─────┬─────┬─────┬─────┬─────┐
    │S_lo  │S_hi  │ k5  │ k4  │ k3  │ k2  │ k1  │ k0  │
    │ 31   │ 30   │29-25│24-20│19-15│14-10│ 9-5 │ 4-0 │
    └──────┴──────┴─────┴─────┴─────┴─────┴─────┴─────┘
    (S_lo S_hi) = 00 all+, 11 all−, 10 −−−+++, 01 +++−−−

  k ∈ [0..31]:  w(k) = 1 − 2^(−k/4) for k<16,  1 + 2^(−(31−k)/4) for k≥16
  32 bits / 6 values = 5.333 bpw → ~6.0× vs FP32

Row padding:
  The tensor is viewed as (rows, cols) with cols = last dimension.  Each row
  is padded to a multiple of 6 with k=0, sign=0; w(0) = 0.0, so padding never
  contributes to a dot product.  Row r starts at word r·ceil(cols/6).

Sign approximation:
  Unsigned layers (RMSNorm γ, RWKV time-decay W): sign bits are always 0 —
  no information lost.  Signed layers: the minority element of a triplet is
  reconstructed with the wrong sign ("wir hoffen drauf dass es klappt").

Layout per tensor block in the GGUF data section:
  ┌──────────────────────────────────────────────────────────┐
  │  rows × ceil(cols/6) × uint32  (little-endian words)     │
  │  4 bytes float32  scale                                  │
  └──────────────────────────────────────────────────────────┘

GGUF type assignment:
  0x0000–0x00FF  reserved (llama.cpp official)
  0x0105         Q5_FIVER  ← this file writes this value; future llama.cpp
                             patch can detect and dequantize via the formula above

Usage:
  python fiver_gguf.py --manifest ./llama_fiver/manifest.txt --out ./llama_fiver.gguf
  python fiver_gguf.py --manifest ./rwkv_fiver/manifest.txt  --out ./rwkv_fiver.gguf
  python fiver_gguf.py --npz layer1.npz layer2.npz           --out ./custom.gguf
"""
import argparse
import struct
import sys
import time
from pathlib import Path

import numpy as np

# ── GGUF constants ────────────────────────────────────────────────────────────
GGUF_MAGIC   = 0x46554747           # "GGUF" little-endian
GGUF_VERSION = 3

GGUF_TYPE_UINT32 = 4
GGUF_TYPE_STRING = 8

GGML_TYPE_F32      = 0
GGML_TYPE_Q5_FIVER = 0x0105        # Fiver word format; not yet in llama.cpp main


# ── Packing / unpacking ───────────────────────────────────────────────────────

LAYOUTS = ("u16", "u32")


def _k_shifts(layout: str) -> list[int]:
    return [0, 5, 10, 16, 21, 26] if layout == "u16" else [0, 5, 10, 15, 20, 25]


def _sign_shifts(layout: str) -> tuple[int, int]:
    """(S_lo, S_hi) bit positions."""
    return (15, 31) if layout == "u16" else (31, 30)


def _as_rows(a: np.ndarray) -> np.ndarray:
    return a.reshape(1, -1) if a.ndim <= 1 else a.reshape(-1, a.shape[-1])


def pack_fiver(k_data: np.ndarray, sign_bits: np.ndarray | None,
               layout: str = "u16") -> bytes:
    """
    Pack k_data (uint8, 0..31) + optional per-element sign_bits into
    row-padded uint32 words (6 weights each).  See module docstring.
    """
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")
    k = _as_rows(k_data).astype(np.uint32)
    s = (_as_rows(sign_bits).astype(np.uint32) if sign_bits is not None
         else np.zeros_like(k))

    rows, cols = k.shape
    pad = (-cols) % 6
    real = np.ones_like(k)
    if pad:
        k    = np.pad(k,    ((0, 0), (0, pad)))
        s    = np.pad(s,    ((0, 0), (0, pad)))
        real = np.pad(real, ((0, 0), (0, pad)))

    k6, s6, r6 = (a.reshape(rows, -1, 6) for a in (k, s, real))
    # Majority over the real (non-padding) elements of each triplet; ties → +
    s_lo = (2 * s6[..., :3].sum(axis=-1) > r6[..., :3].sum(axis=-1)).astype(np.uint32)
    s_hi = (2 * s6[..., 3:].sum(axis=-1) > r6[..., 3:].sum(axis=-1)).astype(np.uint32)

    sh_lo, sh_hi = _sign_shifts(layout)
    words = (s_lo << sh_lo) | (s_hi << sh_hi)
    for i, sh in enumerate(_k_shifts(layout)):
        words |= (k6[..., i] & 0x1F) << sh

    return words.astype("<u4").tobytes()


def unpack_fiver(buf: bytes, shape: tuple, layout: str = "u16") -> tuple[np.ndarray, np.ndarray]:
    """Inverse of pack_fiver → (k_data, sign_bits), both uint8 with `shape`."""
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")
    cols = int(shape[-1]) if len(shape) else 1
    rows = int(np.prod(shape)) // max(cols, 1)
    wpr  = (cols + 5) // 6

    words = np.frombuffer(buf, dtype="<u4", count=rows * wpr).reshape(rows, wpr)
    sh_lo, sh_hi = _sign_shifts(layout)
    s_lo = ((words >> sh_lo) & 1).astype(np.uint8)
    s_hi = ((words >> sh_hi) & 1).astype(np.uint8)

    k = np.stack([((words >> sh) & 0x1F).astype(np.uint8) for sh in _k_shifts(layout)],
                 axis=-1)                                      # (rows, wpr, 6)
    s = np.concatenate([np.repeat(s_lo[..., None], 3, axis=-1),
                        np.repeat(s_hi[..., None], 3, axis=-1)], axis=-1)

    k = k.reshape(rows, wpr * 6)[:, :cols].reshape(shape)
    s = s.reshape(rows, wpr * 6)[:, :cols].reshape(shape)
    return k, s


# ── GGUF helpers ─────────────────────────────────────────────────────────────

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


# ── Main converter ────────────────────────────────────────────────────────────

def convert_to_gguf(args: argparse.Namespace) -> None:
    t0 = time.time()

    # Collect NPZ sources
    entries: list[tuple[str, Path]] = []
    if args.manifest:
        entries = _load_manifest(Path(args.manifest))
        print(f"[fiver_gguf] manifest: {args.manifest} → {len(entries)} tensors")
    elif args.npz:
        for npz_path in args.npz:
            p = Path(npz_path)
            entries.append((p.stem.replace("__", "/").replace("_", "."), p))
        print(f"[fiver_gguf] {len(entries)} NPZ files specified directly")
    else:
        print("[fiver_gguf] ERROR: provide --manifest or --npz", file=sys.stderr)
        sys.exit(1)

    if not entries:
        print("[fiver_gguf] ERROR: no tensors to convert", file=sys.stderr)
        sys.exit(1)

    # Load and pack
    packed_tensors: list[dict] = []
    for orig_name, npz_path in entries:
        try:
            d = np.load(str(npz_path), allow_pickle=False)
        except Exception as exc:
            print(f"[fiver_gguf] WARNING: skipping {npz_path}: {exc}", file=sys.stderr)
            continue

        k_data    = d["k_data"]
        scale_val = float(d["scale"][0]) if d["scale"].ndim > 0 else float(d["scale"])
        sign_bits = d["sign_bits"] if "sign_bits" in d else None
        shape     = tuple(int(x) for x in d["shape"]) if "shape" in d else k_data.shape
        n         = int(np.prod(shape))

        k_data = k_data.reshape(shape)
        if sign_bits is not None:
            sign_bits = sign_bits.reshape(shape)

        packed  = pack_fiver(k_data, sign_bits, args.pack)
        n_words = len(packed) // 4
        bpw     = len(packed) * 8 / n         # 5.333 when cols % 6 == 0

        # Tensor data block: packed uint32 words + float32 scale
        scale_bytes = struct.pack("<f", scale_val)
        tensor_data = packed + scale_bytes

        print(f"  {orig_name[:50]:50s}  n={n:>10,}  words={n_words:>7,}"
              f"  bpw={bpw:.3f}  packed={len(packed)/1024:.1f}KB"
              f"  total={len(tensor_data)/1024:.1f}KB"
              f"  compress={n*4/max(len(tensor_data),1):.1f}×")

        packed_tensors.append({
            "name":  orig_name,
            "shape": shape,
            "data":  tensor_data,
        })

    if not packed_tensors:
        print("[fiver_gguf] ERROR: nothing to write", file=sys.stderr)
        sys.exit(1)

    # Build GGUF file
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    source_label = args.manifest or (args.npz[0] if args.npz else "unknown")
    kv_block  = b""
    kv_block += _gguf_kv_string("general.architecture",  "fiver_quant")
    kv_block += _gguf_kv_string("general.quantization",  "Q5_FIVER")
    kv_block += _gguf_kv_uint32("general.quant_type",    GGML_TYPE_Q5_FIVER)
    kv_block += _gguf_kv_string("general.source",        str(source_label))
    kv_block += _gguf_kv_string("fiver.packing",
        args.pack + ": uint32 word = 6 weights + S_lo/S_hi; rows padded to 6 (k=0)")
    kv_block += _gguf_kv_string("fiver.formula",
        "w(k)=1+2^(-(31-k)/4) if k>=16 else 1-2^(-k/4); reconstructed *= sign*scale")
    kv_block += _gguf_kv_string("fiver.bpw",             "5.333 (16/3)")
    kv_block += _gguf_kv_string("fiver.compression_vs_fp32", "~6.0x")
    n_kv = 8

    # Tensor offsets
    tensor_offsets: list[int] = []
    cursor = 0
    for t in packed_tensors:
        tensor_offsets.append(cursor)
        cursor += len(t["data"])

    tensor_info_block = b""
    for t, offset in zip(packed_tensors, tensor_offsets):
        tensor_info_block += _gguf_str(t["name"])
        ndim = len(t["shape"])
        tensor_info_block += struct.pack("<I", ndim)
        for dim in t["shape"]:
            tensor_info_block += struct.pack("<Q", dim)
        tensor_info_block += struct.pack("<I", GGML_TYPE_Q5_FIVER)
        tensor_info_block += struct.pack("<Q", offset)

    header  = struct.pack("<I", GGUF_MAGIC)
    header += struct.pack("<I", GGUF_VERSION)
    header += struct.pack("<Q", len(packed_tensors))
    header += struct.pack("<Q", n_kv)

    with open(out_path, "wb") as f:
        f.write(header)
        f.write(kv_block)
        f.write(tensor_info_block)
        header_size = f.tell()
        pad = (32 - header_size % 32) % 32
        f.write(b"\x00" * pad)
        for t in packed_tensors:
            f.write(t["data"])

    total_bytes = out_path.stat().st_size
    print(f"\n[fiver_gguf] wrote {out_path}  ({total_bytes/1e6:.2f} MB, "
          f"{len(packed_tensors)} tensors)")
    print(f"[fiver_gguf] format: {args.pack}, 6 k-values + 2 sign bits per uint32, "
          f"5.333 bpw, ~6.0× vs FP32")
    print(f"[fiver_gguf] quant_type=0x{GGML_TYPE_Q5_FIVER:04X} (Q5_FIVER)")
    print(f"[fiver_gguf] total time: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Convert Fiver .npz bundles to GGUF (Q5_FIVER, 3-per-uint16 format)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", metavar="MANIFEST_TXT",
                     help="manifest.txt written by quant_llama.py / quant_rwkv.py")
    src.add_argument("--npz",      nargs="+", metavar="FILE.npz",
                     help="One or more .npz files to include directly")
    ap.add_argument("--out", default="./fiver_quant.gguf",
                    help="Output .gguf file path (default: ./fiver_quant.gguf)")
    ap.add_argument("--pack", choices=["u16", "u32"], default="u16",
                    help="u16: 3 values per uint16 + 1-bit sign (default); "
                         "u32: 6 values per uint32 + 2-bit sign (4 patterns)")
    convert_to_gguf(ap.parse_args())
