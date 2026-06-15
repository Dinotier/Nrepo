"""
fiver_gguf.py — Convert Fiver-quantized .npz files to a GGUF-compatible layout.

This is a documentation-forward converter.  GGUF (GPT-Generated Unified Format,
used by llama.cpp) stores tensors with a quant_type enum.  No official Q5_FIVER
type exists yet; this tool:

  1. Reads .npz files produced by quant_llama.py / quant_rwkv.py
  2. Packs 5-bit k_data values into a raw byte buffer (5 bits/element, big-endian
     bit-packing within each byte, zero-padded to 8-bit boundary)
  3. Writes a minimal .gguf file with:
       - magic + version header
       - metadata KV pairs (architecture, quant_type=Q5_FIVER=0x0105, source path)
       - tensor directory (name, shape, type, offset)
       - tensor data block (packed bits + scale f32 + optional sign_bits)

GGUF type assignment:
  0x0000–0x00FF  reserved (llama.cpp official)
  0x0100         Q5_K (llama.cpp)
  0x0105         Q5_FIVER  ← this tool writes this value in the header
                             so future llama.cpp patches can detect + load it

Bit-packing layout (per tensor block of 8 elements → 5 bytes):
  byte 0:  k[0][7:3]  k[1][7:3]    → bits 7-3 from k[0], bits 7-3 from k[1] NOT right
  Correct: pack 8×5-bit values into 5 bytes, MSB first within each k:
    byte 0: k0[4] k0[3] k0[2] k0[1] k0[0] k1[4] k1[3] k1[2]
    byte 1: k1[1] k1[0] k2[4] k2[3] k2[2] k2[1] k2[0] k3[4]
    byte 2: k3[3] k3[2] k3[1] k3[0] k4[4] k4[3] k4[2] k4[1]
    byte 3: k4[0] k5[4] k5[3] k5[2] k5[1] k5[0] k6[4] k6[3]
    byte 4: k6[2] k6[1] k6[0] k7[4] k7[3] k7[2] k7[1] k7[0]

Scale and optional sign_bits follow the packed data as:
  4 bytes  float32 scale
  ceil(n/8) bytes  sign_bits packed 1-bit-per-element (if present, else omitted)

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

# KV value type tags (subset used here)
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_STRING = 8

# Tensor quant types
GGML_TYPE_F32      = 0
GGML_TYPE_Q5_FIVER = 0x0105        # Fiver-specific; not yet in llama.cpp main


def _pack_5bit(k_data: np.ndarray) -> bytes:
    """
    Pack uint8 array of 5-bit values into a tightly packed byte buffer.
    Groups of 8 k-values → 5 bytes.  Tail group zero-padded.
    """
    flat = k_data.ravel().astype(np.uint8)
    n = len(flat)
    # Pad to multiple of 8
    pad = (-n) % 8
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.uint8)])

    groups = flat.reshape(-1, 8)       # (G, 8)
    out = bytearray()
    for g in groups:
        # Shift each k into its bit position within 5 bytes (40 bits)
        bits = 0
        for i, k in enumerate(g):
            bits = (bits << 5) | int(k & 0x1F)
        # bits is now a 40-bit integer; emit 5 bytes big-endian
        out += bits.to_bytes(5, "big")
    return bytes(out)


def _pack_signbits(sign_bits: np.ndarray) -> bytes:
    """Pack boolean sign_bits array into 1-bit-per-element bytes."""
    flat = sign_bits.ravel().astype(np.uint8)
    n    = len(flat)
    pad  = (-n) % 8
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.uint8)])
    # Pack 8 bits per byte, MSB first
    packed = np.packbits(flat, bitorder="big")
    return packed.tobytes()


def _gguf_str(s: str) -> bytes:
    """Encode a string as GGUF uint64-length-prefixed UTF-8."""
    enc = s.encode("utf-8")
    return struct.pack("<Q", len(enc)) + enc


def _gguf_kv_uint32(key: str, value: int) -> bytes:
    return _gguf_str(key) + struct.pack("<I", GGUF_TYPE_UINT32) + struct.pack("<I", value)


def _gguf_kv_string(key: str, value: str) -> bytes:
    return _gguf_str(key) + struct.pack("<I", GGUF_TYPE_STRING) + _gguf_str(value)


def _load_manifest(manifest_path: Path) -> list[tuple[str, Path]]:
    """Read manifest.txt → list of (original_key, npz_path)."""
    entries: list[tuple[str, Path]] = []
    base = manifest_path.parent
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            entries.append((parts[0], base / parts[1]))
    return entries


def convert_to_gguf(args: argparse.Namespace) -> None:
    t0 = time.time()

    # ── Collect NPZ sources ───────────────────────────────────────────────────
    entries: list[tuple[str, Path]] = []

    if args.manifest:
        entries = _load_manifest(Path(args.manifest))
        print(f"[fiver_gguf] manifest: {args.manifest} → {len(entries)} tensors")
    elif args.npz:
        for npz_path in args.npz:
            p = Path(npz_path)
            # Use filename stem as tensor name
            entries.append((p.stem.replace("__", "/").replace("_", "."), p))
        print(f"[fiver_gguf] {len(entries)} NPZ files specified directly")
    else:
        print("[fiver_gguf] ERROR: provide --manifest or --npz", file=sys.stderr)
        sys.exit(1)

    if not entries:
        print("[fiver_gguf] ERROR: no tensors to convert", file=sys.stderr)
        sys.exit(1)

    # ── Load and pack all tensors ─────────────────────────────────────────────
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

        packed_k   = _pack_5bit(k_data)
        packed_sgn = _pack_signbits(sign_bits) if sign_bits is not None else b""
        scale_bytes = struct.pack("<f", scale_val)

        # Tensor data block: packed_k + scale_f32 + sign_bytes
        tensor_data = packed_k + scale_bytes + packed_sgn

        n = int(np.prod(shape))
        print(f"  {orig_name:50s}  shape={str(shape):20s}"
              f"  packed={len(packed_k)/1024:.1f}KB"
              f"  total={len(tensor_data)/1024:.1f}KB"
              f"  compression={n*4/max(len(tensor_data),1):.1f}×")

        packed_tensors.append({
            "name":        orig_name,
            "shape":       shape,
            "data":        tensor_data,
            "has_sign":    sign_bits is not None,
        })

    if not packed_tensors:
        print("[fiver_gguf] ERROR: nothing to write after loading", file=sys.stderr)
        sys.exit(1)

    # ── Build GGUF file ───────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # KV metadata
    source_label = args.manifest or (args.npz[0] if args.npz else "unknown")
    kv_block = b""
    kv_block += _gguf_kv_string("general.architecture",  "fiver_quant")
    kv_block += _gguf_kv_string("general.quantization",  "Q5_FIVER")
    kv_block += _gguf_kv_uint32("general.quant_type",    GGML_TYPE_Q5_FIVER)
    kv_block += _gguf_kv_string("general.source",        str(source_label))
    kv_block += _gguf_kv_string("fiver.bit_depth",       "5")
    kv_block += _gguf_kv_string("fiver.encoding",        "k∈[0..31] w(k)=1±2^(-mag/4) per-tensor-scale")
    n_kv = 6

    # Tensor directory: compute offsets
    tensor_offsets: list[int] = []
    cursor = 0
    for t in packed_tensors:
        tensor_offsets.append(cursor)
        cursor += len(t["data"])

    # Tensor info block
    tensor_info_block = b""
    for t, offset in zip(packed_tensors, tensor_offsets):
        tensor_info_block += _gguf_str(t["name"])
        ndim = len(t["shape"])
        tensor_info_block += struct.pack("<I", ndim)
        for dim in t["shape"]:
            tensor_info_block += struct.pack("<Q", dim)
        tensor_info_block += struct.pack("<I", GGML_TYPE_Q5_FIVER)
        tensor_info_block += struct.pack("<Q", offset)

    # GGUF header
    header = struct.pack("<I", GGUF_MAGIC)
    header += struct.pack("<I", GGUF_VERSION)
    header += struct.pack("<Q", len(packed_tensors))   # tensor_count
    header += struct.pack("<Q", n_kv)                  # metadata_kv_count

    with open(out_path, "wb") as f:
        f.write(header)
        f.write(kv_block)
        f.write(tensor_info_block)
        # 32-byte alignment padding before tensor data (GGUF spec requirement)
        header_size = f.tell()
        pad = (32 - header_size % 32) % 32
        f.write(b"\x00" * pad)
        for t in packed_tensors:
            f.write(t["data"])

    total_bytes = out_path.stat().st_size
    print(f"\n[fiver_gguf] wrote {out_path}  ({total_bytes/1e6:.2f} MB, "
          f"{len(packed_tensors)} tensors)")
    print(f"[fiver_gguf] quant_type=0x{GGML_TYPE_Q5_FIVER:04X} (Q5_FIVER, "
          f"not yet in llama.cpp — patch needed)")
    print(f"[fiver_gguf] total time: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Convert Fiver .npz bundles to GGUF format")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", metavar="MANIFEST_TXT",
                     help="manifest.txt written by quant_llama.py / quant_rwkv.py")
    src.add_argument("--npz",      nargs="+", metavar="FILE.npz",
                     help="One or more .npz files to include directly")
    ap.add_argument("--out", default="./fiver_quant.gguf",
                    help="Output .gguf file path (default: ./fiver_quant.gguf)")
    convert_to_gguf(ap.parse_args())
