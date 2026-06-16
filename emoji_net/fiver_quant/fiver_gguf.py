"""
fiver_gguf.py — Convert Fiver-quantized .npz files to a GGUF-compatible layout.

Storage format: Q5_FIVER / "Fiver-Word"
═══════════════════════════════════════════════════════════════════════════════

Three k-values share one uint16_t word, with a single shared sign bit:

    ┌────┬─────────┬─────────┬─────────┐
    │ S  │   k2    │   k1    │   k0    │
    │ 15 │ 14 – 10 │  9 – 5  │  4 – 0  │
    └────┴─────────┴─────────┴─────────┘

  S   = shared sign bit for all three weights in this word
        0 → all three positive,  1 → all three negative
  k0–k2 ∈ [0..31]:  w(k) = 1 ± 2^(−|dir(k)|/4)

Effective storage cost: 16 bits / 3 values = 5.333 bpw
Compression vs FP32: 32 / 5.333 ≈ 6.0×

Sign approximation:
  For purely unsigned layers (RMSNorm γ, RWKV time-decay W) S is always 0 —
  no information is lost.
  For signed layers (embeddings, projections) a majority-vote sign is used per
  triplet: if ≥2 of the three values are negative, S=1.  The minority element
  is reconstructed with the wrong sign — this is the intended approximation.
  ("wir hoffen drauf dass es klappt")

Tail padding:
  If len(k_data) % 3 ≠ 0, the last word is zero-padded (k_pad=0, S=0).

Layout per tensor block in the GGUF data section:
  ┌──────────────────────────────────────────────┐
  │  ceil(n/3) × uint16_t  (little-endian words) │
  │  4 bytes float32  scale                      │
  └──────────────────────────────────────────────┘

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

def pack_triplets(k_data: np.ndarray, sign_bits: np.ndarray | None) -> bytes:
    """
    Pack k_data (uint8, values 0–31) + optional per-element sign_bits into
    Fiver-Word uint16_t array.

    Layout per word (little-endian):
      bit 15   : shared sign (majority vote over the triplet)
      bits 14-10: k2
      bits  9- 5: k1
      bits  4- 0: k0

    Parameters
    ----------
    k_data    : uint8 ndarray, any shape, values in [0..31]
    sign_bits : uint8 ndarray same shape (0=pos, 1=neg), or None for unsigned

    Returns
    -------
    bytes  — little-endian uint16_t array, length = ceil(n/3) × 2 bytes
    """
    flat_k = k_data.ravel().astype(np.uint32)
    n = len(flat_k)

    if sign_bits is not None:
        flat_s = sign_bits.ravel().astype(np.uint32)
    else:
        flat_s = np.zeros(n, dtype=np.uint32)

    # Pad to multiple of 3
    pad = (-n) % 3
    if pad:
        flat_k = np.concatenate([flat_k, np.zeros(pad, dtype=np.uint32)])
        flat_s = np.concatenate([flat_s, np.zeros(pad, dtype=np.uint32)])

    k_mat = flat_k.reshape(-1, 3)   # (G, 3): k_mat[:,0]=k0, [:,1]=k1, [:,2]=k2
    s_mat = flat_s.reshape(-1, 3)   # (G, 3): sign per element

    # Majority-vote shared sign: S=1 if ≥2 of the three signs are 1
    shared_sign = (s_mat.sum(axis=1) >= 2).astype(np.uint32)  # (G,)

    # Pack: word = (S<<15) | (k2<<10) | (k1<<5) | k0
    words = ((shared_sign << 15)
             | ((k_mat[:, 2] & 0x1F) << 10)
             | ((k_mat[:, 1] & 0x1F) << 5)
             |  (k_mat[:, 0] & 0x1F)).astype(np.uint16)

    return words.tobytes()          # little-endian on all x86/ARM hosts


def unpack_triplets(buf: bytes, n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Inverse of pack_triplets.

    Parameters
    ----------
    buf : bytes produced by pack_triplets
    n   : number of original elements (to strip tail padding)

    Returns
    -------
    k_data    : uint8 ndarray, shape (n,)
    sign_bits : uint8 ndarray, shape (n,)
    """
    words = np.frombuffer(buf, dtype=np.uint16).astype(np.uint32)
    shared_sign = ((words >> 15) & 1).astype(np.uint8)
    k2 = ((words >> 10) & 0x1F).astype(np.uint8)
    k1 = ((words >>  5) & 0x1F).astype(np.uint8)
    k0 = ( words        & 0x1F).astype(np.uint8)

    # Interleave: [k0, k1, k2, k0, k1, k2, ...]
    g = len(words)
    k_flat    = np.empty(g * 3, dtype=np.uint8)
    sign_flat = np.empty(g * 3, dtype=np.uint8)
    k_flat[0::3] = k0;  k_flat[1::3] = k1;  k_flat[2::3] = k2
    sign_flat[0::3] = shared_sign
    sign_flat[1::3] = shared_sign
    sign_flat[2::3] = shared_sign

    return k_flat[:n], sign_flat[:n]


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

        packed = pack_triplets(k_data, sign_bits)
        n_words = len(packed) // 2
        bpw     = len(packed) * 8 / n         # ≈ 5.333 for non-padded tensors

        # Tensor data block: packed uint16 words + float32 scale
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
        "3-per-uint16: [S:1|k2:5|k1:5|k0:5], S=shared sign (majority vote)")
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
    print(f"[fiver_gguf] format: 3 k-values per uint16, shared sign bit, "
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
    convert_to_gguf(ap.parse_args())
