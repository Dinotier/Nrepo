"""
data_loader.py — OpenEmoji download and preprocessing.

Downloads:
  openmoji.csv             — metadata (hexcode, annotation, group, subgroups)
  openmoji-72x72-black.zip — PNG images (72×72 black, MIT license)

Preprocesses:
  Image  : resize to 100×100, grayscale, normalize [0, 1]
  Text   : 100-word vocabulary, 100-feature BoW matrix per annotation
  Labels : multi-hot (n_cat × max_elem) from group/subgroups fields
"""
import os
import csv
import zipfile
import numpy as np
import urllib.request
from pathlib import Path
from collections import Counter

OPENMOJI_CSV_URL = (
    "https://raw.githubusercontent.com/hfg-gmuend/openmoji/master/data/openmoji.csv"
)
OPENMOJI_ZIP_URL = (
    "https://github.com/hfg-gmuend/openmoji/releases/download/15.0.0/"
    "openmoji-72x72-black.zip"
)

VOCAB_SIZE   = 100
FEATURE_SIZE = 100
IMG_SIZE     = 100


# ── Download helpers ──────────────────────────────────────────────────────

def _download(url: str, dest: Path, label: str) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[data] downloading {label}...", flush=True)
    urllib.request.urlretrieve(url, str(dest))
    print(f"[data] saved → {dest}")


def load_openmoji(data_dir: str = "data") -> list[dict]:
    """Download metadata + images; return list of record dicts."""
    root    = Path(data_dir)
    root.mkdir(exist_ok=True)

    csv_path = root / "openmoji.csv"
    zip_path = root / "openmoji-72x72-black.zip"
    img_dir  = root / "openmoji-72x72-black"

    _download(OPENMOJI_CSV_URL, csv_path, "openmoji.csv")

    if not img_dir.exists():
        _download(OPENMOJI_ZIP_URL, zip_path, "openmoji PNG archive")
        print("[data] extracting images...", flush=True)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(root)

    records: list[dict] = []
    with open(csv_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            hexcode    = row.get("hexcode",    "").strip()
            annotation = row.get("annotation", "").strip()
            group      = row.get("group",      "").strip()
            subgroups  = row.get("subgroups",  "").strip()
            if not hexcode or not annotation:
                continue
            img_path = img_dir / f"{hexcode}.png"
            if img_path.exists():
                records.append({
                    "hexcode":    hexcode,
                    "annotation": annotation,
                    "group":      group,
                    "subgroups":  subgroups,
                    "img_path":   str(img_path),
                })

    print(f"[data] {len(records)} emoji records loaded.")
    return records


# ── Vocabulary ────────────────────────────────────────────────────────────

def build_vocabulary(records: list[dict]) -> list[str]:
    """Top-VOCAB_SIZE words across all annotations."""
    counter: Counter = Counter()
    for r in records:
        counter.update(r["annotation"].lower().split())
    return [w for w, _ in counter.most_common(VOCAB_SIZE)]


# ── BoW feature matrix ────────────────────────────────────────────────────

def annotation_to_bow(annotation: str, vocab: list[str]) -> np.ndarray:
    """
    Encode an annotation as a (VOCAB_SIZE × FEATURE_SIZE) float32 matrix.

    Row i represents vocab word i:
      col 0 : TF presence (0 or 1)
      col 1 : normalized first-occurrence position
      col 2 : normalized word length
      col 3+ : deterministic character-hash features (no randomness)
    Remaining columns padded with 0.
    """
    words   = annotation.lower().split()
    word_set = set(words)
    mat = np.zeros((VOCAB_SIZE, FEATURE_SIZE), dtype=np.float32)
    for i, vw in enumerate(vocab):
        mat[i, 0] = float(vw in word_set)
        if vw in words:
            mat[i, 1] = words.index(vw) / max(len(words) - 1, 1)
        mat[i, 2] = min(len(vw) / 20.0, 1.0)
        for j, ch in enumerate(vw[:FEATURE_SIZE - 3], start=3):
            mat[i, j] = (ord(ch) % 97) / 97.0
    return mat


# ── Image loading ─────────────────────────────────────────────────────────

def load_image(img_path: str) -> np.ndarray:
    """Load PNG → 100×100 float32 grayscale, normalized [0, 1]."""
    try:
        from PIL import Image
        img = (Image.open(img_path)
                    .convert("L")
                    .resize((IMG_SIZE, IMG_SIZE)))
        return np.array(img, dtype=np.float32) / 255.0
    except Exception:
        return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)


# ── Label construction ────────────────────────────────────────────────────

def build_label_maps(records: list[dict]) -> tuple:
    groups    = sorted({r["group"]     for r in records if r["group"]})
    subgroups = sorted({r["subgroups"] for r in records if r["subgroups"]})
    return (groups, subgroups,
            {g: i for i, g in enumerate(groups)},
            {s: i for i, s in enumerate(subgroups)})


def make_label_vector(record: dict, group_map: dict, subgroup_map: dict,
                      n_cat: int, max_elem: int) -> np.ndarray:
    lbl   = np.zeros((n_cat, max_elem), dtype=np.float32)
    g_idx = group_map.get(record["group"],     -1)
    s_idx = subgroup_map.get(record["subgroups"], -1)
    if 0 <= g_idx < n_cat and 0 <= s_idx < max_elem:
        lbl[g_idx, s_idx] = 1.0
    return lbl


# ── Dataset wrapper ───────────────────────────────────────────────────────

class EmojiDataset:
    """Iterable dataset, returns (img_flat, txt_flat, label) tuples."""

    def __init__(self, records, vocab, group_map, subgroup_map, n_cat, max_elem):
        self.records      = records
        self.vocab        = vocab
        self.group_map    = group_map
        self.subgroup_map = subgroup_map
        self.n_cat        = n_cat
        self.max_elem     = max_elem

    def __len__(self) -> int:
        return len(self.records)

    def get_item(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r   = self.records[idx]
        img = load_image(r["img_path"]).flatten()
        txt = annotation_to_bow(r["annotation"], self.vocab).flatten()
        lbl = make_label_vector(r, self.group_map, self.subgroup_map,
                                self.n_cat, self.max_elem)
        return img, txt, lbl

    def batch(self, indices: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        imgs, txts, lbls = [], [], []
        for i in indices:
            img, txt, lbl = self.get_item(i)
            imgs.append(img); txts.append(txt); lbls.append(lbl)
        return np.stack(imgs), np.stack(txts), np.stack(lbls)
