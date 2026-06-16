# 5-Bit Quantization for LLMs — Research Report
*Recherche zum Stand der Technik und Einordnung des Fiver-Ansatzes*
*Erstellt: 2026-06-16*

---

## 1. Kurzfassung

5-Bit-Quantisierung existiert in der Praxis — vor allem in **llama.cpp (Q5_K_M)** — ist
aber in der akademischen PTQ-Literatur ein nahezu unbeschriebenes Blatt.
Forschung springt direkt von 4-Bit zu sub-3-Bit-Verfahren.
Das ist die Lücke, in die **Fiver** passt: ein strukturiertes, hardware-freundliches
5-Bit-Encoding, das insbesondere für vorzeichenlose Layer (RMSNorm, Embedding, RWKV-W)
optimiert ist.

---

## 2. llama.cpp — Q5-Formate (einzige verbreitete 5-Bit-Implementierung)

### 2.1 Verfügbare Typen

| Typ      | Bits/Weight (eff.) | Beschreibung                                      | Status     |
|----------|--------------------|---------------------------------------------------|------------|
| Q5_0     | 5.0                | Legacy: 32-Weight-Blöcke, FP16-Scale              | veraltet   |
| Q5_1     | 5.3                | Legacy: 32-Weight-Blöcke, FP16-Scale + FP16-Min  | veraltet   |
| Q5_K_S   | 5.57               | K-Quant: 8 × 32 Gewichte/Superblock, 6-bit Scales| empfohlen  |
| Q5_K_M   | 5.70               | K-Quant: wie K_S, aber einige Layer bei Q6_K      | **best**   |

### 2.2 Internes Format Q5_K (K-Quant-Familie)

```
Superblock = 8 Blöcke × 32 Gewichte = 256 Gewichte
  → 256 × 5 bit = 1280 bit Nutz-Daten
  + 8 × 6 bit scale + 8 × 6 bit min = 96 bit Overhead
  + 2 × FP16 (Superblock-Scale + Min) = 32 bit
  ─────────────────────────────────────────────────
  Effektiv: 1408 bit / 256 ≈ 5.50 bpw
```

Zum Vergleich:

| Format  | Nom. Bits | Eff. bpw | Ppl-Delta vs FP16 (Llama-3.1-8B) | Größe (GiB) |
|---------|-----------|----------|-----------------------------------|-------------|
| Q4_K_M  | 4         | 4.89     | +1.20%                            | 4.58        |
| Q5_K_S  | 5         | 5.57     | —                                 | 5.21        |
| Q5_K_M  | 5         | 5.70     | **+0.40%**                        | 5.33        |
| Fiver-W | 5.333 eff | 5.333    | TBD (triplet-sign approx.)        | —           |
| Q6_K    | 6         | 6.56     | +0.16%                            | 5.21        |
| Q8_0    | 8         | 8.50     | ~0.0%                             | 7.95        |

→ **Q5_K_M halbiert den Perplexitäts-Fehler von Q4_K_M** bei nur 16% mehr Speicher.

### 2.3 Wichtig: Layer-Behandlung in Q5_K_M

llama.cpp quantisiert **Embeddings und Layer-Norms nicht** mit Q5_K.
Sie bleiben in FP16/BF16 — genau die Layer, für die Fiver am besten geeignet ist.
Das ist kein Zufall: diese Layer enthalten entweder outlier-reiche Verteilungen
(Embeddings) oder Werte nahe 1.0 (RMSNorm-Gains), für die ein allgemeines Q5_K
suboptimal kalibriert ist.

---

## 3. Akademische PTQ-Literatur: 5-Bit ist eine Lücke

Eine kuratierte Liste der wichtigsten LLM-Quantisierungsverfahren
(Awesome-LLM-Quantization, ~200 Einträge) enthält **keinen einzigen 5-Bit-Ansatz**.
Die Forschung konzentriert sich auf:

- **4-Bit**: GPTQ, AWQ, QuaRot, QUIK, MixLLM, SpQR
- **3-Bit**: OWQ, RPTQ, SqueezeLLM
- **2-Bit**: BiLLM, AQLM, VPTQ, QuIP#
- **1-Bit**: BitNet b1.58

### 3.1 GPTQ / AWQ bei 5-Bit

- GPTQ und AWQ wurden primär für 4-Bit entwickelt; 5-Bit-Experimente sind selten publiziert
- RTN W5A16 (Round-to-Nearest, 5-Bit-Gewichte) übertrifft GPTQ W4A16 messbar
- MixLLM zeigt: **4.4 bit mixed ≈ 5 bit RTN** — Qualitäts-sweet-spot liegt bei ~5 bit

### 3.2 SqueezeLLM / QuIP# / AQLM / SpQR

| Methode    | Ziel-Bits | Ppl-Delta (LLaMA-7B, WikiText2) | Besonderheit                     |
|------------|-----------|---------------------------------|----------------------------------|
| SqueezeLLM | 4-bit     | +0.67 (vs FP16 3.53)           | Dense+Sparse; 0.45% Ausreißer    |
| QuIP#      | 3-bit     | besser als 4-bit RTN            | Hadamard-Rotation, Codebook      |
| AQLM       | 4-bit     | nahe FP16                       | Additive Codebook; 1.2-3× Speed  |
| SpQR       | 4-bit     | <1% rel. Fehler                 | Outlier-separate Sparse Format   |

Keine dieser Methoden zielt auf 5-Bit als primäres Kompressionsniveau.

---

## 4. Prior Art: Power-of-Two-Quantisierung

### 4.1 APoT — Additive Powers-of-Two (Li et al., 2019)

**Paper**: arXiv:1909.13144, ICLR 2020

Kernidee: Alle Quantisierungsstufen sind Summen von Zweierpotenzen:

```
w_q = Σ_{i=0}^{p-1}  α_i · 2^{-i}   mit α_i ∈ {0, +1, -1}
```

- **Hardware-Vorteil**: Multiplikation → Shift-Operationen (~2× schneller)
- **Ergebnis**: 4-Bit ResNet-50 auf ImageNet = 76.6% Top-1 (≈ FP32)
- **Verteilung**: nicht-uniform, passt gut zu glockenförmigen Gewichtsverteilungen

**Unterschied zu Fiver**: APoT ist eine Summe von PoT-Termen ohne festen Anker.
Fiver verwendet `w(k) = 1 ± 2^(-k/4)` — der Anker bei 1.0 ist bewusst gewählt
für Layer, deren Gewichte intrinsisch nahe 1.0 liegen (RMSNorm-Gains).

### 4.2 Compact Powers-of-Two (CPT, 2024)

**Quelle**: ResearchGate / arXiv 2024

Ähnliches Konzept wie APoT, speziell für Deep Neural Networks optimiert.
Keine bekannte Anwendung auf LLM-Layer-Typen dokumentiert.

### 4.3 Low-Bit PoT Quantization for LLMs (Wang, McGill 2024)

Masterarbeit, McGill University — direkte Anwendung von Power-of-Two-Quantisierung auf
LLMs. Bestätigt Lücke: kein etablierter Standard für PoT-basierte LLM-Quantisierung.

---

## 5. RWKV — Besonders geeignete Ziel-Architektur

### 5.1 Die time-decay W-Vektoren

In RWKV (alle Versionen 4–7) steuert ein trainierter Vektor W die Vergessensrate:

```
state_t = exp(-exp(W)) · state_{t-1}  +  K_t · V_t
```

W-Werte: real-wertig, typisch in [−5, 0], entspricht Decay-Rates in [0, 1).
Die Werte sind **vorzeichenlos bezüglich der Größe**, bounded, und folgen einer
Exponentialverteilung — perfektes Ziel für Fiver k ∈ [0..15] (untere Hälfte).

### 5.2 RWKVQuant (arXiv:2505.03803, 2025)

Neuestes Paper zur RWKV-Quantisierung:
- PTQ-Framework mit Proxy-gesteuerter Scalar/Vector-Quantisierung
- Ergebnis: RWKV-6-14B → **3-Bit mit <1% Accuracy-Verlust**, 2.14× Speed-up
- Getestete Bit-Breiten: **3 und 4 bit** — 5-Bit nicht untersucht

**→ Fiver bei 5-Bit für W-Vektoren ist unerforscht und wohlmotiviert.**

---

## 6. Einordnung des Fiver-Ansatzes

### 6.1 Speicherformat: Fiver-Word (3-per-uint16)

Drei k-Werte teilen sich ein 16-Bit-Wort mit **einem gemeinsamen Vorzeichenbit**:

```
┌────┬─────────┬─────────┬─────────┐
│ S  │   k2    │   k1    │   k0    │
│ 15 │ 14 – 10 │  9 – 5  │  4 – 0  │
└────┴─────────┴─────────┴─────────┘

S   = shared sign für alle drei Gewichte im Wort
      0 → alle positiv,  1 → alle negativ
k ∈ [0..31]:  w(k) = 1 ± 2^(−mag(k)/4)
```

Effektiver Speicherbedarf: **16 bit / 3 Werte = 5.333 bpw → 6.0× Kompression vs FP32**

Das gemeinsame VZ-Bit ist eine bewusste Approximation:
- **Unsigned Layer** (RMSNorm, RWKV-W): S = 0 immer → kein Informationsverlust
- **Signed Layer** (Embeddings, Projektionen): Mehrheitsentscheid über das Triplet;
  das Minoritäts-Element wird mit falschem Vorzeichen rekonstruiert

Rekonstruktion pro Wort:
```python
sign  = -1.0 if (word >> 15) else +1.0
k0    =  word        & 0x1F
k1    = (word >>  5) & 0x1F
k2    = (word >> 10) & 0x1F
w_i   = sign * scale * fiver_values[k_i]   # für i in {0, 1, 2}
```

### 6.2 Was Fiver anders macht

| Eigenschaft           | llama.cpp Q5_K      | APoT              | Fiver                          |
|-----------------------|---------------------|-------------------|-------------------------------|
| Eff. Bit-Breite       | 5.5 – 5.7 bpw       | 4 (primär)        | **5.333 bpw** (16/3)          |
| Kompression vs FP32   | 5.6×                | ~8×               | **6.0×**                      |
| Encoding              | Uniform + Scale     | Σ 2^(-i)          | 1 ± 2^(-k/4), monoton         |
| Anker                 | keiner              | keiner            | w(k=15)=1.0 explizit          |
| Unsigned-Modus        | nein                | nein              | ja (use_sign=False, S=0)      |
| Vorzeichen-Granularität | pro Block          | pro Wert          | **pro Triplet** (shared S)    |
| Ziel-Layer            | alle Linear         | CNN-Gewichte      | Embeddings, Norms, RWKV-W     |
| Hardware              | GGML/CPU            | ASIC-optimiert    | CUDA + CPU (diese Arbeit)     |
| GGUF-Integration      | ja (nativ)          | nein              | Q5_FIVER=0x0105 vorgeschlagen |

### 6.2 Wo Fiver am besten passt (nach Recherche)

1. **RMSNorm / LayerNorm Gains** (LLaMA, Mistral, Falcon)
   - Werte bei 1.0 ± kleines δ → Fiver-Mitte k=15 ist natürlicher Ruhepunkt
   - Typisch: d_model=4096 Werte × 32 Layer = 131K Parameter pro Modell
   - Einsparung: FP32 → 5-bit = **6.4× Kompression**

2. **Embedding-Gewichte** (embed_tokens, lm_head)
   - LLaMA-3.1-8B: 128K × 4096 = 524M Parameter = **2 GB in FP32**
   - Fiver-Word: ~333 MB (6.0× Kompression, 16/3 bpw)
   - Q5_K_M lässt diese Layer bei FP16 → Fiver füllt diese Lücke

3. **RWKV time-decay W-Vektoren**
   - Exponentialverteilung in [0,1) → untere Fiver-Hälfte (k ∈ [0..15])
   - RWKVQuant (2025) zeigt Quantisierbarkeit; 5-Bit nicht getestet

4. **Ausgabe-Projektionen** (o_proj, down_proj)
   - Post-Aktivierungs-Gewichte: oft vorzeichenlos oder schwach bipolar
   - llama.cpp quantisiert diese mit Q5_K_M; Fiver bietet strukturierte Alternative

---

## 7. Kompressionsraten im Überblick

```
Format        Bits    vs FP32   Ppl-Delta (7–8B Modell)
──────────────────────────────────────────────────────────
FP32          32      1.0×      Referenz (0.0)
FP16          16      2.0×      ~0.0
Q8_0           8.5   3.8×      ~0.0
Q6_K           6.6   4.8×      +0.16%
Q5_K_M         5.7   5.6×      +0.40%   ← Fiver-Zielbereich
Fiver-Word     5.333 6.0×      TBD (triplet-sign approx.)
Q4_K_M         4.9   6.6×      +1.20%
RTN W4A16      4.0   8.0×      +~2%
GPTQ W4A16     4.0   8.0×      +1.5%
AWQ W4A16      4.0   8.0×      +1.2%
QuIP# W3       3.0  10.7×      +~2% (besser als RTN)
SqueezeLLM W3  3.0  10.7×      +0.67 ppl abs.
```

**Fiver-Word positioniert sich zwischen Q5_K_M und Q4_K_M**: 6.0× Kompression
bei 5.333 bpw (16 bit / 3 Werte), mit word-aligntem Speicher für GPU-Effizienz.
Der Overhead gegenüber tight-packed 5.0 bpw beträgt nur 6.7% mehr Speicher
(5.333 vs 5.0 bpw), dafür sind Lesezugriffe natürlich wort-aligned.

---

## 8. Quellen

- [Which Quantization Should I Use? llama.cpp on Llama-3.1-8B (arXiv:2601.14277)](https://arxiv.org/html/2601.14277v1)
- [llama.cpp k-quants PR #1684 (ikawrakow)](https://github.com/ggml-org/llama.cpp/pull/1684)
- [llama.cpp quantize README](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md)
- [Perplexity Discussion #406 — llama.cpp](https://github.com/ggml-org/llama.cpp/discussions/406)
- [APoT: Additive Powers-of-Two Quantization (arXiv:1909.13144, ICLR 2020)](https://arxiv.org/pdf/1909.13144)
- [RWKVQuant: Proxy Guided Hybrid Quantization (arXiv:2505.03803)](https://arxiv.org/abs/2505.03803)
- [RWKV-LM GitHub — BlinkDL](https://github.com/BlinkDL/RWKV-LM)
- [SqueezeLLM: Dense-and-Sparse Quantization (ICML 2024)](https://arxiv.org/pdf/2306.07629)
- [QuIP#: LLM Quantization with Hadamard Incoherence (arXiv:2402.04396)](https://arxiv.org/pdf/2402.04396)
- [AWQ: Activation-aware Weight Quantization (arXiv:2306.00978)](https://arxiv.org/pdf/2306.00978)
- [MixLLM: Global Mixed-Precision Quantization (arXiv:2412.14590)](https://arxiv.org/pdf/2412.14590)
- [ParetoQ: Scaling Laws for Low-bit LLMs (arXiv:2502.02631)](https://arxiv.org/pdf/2502.02631)
- [Awesome-LLM-Quantization (pprp/GitHub)](https://github.com/pprp/Awesome-LLM-Quantization)
- [Quantizing Attention and Normalization Layers — apxml.com](https://apxml.com/courses/quantized-llm-deployment/chapter-5-addressing-advanced-challenges/quantizing-specific-llm-components)
- [Compact Powers-of-Two (CPT) — ResearchGate 2024](https://www.researchgate.net/publication/383130988_Compact_Powers-of-Two_An_Efficient_Non-Uniform_Quantization_for_Deep_Neural_Networks)
- [Low-Bit PoT Quantization for LLMs — McGill 2024](https://escholarship.mcgill.ca/downloads/w3763d65g)
