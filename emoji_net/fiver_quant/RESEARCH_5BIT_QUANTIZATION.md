# 5-Bit Quantization for LLMs — Research Report
*Recherche zum Stand der Technik und Einordnung des Fiver-Ansatzes*
*Erstellt: 2026-06-16 · Abschnitte 1, 6–8 überarbeitet nach Messung an SmolLM-135M*

---

## 1. Kurzfassung

5-Bit-Quantisierung existiert in der Praxis — vor allem in **llama.cpp (Q5_K_M)** — ist
aber in der akademischen PTQ-Literatur ein nahezu unbeschriebenes Blatt.
Forschung springt direkt von 4-Bit zu sub-3-Bit-Verfahren.
Das ist die Lücke, in die **Fiver** passt.

Die Messung an echten Gewichten (Abschnitt 6) hat die ursprüngliche Annahme
korrigiert: Große vorzeichenlose Layer gibt es in LLMs nicht — die
Gewichtsmatrizen sind nullzentrierte Glockenkurven. Fiver um 0 zentriert
(`v = ∓2^(−m/4)`, Code 16 = 0, Vorzeichen im MSB) trifft diese Verteilung
und ist mit 5,333 bpw gleichauf mit uniformem int5; das ursprüngliche,
bei 1 verankerte Gitter bleibt für positive Vektoren wie RMSNorm γ.

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

**→ Fiver bei 5-Bit für W-Vektoren ist unerforscht** — aber die Vektoren sind klein
(n_embd pro Layer), die Speicherersparnis ist daher gering (siehe Abschnitt 6).

---

## 6. Messung an echten Gewichten (SmolLM-135M)

Modell: `HuggingFaceTB/SmolLM-135M` (LLaMA-Architektur, 30 Layer, RMSNorm).
Werkzeug: `analyze_weights.py`. Fehler = relativer RMS-Fehler ‖x − q‖ / ‖x‖.

### 6.1 Verteilung

| Tensor | Mittel/σ | negativ | Kurtosis (Gauß 3) | max/σ |
|---|---|---|---|---|
| L15 q / gate / down_proj | ≈ 0 | 50 % | 4–5 | 11–13 |
| L15 v_proj | ≈ 0 | 50 % | 3,3 | 5 |
| L0 q_proj | ≈ 0 | 50 % | 55 | 52 |
| embed_tokens | ≈ 0 | 51 % | 15 | 21 |
| RMSNorm γ | 3–4,6 | 0 % | – | – |

Lehrbuch bestätigt: nullzentriert, im Kern gaußförmig, leicht schwere Ränder,
einzelne starke Ausreißer (Layer 0, Embedding). Vorzeichen in Triplets sind
exakt zufällig verteilt (25 % gleiches Vorzeichen, wie bei Zufall).
Vorzeichenlose Tensoren (RMSNorm) sind ~0,03 % der Parameter.

### 6.2 Ergebnisse

| Variante | Bit/Gewicht | L15 gate | L29 down | L0 q (Ausreißer) |
|---|---|---|---|---|
| Fiver original 1 ± 2^(−m/4), VZ pro Triplet | 5,33 | **100 %** | 100 % | – |
| Fiver original + VZ pro Gewicht | 6,4 | 5,4 % | 6,4 % | 12,7 % |
| **Fiver zentriert, Code 16 = 0** | **5,33** | **6,1 %** | **6,9 %** | 12,6 % |
| Fiver zentriert ohne Null-Code | 5,33 | 7,7 % | 8,8 % | 21 % |
| uniform int5 | 5,33 | 6,2 % | 7,0 % | 11,9 % |
| Fiver zentriert 6 Bit (±2^(−m/4), m < 32) | 6,4 | 4,8 % | 4,9 % | **4,6 %** |
| uniform int6 | 6,4 | **3,2 %** | **3,7 %** | 6,3 % |

Skalierung (Fiver 5 Bit + VZ, L15 gate): max pro Tensor 29 % · 2^E pro Tensor
8,9 % · 2^E pro Zeile 7,0 % · float pro Zeile 5,4 %. Den Gewinn bringt die
MSE-optimale Wahl; die 2^n-Einschränkung kostet pro Zeile 15–35 % mehr Fehler
und spart keine Rechenzeit (Skala einmal pro Ausgabewert).

6 Bit (anchored, δ/16): RMSNorm L29 von 3,5 % auf 0,46 %; Projektionen nur ~3 %.
Die frühere Idee „Bit 5 = δ/4“ liefert nur 48 verschiedene Werte.

---

## 7. Gewähltes Format: Fiver zentriert

```
v(k) = −2^(−k/4)        k =  0..15     (−1.000 … −0.074)
v(16) = 0
v(k) = +2^(−(31−k)/4)   k = 17..31     (+0.088 … +1.000)
x[r, c] = scale[r] · v(k)              eine float-Skala pro Zeile (MSE-optimal)

uint32: │ r │ r │ k5 │ k4 │ k3 │ k2 │ k1 │ k0 │   k_i an Bit 5i, Bits 30-31 reserviert
        6 Codes pro Wort → 5,333 bpw + 32 Bit pro Zeile; Zeilen mit Code 16 aufgefüllt
```

- Vorzeichen = MSB von k, kein Vorzeichen-Bit, kein geteiltes Vorzeichen.
- Streng monoton in k: STE-Update k ∓ 1, Vorzeichenwechsel über Code 16.
- int8: e = rn(127·v) (−127 … −9, 0, 11 … 127, alle verschieden); `__dp4a(E, Xq, acc)`
  ohne Σx-Term, eine MMA auf den Tensor Cores.
- Das ursprüngliche Gitter bleibt als `anchored` (`w = 1 ∓ 2^(−m/4)`, Code 0 = 0)
  für positive Tensoren mit Mittel > σ; `--grid auto` probiert dann beide.
  int8: w = 1 + e/127, der Σx-Term ist eine Zahl pro Eingabezeile.
- GGUF-Typ 0x0105 (centered) / 0x0106 (anchored).

## 7.1 Offene Optionen

- 6-Bit-Codes (5 pro uint32, 6,4 bpw): lohnt bei Ausreißer-Tensoren (L0) und Norm-Vektoren.
- Gitter pro Tensor nach Kurtosis wählen (uniform bei Gauß, Fiver-Log bei Ausreißern):
  die Kernels tauschen nur die 32-Werte-Tabelle.
- Blockskalen (z. B. 32 Gewichte wie Q5_K) statt Zeilenskalen: nicht gemessen.
- Perplexity-Messung mit Forward-Pass steht aus; alle Zahlen hier sind Gewichtsraum-Fehler.

---

## 8. Quellen

- [SmolLM-135M — Hugging Face](https://huggingface.co/HuggingFaceTB/SmolLM-135M) (Messgrundlage Abschnitt 6)

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
