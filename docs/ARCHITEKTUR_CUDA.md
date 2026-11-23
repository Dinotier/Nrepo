# Fiver CUDA — Architektur für den Um- und Neubau

*Stand: Branch `claude/compact-weight-format-nKv8j`, `compact_weights.cu` mit drei Gittern
(centered / anchored / magic). Ziel-Hardware zuerst: GTX 1080 Ti (sm_61), danach Turing+.*

> **Wichtig vorab:** Keine Zahl in diesem Dokument ist auf einer GPU gemessen. Der bisherige
> Code wurde nur kompiliert (CUDA 12.0) und auf dem Host geprüft. Leistungsangaben sind
> Roofline-Abschätzungen oder Zielwerte und als solche markiert. Jede Phase im Migrationsplan
> (Abschnitt 10) endet deshalb mit einem Messlauf auf echter Hardware.

---

## 1. Ausgangslage in einem Absatz

Das Format steht und ist geprüft: 6 Codes à 5 Bit pro `uint32`, eine float-Skala pro Zeile,
drei Wertegitter hinter einer 32-Einträge-Tabelle. Die Kernels sind fusioniert, die Gewichte
bleiben also gepackt. Sie sind aber als *Proof of Concept* gebaut: eine Datei mit Format,
Kernels, Tests, Benchmarks und `main`, ein Warp pro Ausgabezeile, keine Kachelung, keine
Bibliothek und keine Anbindung an Python. Das EmojiNet-Training läuft deshalb weiter in numpy
auf der CPU. Der Neubau soll daraus eine Bibliothek machen, die ein Trainingsschritt
tatsächlich auf der GPU ausführt und dabei nahe an die Hardwaregrenzen kommt.

---

## 2. Hardware-Grenzen (Roofline)

| | GTX 1080 Ti (sm_61) | Anmerkung |
|---|---|---|
| DRAM | 484 GB/s, 11 GB | |
| FP32 | ≈ 11,3 TFLOP/s | 28 SM × 128 Lanes × 1,58 GHz × 2 |
| INT8 `__dp4a` | ≈ 45 TOP/s | 4 MACs pro Lane und Befehl |
| FP16-Arithmetik | 1/64 von FP32 | auf Pascal unbrauchbar, nur als Speicherformat |
| Tensor Cores | keine | ab sm_75 (INT8), sm_89 (FP8) |
| Shared Memory | 96 KB/SM, 48 KB/Block | |
| L2 | 2,75 MB | |

**Arithmetische Intensität** einer Fiver-Schicht mit Batch B: pro Gewicht 0,67 Byte gelesen,
2·B FLOP gerechnet → **3·B FLOP/Byte**.

| Bereich | FP32-Pfad | `__dp4a`-Pfad |
|---|---|---|
| speichergebunden bis | B ≈ 8 (Knick 23 FLOP/B) | B ≈ 31 (Knick 93 OP/B) |
| Konsequenz | GEMV bis Batch ~8 | INT8 lohnt ab Batch ~8, bis ~31 sogar bis zu 4× |

Daraus folgen **zwei Regime**, die getrennte Kernels brauchen:
- **GEMV** (B = 1…8): Inferenz, LLM-Token für Token. Hier zählt nur die Bandbreite.
- **GEMM** (B ≥ 16): Training (EmojiNet mit Batch 32), Prefill. Hier zählt die Rechenleistung,
  und das Entpacken muss sich auf viele Batch-Spalten verteilen.

---

## 3. Bestandsaufnahme

| Baustein | heute | Problem | Folge |
|---|---|---|---|
| Struktur | 1 Datei, 1 Executable | keine Bibliothek, keine C-Schnittstelle | Python kann nichts aufrufen → kein GPU-Training |
| GEMV float | Warp pro Zeile, BT = 4, x per `__ldg` mit 6 Einzel-Loads je Wort | pro Wort 7 Loads + 6 Shuffles über dieselbe Load/Shuffle-Einheit: grob 15 Gewichte/Takt/SM ≈ 650 G/s, also knapp *unter* der DRAM-Grenze (718 G/s) | vermutlich Load-Issue-gebunden statt bandbreitengebunden (Schätzung, mit Nsight prüfen) |
| GEMV Batch > 4 | Grid-y über Batch-Kacheln | W wird pro 4 Batch-Zeilen neu gelesen und neu entpackt | bei Batch 64: 16× W-Traffic |
| `__dp4a` | wie GEMV, 2 dp4a/Wort | x-Quantisierung als eigener Kernel bei jedem Aufruf | 2–3 Extra-Launches pro Schicht |
| Tensor Cores | WMMA, 1 Warp pro 16×16-Kachel, Byte-Stores in Shared Memory | Entpacken (256 Wörter) dominiert gegenüber 6 MMAs, keine Latenzverdeckung, kein Double Buffering | TCs größtenteils ungenutzt |
| dX (Backward) | Thread pro Wort, atomicAdd über 64er-Zeilenstreifen | nicht deterministisch, viele Atomics | Reproduzierbarkeit fehlt |
| STE-Update | Warp pro 4 Zeilen, Gradient per skalarer FMA | g = dYᵀX ist eine GEMM (rows×cols×batch), läuft aber ohne Kachelung, jedes X-Element nur 4× genutzt | weit unter FP32-Peak (Schätzung: < 10 %) |
| Epiloge | Skala im Kernel, sonst nichts | Bias, Aktivierung, Residual extra | zusätzliche Speicherpässe |
| act_bits | 2 Bit Aktivierung pro Gewicht | bricht die int8-Zerlegung, kostet Loads | Sonderweg in jedem Kernel → künftig nur float-GEMV (11.2) |
| Tests | im Executable, Host-Referenz in double | gut, aber nur manuell auf GPU ausführbar | kein CI |

**Was bleibt, weil es sich bewährt hat:**
- Das Format (Wort-Layout, Zeilen-Padding mit Null-Code, Zeilenskala).
- Das Entpacken per Register-Shuffle (32 Codes = 32 Lanes).
- Die int8-Tabellen und die exakte Integer-Prüfung in den Tests.

---

## 4. Zielarchitektur

```
                ┌──────────────────────────────────────────────────────────┐
  C# Controller │ benchA/benchB → compact_weights_test (unverändert)       │
                │ pyTorch_nv   → python train.py --backend cuda (existiert) │
                └───────────────┬──────────────────────────────────────────┘
                                │
                ┌───────────────▼──────────────────────────────────────────┐
  Python        │ fiver.py  FiverLayer(backend="numpy" | "cuda")            │
                │ fiver_cuda.py  ctypes-Wrapper, Handles, Device-Puffer     │
                └───────────────┬──────────────────────────────────────────┘
                                │  C-ABI (extern "C")
                ┌───────────────▼──────────────────────────────────────────┐
  libfiver_cuda │ api.cpp        Handles, Streams, Workspace, Fehlercodes   │
                │ dispatch.cpp   (Arch, Batch, Shape) → Kernel-Variante     │
                │ ┌────────────┬──────────────┬─────────────┬────────────┐ │
                │ │ gemv.cu    │ gemm_*.cu    │ backward.cu │ quant.cu   │ │
                │ └────────────┴──────────────┴─────────────┴────────────┘ │
                │ include/fiver/format.h   Format, Gitter, Tabellen, Packen │
                └──────────────────────────────────────────────────────────┘
```

**Verzeichnisstruktur:**
```
cuda/
  include/fiver/format.h      header-only, Host + Device + reines C (auch für CPU-Backend)
  include/fiver/fiver.h       C-ABI
  include/fiver/compat.h      wählt cuda_runtime.h oder den CPU-Emulator
  include/fiver/cuda_emu.h    CPU-Emulator für Tests ohne GPU (Abschnitt 9)
  src/api.cpp  src/dispatch.cpp
  src/gemv.cu  src/gemm_sm61.cu  src/gemm_mma.cu  src/quant.cu  src/backward.cu
  tests/test_main.cu          + gemeinsame CPU-Referenz (tests/reference.c)
  bench/bench_main.cu
compact_weights.cu            dünnes CLI: --bench-a / --bench-b / --test / --bench-*
                              (Flags bleiben, C# muss nichts ändern)
bench/cpu_fiver_decode.c      CPU-Messung (Abschnitt 8)
```

`format.h` ist die einzige Quelle für das Format. Die Python-Seite (`fiver_core.py`) wird in
den Tests gegen dieselben Tabellen geprüft. Das macht `xcheck` heute schon, künftig fest im
Test-Target.

---

## 5. Kernel-Design

### 5.1 Forward GEMV (Batch 1–8)

- **CTA:** 4 Warps, jeder Warp 2–4 Ausgabezeilen (R). Das wird pro Arch per Benchmark gewählt.
- **x in Shared Memory:** einmal pro CTA in Spaltenblöcken von z. B. 3072 floats (12 KB), mit
  `float4`-Loads. Alle R × 4 Zeilen lesen x danach aus Shared Memory statt aus L1. Das senkt
  die Load-Befehle pro Gewicht von heute 1 auf etwa 1/4 (Lesen per `LDS.128`).
- **Gewichte:** pro Lane ein `uint4` = 4 Wörter = 24 Gewichte, ein Warp-Request = 512 Byte.
- **Entpacken:** unverändert per Register-Shuffle. Die Tabelle wird pro Kernel einmal geladen.
- **Epilog:** Zeilenskala, Bias, Aktivierung (GELU, Softsign, Identität) und optional
  int8-Quantisierung der Ausgabe für die nächste Schicht (5.3).
- **Ziel:** ≥ 80 % der DRAM-Bandbreite bei B = 1, also ≥ 570 G Gewichte/s auf der 1080 Ti.

### 5.2 Forward GEMM (Batch ≥ 16)

Gemeinsames Muster für alle Archs: **eine CTA-Kachel W (M×K) wird einmal entpackt und von
allen N Batch-Spalten der Kachel genutzt.** Das Entpacken verteilt sich auf N statt auf 4.

| Arch | Kachel (M×N×K) | Entpacken nach | Rechenwerk | Laden |
|---|---|---|---|---|
| sm_61 FP32 | 128×64×96 | Register (float) | FFMA, 8×8 pro Thread | `ld.global` + Double Buffer in Shared Memory |
| sm_61 INT8 | 128×64×96 | Shared Memory (int8 e) | `__dp4a`, 8×4 pro Thread | wie oben, x vorquantisiert |
| sm_75 | 128×64×96 | Shared Memory (int8 e) | `mma.sync m8n8k16.s8` | `ldmatrix` |
| sm_80/86 | 128×128×96 | Shared Memory (int8 e / fp16) | `mma.sync m16n8k32.s8` | `cp.async`, 3 Stufen |
| sm_89/90 | wie sm_80 | Shared Memory (FP8) | FP8-MMA | `magic`-Gitter ist in FP8 exakt (nur 2 Mantissenbits) |

- **K = 96:** 16 Wörter pro Zeile, damit fallen Kachelgrenzen auf Wortgrenzen. Das ist auch
  die Bedingung für das spätere In-place-Update in 5.6.
- **Entpacken in Shared Memory:** konfliktfrei mit r = idx % 16 (heute schon so). Dazu
  32-Bit-Stores statt Byte-Stores: 4 int8-Codes zusammenpacken, dann ein Store.
- **Ziel sm_61:** FP32 ≥ 60 % Peak, `__dp4a` ≥ 50 % Peak bei B = 64…256.

### 5.3 Aktivierungs-Quantisierung (INT8-Pfade)

Heute läuft vor jeder int8-Schicht ein eigener Kernel: Maximum pro Zeile, quantisieren,
Triplets packen. Neu:
- Der **Epilog der vorherigen Schicht** schreibt direkt int8 + `sx` + `Σxq`.
- Das Maximum pro Zeile braucht dafür eine Reduktion über alle CTAs der Zeile: Teilmaxima pro
  CTA, dann ein kleiner Abschluss-Kernel, oder dynamisch pro Kachel mit Kachelskala statt
  Zeilenskala. **Offen, hängt an 12.1 und 12.2.**
- Bei LayerNorm davor (EmojiNet) ist ein fester Bereich nach der Norm eine einfache Alternative.

### 5.4 Epilog-Fusion

Ein Epilog-Template mit Bausteinen, der Reihe nach: Skala → Bias → Aktivierung → Residual →
Ausgabeformat (fp32 / int8 + Skala). Spart pro Schicht 1–3 Speicherpässe über die Ausgabe.

### 5.5 Backward dX = Wᵀ · (s ∘ dY)

- **Dieselbe Kachel-Maschinerie wie 5.2,** nur in transponierter Rolle: Ausgabe über Spalten c,
  Reduktion über Zeilen r. Die W-Kachel (r-Block × c-Block) wird genauso entpackt.
- **Keine Atomics:** Die Reduktion über r läuft innerhalb der CTA. Bei sehr vielen Zeilen kommt
  Split-K mit fester Reihenfolge dazu (Teilsummen in einen Puffer, zweiter Kernel summiert
  deterministisch).

### 5.6 Backward Gewichts-Gradient + STE (fusioniert)

g = dYᵀ · X ist eine GEMM mit **K = Batch** (16…256). Sie wird gekachelt wie 5.2:
- **CTA-Kachel:** 64 Zeilen × 96 Spalten (16 Wörter), K-Schleife über den Batch.
- **Akkumulation:** in Registern bzw. MMA-Fragmenten. Auf Turing+ reicht fp16/bf16-MMA, weil nur
  das **Vorzeichen** von g gebraucht wird. Fehlerhaft ist es nur dort, wo |g| ≈ 0 ist, und dort
  ist das Vorzeichen sowieso zufällig.
- **Epilog:** die 16 gepackten Wörter jeder Zeile der Kachel lesen, k ∓ 1 sättigend anwenden
  (Padding-Spalten nicht anfassen), zurückschreiben. **Die Gradient-Matrix existiert nie im
  Speicher,** wie heute, nur jetzt mit Kachel-Wiederverwendung von X und dY.
- **Bias-Gradient:** Σ_b dY im selben Kernel (Zeilenblock pro CTA, kein Atomic, weil die
  Zeilen-CTAs disjunkt sind; Split über Spalten muss die Bias-Summe einem Block zuordnen).
- **Ziel:** ≥ 40 % FP32-Peak auf sm_61; Backward-Schritt ≤ 3× Forward-Zeit.

### 5.7 Determinismus

Alle Standardpfade sind deterministisch, ohne Atomics auf float. Optionale schnellere Pfade
mit Atomics bekommen ein Flag.

---

## 6. Host-Seite

- **Handles:** `fiver_matrix_t` (Device-Wörter, Zeilenskalen, Gitter, Form) und
  `fiver_ctx_t` (Stream, Workspace-Pool, Arch-Info, Dispatch-Tabelle).
- **Workspace:** einmal pro Kontext, wächst bei Bedarf. Kein `cudaMalloc` im Hot Path.
- **Fehler:** alle API-Funktionen geben `fiver_status_t` zurück, kein `exit()` in der
  Bibliothek. `CW_CUDA_CHECK` mit `exit` bleibt nur im CLI und in den Tests.
- **CUDA Graphs:** Ein Trainingsschritt (Forward aller Schichten, Backward, Update) wird nach
  dem ersten Lauf als Graph aufgezeichnet. EmojiNet hat 4 Fiver-Schichten, also ≈ 12 Fiver-Kernels
  pro Schritt plus Norm und Loss. Bei kurzen Kernels ist die Launch-Latenz (einige µs je Launch)
  ein merklicher Anteil, den Graphs größtenteils einsparen.
- **Dispatch:** Tabelle (Arch, Batch-Bereich, Gitter) → Kernel-Variante. Erster Wurf per
  Heuristik, später per Autotuning beim ersten Aufruf mit Cache-Datei.

**C-ABI (Entwurf):**
```c
fiver_status_t fiver_ctx_create(fiver_ctx_t** ctx, int device);
fiver_status_t fiver_matrix_upload(fiver_ctx_t*, const uint32_t* words, const float* row_scale,
                                   uint32_t rows, uint32_t cols, int grid, fiver_matrix_t** out);
fiver_status_t fiver_matrix_download(fiver_ctx_t*, const fiver_matrix_t*, uint32_t* words);
fiver_status_t fiver_forward(fiver_ctx_t*, const fiver_matrix_t*, const float* x, float* y,
                             uint32_t batch, const fiver_epilogue_t* epi);
fiver_status_t fiver_backward_step(fiver_ctx_t*, fiver_matrix_t*, const float* x, const float* dy,
                                   float* dx /*nullable*/, float* bias /*nullable*/,
                                   uint32_t batch, float lr);
```

---

## 7. Anbindung an Python und C#

- **Python:** `fiver_cuda.py` lädt `libfiver_cuda.so` per `ctypes`. Ohne Build-Abhängigkeit
  und ohne PyTorch. `FiverLayer(backend="cuda")` hält die Matrix dauerhaft auf der GPU.
  - **Phase 1:** Nur die Fiver-Schichten laufen auf der GPU. Aktivierungen werden pro Schicht
    kopiert, bei EmojiNet sind das Batch × 512 floats, also klein. LayerNorm, Loss und Attention
    bleiben vorerst in numpy.
  - **Später:** Aktivierungen bleiben auf dem Gerät, dafür kommen Device-Puffer-Handles in die
    Python-API.
- **C#:** `pyTorch_nv` ruft heute schon `train.py --backend cuda` auf (`pyTorch_i86` →
  `cpu`, `pyTorch_risc` → `mps`). `train.py` loggt den Wert bisher nur. In P1 wählt
  `--backend cuda` die neue Bibliothek, `benchA/benchB` bleiben beim CLI-Binary mit denselben
  Flags. **Der Controller muss nicht geändert werden.**
- **Alternative:** eine PyTorch-Extension (`torch.autograd.Function`). Die Namen
  `pyTorch_nv/i86/risc` legen das nahe, der Python-Code nutzt PyTorch aber bisher nicht.
  **Entschieden: ctypes (11.1).**

---

## 8. Die drei Backends und das Entpacken

### 8.1 Gemessen: CPU (`bench/cpu_fiver_decode.c`)

Emerald-Rapids-Xeon (Cloud-VM, 1 Kern, Takt unbekannt, Läufe schwanken um ±20 %), GEMV-Zeile
4608 Spalten, `magic`-Gitter, ganze SIMD-Blöcke:

| Variante | Befehle pro Block | llvm-mca Takte/Gewicht (Skylake … Zen 4) | gemessen ns/Gewicht (0,8 MB / 25 MB) |
|---|---|---|---|
| AVX2 LUT (4× `vpermps` + Blends) | 40 / 24 Gew. | 0,75 – 0,88 | 0,31 / 0,35 |
| AVX2 LUT per Gather | 25 / 24 Gew. | Intel 1,13 – 1,55 (Zen 0,2, vermutlich unrealistisch) | 0,23 / 0,28 |
| **AVX2 magic per Formel** | 46 / 24 Gew. | **0,59 – 0,69** | **0,24 / 0,25** |
| **AVX-512 LUT (1× `vpermt2ps`)** | **13 / 48 Gew.** | **0,13 – 0,20** | **0,05 / 0,05** |
| AVX-512 LUT per Gather | 22 / 48 Gew. | 0,56 – 0,65 (Zen 0,19) | 0,20 / 0,22 |
| AVX-512 magic per Formel | 43 / 48 Gew. | 0,44 – 0,45 | 0,16 / 0,17 |

**Zweite Messung, Granite Rapids** (Xeon 6, Cloud-VM, 1 Kern gebunden, 7 Durchläufe, Median
und Streuung max−min, ns pro Gewicht):

| Variante | 0,8 MB (L2) | Streuung | 25 MB (L3/RAM) | Streuung |
|---|---|---|---|---|
| AVX2 LUT per Permute | 0,226 | 4 % | 0,444 | 12 % |
| AVX2 LUT per Gather | 0,169 | 6 % | 0,377 | 20 % |
| AVX2 magic per Formel | 0,174 | 10 % | 0,412 | 20 % |
| AVX-512 LUT `vpermt2ps` | **0,038** | 34 % (1 Ausreißer) | **0,055** | 20 % |
| AVX-512 LUT per Gather | 0,152 | 20 % | 0,337 | 60 % |
| AVX-512 magic per Formel | 0,130 | 3 % | 0,400 | 28 % |

Gleiche Rangfolge wie auf Emerald Rapids: AVX2 Formel vor Permute-Tabelle (−23 % im L2-Fall),
AVX-512 Tabelle 3,4–7× vor der Formel. Gather ist auf beiden neuen Xeons schnell, auf älteren
Kernen (bis Ice Lake, Downfall-Mitigation) und auf AMD nicht. Die 25-MB-Werte streuen stark,
vermutlich durch L3/RAM, den sich die VM mit anderen teilt.

- **AVX2:** Die Formel braucht *mehr* Befehle als die Tabelle, ist aber schneller. Die
  32er-Tabelle braucht 15 Shuffles pro Block, und die laufen auf Intel alle über Port 5. Die
  Formel verteilt sich auf die normalen ALUs. **→ AVX2-Backend entpackt `magic` per Formel.**
- **AVX-512:** `vpermt2ps` liest genau die Bits 4:0, also den 5-Bit-Code, und schlägt alle 32
  Einträge in einem Befehl nach. Das ist 3× schneller als die Formel. **→ AVX-512 per Tabelle.**
- **Gather:** ein Befehl, aber stark CPU- und Microcode-abhängig. Auf Skylake bis Ice Lake hat
  Intel Gathers per Downfall-Mitigation deutlich verlangsamt. **→ nicht als Standard.**
- **CPU-spezifisch:** Auf der CPU ist das Entpacken *der* Engpass. AVX2 liest nur 2–3 GB/s
  gepackt, weit unter der Speicherbandbreite. Deshalb zählt der Entpack-Weg dort, anders als bei
  GPU-GEMV.

### 8.2 Empfehlung je Backend

| Backend | Entpacken | int8-Pfad | Bemerkung |
|---|---|---|---|
| `nv` (CUDA) | Register-Shuffle-LUT, für alle Gitter | `__dp4a` / IMMA, `magic` exakt mit Skala 64 | Formel-Variante nur als Benchmark (`--bench-gemv`) |
| `i86` AVX-512 | `vpermt2ps` | VNNI `vpdpbusd`, AMX bei großem Batch | |
| `i86` AVX2 | `magic`-Formel | `vpshufb` (2× 16 Byte) + `vpmaddubsw` | `magic` als Standardgitter für AVX2-Rechner |
| `risc` ARM NEON | Formel (float) | `vqtbl2q_s8`: 32-Byte-Tabelle = genau 32 Codes, dann `sdot` | nicht gemessen |
| `risc` FPGA | Formel als Shifter + Subtrahierer | Gewicht · x als Exponenten-Addition (nur `magic`) | kein Tabellenspeicher, kein Multiplizierer nötig |

Weil alle Backends dasselbe `magic`-Gitter bitgleich erzeugen, kann ein auf der GPU trainiertes
Modell ohne Umrechnung auf CPU oder FPGA laufen.


### 8.3 Warum 5 Bit: die Kosten des Entpackens nach Codebreite

32 Einträge sind auf beiden Plattformen die größte Tabelle, die **ein** Befehl aus Registern
nachschlägt: ein Warp-Shuffle über 32 Lanes (GPU), `vpermt2ps` über 2 × 16 floats (AVX-512).

| Codebreite | AVX-512 gemessen, ns/Code (Granite Rapids) | GPU sm_61, Befehle für 8 Codes (SASS) |
|---|---|---|
| 4 Bit | 0,027 (1 × `vpermps`) | 46 (8 Shuffles) |
| **5 Bit** | **0,027** (1 × `vpermt2ps`) | **46** (8 Shuffles) |
| 6 Bit | 0,053 (2 × `vpermt2ps` + Blend) | 76 (16 Shuffles + Auswahl) |
| 8 Bit | 0,151 (Gather) | 64 (Shared Memory, plus Bank-Konflikte, Füllschleife, Barriere) |
| int5 gleichmäßig, ohne Tabelle | 0,035 | 52 |

- 5 Bit ist die **größte** Breite mit Kosten von einem Befehl, 4 Bit ist genauso billig.
  Ab 6 Bit verdoppelt sich der Aufwand auf beiden Plattformen.
- Die Tabelle macht ein frei geformtes Gitter (Fiver) so billig wie ein gleichmäßiges
  int-Gitter, nicht billiger.
- Die Grenze gilt für float-Tabellen. Hält die Tabelle int8-Werte (`__dp4a`-Pfad), reichen
  `vpermi2b` (AVX-512-VBMI, 128 Bytes) bzw. Shuffle + `PRMT` (GPU, 4 Bytes pro Lane) bis 7 Bit
  (nicht gemessen).
- 6 Codes à 5 Bit füllen ein 32-Bit-Wort bis auf 2 Bit (6 % Verschnitt).
- Messprogramm: `bench/cpu_lut_width.c`.
**int8-Tabellen: `vpermb` / `vpermi2b` (CPU) und Shuffle + `PRMT` (GPU)** — gemessen bzw. gezählt:

| Tabelle | AVX-512-VBMI, Granite Rapids | GPU sm_61, Shuffles pro Code |
|---|---|---|
| 6 Bit (64 int8) | `vpermb` 0,0086 ns/Code (116 G/s) | 1 |
| 7 Bit (128 int8) | `vpermi2b` 0,0115 ns/Code (87 G/s) | 1 (Lane hält 4 Bytes) |
| 8 Bit (256 int8) | 2 × `vpermi2b` + Blend 0,0202 ns/Code | 2 |
| Fiver 5 Bit aus gepackten Wörtern → int8 | `vpmovzxdq` + `vpmultishiftqb` + `vpermi2b`: 0,0227 ns/Code (44 G/s) | 1 |

- Für int8-Werte liegt die Ein-Befehl-Grenze bei 7 Bit statt 5.
- `vpmultishiftqb` zieht die 6 Codes eines Worts in einem Befehl an ihre Byte-Positionen. Die
  32er-Tabelle steht viermal wiederholt in den 128 Bytes von `vpermi2b`, damit stören die
  mitgezogenen Nachbarbits nicht: 3 Befehle für 48 Codes.
- GPU: nvcc erzeugt `PRMT` für Byte-Packen schon selbst. Explizites `__byte_perm` spart bei 7/8 Bit
  ~8 % Befehle (44 statt 48, 63 statt 68), bei 5 Bit kostet es mehr (55 statt 42), weil die
  Selektoren zur Laufzeit entstehen. Der `__dp4a`-Kernel bleibt bei Shift/Or.
- Option, keine Empfehlung: Ein int8-Gitter mit 6 oder 7 Bit wäre gleich billig zu entpacken, kostet
  aber 6,4 bzw. 8 Bit pro Gewicht (5 bzw. 4 Codes pro Wort).
- Programme: `bench/cpu_int8_lut.c` (CPU, mit Korrektheitsprüfung), `cuda/bench/prmt_lut.cu`
  (`build/prmt_lut_test`, Korrektheit auf GPU oder im Emulator).
**fp16-Tabellen für den float-Pfad: `vpermw` / `vpermi2w` + `vcvtph2ps`** (AVX512BW, ohne VBMI;
Codes als 16-Bit-Lanes, Granite Rapids, ns/Code):

| Breite | float32-Tabelle | fp16-Tabelle |
|---|---|---|
| 5 Bit | **0,0351** (`vpermt2ps`) | 0,0416 (`vpermw` + 2 × `vcvtph2ps`) |
| 6 Bit | 0,0706 (2 × `vpermt2ps` + Blend) | **0,0494** (`vpermi2w` + 2 × `vcvtph2ps`) |
| 7 Bit | — | 0,0721 (2 × `vpermi2w` + Blend + Umwandlung) |

- 5 Bit bleibt bei float32. Für 6 Bit im float-Pfad ist fp16 30 % billiger, 7 Bit gehört in den
  int8-Pfad (`vpermi2b`).
- `magic` kommt über fp16 bitgenau an, `centered` rundet um höchstens 2,1e-4 relativ.
- GPU-Gegenstück (nicht getestet): zwei fp16 pro Lane, also 64 Einträge pro Shuffle, danach eine
  Umwandlung.
- Programm: `bench/cpu_fp16_lut.c` (mit Korrektheits- und Exaktheitsprüfung).
**Direkt in fp16 weiterrechnen** (AVX512-FP16 / -BF16, Skalarprodukt über 4608 Gewichte, 5-Bit
`magic`, L2-resident, Median aus 9 Läufen, Streuung < 5 %):

| Variante | ns/Gewicht | vs. fp32 | Fehler (rel. zu Σ\|w·x\|) |
|---|---|---|---|
| fp32: `vpermt2ps` + `vfmadd231ps` | 0,0388 | 1,00× | 9e-9 |
| **fp16: `vpermw` + `vfmadd231ph`** | **0,0208** | **1,87×** | 5,9e-5 |
| fp16, alle 8 FMAs in fp32-Summe | 0,0236 | 1,64× | 2,5e-5 |
| bf16: `vpermw` + `vdpbf16ps` | 0,0326 | 1,19× | 6,5e-5 |

- `vpermw` liefert 32 fp16-Werte, `vfmadd231ph` rechnet 32 FMAs: gleiche Breite, keine Umwandlung.
- Der Fehler (Rundung von x auf fp16 und fp16-Summe) liegt weit unter dem Quantisierungsfehler der
  5-Bit-Gewichte (~6 %). Für Inferenz unkritisch, für Training (dX, Gewichts-Gradient) fp32 behalten.
- GPU-Gegenstück: `half2` pro Lane (64 Einträge pro Shuffle) + `HFMA2`. Auf der 1080 Ti nutzlos
  (fp16 mit 1/64 der FP32-Rate), ab Turing doppelt so schnell wie FP32. Nicht getestet.
- Programm: `bench/cpu_fp16_compute.c`.
---

## 9. Test- und Messstrategie

- **Referenz:** eine gemeinsame CPU-Referenz in C (Format, GEMV, GEMM, Backward, STE) für die
  CUDA-Tests und den Python-Abgleich. Die heutigen Host-Referenzen wandern dorthin.
- **Korrektheit:**
  - Integer-Pfade bitgenau.
  - Float-Pfade ≤ 1e-5 relativ zu Σ|w·x|.
  - STE-Update exakt bis auf die heute schon gezählten „ambiguous“-Fälle mit |g| ≈ 0.
- **Formen:** ungerade Spaltenzahlen (Padding), Batch-Reste, Zeilen < Kachel, cols > 65536
  (INT8-Grenze), leere Zeilen (Skala 0).
- **Gitter:** jeder Pfad (GEMV, GEMM float/int8/MMA, dX, Update) × jedes Gitter (`centered`,
  `anchored`, `magic`) ist ein eigener Testfall (Entscheidung 11.3).
- **Werkzeuge:** `compute-sanitizer --tool memcheck|racecheck` in jeder Phase, Nsight Compute
  für die Roofline-Kennzahlen (`dram__throughput`, `sm__inst_executed_pipe_*`).
- **CPU-Emulator (`cuda/include/fiver/cuda_emu.h`):** Dieselben Kernel-Quellen laufen mit
  `-DFIVER_EMU` auf der CPU. Jeder CUDA-Thread ist ein echter Thread, `__syncthreads` und
  `__syncwarp` sind Barrieren, aus denen beendete Threads ausscheiden. Warp-Shuffles laufen über
  einen Austauschpuffer pro Warp. Ein nicht konvergenter Shuffle oder eine nicht erreichte
  Barriere führt dort zum Hängen statt zu falschen Ergebnissen, der Fehler wird also sichtbar.
  Abgedeckt sind alle sm_61-Pfade (float, `__dp4a`, Kacheln, Backward). WMMA/MMA-Pfade (sm_75+)
  laufen nur auf echter Hardware. Der Emulator ersetzt keine GPU-Messung, aber er findet
  Index-, Padding- und Synchronisationsfehler, bevor die Kernels auf die Karte kommen.
- **CI:** Build-Job ohne GPU (Kompilieren für alle Archs, Python-Tests, `xcheck`). GPU-Tests
  laufen auf deiner 1080 Ti oder auf einem Self-Hosted-Runner. Aus dieser Cloud-Umgebung heraus
  kann ich keine GPU nutzen.
- **Benchmarks:** jeweils gegen die Roofline aus Abschnitt 2 berichten: Anteil an Bandbreite
  bzw. Peak, nicht nur ms.

---

## 10. Migrationsplan

| Phase | Inhalt | Ergebnis / Abnahme |
|---|---|---|
| **P0** Struktur | `format.h`, Bibliothek + CLI, Tests und Benchmarks auslagern. Keine Verhaltensänderung | `--test` grün, Benchmark-Zahlen ±5 % gleich, CLI-Flags unverändert |
| **P1** Anbindung | C-ABI, `fiver_cuda.py`, `FiverLayer(backend="cuda")` mit den heutigen Kernels; `train.py --backend cuda` (von C# schon übergeben) nutzt sie | erstes EmojiNet-Training auf der GPU. Loss-Verlauf wie numpy (gleiche Seeds, Toleranz). Basiszeit pro Schritt gemessen |
| **P2** GEMV v2 | 5.1: x in Shared Memory, `uint4`-Loads, R Zeilen pro Warp, Epilog | ≥ 80 % DRAM-Bandbreite bei B = 1 |
| **P3** GEMM sm_61 | 5.2: FP32-SIMT- und `__dp4a`-Kachelkernel, Dispatch nach Batch | FP32 ≥ 60 %, dp4a ≥ 50 % Peak bei B = 64 |
| **P4** Backward v2 | 5.5 und 5.6: gekacheltes dX ohne Atomics, fusioniertes dYᵀX + STE | Backward ≤ 3× Forward, deterministisch |
| **P5** Fusion | 5.3/5.4: Epiloge, int8-Ausgabe, CUDA Graph pro Schritt | Kernels pro Schritt halbiert, Schrittzeit gemessen |
| **P6** Turing+ | `mma.sync`-Varianten sm_75/80, `cp.async` | auf Turing+ gemessen (braucht entsprechende Karte) |
| **P7** optional | FP8 für `magic` (sm_89/90), 6-Bit-Codes, CPU-Backend aus Abschnitt 8 | jeweils eigene Messung |

P0 und P1 bringen keine Beschleunigung, sind aber die Voraussetzung dafür, dass jede weitere
Phase am echten Trainingsschritt gemessen werden kann. Sinnvolle Reihenfolge danach: P3 vor P2,
denn EmojiNet trainiert mit Batch 32 im GEMM-Regime, GEMV ist eher für LLM-Inferenz.

---

## 11. Entscheidungen

| # | Thema | Entscheidung |
|---|---|---|
| 11.1 | Python-Anbindung | **ctypes** gegen die C-ABI aus Abschnitt 6. Keine PyTorch-Abhängigkeit. |
| 11.2 | act_bits (Aktivierung pro Gewicht) | Nur im **float-GEMV**. Alle anderen Pfade (int8, GEMM, Backward) lehnen Matrizen mit act_bits ab. Hinweis: Wer mit act_bits *trainieren* will, braucht dazu auch ein float-dX, sonst passt der Gradient nicht zum Forward. Das kommt erst, wenn ein Modell es nutzt. |
| 11.3 | Gitter | **Alle drei Gitter (`centered`, `anchored`, `magic`) werden in jedem Pfad getrennt aufgebaut und getestet.** Keine Variante wird zugunsten einer anderen eingespart. Die Testmatrix ist Pfad × Gitter (Abschnitt 9). Welches Gitter EmojiNet standardmäßig nutzt, ist damit eine reine Konfigurationsfrage. |
| 11.4 | dX ohne Atomics | **Entschieden.** Gekachelter dX mit Reduktion in fester Reihenfolge (5.5) ist der Standard. Grund: Mit Atomics landen bei 4096 Zeilen und 64er-Streifen 64 Atomics auf jedem dX-Element, bei Batch 32 × 4096 Spalten ≈ 8,4 Mio. Diese werden im L2 serialisiert, wo sie dieselbe Adresse treffen. Geschätzt kostet das so viel wie die Rechnung selbst oder mehr, dazu kommt das Nullsetzen. Außerdem kippen nicht reproduzierbare Rundungen beim STE-Training Vorzeichen nahe 0, und Läufe mit gleichem Seed laufen auseinander. Der Atomics-Pfad bleibt nur als Messvergleich. |

## 12. Offen und Risiken

1. **int8-Aktivierungen im Training.** Im int8-Pfad wird neben den Gewichten auch x gerundet
   (Skala pro Zeile max|x|/127, zusätzliches Rauschen bis ~0,4 % von max|x| je Element). Für
   Inferenz ist das gut belegt. Ungeprüft ist, ob STE-Training von Grund auf damit genauso
   konvergiert, denn die Schritte hängen am Vorzeichen des Gradienten, und der hängt an den
   gerundeten Vorwärtswerten. Beim `magic`-Gitter sind die Gewichte in int8 exakt, dort rundet
   nur x.
   *Test (vergrößert):* Die int8-Vorwärtsrechnung lässt sich in numpy exakt nachbilden, gleiche
   Rundung wie `cw_quantize_x` und `cw_grid_e`. Der Vergleich braucht deshalb keine GPU und läuft
   vor P3. Er nutzt viele Batches und mehrere Seeds, damit Unterschiede zwischen einzelnen Batches
   herausgemittelt werden: Lehrer-Schüler-Aufgabe, je Seed float- und int8-Pfad mit identischen
   Daten und Initialisierung, berichtet als Mittelwert ± Streuung der Loss-Kurven und als gepaarte
   Differenz. Skript: `emoji_net/experiments/int8_training_compare.py`. Liegen die Kurven
   innerhalb der Seed-Streuung, darf das Training den int8-Pfad nutzen. Sonst nutzt das Training
   float, und int8 bleibt für die Inferenz. Auf der GPU wird das in P3 mit echten Kernels bestätigt.
   **Ergebnis (512 → 256 → 64, Batch 32, 2000 Schritte, 6 Seeds, 4096 zurückgehaltene Beispiele):**

   | Gitter | float MSE | int8 MSE | gepaarte Differenz | Streuung zwischen Seeds |
   |---|---|---|---|---|
   | centered | 0,10878 ± 0,00098 | 0,10894 ± 0,00134 | +0,15 % ± 0,45 % (s.e.) | 0,90 % |
   | magic | 0,10978 ± 0,00133 | 0,11002 ± 0,00112 | +0,22 % ± 0,50 % (s.e.) | 1,21 % |

   Kein messbarer Unterschied: Die Differenz liegt weit innerhalb der Seed-Streuung und
   innerhalb ihres Standardfehlers. **Das Training darf den int8-Pfad nutzen** (nur mit
   `centered`/`magic`, siehe 13.3). Einschränkung: Der Vergleich deckt das Plateau ab, auf dem
   das Sign-STE-Training nach ~250 Schritten steht (12.6).
2. **Zeilenskala vs. Blockskala:** bewusst offen. Blockskalen (32–96 Gewichte) wären vermutlich
   genauer (nicht gemessen) und passen zur Kachelung mit K = 96, ändern aber das Format. Die
   neue Bibliothek kapselt die Skala hinter dem Matrix-Handle, damit die Entscheidung später
   ohne API-Bruch fallen kann.
3. **Treiber/Toolkit:** Die 1080 Ti braucht CUDA 12.x (CUDA 13 hat sm_61 entfernt). Laut
   NVIDIA ist die Treiberreihe R580 die letzte mit Pascal-Unterstützung. Turing+-Pfade
   langfristig separat pflegen.
4. **Kein GPU-Zugang in der Entwicklungsumgebung:** Jede Phase braucht einen Messlauf auf
   echter Hardware, bevor die nächste beginnt. Das ist der größte Zeitfaktor.
5. **Aufwand der Kachel-Kernels (P3/P4):** grob 600–1000 Zeilen CUDA. Korrektheit ist schnell
   erreicht. Die Leistung (Kachelgrößen, Pipelining, Bank-Konflikte) braucht mehrere
   Mess-Iterationen auf der Zielkarte. Die heutigen Kernels bleiben als Referenz und für kleine
   Batches erhalten, der Umbau ist dadurch inkrementell.

6. **Plateau des Sign-STE-Trainings:** Im int8-Vergleich steht der Fehler ab ~250 Schritten
   bei ~0,108 (Zielvarianz ~0,39) und sinkt nicht weiter, in beiden Pfaden gleich. Die Regel
   „jedes Gewicht mit Gradient ≠ 0 bewegt sich jeden Schritt um ±1 Code“ lässt die Gewichte um
   ihr Optimum pendeln. Das ist kein Kernel-Thema, sondern eine Eigenschaft des Updates.
   Kandidaten, je mit eigenem Vergleich: Update nur ab einer Schwelle |G| (Totzone),
   Gradienten über mehrere Schritte ansammeln (Akkumulator pro Gewicht, kostet Speicher),
   stochastisches Runden des Schritts. Der fusionierte Kernel kann jede dieser Regeln im
   Epilog umsetzen.

---

## 13. Umsetzungsstand

### 13.1 Was gebaut ist

| Phase | Stand | Anmerkung |
|---|---|---|
| P0 Struktur | **fertig** | `cuda/` mit `format.h`, `compat.h`, `cuda_emu.h`, Bibliothek `libfiver_cuda.so`, CLI `build/compact_weights_test` (Flags unverändert) |
| P1 Anbindung | **fertig, nur im Emulator geprüft** | C-ABI (`fiver.h`), `emoji_net/fiver_cuda.py`, `FiverLayer(backend="cuda")`, `train.py --backend cuda` mit numpy-Rückfall |
| P2 GEMV v2 | **fertig, nicht gemessen** | siehe 13.2 |
| P3 GEMM sm_61 | **fertig, nicht gemessen** | FP32-SIMT und `__dp4a`, Split-K deterministisch |
| P4 Backward v2 | **fertig, nicht gemessen** | dX gekachelt ohne Atomics, wgrad + STE fusioniert |
| P5–P7 | offen | Epiloge über Bias hinaus, CUDA Graphs, Turing+-MMA, FP8 |

### 13.2 Tatsächliche Kernel-Auslegung (sm_61, `nvcc -Xptxas -v`)

| Kernel | Kachel | Threads | Register | Shared | CTAs/SM (Reg. / Shared) |
|---|---|---|---|---|---|
| `k_gemv_v2<4,4>` (Batch ≤ 4) | 16 Zeilen, x-Block 768 Spalten | 128 | 72 | 12,4 KB | 7 / 7 |
| `k_gemv_v2<2,8>` (Batch 5–8) | 8 Zeilen | 128 | 91 | 24,8 KB | 5 / 3 |
| `k_gemm_f32` | 128 Zeilen × 32 Batch × 48 Spalten | 128 | 94 | 30,0 KB | 5 / 3 |
| `k_gemm_dp4a` | 128 × 32 × 48 (12 int32-Gruppen) | 128 | 80 | 7,5 KB | 6 / 12 |
| `k_dgrad` | 192 Spalten × 32 Batch, Zeilen in 32er-Blöcken | 128 | 115 | 28,8 KB | 4 / 3 |
| `k_wgrad_ste` | 64 Zeilen × 96 Spalten, Batch in 32er-Blöcken | 128 | 104 | 20,0 KB | 4 / 4 |

Abweichungen vom Entwurf in Abschnitt 5, mit Grund:
- **GEMM-Kachel 128 × 32 × 48 statt 128 × 64 × 96:** Eine entpackte 96er-Kachel braucht 48 KB
  und sprengt die 48-KB-Grenze pro Block. N = 32 passt zur EmojiNet-Batch. Damit sich die
  B-Werte als Warp-Broadcast lesen lassen, teilt sich jeder Warp dieselben 8 Batch-Spalten. Pro
  k ergeben sich ~6 Shared-Memory-Wavefronts gegen 8 FMA-Takte, ausgelegt auf FMA-Grenze.
- **GEMV v2 mit Wort-Verschränkung statt `uint4`-Loads:** Lane l liest Wort l + 32j. Die
  Gewichts-Loads bleiben 128-Byte-zusammenhängend, und die x-Werte in Shared Memory
  (wortweise transponiert) liest jede Lane ohne Bank-Konflikt. Mit `uint4` pro Lane wären die
  x-Reads 4–8-fach konfliktbehaftet. Den Load-Engpass aus Abschnitt 3 löst die Wiederverwendung
  von x über RPW Zeilen pro Warp (Lade-Einheit grob 1,3 statt 2 Befehle pro Gewicht).
- **Zeilenbreite auf 16 Wörter (96 Spalten) gepolstert** (`fiver_row_pitch`): Keine Kachel
  überschreitet ein Zeilenende, die Schleifen kommen ohne Randprüfung aus. Das externe Format
  (Python, GGUF) bleibt unverändert, die Bibliothek polstert beim Hochladen.
- **Tensor-Core-Pfad** (`k_gemm_tc`): nur auf ausdrückliche Anfrage (`FIVER_PATH_INT8_TC`).
  Im Automatikbetrieb werden ausschließlich Pfade genutzt, die der Emulator geprüft hat.
- **Stellschrauben für die erste GPU-Messung:** `__launch_bounds__` für `k_dgrad` und
  `k_gemm_f32` (Belegung 3 CTAs/SM), Split-K-Ziel (heute 2 × SMs), RPW/BT im GEMV v2.

### 13.3 Was geprüft ist und was nicht

- **CPU-Emulator:** 237 Prüfungen grün. Alle Pfade × alle Gitter, ungerade Formen,
  Batch-Reste, erzwungene Splits, bitgleiche Wiederholung, STE-Codes einzeln, C-ABI mit
  Fehlerfällen.
- **Python:** `check_cuda_backend.py` gegen die Emulator-Bibliothek. Float-Pfade weichen um
  ~1e-7 von numpy ab, die Codes nach jedem STE-Schritt sind identisch, für alle drei Gitter.
- **int8 im Training (numpy-Simulation des GPU-Pfads):** kein messbarer Unterschied zu float,
  Details in 12.1.
- **Nicht geprüft:** jede Laufzeit, der Tensor-Core-Pfad, echtes GPU-Speicherverhalten
  (Caches, Wettläufe zwischen Blöcken — der Emulator führt Blöcke nacheinander aus).

**Befund int8 + `anchored`:** Die int8-Rundung von x erzeugt einen absoluten Fehler, der mit der
Gewichtsgröße wächst. Beim `anchored`-Gitter liegen alle Gewichte bei etwa 1 (Skala 1), die
Vorwärtswerte sind Summen großer Beiträge bis ±26. Ausgänge nahe 0 tragen deshalb einen großen
relativen Fehler, den Softsign fast ungedämpft weitergibt. Im Testnetz sind das 8 % nach der
Aktivierung, bei `centered` und `magic` 1–2 %. **Für das Training mit int8-Aktivierungen
eignen sich nur die nullzentrierten Gitter.**

### 13.4 Auf der GTX 1080 Ti ausführen

```
cmake -S . -B build && cmake --build build -j
./build/compact_weights_test --test                  # muss "TEST PASSED" zeigen
./build/compact_weights_test --bench-gemv --bench-gemm --bench-bwd
python3 emoji_net/check_cuda_backend.py              # Python-Anbindung gegen die GPU
python3 emoji_net/train.py --backend cuda --epochs 1 # erstes GPU-Training
```

Ohne GPU, für Tests auf jedem Rechner:

```
cmake -S . -B build-emu -DFIVER_EMU=ON && cmake --build build-emu -j
./build-emu/compact_weights_test --test
```
