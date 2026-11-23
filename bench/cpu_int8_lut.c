/*
 * cpu_int8_lut.c — int8 table decodes with AVX-512 VBMI (vpermb / vpermi2b).
 *
 * Part A — cost versus code width for int8 tables, codes stored one per byte:
 *   6 bit  vpermb    (_mm512_permutexvar_epi8)   64-entry table, 64 codes / instruction
 *   7 bit  vpermi2b  (_mm512_permutex2var_epi8)  128-entry table from two registers
 *   8 bit  2 × vpermi2b + vpmovb2m + blend
 *   reference: 5-bit float decode with vpermt2ps (16 codes / instruction)
 *
 * Part B — complete Fiver int8 decode straight from the packed words
 * (6 codes per uint32, k_i at bits 5i):
 *   vpmovzxdq       8 words → 8 qword lanes
 *   vpmultishiftqb  per qword: bytes = bits [0,5,10,15,20,25) → the 6 codes
 *                   (plus 3 neighbour bits above each code)
 *   vpermi2b        lookup in the 32-entry e table repeated 4× over 128 bytes,
 *                   so the neighbour bits 5–6 never matter (bit 7 is ignored)
 *   → 48 codes in 3 instructions (bytes 6–7 of each qword are don't-care).
 *   With one extra vpermb the 48 codes are compacted into natural order.
 *
 * Every variant is checked against a scalar reference before timing.
 *
 *   clang -O3 -o cpu_int8_lut bench/cpu_int8_lut.c && taskset -c 1 ./cpu_int8_lut
 * Needs AVX-512 VBMI (Ice Lake and newer Intel, AMD Zen 4+).
 */
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TV __attribute__((target("avx512f,avx512bw,avx512vl,avx512vbmi"), noinline))
#define IV __attribute__((target("avx512f,avx512bw,avx512vl,avx512vbmi"), always_inline)) inline

/* Fiver centered int8 codes e = rn(127·v), identical to FIVER_E_INIT */
static const int8_t E_CENTERED[32] = {
    -127, -107, -90, -76, -64, -53, -45, -38, -32, -27, -22, -19, -16, -13, -11, -9,
       0,   11,  13,  16,  19,  22,  27,  32,  38,  45,  53,  64,  76,  90, 107, 127 };

static int8_t TAB[256] __attribute__((aligned(64)));
static float  FTAB[32] __attribute__((aligned(64)));

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

/* ── Part A: byte codes → int8 values ─────────────────────────────────── */
static IV __m512i dec6(__m512i k, __m512i t0)                 { return _mm512_permutexvar_epi8(k, t0); }
static IV __m512i dec7(__m512i k, __m512i t0, __m512i t1)     { return _mm512_permutex2var_epi8(t0, k, t1); }
static IV __m512i dec8(__m512i k, __m512i t0, __m512i t1, __m512i t2, __m512i t3) {
    const __mmask64 hi = _mm512_movepi8_mask(k);              /* bit 7 */
    return _mm512_mask_blend_epi8(hi, _mm512_permutex2var_epi8(t0, k, t1), _mm512_permutex2var_epi8(t2, k, t3));
}

#define LOADT __m512i t0 = _mm512_load_si512(TAB), t1 = _mm512_load_si512(TAB + 64), \
                      t2 = _mm512_load_si512(TAB + 128), t3 = _mm512_load_si512(TAB + 192); \
              (void)t0; (void)t1; (void)t2; (void)t3;

#define DEC_FN(NAME, BODY)                                                                   \
static TV void NAME##_store(const uint8_t* c, int8_t* out, int n) {                          \
    LOADT                                                                                    \
    for (int i = 0; i < n; i += 64) _mm512_store_si512(out + i, BODY(_mm512_load_si512(c + i))); \
}                                                                                            \
static TV int NAME##_sum(const uint8_t* c, int n) {                                          \
    LOADT                                                                                    \
    __m512i a0 = _mm512_setzero_si512(), a1 = a0, a2 = a0, a3 = a0;                          \
    for (int i = 0; i < n; i += 256) {                                                       \
        a0 = _mm512_add_epi8(a0, BODY(_mm512_load_si512(c + i)));                            \
        a1 = _mm512_add_epi8(a1, BODY(_mm512_load_si512(c + i + 64)));                       \
        a2 = _mm512_add_epi8(a2, BODY(_mm512_load_si512(c + i + 128)));                      \
        a3 = _mm512_add_epi8(a3, BODY(_mm512_load_si512(c + i + 192)));                      \
    }                                                                                        \
    __m512i a = _mm512_add_epi8(_mm512_add_epi8(a0, a1), _mm512_add_epi8(a2, a3));            \
    int8_t r[64]; _mm512_storeu_si512(r, a);                                                 \
    int s = 0; for (int j = 0; j < 64; j++) s += r[j];                                       \
    return s;                                                                                \
}
#define B6(k) dec6(k, t0)
#define B7(k) dec7(k, t0, t1)
#define B8(k) dec8(k, t0, t1, t2, t3)
DEC_FN(lut6, B6)
DEC_FN(lut7, B7)
DEC_FN(lut8, B8)

/* reference: 5-bit float decode, 16 codes per vpermt2ps (codes widened from bytes) */
static TV float f5_sum(const uint8_t* c, int n) {
    const __m512 lo = _mm512_load_ps(FTAB), hi = _mm512_load_ps(FTAB + 16);
    __m512 a0 = _mm512_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
    for (int i = 0; i < n; i += 64) {
        a0 = _mm512_add_ps(a0, _mm512_permutex2var_ps(lo, _mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*)(c + i))), hi));
        a1 = _mm512_add_ps(a1, _mm512_permutex2var_ps(lo, _mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*)(c + i + 16))), hi));
        a2 = _mm512_add_ps(a2, _mm512_permutex2var_ps(lo, _mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*)(c + i + 32))), hi));
        a3 = _mm512_add_ps(a3, _mm512_permutex2var_ps(lo, _mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*)(c + i + 48))), hi));
    }
    return _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3)));
}

/* ── Part B: packed Fiver words → int8 e codes ────────────────────────── */
static IV __m512i fiver_e48(const uint32_t* w, __m512i ctrl, __m512i t0, __m512i t1) {
    const __m512i q = _mm512_cvtepu32_epi64(_mm256_loadu_si256((const __m256i*)w));   /* vpmovzxdq */
    return _mm512_permutex2var_epi8(t0, _mm512_multishift_epi64_epi8(ctrl, q), t1);   /* multishift + vpermi2b */
}

static IV __m512i make_ctrl(void) {
    int8_t c[64];
    for (int l = 0; l < 8; l++)
        for (int j = 0; j < 8; j++) c[8 * l + j] = (int8_t)(5 * (j < 6 ? j : 5));
    return _mm512_loadu_si512(c);
}

static IV __m512i make_compact(void) {          /* output byte p ← gapped byte (p/6)·8 + p%6 */
    int8_t c[64];
    for (int p = 0; p < 64; p++) c[p] = (int8_t)(p < 48 ? (p / 6) * 8 + p % 6 : 0);
    return _mm512_loadu_si512(c);
}

static TV void fiver_decode_store(const uint32_t* w, int8_t* out, int nwords) {
    const __m512i t0 = _mm512_load_si512(TAB), t1 = _mm512_load_si512(TAB + 64);
    const __m512i ctrl = make_ctrl(), comp = make_compact();
    for (int i = 0; i < nwords; i += 8)
        _mm512_mask_storeu_epi8(out + 6 * i, ((__mmask64)1 << 48) - 1,
                                _mm512_permutexvar_epi8(comp, fiver_e48(w + i, ctrl, t0, t1)));
}

static TV int fiver_decode_sum(const uint32_t* w, int nwords, int compact) {
    const __m512i t0 = _mm512_load_si512(TAB), t1 = _mm512_load_si512(TAB + 64);
    const __m512i ctrl = make_ctrl(), comp = make_compact();
    __m512i a0 = _mm512_setzero_si512(), a1 = a0;
    if (compact) {
        for (int i = 0; i < nwords; i += 16) {
            a0 = _mm512_add_epi8(a0, _mm512_permutexvar_epi8(comp, fiver_e48(w + i, ctrl, t0, t1)));
            a1 = _mm512_add_epi8(a1, _mm512_permutexvar_epi8(comp, fiver_e48(w + i + 8, ctrl, t0, t1)));
        }
    } else {
        for (int i = 0; i < nwords; i += 16) {
            a0 = _mm512_add_epi8(a0, fiver_e48(w + i, ctrl, t0, t1));
            a1 = _mm512_add_epi8(a1, fiver_e48(w + i + 8, ctrl, t0, t1));
        }
    }
    int8_t r[64]; _mm512_storeu_si512(r, _mm512_add_epi8(a0, a1));
    int s = 0; for (int j = 0; j < 64; j++) s += r[j];
    return s;
}

/* ── driver ───────────────────────────────────────────────────────────── */
static double best_ns(int (*f)(const uint8_t*, int), const uint8_t* c, int n) {
    volatile int sink = 0; double best = 1e9;
    for (int r = 0; r < 7; r++) {
        double t0 = now();
        for (int it = 0; it < 400; it++) sink += f(c, n);
        double dt = (now() - t0) / (400.0 * n);
        if (dt < best) best = dt;
    }
    return best * 1e9;
}

static float f5_wrap_dummy;
static int f5_wrap(const uint8_t* c, int n) { f5_wrap_dummy += f5_sum(c, n); return 0; }

int main(void) {
    srand(7);
    for (int i = 0; i < 256; i++) TAB[i] = (int8_t)(rand() & 0xFF);
    for (int i = 0; i < 32; i++) FTAB[i] = (float)TAB[i];
    const int n = 1 << 18;                                     /* 256 KB of byte codes: L2-resident */
    uint8_t* c = aligned_alloc(64, n);
    int8_t* out = aligned_alloc(64, n);

    printf("Part A — int8 table lookup cost by code width (best of 7, ns per code)\n");
    struct { const char* name; int bits; void (*st)(const uint8_t*, int8_t*, int); int (*sum)(const uint8_t*, int); } A[] = {
        { "6 bit vpermb         (64 codes/instr)", 6, lut6_store, lut6_sum },
        { "7 bit vpermi2b       (64 codes/instr)", 7, lut7_store, lut7_sum },
        { "8 bit 2x vpermi2b + blend          ", 8, lut8_store, lut8_sum } };
    int fails = 0;
    for (int j = 0; j < 3; j++) {
        for (int i = 0; i < n; i++) c[i] = (uint8_t)(rand() & ((1 << A[j].bits) - 1));
        A[j].st(c, out, n);
        int bad = 0;
        for (int i = 0; i < n; i++) bad += out[i] != TAB[c[i]];
        fails += bad != 0;
        const double t = best_ns(A[j].sum, c, n);
        printf("  %s  %6.4f ns/code  %6.1f Gcodes/s   check: %s\n", A[j].name, t, 1.0 / t, bad ? "FAIL" : "ok");
    }
    for (int i = 0; i < n; i++) c[i] = (uint8_t)(rand() & 31);
    printf("  5 bit float vpermt2ps (16 codes/instr)  %6.4f ns/code  (reference, float output)\n", best_ns(f5_wrap, c, n));

    printf("\nPart B — Fiver int8 decode from packed words (centered e codes)\n");
    for (int r = 0; r < 4; r++) memcpy(TAB + 32 * r, E_CENTERED, 32);   /* 4x repeated: bits 5-6 don't matter */
    const int nwords = 1 << 16;                                          /* 256 KB packed, 393216 codes */
    uint32_t* w = aligned_alloc(64, nwords * 4);
    int8_t*   e = aligned_alloc(64, nwords * 6 + 64);
    for (int i = 0; i < nwords; i++) {
        uint32_t v = 0;
        for (int j = 0; j < 6; j++) v |= (uint32_t)(rand() & 31) << (5 * j);
        w[i] = v | ((uint32_t)(rand() & 3) << 30);                       /* reserved bits set on purpose */
    }
    fiver_decode_store(w, e, nwords);
    int bad = 0;
    for (int i = 0; i < nwords; i++)
        for (int j = 0; j < 6; j++) bad += e[6 * i + j] != E_CENTERED[(w[i] >> (5 * j)) & 31];
    fails += bad != 0;
    for (int compact = 0; compact <= 1; compact++) {
        volatile int sink = 0; double best = 1e9;
        for (int r = 0; r < 7; r++) {
            double t0 = now();
            for (int it = 0; it < 400; it++) sink += fiver_decode_sum(w, nwords, compact);
            double dt = (now() - t0) / (400.0 * nwords * 6);
            if (dt < best) best = dt;
        }
        printf("  %-46s %6.4f ns/code  %6.1f Gcodes/s   check: %s\n",
               compact ? "vpmovzxdq + vpmultishiftqb + vpermi2b + vpermb" : "vpmovzxdq + vpmultishiftqb + vpermi2b (gapped)",
               best * 1e9, 1.0 / (best * 1e9), bad ? "FAIL" : "ok");
    }
    printf("\n%s\n", fails ? "CHECK FAILED" : "all checks ok");
    return fails != 0;
}
