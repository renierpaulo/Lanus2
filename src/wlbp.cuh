/*
 * wlbp.cuh - Word List Bit Protocol v1 (GLBP technique applied to BIP39)
 *
 * PATTERN REPLACEMENT (this step):
 *   OLD: thread receives words ONE AT A TIME via global lookups
 *        (char wordlist[2048][16] in global memory, wordlist[indices[w]])
 *   NEW: EACH THREAD RECEIVES THE FULL 2048-WORD VOCABULARY as its own
 *        private compressed blob (5,784 bytes, local memory) and decodes
 *        any word from its private copy. The thread is self-contained:
 *        no global word lookups during decode.
 *
 * Format (see wlbp_blob.h for the generated blob):
 *   - Jump table: 64 groups x uint32 LE absolute bit offset (256 bytes)
 *   - Payload: bit-packed word entries, 32 words per group
 *     [len-3 : 3 bits][prefix_len : 3 bits][suffix : 5 bits/char, a=0..z=25]
 *   - Group head words store prefix_len = 0 (full word)
 *   - Random access: jump table -> forward decode <= 32 entries (~40 ALU ops)
 *
 * Verification: host build compiles the exact same decoder logic with gcc
 * and round-trip checks all 2048 words against wordlist.txt (test_wlbp_host.c).
 */

#ifndef WLBP_CUH
#define WLBP_CUH

#include <stdint.h>
#include "wlbp_blob.h"

#ifdef __CUDACC__
#define WLBP_DEV __device__
#define WLBP_INLINE __host__ __device__ __forceinline__
#define WLBP_LOAD(p) (*(p))
#else
/* Host build: same code compiles for verification with gcc/cl */
#define WLBP_DEV
#define WLBP_INLINE static inline
#define WLBP_LOAD(p) (*(p))
#endif

#define WLBP_GROUP_WORDS 32

/* ---------------------------------------------------------------------------
 * Bit reader (MSB-first, matches the packer in wlbp_prototype.py)
 * ------------------------------------------------------------------------- */
typedef struct {
    const unsigned char* data;
    uint64_t pos; /* absolute bit offset */
} wlbp_reader_t;

WLBP_INLINE uint32_t wlbp_getbits(const unsigned char* data, uint64_t* pos, int width) {
    uint32_t v = 0;
    for (int i = 0; i < width; i++) {
        unsigned int byte = data[*pos >> 3];
        unsigned int bit = (byte >> (7u - (*pos & 7u))) & 1u;
        v = (v << 1) | bit;
        (*pos) += 1;
    }
    return v;
}

/* ---------------------------------------------------------------------------
 * Decode word `index` (0..2047) from a vocabulary blob.
 * `out` must hold at least 9 bytes (8 chars + NUL). Returns word length.
 *
 * PREFIX CHAIN: the previous word of the group must already be in `out`
 * (call sequentially for slots 1..31 of a group). For slot 0 the blob
 * stores the full word, so `out` needs no prior state.
 * ------------------------------------------------------------------------- */
WLBP_INLINE int wlbp_decode_word(const unsigned char* vocab, uint32_t index, unsigned char* out) {
    uint32_t group = index / WLBP_GROUP_WORDS;
    uint32_t slot = index % WLBP_GROUP_WORDS;

    /* jump table: uint32 LE absolute bit offset of the group head */
    const unsigned char* jt = vocab + group * 4;
    uint64_t pos = (uint64_t)jt[0] |
                   ((uint64_t)jt[1] << 8) |
                   ((uint64_t)jt[2] << 16) |
                   ((uint64_t)jt[3] << 24);

    int len = 0;
    for (uint32_t k = 0; k <= slot; k++) {
        len = (int)wlbp_getbits(vocab, &pos, 3) + 3;
        int plen = (int)wlbp_getbits(vocab, &pos, 3);
        int slen = len - plen;
        /* prefix chars are already in out[] from the previous slot */
        for (int c = 0; c < slen; c++) {
            out[plen + c] = (unsigned char)(97 + (int)wlbp_getbits(vocab, &pos, 5));
        }
        out[len] = 0;
    }
    return len;
}

/* ---------------------------------------------------------------------------
 * THE NEW PATTERN: per-thread full vocabulary.
 *
 * Each thread declares its own compressed vocabulary in local memory and
 * copies the shared source ONCE. The init loop is perfectly coalesced:
 * every lane reads d_source[i] at the same i, so the warp drains the
 * source 32 bytes per transaction.
 *
 * After init, the thread NEVER touches global memory for words again.
 * ------------------------------------------------------------------------- */
#define WLBP_VOCAB_BYTES WLBP_SIZE /* 5784 */

WLBP_INLINE void wlbp_thread_vocab_init(unsigned char* vocab /* local, per-thread */,
                                        const unsigned char* d_source /* global or const */) {
#pragma unroll 8
    for (int i = 0; i < WLBP_VOCAB_BYTES; i++) {
        vocab[i] = d_source[i];
    }
}

/* ---------------------------------------------------------------------------
 * PROCESS-READY LAYER (WLBP-PR)
 * The vocabulary is no longer a decode stream - it is the thread's compute
 * table. A phrase = selection of 12 entries + register-only assembly.
 * No ASCII string, no byte buffers, no global/shared traffic after init.
 * ------------------------------------------------------------------------- */

/*
 * Extract one word from the thread's private vocabulary as a byte-expanded
 * big-endian uint64 (char0 = most significant byte of the low `len` bytes)
 * plus its length. Pure register result.
 */
WLBP_INLINE int wlbp_word_bytes(const unsigned char* vocab, uint32_t index, uint64_t* out_bytes) {
    unsigned char tmp[9];
    int len = wlbp_decode_word(vocab, index, tmp);
    uint64_t v = 0;
    for (int c = 0; c < len; c++) v = (v << 8) | (uint64_t)tmp[c];
    *out_bytes = v;
    return len;
}

/*
 * Append nbits_val (byte-granular) of `value` (value's bits start at its MSB
 * of the low nbits_val/8 bytes) into the streaming key assembly.
 */
WLBP_INLINE void wlbp_append_bits(uint64_t* cur, uint32_t* bits, int* wi, uint64_t w[16],
                                  uint64_t value, uint32_t nbits_val) {
    uint32_t avail = 64u - *bits;
    if (nbits_val <= avail) {
        *cur |= value << (avail - nbits_val);
        *bits += nbits_val;
        if (*bits == 64u) { w[(*wi)++] = *cur; *cur = 0; *bits = 0; }
    } else {
        uint32_t overflow = nbits_val - avail;    /* whole bytes, byte-granular */
        *cur |= (value >> overflow);              /* top `avail` bits fill cur */
        w[(*wi)++] = *cur;
        *cur = value << (64u - overflow);         /* remainder at top of new cur */
        *bits = overflow;
    }
}

/*
 * Assemble the 128-byte HMAC key block (mnemonic zero-padded to 128 bytes)
 * as 16 big-endian uint64 W-words - the exact input format of the SHA-512
 * first compression stage. All register arithmetic, byte-granular.
 * Includes the 0x20 separator byte between words (11 spaces total).
 * Worst case: 12 x (8+1) - 1 = 107 bytes < 128 -> always fits w[0..15].
 */
WLBP_INLINE void wlbp_assemble_key12(const unsigned char* vocab,
                                     const uint16_t* indices /* 12 entries */,
                                     uint64_t w[16]) {
    uint64_t cur = 0;
    uint32_t bits = 0; /* valid bits in cur, byte-granular, < 64 */
    int wi = 0;

#pragma unroll 1
    for (int k = 0; k < 12; k++) {
        if (k > 0) {
            wlbp_append_bits(&cur, &bits, &wi, w, 0x20ULL, 8); /* ' ' separator (LSB-aligned) */
        }
        uint64_t wb;
        int len = wlbp_word_bytes(vocab, indices[k], &wb);
        wlbp_append_bits(&cur, &bits, &wi, w, wb, (uint32_t)len * 8u);
    }
    if (bits > 0u) w[wi++] = cur;
    for (; wi < 16; wi++) w[wi] = 0;                  /* zero-pad to 128 bytes */
}

/* ---------------------------------------------------------------------------
 * ETAPA B: REPRESENTATIVE TABLE
 * The thread transforms its whole 2048-word vocabulary ONCE into process-
 * ready uint64 representatives (byte-expanded, big-endian). The word length
 * is DERIVABLE from the representative itself (chars are 0x61..0x7a, never
 * zero) -> the table is exactly 2048 x uint64 = 16KB, zero metadata.
 * ------------------------------------------------------------------------- */

/* Word length from its representative: count leading zero bytes. */
WLBP_INLINE int wlbp_wb_len(uint64_t wb) {
    int n = 0;
    uint64_t m = wb;
    while (!(m & 0xFF00000000000000ULL)) { m <<= 8; n++; }
    return 8 - n;
}

/* Build the representative table from the thread's private blob.
 * This is the in-thread processing of ALL 2048 words (one pass). */
WLBP_INLINE void wlbp_build_repr_table(const unsigned char* vocab, uint64_t table[2048]) {
#pragma unroll 1
    for (int i = 0; i < 2048; i++) {
        uint64_t wb;
        wlbp_word_bytes(vocab, (uint32_t)i, &wb);
        table[i] = wb;
    }
}

/* Register-only assembly v2: pure selection from the representative table.
 * Same output as wlbp_assemble_key12 (byte-exact), plus the key length
 * (mnemonic bytes incl. separators) used for round folding in ETAPA B3. */
WLBP_INLINE void wlbp_assemble_key12_rt(const uint64_t table[2048],
                                        const uint16_t* indices,
                                        uint64_t w[16], uint32_t* key_len) {
    uint64_t cur = 0;
    uint32_t bits = 0;
    int wi = 0;
    uint32_t total = 0;

#pragma unroll 1
    for (int k = 0; k < 12; k++) {
        if (k > 0) {
            wlbp_append_bits(&cur, &bits, &wi, w, 0x20ULL, 8); /* ' ' separator */
            total++;
        }
        uint64_t wb = WLBP_LOAD(&table[indices[k]]);
        int len = wlbp_wb_len(wb);
        wlbp_append_bits(&cur, &bits, &wi, w, wb, (uint32_t)len * 8u);
        total += (uint32_t)len;
    }
    if (bits > 0u) w[wi++] = cur;
    for (; wi < 16; wi++) w[wi] = 0;
    *key_len = total;
}

/* ---------------------------------------------------------------------------
 * E2: BASE PREFIX + TAIL SWAP
 * The thread's 2048 candidates share an 11-word base. Assemble the base
 * prefix ONCE per thread (11 words + 11 separators, stream state kept);
 * per candidate, only the tail (12th word) is appended (~10 register ops).
 * ------------------------------------------------------------------------- */

/* Assemble the base prefix: words 1..11 + all 11 separator bytes (the last
 * separator precedes the 12th word). Stream state (cur/bits/wi) is returned
 * so the per-candidate tail can continue it. */
WLBP_INLINE void wlbp_base_prefix_build(const uint64_t* table, const uint16_t* base11,
                                        uint64_t prefix_w[16],
                                        uint64_t* cur_out, uint32_t* bits_out, int* wi_out,
                                        uint32_t* prefix_len) {
    uint64_t cur = 0;
    uint32_t bits = 0;
    int wi = 0;
    uint32_t total = 0;

#pragma unroll 1
    for (int k = 0; k < 11; k++) {
        if (k > 0) {
            wlbp_append_bits(&cur, &bits, &wi, prefix_w, 0x20ULL, 8);
            total++;
        }
        uint64_t wb = WLBP_LOAD(&table[base11[k]]);
        int len = wlbp_wb_len(wb);
        wlbp_append_bits(&cur, &bits, &wi, prefix_w, wb, (uint32_t)len * 8u);
        total += (uint32_t)len;
    }
    /* trailing separator: the one that precedes the 12th word */
    wlbp_append_bits(&cur, &bits, &wi, prefix_w, 0x20ULL, 8);
    total++;

    /* stream state at the prefix boundary: wi = index of the partial word */
    *cur_out = cur;
    *bits_out = bits;
    *wi_out = wi;
    *prefix_len = total;

    if (bits > 0u) prefix_w[wi++] = cur;
    for (; wi < 16; wi++) prefix_w[wi] = 0;
}

/* Per-candidate: continue the prefix stream with the 12th word, zero-pad.
 * Cost ~10 register ops (copy full words + one append) vs full re-assembly. */
WLBP_INLINE void wlbp_keyblock_tail_swap(const uint64_t prefix_w[16],
                                         uint64_t cur, uint32_t bits, int wi,
                                         uint64_t wb12, uint32_t len12,
                                         uint64_t key_w[16]) {
    #pragma unroll
    for (int i = 0; i < 16; i++) key_w[i] = prefix_w[i];
    /* restore live stream state at the prefix boundary */
    uint32_t b = bits;
    int w_i = wi;
    uint64_t c = cur;
    wlbp_append_bits(&c, &b, &w_i, key_w, wb12, len12 * 8u);
    if (b > 0u) key_w[w_i++] = c;
    for (; w_i < 16; w_i++) key_w[w_i] = 0;
}

#ifdef __CUDACC__

/* Global read-only source (uploaded once by the host) */
__device__ unsigned char d_wlbp_source[WLBP_SIZE];

/* Host-side upload: call once before any kernel launch */
static inline void wlbp_upload_source(void) {
    cudaMemcpyToSymbol(d_wlbp_source, WLBP_BLOB, WLBP_SIZE);
}

/* ---------------------------------------------------------------------------
 * ETAPA B runtime delivery: the representative table is CONSTANT for all
 * threads -> the HOST builds it once (microseconds) and uploads it as a
 * 16KB read-only table. Threads read representatives via __ldg (L1/L2) and
 * materialize the active words in registers. No per-thread copy, no decode,
 * no local array - zero regression, pure win.
 * ---------------------------------------------------------------------- */
__device__ uint64_t d_repr_table[2048];

static inline void wlbp_upload_repr_table(void) {
    unsigned char vocab[WLBP_VOCAB_BYTES];
    wlbp_thread_vocab_init(vocab, WLBP_BLOB);
    uint64_t table[2048];
    wlbp_build_repr_table(vocab, table);
    cudaMemcpyToSymbol(d_repr_table, table, sizeof(table));
}

/* -------------------------------------------------------------------------
 * DEMO of the pattern replacement (NOT integrated into main.cu yet):
 * every thread holds all 2048 words privately and decodes its 12-word
 * phrase from its own copy. Drop-in shape for kernel 2 later.
 * ---------------------------------------------------------------------- */
__global__ void wlbp_pattern_demo_kernel(const uint16_t* phrases, /* packed 12 indices per thread */
                                         uint32_t num_threads,
                                         unsigned char* out_mnemonics /* 256B per thread */) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_threads) return;

    /* 1. THREAD RECEIVES THE FULL 2048-WORD VOCABULARY (private, local) */
    unsigned char vocab[WLBP_VOCAB_BYTES];
    wlbp_thread_vocab_init(vocab, d_wlbp_source);

    /* 2. decode the phrase from the thread's OWN vocabulary - zero global reads */
    const uint16_t* indices = phrases + tid * 12;
    unsigned char* mn = out_mnemonics + tid * 256;
    int mn_len = 0;
    unsigned char word[9];

#pragma unroll 1
    for (int w = 0; w < 12; w++) {
        int len = wlbp_decode_word(vocab, indices[w], word);
        if (w > 0) mn[mn_len++] = ' ';
        for (int c = 0; c < len; c++) mn[mn_len++] = word[c];
    }
    mn[mn_len] = 0;
}

#endif /* __CUDACC__ */

#endif /* WLBP_CUH */
