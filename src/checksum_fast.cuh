/*
 * checksum_fast.cuh - Mini-midstate BIP39 12-word checksum filter
 *
 * Replaces verify_checksum_12 (main.cu) in the per-base w12 enumeration:
 *
 *     ChecksumFastState_t cst;
 *     checksum_fast_init(base11, &cst);              // ONCE per base
 *     for (uint32_t w12 = 0; w12 < 2048; w12++) {
 *         if (checksum_fast_check(&cst, (uint16_t)w12)) { ... }
 *     }
 *
 * THE MATH
 * --------
 * A 12-word BIP39 phrase packs 132 bits: words 0..10 contribute bits 0..120
 * (11 words x 11 bits, FIXED per base), word 12 contributes bits 121..131.
 * Entropy = bits 0..127, checksum = the last 4 bits of the phrase (w12 & 0xF)
 * which must equal the top 4 bits of SHA-256(entropy)[0].
 *
 *   - entropy bytes 0..14 (bits 0..119) come from the base alone -> CONSTANT
 *     for all 2048 candidates of a base.
 *   - entropy byte 15 = (base bit 120 << 7) | (w12 >> 4): only the candidate's
 *     TOP 7 BITS enter the entropy; the low 4 bits are the checksum itself.
 *
 * The whole 16-byte message is ONE SHA-256 block:
 *   W[0..2]  = entropy bytes 0..11            (constant per base)
 *   W[3]     = entropy bytes 12..14 | byte15  (only byte15 varies)
 *   W[4]     = 0x80000000, W[5..14] = 0, W[15] = 128 (bit length)
 *
 * Rounds 0..2 consume only W[0..2] -> they are base-constant. checksum_fast_init
 * computes the state after round 2 ONCE per base (mini-midstate) plus the two
 * schedule words that depend only on constants (W[16] = W[0] + gamma0(W[1]),
 * W[17] = W[1] + gamma0(W[2]) + gamma1(128)). checksum_fast_check patches W[3]
 * with (w12 >> 4) and runs rounds 3..63 over a 16-word circular schedule.
 *
 * Per candidate the baseline pays: 132-step bit packing + 48 schedule words +
 * 64 rounds. The fast path pays: 46 schedule words + 61 rounds. The bit
 * packing and rounds 0..2 run once per base instead of 2048 times.
 */

#ifndef CHECKSUM_FAST_CUH
#define CHECKSUM_FAST_CUH

// ---------------------------------------------------------------------------
// Host shim (same pattern as bip32_fast.cuh): lets this header compile with a
// plain C++ compiler for host-side equivalence tests. NVCC defines natively.
// ---------------------------------------------------------------------------
#ifndef __CUDACC__
  #ifndef __device__
    #define __device__
  #endif
  #ifndef __host__
    #define __host__
  #endif
  #ifndef __constant__
    #define __constant__ const
  #endif
  #ifndef __forceinline__
    #define __forceinline__ inline
  #endif
#endif

#include <stdint.h>

#include "sha256.cuh"   // K256, rotr32, ch, maj, sigma0/1_256, gamma0/1_256

// ---------------------------------------------------------------------------
// Per-base mini-midstate (fits in 12 registers)
// ---------------------------------------------------------------------------
typedef struct {
    uint32_t w0, w1, w2;   // entropy bytes 0..11 as big-endian W words (constant)
    uint32_t w3_base;      // bytes 12..14 | (bit120 << 7); per candidate: | (w12 >> 4)
    uint32_t w16, w17;     // message-schedule words 16, 17 (base-constant)
    uint32_t a, b, c, d;   // SHA-256 working state after round 2
    uint32_t e, f, g, h;
} ChecksumFastState_t;

// ---------------------------------------------------------------------------
// checksum_fast_init - ONCE per base (the 11 fixed words)
//
// Packs the 121 base bits exactly like verify_checksum_12, then runs SHA-256
// rounds 0..2 on the IV. Fills: W[0..2], the byte-15 template (w3_base),
// W[16], W[17] and the post-round-2 working state.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void checksum_fast_init(const uint16_t* base11,
                                                   ChecksumFastState_t* st)
{
    uint8_t entropy[16];
    uint32_t bits = 0;
    int bit_count = 0, byte_idx = 0;

    #pragma unroll
    for (int w = 0; w < 11; w++) {
        uint16_t idx = base11[w];
        #pragma unroll
        for (int b = 10; b >= 0; b--) {
            bits = (bits << 1) | ((idx >> b) & 1);
            if (++bit_count == 8) {
                entropy[byte_idx++] = (uint8_t)(bits & 0xFF);
                bits = 0;
                bit_count = 0;
            }
        }
    }
    // 121 bits consumed: bytes 0..14 flushed, one pending bit (= entropy bit 120)

    st->w0 = ((uint32_t)entropy[0]  << 24) | ((uint32_t)entropy[1]  << 16) |
             ((uint32_t)entropy[2]  << 8)  |  (uint32_t)entropy[3];
    st->w1 = ((uint32_t)entropy[4]  << 24) | ((uint32_t)entropy[5]  << 16) |
             ((uint32_t)entropy[6]  << 8)  |  (uint32_t)entropy[7];
    st->w2 = ((uint32_t)entropy[8]  << 24) | ((uint32_t)entropy[9]  << 16) |
             ((uint32_t)entropy[10] << 8)  |  (uint32_t)entropy[11];
    st->w3_base = ((uint32_t)entropy[12] << 24) | ((uint32_t)entropy[13] << 16) |
                  ((uint32_t)entropy[14] << 8)  | ((uint32_t)(bits & 1) << 7);

    // Schedule words 16/17 depend only on constants (W[9], W[10], W[14] = 0,
    // W[15] = 128), so they are base-constant too:
    //   W[16] = W[0] + gamma0(W[1]) + W[9]  + gamma1(W[14])
    //   W[17] = W[1] + gamma0(W[2]) + W[10] + gamma1(W[15])
    st->w16 = st->w0 + gamma0_256(st->w1);
    st->w17 = st->w1 + gamma0_256(st->w2) + gamma1_256(128u);

    // Rounds 0..2 (message words W[0..2] are constant per base)
    uint32_t a = 0x6a09e667u, b = 0xbb67ae85u, c = 0x3c6ef372u, d = 0xa54ff53au;
    uint32_t e = 0x510e527fu, f = 0x9b05688cu, g = 0x1f83d9abu, h = 0x5be0cd19u;
    const uint32_t cw[3] = { st->w0, st->w1, st->w2 };

    #pragma unroll
    for (int i = 0; i < 3; i++) {
        uint32_t t1 = h + sigma1_256(e) + ch(e, f, g) + K256[i] + cw[i];
        uint32_t t2 = sigma0_256(a) + maj(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    st->a = a; st->b = b; st->c = c; st->d = d;
    st->e = e; st->f = f; st->g = g; st->h = h;
}

// ---------------------------------------------------------------------------
// checksum_fast_cs4 - per candidate: the EXPECTED 4-bit checksum for the
// entropy completed by w12's top 7 bits. Primitive for the direct-valid
// enumeration: valid_w12 = (v << 4) | checksum_fast_cs4(&cst, v << 4).
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint32_t checksum_fast_cs4(const ChecksumFastState_t* st,
                                                      uint16_t w12)
{
    // 16-word circular schedule. Slot pre-fill exploits that the values W[0]
    // and W[1] are only consumed by rounds 0/1 (inside the midstate) and by
    // the precomputed W[16]/W[17]: slot 0 holds W[16], slot 1 holds W[17],
    // slots 2..15 hold W[2..15]. Each slot is overwritten exactly when its
    // old value (word i-16) is consumed by schedule word i.
    uint32_t W[16];
    W[0]  = st->w16;
    W[1]  = st->w17;
    W[2]  = st->w2;
    W[3]  = st->w3_base | (uint32_t)(w12 >> 4);  // byte15 = bit120<<7 | top7(w12)
    W[4]  = 0x80000000u;                          // 0x80 padding marker
    W[5]  = 0; W[6]  = 0; W[7]  = 0; W[8]  = 0;
    W[9]  = 0; W[10] = 0; W[11] = 0; W[12] = 0;
    W[13] = 0; W[14] = 0;
    W[15] = 128u;                                 // 16 entropy bytes = 128 bits

    uint32_t a = st->a, b = st->b, c = st->c, d = st->d;
    uint32_t e = st->e, f = st->f, g = st->g, h = st->h;

    #pragma unroll
    for (int i = 3; i < 64; i++) {
        if (i >= 18) {
            W[i & 15] = W[(i - 16) & 15] + gamma0_256(W[(i - 15) & 15]) +
                        W[(i - 7) & 15]  + gamma1_256(W[(i - 2) & 15]);
        }
        uint32_t t1 = h + sigma1_256(e) + ch(e, f, g) + K256[i] + W[i & 15];
        uint32_t t2 = sigma0_256(a) + maj(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    // Single-block compression: hash word 0 = IV0 + a. Checksum = its top 4 bits.
    return (0x6a09e667u + a) >> 28;
}

// ---------------------------------------------------------------------------
// checksum_fast_check - per candidate (w12 = 0..2047): true iff this exact
// w12 carries the checksum its entropy implies.
// ---------------------------------------------------------------------------
__device__ __forceinline__ bool checksum_fast_check(const ChecksumFastState_t* st,
                                                    uint16_t w12)
{
    return checksum_fast_cs4(st, w12) == (uint32_t)(w12 & 0x0Fu);
}

#endif // CHECKSUM_FAST_CUH
