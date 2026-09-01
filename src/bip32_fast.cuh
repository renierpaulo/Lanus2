/*
 * BIP32 fast derivation — ZERO byte arrays (u64-register pipeline)
 *
 * Replaces the per-level pattern of main.cu:derive_child_key
 *   hmac_sha512(parent_chaincode, 32, data, 37, I)   // builds k_ipad[128],
 *                                                    // k_opad[128], W[80]...
 *   secp256k1_scalar_add(I, parent_key, child_key)   // byte round-trips
 * with an all-uint64_t pipeline:
 *   - ipad/opad blocks built directly as 16 SHA-512 W-words:
 *       W[0..3]  = chaincode ^ 0x36/0x5c pattern   (chaincode = 4 u64)
 *       W[4..15] = pure pad constants
 *     compressed with sha512_compress16_rot_opt using fold_from = 4
 *     (rounds 4..15 folded into d_KP36/d_KP5c — W[4..15] are constants).
 *   - message block: W[0..4] = 37B data (ser_P(k)||ser32(i')), W[5..14] = 0,
 *     W[15] = 1320 ((128 + 37) * 8), 0x80 marker OR-ed at byte 37.
 *   - outer finish via sha512_finish64msg_fast (the 64B inner digest words
 *     feed the compression directly — no byte serialization).
 *   - child_key_add: (IL + parent) mod n in 4x uint64 BE words.
 *   - non-hardened: parent pubkey X/prefix produced as u64 words
 *     (scalar_mult_G_window output limbs packed without byte buffers).
 *
 * All 32-byte values are BIG-ENDIAN packed uint64_t[4] (byte 0 of the 32B
 * value is the MSB of word 0) — the same convention as the SHA-512 digest
 * state words produced by sha512_extract_opt, so digest words flow straight
 * into the next compression with no repacking.
 *
 * Usage (replaces the hmac_sha512 + memcpy pattern in derive_and_match):
 *   uint64_t priv4[4];
 *   bip32_chain_seed_fast(seed, priv4);   // seed = 64B BIP39 seed (8 BE u64)
 *   // priv4 = m/44'/0'/0'/0/0 private key
 *
 * Note: scalar_mult_G_window requires the fixed-G window tables to be
 * initialized once (secp256k1_gwin_upload() on device / secp256k1_gwin_build()
 * on host) — main.cu already does this before launching derivation kernels.
 */

#ifndef BIP32_FAST_CUH
#define BIP32_FAST_CUH

// ---------------------------------------------------------------------------
// Host shim: lets this header (and the primitives it pulls in) compile with a
// plain C++ compiler. NVCC defines all of these natively.
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
  #ifndef __align__
    #define __align__(x) alignas(x)
  #endif
#endif

#include <stdint.h>

#include "sha512.cuh"       // base SHA-512 primitives (K512, sigma/gamma, ...)
#include "pbkdf2_opt.cuh"   // sha512_compress16_rot_opt, sha512_finish64msg_fast,
                            // SHA512State_t, d_KP36/d_KP5c, d_min_pre/d_mout_pre
#include "secp256k1.cuh"    // point math for the non-hardened pubkey path

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

// secp256k1 group order n = FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE BAAEDCE6
//                           AF48A03B BFD25E8C D0364141, as 4 big-endian u64.
__constant__ uint64_t d_bip32_n[4] = {
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFEULL,
    0xBAAEDCE6AF48A03BULL, 0xBFD25E8CD0364141ULL
};

#define BIP32_IPAD_W   0x3636363636363636ULL
#define BIP32_OPAD_W   0x5c5c5c5c5c5c5c5cULL
#define BIP32_MSG_PAD  0x00800000ULL   // 0x80 SHA terminator at byte 37 of block 2
#define BIP32_MSG_BITS 1320ULL         // (128 + 37) * 8

// ---------------------------------------------------------------------------
// Non-hardened support: parent pubkey as u64 words (no byte buffers)
// ---------------------------------------------------------------------------

// Compressed pubkey of private scalar k: affine X as 4 BE words + parity
// prefix (0x02 even Y / 0x03 odd Y). Limb packing only — uint256_t (8x u32,
// d[7] = most significant) maps to 4 BE u64 without any byte round-trip.
__device__ __forceinline__ void bip32_pubkey_words(
    const uint64_t k[4],
    uint64_t xw[4],
    uint32_t* prefix
) {
    uint256_t kk;
    kk.d[7] = (uint32_t)(k[0] >> 32); kk.d[6] = (uint32_t)k[0];
    kk.d[5] = (uint32_t)(k[1] >> 32); kk.d[4] = (uint32_t)k[1];
    kk.d[3] = (uint32_t)(k[2] >> 32); kk.d[2] = (uint32_t)k[2];
    kk.d[1] = (uint32_t)(k[3] >> 32); kk.d[0] = (uint32_t)k[3];

    point_t P;
    scalar_mult_G_window(&P, &kk);

    uint256_t x_aff, y_aff;
    if (P.infinity) {
        uint256_clear(&x_aff);
        uint256_clear(&y_aff);
    } else {
        uint256_t zinv, z2inv, z3inv;
        uint256_mod_inv(&zinv, &P.z, &SECP256K1_P);
        uint256_mod_mul(&z2inv, &zinv, &zinv, &SECP256K1_P);
        uint256_mod_mul(&z3inv, &z2inv, &zinv, &SECP256K1_P);
        uint256_mod_mul(&x_aff, &P.x, &z2inv, &SECP256K1_P);
        uint256_mod_mul(&y_aff, &P.y, &z3inv, &SECP256K1_P);
    }

    *prefix = (y_aff.d[0] & 1u) ? 0x03u : 0x02u;
    xw[0] = ((uint64_t)x_aff.d[7] << 32) | (uint64_t)x_aff.d[6];
    xw[1] = ((uint64_t)x_aff.d[5] << 32) | (uint64_t)x_aff.d[4];
    xw[2] = ((uint64_t)x_aff.d[3] << 32) | (uint64_t)x_aff.d[2];
    xw[3] = ((uint64_t)x_aff.d[1] << 32) | (uint64_t)x_aff.d[0];
}

// BIP32 HMAC message ser_P(parent) || ser32(index) — 37 bytes packed into
// 5 BE words: words 0..3 = bytes 0..31, word 4 = bytes 32..36 in the top
// 40 bits (low 24 bits zero — the 0x80 marker goes there at finish time).
// head = 0x00 for hardened (xw = parent key), 0x02/0x03 for non-hardened
// (xw = parent pubkey X).
__device__ __forceinline__ void bip32_msg_words(
    uint32_t head, const uint64_t xw[4], uint32_t index, uint64_t mw[5]
) {
    mw[0] = ((uint64_t)head << 56) | (xw[0] >> 8);
    mw[1] = (xw[0] << 56) | (xw[1] >> 8);
    mw[2] = (xw[1] << 56) | (xw[2] >> 8);
    mw[3] = (xw[2] << 56) | (xw[3] >> 8);
    mw[4] = (xw[3] << 56) | ((uint64_t)index << 24);
}

// ---------------------------------------------------------------------------
// Core HMAC: 2 midstate compressions + 2 finishes, zero byte arrays
// ---------------------------------------------------------------------------

// HMAC-SHA512 over a message block already expressed as 16 SHA-512 W-words
// (W[0..4] = data, W[5..14] = 0, W[15] = 1320). Inner: ipad midstate
// (fold_from = 4) + message block; outer: opad midstate (fold_from = 4) +
// inner digest as a plain 64B message block.
__device__ __forceinline__ void bip32_hmac_block_fast(
    const uint64_t parent_cc[4],     // parent chaincode, 4 BE words
    const uint64_t msg_block[16],    // full 2nd block in W-word form
    uint64_t I[8]                    // HMAC-SHA512 digest, 8 BE words
) {
    uint64_t ip[16], op[16];

    // ipad/opad blocks: W[0..3] = chaincode ^ pad, W[4..15] = pad constants
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        ip[i] = parent_cc[i] ^ BIP32_IPAD_W;
        op[i] = parent_cc[i] ^ BIP32_OPAD_W;
    }
    #pragma unroll
    for (int i = 4; i < 16; i++) {
        ip[i] = BIP32_IPAD_W;
        op[i] = BIP32_OPAD_W;
    }

    // inner: IV -> ipad block (rounds 4..15 folded) -> message block
    SHA512State_t inner;
    sha512_init_state_opt(&inner);
    sha512_compress16_rot_opt(&inner, ip, 4u, d_KP36);
    sha512_compress16_rot_opt(&inner, msg_block, 16u, d_KP36); // fold_from=16: no fold

    // outer: IV -> opad block (rounds 4..15 folded) -> 64B inner digest
    SHA512State_t outer;
    sha512_init_state_opt(&outer);
    sha512_compress16_rot_opt(&outer, op, 4u, d_KP5c);
    sha512_finish64msg_fast(&outer, inner.h);

    #pragma unroll
    for (int i = 0; i < 8; i++) I[i] = outer.h[i];
}

// HMAC-SHA512 of the 37-byte BIP32 message (already packed as 5 W-words by
// bip32_msg_words). Completes SHA padding in W-word form: 0x80 at byte 37,
// zero words 5..14, bit length 1320 in word 15. Byte-exact replacement of
// hmac_sha512(parent_chaincode, 32, data, 37, I).
__device__ __forceinline__ void bip32_hmac_fast(
    const uint64_t parent_cc[4],     // parent chaincode, 4 BE words
    const uint64_t msg_w[5],         // message as 5 BE words (low 24 bits of
                                     // word 4 must be zero)
    uint64_t I[8]                    // HMAC-SHA512 digest, 8 BE words
) {
    uint64_t blk[16];
    blk[0] = msg_w[0];
    blk[1] = msg_w[1];
    blk[2] = msg_w[2];
    blk[3] = msg_w[3];
    blk[4] = msg_w[4] | BIP32_MSG_PAD;
    #pragma unroll
    for (int i = 5; i < 15; i++) blk[i] = 0;
    blk[15] = BIP32_MSG_BITS;

    bip32_hmac_block_fast(parent_cc, blk, I);
}

// ---------------------------------------------------------------------------
// child_key_add: (IL + parent) mod n — 256-bit BE u64 arithmetic
// ---------------------------------------------------------------------------

// Lexicographic compare s >= n over 4 BE words.
__device__ __forceinline__ bool bip32_ge_n(
    uint64_t s0, uint64_t s1, uint64_t s2, uint64_t s3
) {
    if (s0 != d_bip32_n[0]) return s0 > d_bip32_n[0];
    if (s1 != d_bip32_n[1]) return s1 > d_bip32_n[1];
    if (s2 != d_bip32_n[2]) return s2 > d_bip32_n[2];
    return s3 >= d_bip32_n[3];
}

// s -= n in place (mod 2^256; the final borrow is the caller's concern and
// is only set when s < n, where the caller relies on 2^256 wraparound).
__device__ __forceinline__ void bip32_sub_n(
    uint64_t& s0, uint64_t& s1, uint64_t& s2, uint64_t& s3
) {
    uint64_t bw = (s3 < d_bip32_n[3]) ? 1ULL : 0ULL;
    uint64_t r3 = s3 - d_bip32_n[3];
    uint64_t r2 = s2 - d_bip32_n[2] - bw;
    bw = ((s2 < d_bip32_n[2]) || (bw && s2 == d_bip32_n[2])) ? 1ULL : 0ULL;
    uint64_t r1 = s1 - d_bip32_n[1] - bw;
    bw = ((s1 < d_bip32_n[1]) || (bw && s1 == d_bip32_n[1])) ? 1ULL : 0ULL;
    uint64_t r0 = s0 - d_bip32_n[0] - bw;
    s0 = r0; s1 = r1; s2 = r2; s3 = r3;
}

// child = (IL + parent_k) mod n. IL and parent_k are 4 BE words; IL may be
// any 256-bit value (sum < 3n, so at most two subtractions).
__device__ __forceinline__ void child_key_add(
    const uint64_t parent_k[4], const uint64_t IL[4], uint64_t child_k[4]
) {
    // 256-bit add, carry from LSB (word 3) to MSB (word 0)
    uint64_t s3 = IL[3] + parent_k[3];
    uint64_t c  = (s3 < IL[3]) ? 1ULL : 0ULL;

    uint64_t t  = IL[2] + parent_k[2];
    uint64_t s2 = t + c;
    c = ((t < IL[2]) || (s2 < t)) ? 1ULL : 0ULL;

    t = IL[1] + parent_k[1];
    uint64_t s1 = t + c;
    c = ((t < IL[1]) || (s1 < t)) ? 1ULL : 0ULL;

    t = IL[0] + parent_k[0];
    uint64_t s0 = t + c;
    c = ((t < IL[0]) || (s0 < t)) ? 1ULL : 0ULL;

    // reduce mod n (a carry into 2^256 forces the first subtraction)
    if (c || bip32_ge_n(s0, s1, s2, s3)) bip32_sub_n(s0, s1, s2, s3);
    if (bip32_ge_n(s0, s1, s2, s3))      bip32_sub_n(s0, s1, s2, s3);

    child_k[0] = s0; child_k[1] = s1; child_k[2] = s2; child_k[3] = s3;
}

// ---------------------------------------------------------------------------
// One BIP32 level — entirely in u64 registers
// ---------------------------------------------------------------------------

__device__ __forceinline__ void bip32_derive_fast(
    const uint64_t parent_k[4],      // parent private key, 4 BE words
    const uint64_t parent_cc[4],     // parent chaincode, 4 BE words
    uint32_t index,
    bool hardened,
    uint64_t child_k[4],             // out: child private key, 4 BE words
    uint64_t child_cc[4]             // out: child chaincode, 4 BE words
) {
    uint64_t mw[5], I[8];

    if (hardened) {
        // data = 0x00 || ser256(parent_k) || ser32(index')
        bip32_msg_words(0x00u, parent_k, index | 0x80000000u, mw);
    } else {
        // data = serP(point(parent_k)) || ser32(index)
        uint64_t xw[4];
        uint32_t prefix;
        bip32_pubkey_words(parent_k, xw, &prefix);
        bip32_msg_words(prefix, xw, index, mw);
    }

    // I = HMAC-SHA512(parent_cc, data): IL = I[0..3], IR = I[4..7]
    bip32_hmac_fast(parent_cc, mw, I);

    child_key_add(parent_k, I, child_k);
    child_cc[0] = I[4]; child_cc[1] = I[5];
    child_cc[2] = I[6]; child_cc[3] = I[7];
}

// ---------------------------------------------------------------------------
// Master key: HMAC-SHA512("Bitcoin seed", seed64) — u64 in, u64 out
// ---------------------------------------------------------------------------

// Uses the fixed-key "Bitcoin seed" ipad/opad midstates (d_min_pre/d_mout_pre,
// generated from the real K512) — 2 compressions per candidate, no pad work.
// Byte-exact equivalent of master_hmac_fast / hmac_sha512("Bitcoin seed", ...).
__device__ __forceinline__ void bip32_master_fast(
    const uint64_t seed_w[8],        // 64-byte seed, 8 BE words
    uint64_t master_k[4],            // out: m/ private key, 4 BE words
    uint64_t master_cc[4]            // out: m/ chaincode, 4 BE words
) {
    SHA512State_t inner;
    #pragma unroll
    for (int j = 0; j < 8; j++) inner.h[j] = d_min_pre[j];
    sha512_finish64msg_fast(&inner, seed_w);

    SHA512State_t outer;
    #pragma unroll
    for (int j = 0; j < 8; j++) outer.h[j] = d_mout_pre[j];
    sha512_finish64msg_fast(&outer, inner.h);

    master_k[0] = outer.h[0]; master_k[1] = outer.h[1];
    master_k[2] = outer.h[2]; master_k[3] = outer.h[3];
    master_cc[0] = outer.h[4]; master_cc[1] = outer.h[5];
    master_cc[2] = outer.h[6]; master_cc[3] = outer.h[7];
}

// ---------------------------------------------------------------------------
// Full chain m/44'/0'/0'/0/0 — zero byte arrays end to end
// ---------------------------------------------------------------------------

__device__ __forceinline__ void bip32_chain_fast(
    const uint64_t seed_w[8],        // 64-byte BIP39 seed, 8 BE words
    uint64_t priv_out[4]             // out: m/44'/0'/0'/0/0 key, 4 BE words
) {
    uint64_t mk[4], mcc[4], k[4], cc[4], tk[4], tcc[4];

    bip32_master_fast(seed_w, mk, mcc);

    bip32_derive_fast(mk, mcc, 44u, true,  k,  cc);    // m/44'
    bip32_derive_fast(k,  cc,   0u, true,  tk, tcc);   // m/44'/0'
    bip32_derive_fast(tk, tcc,  0u, true,  k,  cc);    // m/44'/0'/0'
    bip32_derive_fast(k,  cc,   0u, false, tk, tcc);   // m/44'/0'/0'/0
    bip32_derive_fast(tk, tcc,  0u, false, priv_out, tcc); // m/44'/0'/0'/0/0
}

// Convenience entry point matching derive_and_match's current input shape:
// loads the 64-byte seed into 8 BE words (registers) and runs the chain.
__device__ __forceinline__ void bip32_chain_seed_fast(
    const uint8_t* seed64,
    uint64_t priv_out[4]
) {
    uint64_t seed_w[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        uint64_t v = 0;
        #pragma unroll
        for (int b = 0; b < 8; b++) v = (v << 8) | (uint64_t)seed64[j * 8 + b];
        seed_w[j] = v;
    }
    bip32_chain_fast(seed_w, priv_out);
}

#endif // BIP32_FAST_CUH
