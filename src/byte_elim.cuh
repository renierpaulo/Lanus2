/*
 * byte_elim.cuh â€” ZERO byte-array round-trips from PBKDF2 output to target.
 *
 * Pipeline (all uint64_t registers, one convention):
 *   PBKDF2 T[8] -> master_hmac_fast_u64 -> I[8]
 *     -> bip32_derive_u64 (m/44'/0'/0'/0/0) -> private key (4 u64)
 *     -> pubkey_hash160_u64 (windowed EC -> SHA-256 -> RIPEMD-160) -> hash160 (5 u64)
 *     -> derive_and_match_u64 -> target compare
 *
 * Word convention: a big-endian byte string is packed MSB-first into uint64_t
 * words â€” word 0 holds bytes 0..7 with byte 0 in bits 63..56. This IS the
 * SHA-512 digest state layout, so values flow between stages as pure register
 * moves: no byte split (pbkdf2_sha512_keyblock_fast's output loop), no byte
 * repack (master_hmac_fast's input loop) â€” the T accumulator words from the
 * PBKDF2 family are consumed directly.
 *
 * hash160 representation: 5 u64; word i (i<4) holds digest bytes 4i..4i+3 with
 * byte 4i in bits 63..56; word 4 holds the last 4 RIPEMD bytes in its low 32
 * bits (upper 32 zero).
 *
 * Requirements: the fixed-G window table must be initialized once before any
 * kernel launch (secp256k1_gwin_upload) â€” main.cu already does this.
 * Include after (or rely on the extern declarations of) d_num_targets /
 * d_target_hashes_ptr from main.cu's device-globals block.
 */

#ifndef BYTE_ELIM_CUH
#define BYTE_ELIM_CUH

// ---------------------------------------------------------------------------
// Host shim (same pattern as bip32_fast.cuh): lets this header compile with a
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
#endif

#include <stdint.h>
#include <string.h>   // size_t for sha512.cuh under host compilation

#include "sha512.cuh"     // K512, sigma/gamma_512, ch64/maj64 â€” round primitives only
#include "secp256k1.cuh"  // windowed fixed-G scalar mult + field arithmetic
#include "keccak256.cuh"  // ETH: keccak256(pubkey uncompressed)[12..32)

// device globals shared with main.cu (single-TU: defined here, include guard protects)
__constant__ uint8_t* d_target_hashes_ptr;
__constant__ uint32_t d_num_targets;

// lanus CLI parity: coin selector + bloom filter state
__constant__ uint32_t d_coin;            // 0 = BTC, 60 = ETH
__constant__ uint32_t d_use_bloom;
__device__ uint8_t* d_bloom_bits = nullptr;
__constant__ uint64_t d_bloom_m_bits;
__constant__ uint32_t d_bloom_k;

// ---------------------------------------------------------------------------
// Small helpers (unique be_ names â€” this TU also hosts sha256.cuh/ripemd160.cuh)
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint32_t be_rotr32(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

__device__ __forceinline__ uint32_t be_rotl32(uint32_t x, uint32_t n) {
    return (x << n) | (x >> (32 - n));
}

__device__ __forceinline__ uint32_t be_bswap32(uint32_t v) {
#ifdef __CUDACC__
    return __byte_perm(v, 0, 0x0123);
#else
    return (v << 24) | ((v << 8) & 0x00FF0000u) | ((v >> 8) & 0x0000FF00u) | (v >> 24);
#endif
}

__device__ __forceinline__ void be_sha512_iv(uint64_t st[8]) {
    st[0] = 0x6a09e667f3bcc908ULL; st[1] = 0xbb67ae8584caa73bULL;
    st[2] = 0x3c6ef372fe94f82bULL; st[3] = 0xa54ff53a5f1d36f1ULL;
    st[4] = 0x510e527fade682d1ULL; st[5] = 0x9b05688c2b3e6c1fULL;
    st[6] = 0x1f83d9abfb41bd6bULL; st[7] = 0x5be0cd19137e2179ULL;
}

// ---------------------------------------------------------------------------
// SHA-512 block compressions (rotating 16-word schedule, registers only).
// Same math as pbkdf2_opt.cuh's proven compressors, without the PBKDF2-only
// round folding (the BIP32 side runs ~20 compressions per candidate vs 4098
// in PBKDF2 â€” schedule folding is irrelevant here, correctness is king).
// ---------------------------------------------------------------------------

// full 128-byte block given as 16 big-endian W-words
__device__ __forceinline__ void be_sha512_compress16(uint64_t st[8], const uint64_t m[16]) {
    uint64_t a = st[0], b = st[1], c = st[2], d = st[3];
    uint64_t e = st[4], f = st[5], g = st[6], h = st[7];
    uint64_t t1, t2;
    uint64_t W[16];

    #pragma unroll
    for (int t = 0; t < 16; t++) W[t] = m[t];

    #pragma unroll
    for (int t = 0; t < 16; t++) {
        t1 = h + sigma1_512(e) + ch64(e, f, g) + K512[t] + W[t];
        t2 = sigma0_512(a) + maj64(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    #pragma unroll
    for (int t = 16; t < 80; t++) {
        W[t & 15] += gamma1_512(W[(t - 2) & 15]) + W[(t - 7) & 15] + gamma0_512(W[(t - 15) & 15]);
        t1 = h + sigma1_512(e) + ch64(e, f, g) + K512[t] + W[t & 15];
        t2 = sigma0_512(a) + maj64(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    st[0] += a; st[1] += b; st[2] += c; st[3] += d;
    st[4] += e; st[5] += f; st[6] += g; st[7] += h;
}

// final block over a 64-byte message given as 8 big-endian words + fixed pad
__device__ __forceinline__ void be_sha512_finish64(uint64_t st[8], const uint64_t u[8]) {
    uint64_t a = st[0], b = st[1], c = st[2], d = st[3];
    uint64_t e = st[4], f = st[5], g = st[6], h = st[7];
    uint64_t t1, t2;
    uint64_t W[16];

    #pragma unroll
    for (int t = 0; t < 8; t++) W[t] = u[t];

    W[8]  = 0x8000000000000000ULL;
    W[9]  = 0; W[10] = 0; W[11] = 0;
    W[12] = 0; W[13] = 0; W[14] = 0;
    W[15] = 0x0000000000000600ULL; // (128 + 64) * 8 bits

    #pragma unroll
    for (int t = 0; t < 16; t++) {
        t1 = h + sigma1_512(e) + ch64(e, f, g) + K512[t] + W[t];
        t2 = sigma0_512(a) + maj64(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    #pragma unroll
    for (int t = 16; t < 80; t++) {
        W[t & 15] += gamma1_512(W[(t - 2) & 15]) + W[(t - 7) & 15] + gamma0_512(W[(t - 15) & 15]);
        t1 = h + sigma1_512(e) + ch64(e, f, g) + K512[t] + W[t & 15];
        t2 = sigma0_512(a) + maj64(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    st[0] += a; st[1] += b; st[2] += c; st[3] += d;
    st[4] += e; st[5] += f; st[6] += g; st[7] += h;
}

// ---------------------------------------------------------------------------
// 1) master_hmac_fast_u64 â€” HMAC-SHA512("Bitcoin seed", seed) with the seed
//    ALREADY in its natural big-endian u64 form (PBKDF2 T[8] words).
//    Fixed-key ipad/opad midstates (values from gen_master_pre.cpp, identical
//    to pbkdf2_opt.cuh d_min_pre/d_mout_pre): 2 compressions, no pad work,
//    no byte split, no byte repack.
// ---------------------------------------------------------------------------

__constant__ uint64_t d_be_m_in_pre[8] = {
    0x2e2af459060c1873ULL, 0x7894b868dc88433aULL, 0xdd1a797ef1a1933aULL, 0xe6486d04fcb412a7ULL,
    0xfbcc67b9a396caa0ULL, 0xa2970b146f49b65eULL, 0xfdf1daabc66f6248ULL, 0x2ff99c812ada6dc3ULL
};
__constant__ uint64_t d_be_m_out_pre[8] = {
    0xbbd27bac212e9dbdULL, 0xdd0bc55e7e4037c1ULL, 0xdfdd3d6890bd6424ULL, 0x2902de663032b34cULL,
    0xa30f8aa6f67899fcULL, 0x69a566c30f88378fULL, 0x0500247985ecb694ULL, 0xf6d70307c6b2d337ULL
};

__device__ __forceinline__ void master_hmac_fast_u64(
    const uint64_t seed_w[8],        // in:  PBKDF2 output T[8], big-endian words
    uint64_t I[8]                    // out: I[0..3] = master key, I[4..7] = chaincode
) {
    uint64_t inner[8], outer[8];

    #pragma unroll
    for (int j = 0; j < 8; j++) inner[j] = d_be_m_in_pre[j];
    be_sha512_finish64(inner, seed_w);       // inner: midstate + seed block

    #pragma unroll
    for (int j = 0; j < 8; j++) outer[j] = d_be_m_out_pre[j];
    be_sha512_finish64(outer, inner);        // outer: midstate + inner digest words

    #pragma unroll
    for (int j = 0; j < 8; j++) I[j] = outer[j];
}

// ---------------------------------------------------------------------------
// 2) bip32_derive_u64 â€” full BIP32 chain m/44'/0'/0'/0/0 in u64 registers.
// ---------------------------------------------------------------------------

// secp256k1 group order n, 4 big-endian u64
__constant__ uint64_t d_be_n[4] = {
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFEULL,
    0xBAAEDCE6AF48A03BULL, 0xBFD25E8CD0364141ULL
};

#define BE_IPAD_W 0x3636363636363636ULL
#define BE_OPAD_W 0x5c5c5c5c5c5c5c5cULL
#define BE_MSG_PAD 0x00800000ULL     // 0x80 SHA terminator at byte 37 of block 2
#define BE_MSG_BITS 1320ULL          // (128 + 37) * 8

// lexicographic compare s >= n over 4 BE words
__device__ __forceinline__ bool be_ge_n(uint64_t s0, uint64_t s1, uint64_t s2, uint64_t s3) {
    if (s0 != d_be_n[0]) return s0 > d_be_n[0];
    if (s1 != d_be_n[1]) return s1 > d_be_n[1];
    if (s2 != d_be_n[2]) return s2 > d_be_n[2];
    return s3 >= d_be_n[3];
}

// s -= n in place
__device__ __forceinline__ void be_sub_n(uint64_t& s0, uint64_t& s1, uint64_t& s2, uint64_t& s3) {
    uint64_t bw = (s3 < d_be_n[3]) ? 1ULL : 0ULL;
    uint64_t r3 = s3 - d_be_n[3];
    uint64_t r2 = s2 - d_be_n[2] - bw;
    bw = ((s2 < d_be_n[2]) || (bw && s2 == d_be_n[2])) ? 1ULL : 0ULL;
    uint64_t r1 = s1 - d_be_n[1] - bw;
    bw = ((s1 < d_be_n[1]) || (bw && s1 == d_be_n[1])) ? 1ULL : 0ULL;
    uint64_t r0 = s0 - d_be_n[0] - bw;
    s0 = r0; s1 = r1; s2 = r2; s3 = r3;
}

// child = (IL + parent) mod n â€” sum < 2^256 + n, so at most two subtractions
__device__ __forceinline__ void be_child_add(
    const uint64_t parent_k[4], const uint64_t IL[4], uint64_t child_k[4]
) {
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

    if (c || be_ge_n(s0, s1, s2, s3)) be_sub_n(s0, s1, s2, s3);
    if (be_ge_n(s0, s1, s2, s3))      be_sub_n(s0, s1, s2, s3);

    child_k[0] = s0; child_k[1] = s1; child_k[2] = s2; child_k[3] = s3;
}

// compressed pubkey of scalar k: affine X as 4 BE words + 0x02/0x03 parity
// prefix. uint256_t (8x u32, d[7] = MSB) maps to/from 4 BE u64 with pure
// shifts â€” no byte buffers anywhere.
__device__ __forceinline__ void be_pubkey_words(
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

// BIP32 message ser_P(k)||ser32(i) â€” 37 bytes packed into 5 BE words
// (low 24 bits of word 4 zero; the 0x80 marker lands there at finish time).
// head = 0x00 hardened (xw = parent key), 0x02/0x03 non-hardened (pubkey X).
__device__ __forceinline__ void be_msg_words(
    uint32_t head, const uint64_t xw[4], uint32_t index, uint64_t mw[5]
) {
    mw[0] = ((uint64_t)head << 56) | (xw[0] >> 8);
    mw[1] = (xw[0] << 56) | (xw[1] >> 8);
    mw[2] = (xw[1] << 56) | (xw[2] >> 8);
    mw[3] = (xw[2] << 56) | (xw[3] >> 8);
    mw[4] = (xw[3] << 56) | ((uint64_t)index << 24);
}

// HMAC-SHA512(chaincode, 37B message) â€” ipad/opad midstates from the 32-byte
// chaincode words, message block completed in W-word form, inner digest words
// feed the outer finish directly. Byte-exact replacement of
// hmac_sha512(parent_chaincode, 32, data, 37, I).
__device__ __forceinline__ void be_bip32_hmac(
    const uint64_t parent_cc[4],     // parent chaincode, 4 BE words
    const uint64_t msg_w[5],         // message as 5 BE words
    uint64_t I[8]                    // out: digest, 8 BE words
) {
    uint64_t blk[16];
    blk[0] = msg_w[0];
    blk[1] = msg_w[1];
    blk[2] = msg_w[2];
    blk[3] = msg_w[3];
    blk[4] = msg_w[4] | BE_MSG_PAD;
    #pragma unroll
    for (int i = 5; i < 15; i++) blk[i] = 0;
    blk[15] = BE_MSG_BITS;

    uint64_t ip[16], op[16];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        ip[i] = parent_cc[i] ^ BE_IPAD_W;
        op[i] = parent_cc[i] ^ BE_OPAD_W;
    }
    #pragma unroll
    for (int i = 4; i < 16; i++) {
        ip[i] = BE_IPAD_W;
        op[i] = BE_OPAD_W;
    }

    uint64_t inner[8];
    be_sha512_iv(inner);
    be_sha512_compress16(inner, ip);
    be_sha512_compress16(inner, blk);

    uint64_t outer[8];
    be_sha512_iv(outer);
    be_sha512_compress16(outer, op);
    be_sha512_finish64(outer, inner);

    #pragma unroll
    for (int i = 0; i < 8; i++) I[i] = outer[i];
}

// one BIP32 level â€” entirely in u64 registers
__device__ __forceinline__ void be_derive_level(
    const uint64_t parent_k[4],      // parent private key, 4 BE words
    const uint64_t parent_cc[4],     // parent chaincode, 4 BE words
    uint32_t index,
    bool hardened,
    uint64_t child_k[4],             // out: child private key, 4 BE words
    uint64_t child_cc[4]             // out: child chaincode, 4 BE words
) {
    uint64_t mw[5], I[8];

    if (hardened) {
        be_msg_words(0x00u, parent_k, index | 0x80000000u, mw);
    } else {
        uint64_t xw[4];
        uint32_t prefix;
        be_pubkey_words(parent_k, xw, &prefix);
        be_msg_words(prefix, xw, index, mw);
    }

    be_bip32_hmac(parent_cc, mw, I);   // IL = I[0..3], IR = I[4..7]

    be_child_add(parent_k, I, child_k);
    child_cc[0] = I[4]; child_cc[1] = I[5];
    child_cc[2] = I[6]; child_cc[3] = I[7];
}

// full chain m/44'/coin'/0'/0/0 â€” zero byte arrays end to end
__device__ __forceinline__ void bip32_derive_u64_coin(
    const uint64_t master_k[4],      // in: master key from I[0..3]
    const uint64_t master_cc[4],     // in: master chaincode from I[4..7]
    uint32_t coin,                   // in: BIP44 coin type (0 = BTC, 60 = ETH)
    uint64_t priv_out[4]             // out: derived key, 4 BE words
) {
    uint64_t k[4], cc[4], tk[4], tcc[4];

    be_derive_level(master_k, master_cc, 44u, true,  k,  cc);   // m/44'
    be_derive_level(k,  cc,  coin, true,  tk, tcc);   // m/44'/coin'
    be_derive_level(tk, tcc,  0u, true,  k,  cc);   // m/44'/coin'/0'
    be_derive_level(k,  cc,   0u, false, tk, tcc);   // m/44'/coin'/0'/0
    be_derive_level(tk, tcc,  0u, false, priv_out, tcc); // m/44'/coin'/0'/0/0
}

// full chain m/44'/0'/0'/0/0 â€” zero byte arrays end to end
__device__ __forceinline__ void bip32_derive_u64(
    const uint64_t master_k[4],      // in: master key from I[0..3]
    const uint64_t master_cc[4],     // in: master chaincode from I[4..7]
    uint64_t priv_out[4]             // out: m/44'/0'/0'/0/0 key, 4 BE words
) {
    bip32_derive_u64_coin(master_k, master_cc, 0u, priv_out);
}

// ---------------------------------------------------------------------------
// 3) pubkey_hash160_u64 â€” private key (4 BE words) -> hash160 (5 u64).
//    Windowed EC (scalar_mult_G_window) -> one-block SHA-256 over the 33-byte
//    compressed key built by bit-shuffling the X words -> one-block RIPEMD-160
//    over the digest words. No uint8_t anywhere.
// ---------------------------------------------------------------------------

__constant__ uint32_t d_be_k256[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

// one-block SHA-256 over a message fully described by 16 prebuilt W words
__device__ __forceinline__ void be_sha256_block(uint32_t st[8], const uint32_t Win[16]) {
    uint32_t W[64];

    #pragma unroll
    for (int i = 0; i < 16; i++) W[i] = Win[i];

    #pragma unroll
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = be_rotr32(W[i - 15], 7) ^ be_rotr32(W[i - 15], 18) ^ (W[i - 15] >> 3);
        uint32_t s1 = be_rotr32(W[i - 2], 17) ^ be_rotr32(W[i - 2], 19) ^ (W[i - 2] >> 10);
        W[i] = W[i - 16] + s0 + W[i - 7] + s1;
    }

    uint32_t a = st[0], b = st[1], c = st[2], d = st[3];
    uint32_t e = st[4], f = st[5], g = st[6], h = st[7];

    #pragma unroll
    for (int i = 0; i < 64; i++) {
        uint32_t S1 = be_rotr32(e, 6) ^ be_rotr32(e, 11) ^ be_rotr32(e, 25);
        uint32_t chv = (e & f) ^ (~e & g);
        uint32_t t1 = h + S1 + chv + d_be_k256[i] + W[i];
        uint32_t S0 = be_rotr32(a, 2) ^ be_rotr32(a, 13) ^ be_rotr32(a, 22);
        uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = S0 + mj;
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    st[0] += a; st[1] += b; st[2] += c; st[3] += d;
    st[4] += e; st[5] += f; st[6] += g; st[7] += h;
}

__constant__ uint32_t d_be_rmd_kl[5] = { 0x00000000u, 0x5A827999u, 0x6ED9EBA1u, 0x8F1BBCDCu, 0xA953FD4Eu };
__constant__ uint32_t d_be_rmd_kr[5] = { 0x50A28BE6u, 0x5C4DD124u, 0x6D703EF3u, 0x7A6D76E9u, 0x00000000u };
__constant__ uint8_t d_be_rmd_rl[80] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    7, 4, 13, 1, 10, 6, 15, 3, 12, 0, 9, 5, 2, 14, 11, 8,
    3, 10, 14, 4, 9, 15, 8, 1, 2, 7, 0, 6, 13, 11, 5, 12,
    1, 9, 11, 10, 0, 8, 12, 4, 13, 3, 7, 15, 14, 5, 6, 2,
    4, 0, 5, 9, 7, 12, 2, 10, 14, 1, 3, 8, 11, 6, 15, 13
};
__constant__ uint8_t d_be_rmd_rr[80] = {
    5, 14, 7, 0, 9, 2, 11, 4, 13, 6, 15, 8, 1, 10, 3, 12,
    6, 11, 3, 7, 0, 13, 5, 10, 14, 15, 8, 12, 4, 9, 1, 2,
    15, 5, 1, 3, 7, 14, 6, 9, 11, 8, 12, 2, 10, 0, 4, 13,
    8, 6, 4, 1, 3, 11, 15, 0, 5, 12, 2, 13, 9, 7, 10, 14,
    12, 15, 10, 4, 1, 5, 8, 7, 6, 2, 13, 14, 0, 3, 9, 11
};
__constant__ uint8_t d_be_rmd_sl[80] = {
    11, 14, 15, 12, 5, 8, 7, 9, 11, 13, 14, 15, 6, 7, 9, 8,
    7, 6, 8, 13, 11, 9, 7, 15, 7, 12, 15, 9, 11, 7, 13, 12,
    11, 13, 6, 7, 14, 9, 13, 15, 14, 8, 13, 6, 5, 12, 7, 5,
    11, 12, 14, 15, 14, 15, 9, 8, 9, 14, 5, 6, 8, 6, 5, 12,
    9, 15, 5, 11, 6, 8, 13, 12, 5, 12, 13, 14, 11, 8, 5, 6
};
__constant__ uint8_t d_be_rmd_sr[80] = {
    8, 9, 9, 11, 13, 15, 15, 5, 7, 7, 8, 11, 14, 14, 12, 6,
    9, 13, 15, 7, 12, 8, 9, 11, 7, 7, 12, 7, 6, 15, 13, 11,
    9, 7, 15, 11, 8, 6, 6, 14, 12, 13, 5, 14, 13, 13, 7, 5,
    15, 5, 8, 11, 14, 14, 6, 14, 6, 9, 12, 9, 12, 5, 15, 8,
    8, 5, 12, 9, 12, 5, 14, 6, 8, 13, 6, 5, 15, 13, 11, 11
};

// one-block RIPEMD-160 over a message fully described by 16 prebuilt X words
__device__ __forceinline__ void be_ripemd160_block(uint32_t st[5], const uint32_t X[16]) {
    uint32_t al = st[0], bl = st[1], cl = st[2], dl = st[3], el = st[4];
    uint32_t ar = st[0], br = st[1], cr = st[2], dr = st[3], er = st[4];

    #pragma unroll
    for (int j = 0; j < 80; j++) {
        const int rnd = j >> 4;
        uint32_t f, t;

        if (rnd == 0)      f = bl ^ cl ^ dl;
        else if (rnd == 1) f = (bl & cl) | (~bl & dl);
        else if (rnd == 2) f = (bl | ~cl) ^ dl;
        else if (rnd == 3) f = (bl & dl) | (cl & ~dl);
        else               f = bl ^ (cl | ~dl);
        t = be_rotl32(al + f + X[d_be_rmd_rl[j]] + d_be_rmd_kl[rnd], d_be_rmd_sl[j]) + el;
        al = el; el = dl; dl = be_rotl32(cl, 10); cl = bl; bl = t;

        if (rnd == 0)      f = br ^ (cr | ~dr);
        else if (rnd == 1) f = (br & dr) | (cr & ~dr);
        else if (rnd == 2) f = (br | ~cr) ^ dr;
        else if (rnd == 3) f = (br & cr) | (~br & dr);
        else               f = br ^ cr ^ dr;
        t = be_rotl32(ar + f + X[d_be_rmd_rr[j]] + d_be_rmd_kr[rnd], d_be_rmd_sr[j]) + er;
        ar = er; er = dr; dr = be_rotl32(cr, 10); cr = br; br = t;
    }

    uint32_t t = st[1] + cl + dr;
    st[1] = st[2] + dl + er;
    st[2] = st[3] + el + ar;
    st[3] = st[4] + al + br;
    st[4] = st[0] + bl + cr;
    st[0] = t;
}

// private key (4 BE words) -> hash160 (5 u64): word i = digest bytes 4i..4i+3
// BE-packed (byte 4i in bits 63..56); word 4 = last 4 RIPEMD bytes in the low
// 32 bits. Requires the G window table (secp256k1_gwin_upload) beforehand.
__device__ __forceinline__ void pubkey_hash160_u64(
    const uint64_t priv[4],          // in: private key, 4 BE words
    uint64_t h160[5]                 // out: hash160, 5 words (word 4 low 32 bits)
) {
    uint256_t kk;
    kk.d[7] = (uint32_t)(priv[0] >> 32); kk.d[6] = (uint32_t)priv[0];
    kk.d[5] = (uint32_t)(priv[1] >> 32); kk.d[4] = (uint32_t)priv[1];
    kk.d[3] = (uint32_t)(priv[2] >> 32); kk.d[2] = (uint32_t)priv[2];
    kk.d[1] = (uint32_t)(priv[3] >> 32); kk.d[0] = (uint32_t)priv[3];

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

    const uint32_t prefix = (y_aff.d[0] & 1u) ? 0x03u : 0x02u;
    const uint64_t xw0 = ((uint64_t)x_aff.d[7] << 32) | (uint64_t)x_aff.d[6];
    const uint64_t xw1 = ((uint64_t)x_aff.d[5] << 32) | (uint64_t)x_aff.d[4];
    const uint64_t xw2 = ((uint64_t)x_aff.d[3] << 32) | (uint64_t)x_aff.d[2];
    const uint64_t xw3 = ((uint64_t)x_aff.d[1] << 32) | (uint64_t)x_aff.d[0];

    // SHA-256 over the 33-byte compressed key: 0x80 marker at byte 33,
    // 264-bit length â€” single block, W built from the X words by shifting.
    uint32_t W[16];
    W[0]  = (prefix << 24) | (uint32_t)(xw0 >> 40);
    W[1]  = (uint32_t)(xw0 >> 8);
    W[2]  = ((uint32_t)xw0 << 24) | (uint32_t)(xw1 >> 40);
    W[3]  = (uint32_t)(xw1 >> 8);
    W[4]  = ((uint32_t)xw1 << 24) | (uint32_t)(xw2 >> 40);
    W[5]  = (uint32_t)(xw2 >> 8);
    W[6]  = ((uint32_t)xw2 << 24) | (uint32_t)(xw3 >> 40);
    W[7]  = (uint32_t)(xw3 >> 8);
    W[8]  = ((uint32_t)(xw3 & 0xFFu) << 24) | 0x00800000u;
    W[9]  = 0; W[10] = 0; W[11] = 0; W[12] = 0; W[13] = 0;
    W[14] = 0;
    W[15] = 264u;                    // 33 * 8 bits

    uint32_t sh[8] = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
    };
    be_sha256_block(sh, W);

    // RIPEMD-160 over the 32-byte digest: X words are LITTLE-endian views of
    // the digest bytes, the SHA-256 state words are BIG-endian packed -> X = bswap.
    uint32_t X[16];
    #pragma unroll
    for (int i = 0; i < 8; i++) X[i] = be_bswap32(sh[i]);
    X[8]  = 0x00000080u;             // 0x80 marker at byte 32
    X[9]  = 0; X[10] = 0; X[11] = 0; X[12] = 0; X[13] = 0;
    X[14] = 0x00000100u;             // 256 bits, little-endian low word
    X[15] = 0;

    uint32_t rs[5] = { 0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u, 0xC3D2E1F0u };
    be_ripemd160_block(rs, X);

    // RIPEMD serializes its state little-endian; hash160 words are the digest
    // bytes big-endian-packed -> one bswap per word, upper 32 bits zero.
    #pragma unroll
    for (int i = 0; i < 5; i++) h160[i] = (uint64_t)be_bswap32(rs[i]);
}

// private key (4 BE words) -> ETH address (20 bytes). ETH: addr =
// keccak256(X||Y uncompressed 64B)[12..32). Same affine normalization as
// pubkey_hash160_u64; the 64-byte serialization feeds keccak directly.
__device__ __forceinline__ void pubkey_eth_addr20(
    const uint64_t priv[4],          // in: private key, 4 BE words
    uint8_t addr20[20]               // out: ETH address, 20 bytes
) {
    uint256_t kk;
    kk.d[7] = (uint32_t)(priv[0] >> 32); kk.d[6] = (uint32_t)priv[0];
    kk.d[5] = (uint32_t)(priv[1] >> 32); kk.d[4] = (uint32_t)priv[1];
    kk.d[3] = (uint32_t)(priv[2] >> 32); kk.d[2] = (uint32_t)priv[2];
    kk.d[1] = (uint32_t)(priv[3] >> 32); kk.d[0] = (uint32_t)priv[3];

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

    // serialize X||Y big-endian (64 bytes)
    uint8_t pubU[64];
    #pragma unroll
    for (int w = 0; w < 8; w++) {
        uint32_t hi = x_aff.d[7 - w], lo = y_aff.d[7 - w];
        uint8_t* px = pubU + w * 4;
        uint8_t* py = pubU + 32 + w * 4;
        px[0] = (uint8_t)(hi >> 24); px[1] = (uint8_t)(hi >> 16);
        px[2] = (uint8_t)(hi >> 8);  px[3] = (uint8_t)(hi);
        py[0] = (uint8_t)(lo >> 24); py[1] = (uint8_t)(lo >> 16);
        py[2] = (uint8_t)(lo >> 8);  py[3] = (uint8_t)(lo);
    }

    uint8_t kh[32];
    keccak256(pubU, 64, kh);
    #pragma unroll
    for (int i = 0; i < 20; i++) addr20[i] = kh[12 + i];
}

// ---------------------------------------------------------------------------
// 4) derive_and_match_u64 â€” T[8] (PBKDF2 output, big-endian words) straight
//    into the master HMAC, through BIP32, EC, hash160, to the target compare.
//    The seed never exists as bytes; the only byte writes are the FINAL
//    serialization of a found private key into the host-visible output buffer.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void derive_and_match_u64(
    const uint64_t T[8],             // PBKDF2 output words (big-endian seed)
    const uint16_t* indices12,
    uint32_t* found_count,
    uint8_t* found_privkeys,
    uint16_t* found_indices
) {
    // master: I[0..3] = key, I[4..7] = chaincode (words in, words out)
    uint64_t I[8];
    master_hmac_fast_u64(T, I);

    // BIP32 m/44'/coin'/0'/0/0 (coin 0 = BTC, 60 = ETH)
    uint64_t priv[4];
    bip32_derive_u64_coin(I, I + 4, d_coin, priv);

    // 20-byte digest as 5 big-endian words (word-shaped; BTC never touches a
    // byte array, ETH serializes the pubkey once for keccak)
    uint32_t dig5[5];
    if (d_coin == 60u) {
        uint8_t addr20[20];
        pubkey_eth_addr20(priv, addr20);
        #pragma unroll
        for (int w = 0; w < 5; w++) {
            dig5[w] = ((uint32_t)addr20[w * 4 + 0] << 24) |
                      ((uint32_t)addr20[w * 4 + 1] << 16) |
                      ((uint32_t)addr20[w * 4 + 2] << 8)  |
                      ((uint32_t)addr20[w * 4 + 3]);
        }
    } else {
        uint64_t h160[5];
        pubkey_hash160_u64(priv, h160);
        #pragma unroll
        for (int w = 0; w < 5; w++) dig5[w] = (uint32_t)h160[w];
    }

    // optional bloom prefilter (FNV-1a over the digest words, device-side
    // bit test identical to the host build in main.cu; miss = early out)
    if (d_use_bloom) {
        uint64_t h1 = 1469598103934665603ULL, h2 = 1469598103934665603ULL;
        #pragma unroll
        for (int w = 0; w < 5; w++) {
            uint32_t v = dig5[w];
            for (int b = 0; b < 4; b++) {
                h1 ^= (v >> 24) & 0xFFu; h1 *= 1099511628211ULL; v <<= 8;
            }
        }
        #pragma unroll
        for (int w = 4; w >= 0; w--) {
            uint32_t v = dig5[w];
            for (int b = 0; b < 4; b++) {
                h2 ^= (v >> 24) & 0xFFu; h2 *= 1099511628211ULL; v <<= 8;
            }
        }
        h2 ^= 0x5A5A5A5A5A5A5A5AULL;
        bool bloom_hit = true;
        for (uint32_t j = 0; j < d_bloom_k && bloom_hit; j++) {
            uint64_t combined = h1 + j * h2;
            uint64_t bit_index = combined % d_bloom_m_bits;
            uint8_t bit_mask = (uint8_t)(1u << (bit_index & 7u));
            if (!(d_bloom_bits[bit_index >> 3] & bit_mask)) bloom_hit = false;
        }
        if (!bloom_hit) return;   // cannot be a target
    }

    // target compare: pack each 20-byte target word on the fly (global reads
    // only â€” our digest never touches a byte array)
    bool found = false;
    for (uint32_t t = 0; t < d_num_targets && !found; t++) {
        const uint8_t* th = d_target_hashes_ptr + t * 20u;
        bool match = true;
        #pragma unroll
        for (int w = 0; w < 5; w++) {
            uint32_t tw = ((uint32_t)th[w * 4 + 0] << 24) |
                          ((uint32_t)th[w * 4 + 1] << 16) |
                          ((uint32_t)th[w * 4 + 2] << 8)  |
                          ((uint32_t)th[w * 4 + 3]);
            if (dig5[w] != tw) { match = false; break; }
        }
        if (match) found = true;
    }

    if (found) {
#ifdef __CUDACC__
        uint32_t slot = atomicAdd(found_count, 1u);
#else
        uint32_t slot = *found_count; (*found_count)++;
#endif
        if (slot < 100u) {
            // final output serialization only â€” the key leaves the pipeline here
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                uint64_t x = priv[j];
                uint8_t* p = found_privkeys + slot * 32u + (uint32_t)j * 8u;
                p[0] = (uint8_t)(x >> 56); p[1] = (uint8_t)(x >> 48);
                p[2] = (uint8_t)(x >> 40); p[3] = (uint8_t)(x >> 32);
                p[4] = (uint8_t)(x >> 24); p[5] = (uint8_t)(x >> 16);
                p[6] = (uint8_t)(x >> 8);  p[7] = (uint8_t)(x);
            }
            #pragma unroll
            for (int i = 0; i < 12; i++) found_indices[slot * 12u + (uint32_t)i] = indices12[i];
        }
    }
}

#endif // BYTE_ELIM_CUH
