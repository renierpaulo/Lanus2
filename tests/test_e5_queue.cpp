/*
 * test_e5_queue.cpp - E5a verification: compressed base queue.
 *   1. pack(base11) -> unpack -> identical 11 indices (100k random bases)
 *   2. full chain: pack -> unpack -> base prefix -> candidate 1666 tail swap
 *      -> PBKDF2 -> derive -> hash160 matches the documented target
 *      (the same proven E3 body, now fed through the compressed queue)
 * Build: cl /O2 /EHsc test_e5_queue.cpp
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>

#define __device__
#define __constant__ static
#define __forceinline__ static
#define __align__(x)

#include "src/sha256.cuh"
#include "src/sha512.cuh"
#include "src/ripemd160.cuh"
#include "src/secp256k1.cuh"
#include "src/pbkdf2_opt.cuh"
#include "src/wlbp.cuh"

/* ---- pack/unpack replicated exactly from main.cu kernels ---- */
static void pack_base(const uint16_t* indices, uint32_t* packed) {
    uint64_t acc = 0;
    uint32_t nb = 0, oi = 0;
    #pragma unroll
    for (int i = 0; i < 11; i++) {
        acc = (acc << 11) | (uint64_t)indices[i];
        nb += 11;
        if (nb >= 32) {
            nb -= 32;
            packed[oi++] = (uint32_t)(acc >> nb);
            acc &= (1ULL << nb) - 1;
        }
    }
    packed[oi] = (nb > 0) ? (uint32_t)(acc << (32 - nb)) : 0u;
    for (int i = oi + 1; i < 4; i++) packed[i] = 0;
}

static void unpack_base(const uint32_t* pw, uint16_t* base11) {
    uint32_t bitpos = 0;
    for (int i = 0; i < 11; i++) {
        uint32_t v = 0;
        for (int bb = 0; bb < 11; bb++) {
            uint32_t word = pw[bitpos >> 5];
            uint32_t bit = (word >> (31u - (bitpos & 31u))) & 1u;
            v = (v << 1) | bit;
            bitpos++;
        }
        base11[i] = (uint16_t)v;
    }
}

/* checksum + derivation (proven in E1/E3) */
__forceinline__ uint32_t rotr32(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

static void sha256_checksum_only(const uint8_t* entropy, int ent_bytes, uint8_t* first_byte) {
    uint32_t h0 = 0x6a09e667, h1 = 0xbb67ae85, h2 = 0x3c6ef372, h3 = 0xa54ff53a;
    uint32_t h4 = 0x510e527f, h5 = 0x9b05688c, h6 = 0x1f83d9ab, h7 = 0x5be0cd19;
    uint32_t w[64];
    for (int i = 0; i < 16; i++) {
        if (i < (ent_bytes + 3) / 4) {
            int base = i * 4;
            w[i] = 0;
            if (base < ent_bytes) w[i] |= (uint32_t)entropy[base] << 24;
            if (base + 1 < ent_bytes) w[i] |= (uint32_t)entropy[base + 1] << 16;
            if (base + 2 < ent_bytes) w[i] |= (uint32_t)entropy[base + 2] << 8;
            if (base + 3 < ent_bytes) w[i] |= (uint32_t)entropy[base + 3];
        } else if (i == ent_bytes / 4) {
            w[i] = 0x80000000 >> ((ent_bytes % 4) * 8);
        } else {
            w[i] = 0;
        }
    }
    w[15] = ent_bytes * 8;
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = rotr32(w[i-15], 7) ^ rotr32(w[i-15], 18) ^ (w[i-15] >> 3);
        uint32_t s1 = rotr32(w[i-2], 17) ^ rotr32(w[i-2], 19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    uint32_t a = h0, b = h1, c = h2, d = h3;
    uint32_t e = h4, f = h5, g = h6, h = h7;
    for (int i = 0; i < 64; i++) {
        uint32_t S1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t t1 = h + S1 + ch + K256[i] + w[i];
        uint32_t S0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = S0 + maj;
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    h0 += a;
    *first_byte = (h0 >> 24) & 0xFF;
}

static bool verify_checksum_12(const uint16_t* indices) {
    uint8_t entropy[16];
    uint32_t bits = 0;
    int bit_count = 0, byte_idx = 0;
    for (int w = 0; w < 12; w++) {
        for (int b = 10; b >= 0; b--) {
            bits = (bits << 1) | ((indices[w] >> b) & 1);
            if (++bit_count == 8) {
                if (byte_idx < 16) entropy[byte_idx++] = bits & 0xFF;
                bits = 0; bit_count = 0;
            }
        }
    }
    uint8_t cs;
    sha256_checksum_only(entropy, 16, &cs);
    return (uint8_t)(cs >> 4) == (uint8_t)(indices[11] & 0x0F);
}

static void derive_child_key(const uint8_t* parent_key, const uint8_t* parent_chaincode,
                             uint32_t index, uint8_t* child_key, uint8_t* child_chaincode,
                             bool hardened) {
    uint8_t data[37];
    uint8_t I[64];
    if (hardened) {
        index |= 0x80000000u;
        data[0] = 0x00;
        memcpy(data + 1, parent_key, 32);
    } else {
        uint8_t pubkey[33];
        secp256k1_get_pubkey_compressed(parent_key, pubkey);
        memcpy(data, pubkey, 33);
    }
    data[33] = (index >> 24) & 0xFF;
    data[34] = (index >> 16) & 0xFF;
    data[35] = (index >> 8) & 0xFF;
    data[36] = index & 0xFF;
    hmac_sha512(parent_chaincode, 32, data, 37, I);
    secp256k1_scalar_add(I, parent_key, child_key);
    memcpy(child_chaincode, I + 32, 32);
}

static uint64_t rng_state = 0xC0FFEE123456789ULL;
static uint64_t rng_next(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

int main(void) {
    int fails = 0;

    /* thread setup */
    unsigned char vocab[WLBP_VOCAB_BYTES];
    wlbp_thread_vocab_init(vocab, WLBP_BLOB);
    uint64_t table[2048];
    wlbp_build_repr_table(vocab, table);
    static uint256_t gxs[SECP_GWIN_ENTRIES], gys[SECP_GWIN_ENTRIES];
    secp256k1_gwin_build(gxs, gys);
    memcpy(d_Gwin_x, gxs, sizeof(gxs));
    memcpy(d_Gwin_y, gys, sizeof(gys));

    /* ---- check 1: pack/unpack round-trip (100k random bases) ---- */
    for (int t = 0; t < 100000; t++) {
        uint16_t base11[11];
        for (int i = 0; i < 11; i++) base11[i] = (uint16_t)(rng_next() % 2048);
        uint32_t packed[4];
        pack_base(base11, packed);
        uint16_t out[11];
        unpack_base(packed, out);
        if (memcmp(base11, out, sizeof(base11)) != 0) {
            printf("check1 FAIL t=%d\n", t);
            fails++;
            break;
        }
    }
    printf("check1 pack/unpack round-trip (100k bases): %s\n", fails == 0 ? "PASS" : "FAIL");

    /* ---- check 2: full chain through the compressed queue ---- */
    {
        uint16_t target[12] = {759, 1078, 213, 623, 521, 319, 416, 302, 566, 1104, 191, 1666};
        uint32_t packed[4];
        pack_base(target, packed);
        uint16_t base11[11];
        unpack_base(packed, base11);

        uint64_t prefix_w[16], cur;
        uint32_t bits = 0, prefix_len = 0;
        int wi = 0;
        wlbp_base_prefix_build(table, base11, prefix_w, &cur, &bits, &wi, &prefix_len);

        int found = 0;
        for (uint32_t w12 = 0; w12 < 2048; w12++) {
            uint16_t cand[12];
            memcpy(cand, base11, 11 * sizeof(uint16_t));
            cand[11] = (uint16_t)w12;
            if (!verify_checksum_12(cand)) continue;

            uint64_t wb12 = table[w12];
            uint64_t key_w[16];
            wlbp_keyblock_tail_swap(prefix_w, cur, bits, wi, wb12, (uint32_t)wlbp_wb_len(wb12), key_w);

            uint8_t seed[64];
            pbkdf2_sha512_keyblock_fast(key_w, prefix_len + (uint32_t)wlbp_wb_len(wb12), 2048, seed);

            uint8_t master_key[32], master_chaincode[32], I[64];
            hmac_sha512((const uint8_t*)"Bitcoin seed", 12, seed, 64, I);
            memcpy(master_key, I, 32);
            memcpy(master_chaincode, I + 32, 32);

            uint8_t key[32], cc[32], tk[32], tc[32];
            derive_child_key(master_key, master_chaincode, 44, key, cc, true);
            derive_child_key(key, cc, 0, tk, tc, true);  memcpy(key, tk, 32); memcpy(cc, tc, 32);
            derive_child_key(key, cc, 0, tk, tc, true);  memcpy(key, tk, 32); memcpy(cc, tc, 32);
            derive_child_key(key, cc, 0, tk, tc, false); memcpy(key, tk, 32); memcpy(cc, tc, 32);
            uint8_t priv[32];
            derive_child_key(key, cc, 0, priv, tc, false);

            uint8_t pub[33], sh[32], h160[20];
            secp256k1_get_pubkey_compressed(priv, pub);
            sha256(pub, 33, sh);
            ripemd160(sh, 32, h160);

            if (w12 == 1666) {
                const uint8_t expect[20] = {0x23,0x2f,0xb8,0xa4,0xbb,0x0b,0x8b,0xe8,0xda,0xeb,
                                            0x78,0xd9,0x02,0x2d,0x12,0x60,0x06,0x30,0x9c,0x5c};
                found = (memcmp(h160, expect, 20) == 0);
            }
        }
        printf("check2 compressed-queue base -> full pipeline -> target hash160: %s\n",
               found ? "MATCH" : "MISMATCH");
        if (!found) fails++;
    }

    printf("RESULT: %s\n", fails == 0 ? "PASS - E5a compressed queue verified" : "FAIL");
    return fails == 0 ? 0 : 1;
}
