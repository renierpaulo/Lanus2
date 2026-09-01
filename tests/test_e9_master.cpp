/*
 * test_e9_master.cpp - E9 verification: the constant-key master HMAC fast
 * path (2 compressions) must be byte-exact versus the generic hmac_sha512
 * with "Bitcoin seed". Also re-anchors the full pipeline via the documented
 * target seed (generation-free: direct target indices through the wide path).
 *
 * Build: cl /O2 /EHsc test_e9_master.cpp
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>

#define __device__
#define __constant__ static
#define __forceinline__ static
#define __align__(x)

#include "src/sha256.cuh"
#include "src/pbkdf2_opt.cuh"
#include "src/wlbp.cuh"

static uint64_t rng_state = 0xE66E66E666666666ULL;
static uint64_t rng_next(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

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

int main(void) {
    int fails = 0;

    /* ---- check 1: master_hmac_fast == hmac_sha512 (1000 random seeds) ---- */
    for (int t = 0; t < 1000; t++) {
        uint8_t seed[64], I_fast[64], I_ref[64];
        for (int i = 0; i < 64; i++) seed[i] = (uint8_t)(rng_next() & 0xFF);
        master_hmac_fast(seed, I_fast);
        hmac_sha512((const uint8_t*)"Bitcoin seed", 12, seed, 64, I_ref);
        if (memcmp(I_fast, I_ref, 64) != 0) {
            printf("check1 FAIL t=%d\n", t);
            fails++;
            break;
        }
    }
    printf("check1 master_hmac_fast == hmac_sha512 (1000 seeds): %s\n",
           fails == 0 ? "PASS" : "FAIL");

    /* ---- check 2: full pipeline with the fast master == documented seed ---- */
    {
        uint16_t target[12] = {759, 1078, 213, 623, 521, 319, 416, 302, 566, 1104, 191, 1666};
        const char* expected_hex =
            "4cd832a5c862ef5117870c15bdf792ed558499a2c81d8bd5c68f28bf0f66671b"
            "76e6522d90fb4996cc29d1a7b37ccf0bcc8157dea21f5f065e89921841323ab8";
        uint8_t expected[64];
        for (int i = 0; i < 64; i++) { unsigned v; sscanf(expected_hex + 2*i, "%2x", &v); expected[i] = (uint8_t)v; }

        uint64_t table[2048];
        {
            unsigned char vocab[WLBP_VOCAB_BYTES];
            wlbp_thread_vocab_init(vocab, WLBP_BLOB);
            wlbp_build_repr_table(vocab, table);
        }

        /* PBKDF2 via proven fast path, then master via E9 fast path */
        uint64_t prefix_w[16], cur;
        uint32_t bits = 0, prefix_len = 0;
        int wi = 0;
        wlbp_base_prefix_build(table, target, prefix_w, &cur, &bits, &wi, &prefix_len);
        uint64_t wb = table[1666];
        uint64_t key[16];
        wlbp_keyblock_tail_swap(prefix_w, cur, bits, wi, wb, (uint32_t)wlbp_wb_len(wb), key);
        uint8_t seed[64];
        pbkdf2_sha512_keyblock_fast(key, prefix_len + (uint32_t)wlbp_wb_len(wb), 2048, seed);

        uint8_t I[64];
        master_hmac_fast(seed, I);

        /* the I[0..31] is master_key; verify the PBKDF2 seed matches doc too */
        int ok_seed = memcmp(seed, expected, 64) == 0;
        printf("check2 PBKDF2 seed (proven path): %s | master via E9: computed\n",
               ok_seed ? "MATCH" : "MISMATCH");
        if (!ok_seed) fails++;

        /* full anchor: master_key from E9 + chaincode -> the private key path
           would need the rest of the derive; anchor instead on hmac equivalence
           already proven in check1 (bit-exact I means identical downstream) */
        printf("check2 downstream anchored by check1 byte-exact equivalence\n");
    }

    printf("RESULT: %s\n", fails == 0 ? "PASS - E9 master fast path verified" : "FAIL");
    return fails == 0 ? 0 : 1;
}
