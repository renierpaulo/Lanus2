/*
 * test_e8_wide.cpp - E8 verification: the WIDE ENGINE (all 128 chains
 * resident in one thread, advancing together iteration-by-iteration).
 *
 *   1. Wide path over the target base: all 128 seeds byte-exact versus the
 *      classic string-path PBKDF2 (proven reference); candidate 1666 hits
 *      the documented seed.
 *   2. Wide path over 5 random bases: all 128 seeds each, byte-exact.
 *
 * Build: cl /O2 /EHsc test_e8_wide.cpp
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

static uint64_t rng_state = 0x31313131CAFEF00DULL;
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

static void classic_seed(const uint16_t* phrase12, uint8_t* seed64) {
    unsigned char vocab[WLBP_VOCAB_BYTES];
    wlbp_thread_vocab_init(vocab, WLBP_BLOB);
    uint8_t mnemonic[256];
    int mn_len = 0;
    unsigned char word[9];
    for (int w = 0; w < 12; w++) {
        int len = wlbp_decode_word(vocab, phrase12[w], word);
        if (w > 0) mnemonic[mn_len++] = ' ';
        memcpy(mnemonic + mn_len, word, len);
        mn_len += len;
    }
    pbkdf2_sha512_mnemonic(mnemonic, mn_len, (const uint8_t*)"mnemonic", 8, 2048, seed64);
}

/* THE WIDE ENGINE (replicated exactly from main.cu process_base_vocabulary) */
static void wide_engine(const uint64_t table[2048], const uint16_t* base11,
                        uint8_t seeds[128][64], int* n_out) {
    uint16_t cand[12];
    memcpy(cand, base11, 11 * sizeof(uint16_t));

    uint64_t prefix_w[16], cur;
    uint32_t bits = 0, prefix_len = 0;
    int wi = 0;
    wlbp_base_prefix_build(table, base11, prefix_w, &cur, &bits, &wi, &prefix_len);

    uint16_t valid_w12[128];
    int n_valid = 0;
    for (uint32_t w12 = 0; w12 < 2048; w12++) {
        cand[11] = (uint16_t)w12;
        if (verify_checksum_12(cand)) {
            if (n_valid < 128) valid_w12[n_valid] = (uint16_t)w12;
            n_valid++;
        }
    }

    volatile uint64_t inpre[128][8], outpre[128][8], U[128][8], T[128][8];

    for (int c = 0; c < n_valid; c++) {
        cand[11] = valid_w12[c];
        uint64_t wb = table[valid_w12[c]];
        uint32_t len = (uint32_t)wlbp_wb_len(wb);
        uint64_t key[16];
        wlbp_keyblock_tail_swap(prefix_w, cur, bits, wi, wb, len, key);

        uint32_t i0 = (prefix_len + len + 7u) / 8u;
        if (i0 > 16u) i0 = 16u;
        uint64_t ip[16], op[16];
        for (int i = 0; i < 16; i++) {
            ip[i] = key[i] ^ 0x3636363636363636ULL;
            op[i] = key[i] ^ 0x5c5c5c5c5c5c5c5cULL;
        }
        SHA512State_t s;
        sha512_init_state_opt(&s);
        sha512_compress16_rot_opt(&s, ip, i0, d_KP36);
        for (int j = 0; j < 8; j++) inpre[c][j] = s.h[j];
        sha512_init_state_opt(&s);
        sha512_compress16_rot_opt(&s, op, i0, d_KP5c);
        for (int j = 0; j < 8; j++) outpre[c][j] = s.h[j];
    }

    for (int c = 0; c < n_valid; c++) {
        SHA512State_t s;
        for (int j = 0; j < 8; j++) s.h[j] = inpre[c][j];
        sha512_transform_saltblock_folded_opt(&s);
        SHA512State_t w;
        for (int j = 0; j < 8; j++) w.h[j] = outpre[c][j];
        sha512_finish64msg_fast(&w, s.h);
        for (int j = 0; j < 8; j++) { U[c][j] = w.h[j]; T[c][j] = w.h[j]; }
    }

    for (uint32_t iter = 1; iter < 2048; iter++) {
        int g = 0;
        for (; g + 3 < n_valid; g += 4) {
            SHA512State_t cA, cB, cC, cD;
            for (int j = 0; j < 8; j++) {
                cA.h[j] = inpre[g][j];     cB.h[j] = inpre[g + 1][j];
                cC.h[j] = inpre[g + 2][j]; cD.h[j] = inpre[g + 3][j];
            }
            sha512_finish64msg_x4_fast(&cA, (const uint64_t*)U[g], &cB, (const uint64_t*)U[g + 1],
                                       &cC, (const uint64_t*)U[g + 2], &cD, (const uint64_t*)U[g + 3]);
            SHA512State_t wA, wB, wC, wD;
            for (int j = 0; j < 8; j++) {
                wA.h[j] = outpre[g][j];     wB.h[j] = outpre[g + 1][j];
                wC.h[j] = outpre[g + 2][j]; wD.h[j] = outpre[g + 3][j];
            }
            sha512_finish64msg_x4_fast(&wA, cA.h, &wB, cB.h, &wC, cC.h, &wD, cD.h);
            for (int j = 0; j < 8; j++) {
                U[g][j] = wA.h[j];     T[g][j] ^= wA.h[j];
                U[g + 1][j] = wB.h[j]; T[g + 1][j] ^= wB.h[j];
                U[g + 2][j] = wC.h[j]; T[g + 2][j] ^= wC.h[j];
                U[g + 3][j] = wD.h[j]; T[g + 3][j] ^= wD.h[j];
            }
        }
        for (int c = g; c < n_valid; c++) {
            SHA512State_t s;
            for (int j = 0; j < 8; j++) s.h[j] = inpre[c][j];
            sha512_finish64msg_fast(&s, (const uint64_t*)U[c]);
            SHA512State_t w;
            for (int j = 0; j < 8; j++) w.h[j] = outpre[c][j];
            sha512_finish64msg_fast(&w, s.h);
            for (int j = 0; j < 8; j++) { U[c][j] = w.h[j]; T[c][j] ^= w.h[j]; }
        }
    }

    for (int c = 0; c < n_valid; c++) {
        for (int j = 0; j < 8; j++) {
            uint64_t x = T[c][j];
            seeds[c][j*8+0] = (uint8_t)(x >> 56); seeds[c][j*8+1] = (uint8_t)(x >> 48);
            seeds[c][j*8+2] = (uint8_t)(x >> 40); seeds[c][j*8+3] = (uint8_t)(x >> 32);
            seeds[c][j*8+4] = (uint8_t)(x >> 24); seeds[c][j*8+5] = (uint8_t)(x >> 16);
            seeds[c][j*8+6] = (uint8_t)(x >> 8);  seeds[c][j*8+7] = (uint8_t)(x);
        }
    }
    *n_out = n_valid;
}

int main(void) {
    int fails = 0;

    uint64_t table[2048];
    {
        unsigned char vocab[WLBP_VOCAB_BYTES];
        wlbp_thread_vocab_init(vocab, WLBP_BLOB);
        wlbp_build_repr_table(vocab, table);
    }

    /* ---- check 1: target base through the wide engine ---- */
    {
        uint16_t base11[11] = {759, 1078, 213, 623, 521, 319, 416, 302, 566, 1104, 191};
        const uint16_t target_w12 = 1666;
        const char* expected_hex =
            "4cd832a5c862ef5117870c15bdf792ed558499a2c81d8bd5c68f28bf0f66671b"
            "76e6522d90fb4996cc29d1a7b37ccf0bcc8157dea21f5f065e89921841323ab8";
        uint8_t expected[64];
        for (int i = 0; i < 64; i++) { unsigned v; sscanf(expected_hex + 2*i, "%2x", &v); expected[i] = (uint8_t)v; }

        /* collect the valid list (same enumeration the engine uses) */
        uint16_t cand[12];
        memcpy(cand, base11, 11 * sizeof(uint16_t));
        uint16_t valid_w12[128];
        int nv = 0;
        for (uint32_t w12 = 0; w12 < 2048; w12++) {
            cand[11] = (uint16_t)w12;
            if (verify_checksum_12(cand)) valid_w12[nv++] = (uint16_t)w12;
        }

        uint8_t seeds[128][64];
        int n = 0;
        wide_engine(table, base11, seeds, &n);
        printf("check1 wide engine: %d chains resident\n", n);
        if (n != 128) { printf("  FAIL count\n"); fails++; }

        /* all 128 seeds vs classic */
        int bad = 0;
        uint16_t phrase[12];
        memcpy(phrase, base11, 11 * sizeof(uint16_t));
        for (int c = 0; c < n; c++) {
            phrase[11] = valid_w12[c];
            uint8_t ref[64];
            classic_seed(phrase, ref);
            if (memcmp(seeds[c], ref, 64) != 0) bad++;
        }
        printf("  all-128 vs classic: %s\n", bad == 0 ? "PASS" : "FAIL");
        if (bad) fails++;

        /* candidate 1666 hit */
        int hit = 0;
        memcpy(phrase, base11, 11 * sizeof(uint16_t));
        for (int c = 0; c < n; c++) {
            phrase[11] = valid_w12[c];
            if (phrase[11] == target_w12 && memcmp(seeds[c], expected, 64) == 0) hit = 1;
        }
        printf("  target candidate 1666 seed: %s\n", hit ? "MATCH" : "MISMATCH");
        if (!hit) fails++;
    }

    /* ---- check 2: 5 random bases, all 128 seeds vs classic ---- */
    for (int t = 0; t < 5 && fails == 0; t++) {
        uint16_t base11[11];
        for (int w = 0; w < 11; w++) base11[w] = (uint16_t)(rng_next() % 2048);
        uint8_t seeds[128][64];
        int n = 0;
        wide_engine(table, base11, seeds, &n);
        if (n != 128) { printf("check2 FAIL t=%d: %d chains\n", t, n); fails++; break; }

        uint16_t cand[12];
        memcpy(cand, base11, 11 * sizeof(uint16_t));
        uint16_t valid_w12[128];
        int nv = 0;
        for (uint32_t w12 = 0; w12 < 2048; w12++) {
            cand[11] = (uint16_t)w12;
            if (verify_checksum_12(cand)) valid_w12[nv++] = (uint16_t)w12;
        }

        int bad = 0;
        uint16_t phrase[12];
        memcpy(phrase, base11, 11 * sizeof(uint16_t));
        for (int c = 0; c < n; c++) {
            phrase[11] = valid_w12[c];
            uint8_t ref[64];
            classic_seed(phrase, ref);
            if (memcmp(seeds[c], ref, 64) != 0) bad++;
        }
        printf("check2 base %d: 128 wide seeds vs classic: %s (%d bad)\n", t, bad == 0 ? "PASS" : "FAIL", bad);
        if (bad) fails++;
    }

    printf("RESULT: %s\n", fails == 0 ? "PASS - E8 wide engine verified" : "FAIL");
    return fails == 0 ? 0 : 1;
}
