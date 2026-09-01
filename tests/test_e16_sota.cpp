/*
 * test_e16_sota.cpp - E16 verification: the SOTA compression (LOP3 + fused
 * KW + __rotrll) must be byte-exact versus the existing fast compression,
 * and pbkdf2_sha512_solo must match pbkdf2_sha512_keyblock_fast.
 *
 * Build: cl /O2 /EHsc test_e16_sota.cpp /Fe:test_e16_sota.exe
 */
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#define __device__
#define __constant__ static
#define __forceinline__ static
#define __align__(x)

#include "src/pbkdf2_opt.cuh"
#include "src/wlbp.cuh"

static uint64_t rng_state = 0x50DA7A0000000001ULL;
static uint64_t rng_next(void) {
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7; rng_state ^= rng_state << 17;
    return rng_state;
}

int main(void) {
    int fails = 0;

    /* ---- check 1: SOTA compression == fast compression (500 random) ---- */
    for (int t = 0; t < 500; t++) {
        SHA512State_t state_f, state_s;
        uint64_t msg[8];
        for (int i = 0; i < 8; i++) msg[i] = rng_next();
        for (int j = 0; j < 8; j++) { state_f.h[j] = rng_next(); state_s.h[j] = state_f.h[j]; }

        sha512_finish64msg_fast(&state_f, msg);
        sha512_finish64msg_sota(&state_s, msg);

        if (memcmp(state_f.h, state_s.h, 64) != 0) {
            printf("check1 FAIL t=%d\n", t);
            fails++;
            break;
        }
    }
    printf("check1 SOTA compression == fast (500 rounds): %s\n", fails == 0 ? "PASS" : "FAIL");

    /* ---- check 2: solo PBKDF2 == keyblock_fast (200 random keys) ---- */
    for (int t = 0; t < 200; t++) {
        uint64_t key[16];
        for (int i = 0; i < 16; i++) key[i] = rng_next();
        uint32_t klen = 60 + (uint32_t)(rng_next() % 48); /* 60-107 */

        uint8_t s_fast[64], s_solo[64];
        pbkdf2_sha512_keyblock_fast(key, klen, 2048, s_fast); // uses fast directly
        pbkdf2_sha512_solo(key, klen, 2048, s_solo);

        if (memcmp(s_fast, s_solo, 64) != 0) {
            printf("check2 FAIL t=%d klen=%u\n", t, klen);
            fails++;
            break;
        }
    }
    printf("check2 solo PBKDF2 == keyblock_fast (200 keys): %s\n", fails == 0 ? "PASS" : "FAIL");

    /* ---- check 3: documented target seed through the SOTA path ---- */
    {
        uint16_t target[12] = {759, 1078, 213, 623, 521, 319, 416, 302, 566, 1104, 191, 1666};
        unsigned char vocab[WLBP_VOCAB_BYTES];
        wlbp_thread_vocab_init(vocab, WLBP_BLOB);
        uint64_t table[2048];
        wlbp_build_repr_table(vocab, table);

        uint64_t prefix_w[16], cur;
        uint32_t bits = 0, prefix_len = 0;
        int wi = 0;
        wlbp_base_prefix_build(table, target, prefix_w, &cur, &bits, &wi, &prefix_len);
        uint64_t wb = table[1666];
        uint64_t key[16];
        wlbp_keyblock_tail_swap(prefix_w, cur, bits, wi, wb, (uint32_t)wlbp_wb_len(wb), key);

        uint8_t seed[64];
        pbkdf2_sha512_solo(key, prefix_len + (uint32_t)wlbp_wb_len(wb), 2048, seed);

        const char* expected_hex =
            "4cd832a5c862ef5117870c15bdf792ed558499a2c81d8bd5c68f28bf0f66671b"
            "76e6522d90fb4996cc29d1a7b37ccf0bcc8157dea21f5f065e89921841323ab8";
        uint8_t expected[64];
        for (int i = 0; i < 64; i++) { unsigned v; sscanf(expected_hex + 2*i, "%2x", &v); expected[i] = (uint8_t)v; }

        int ok = memcmp(seed, expected, 64) == 0;
        printf("check3 target seed via SOTA solo: %s\n", ok ? "MATCH" : "MISMATCH");
        if (!ok) fails++;
    }

    printf("RESULT: %s\n", fails == 0 ? "PASS - E16 SOTA verified" : "FAIL");
    return fails == 0 ? 0 : 1;
}
