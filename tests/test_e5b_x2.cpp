/*
 * test_e5b_x2.cpp - E5b verification: x2-interleaved PBKDF2 must be
 * byte-exact versus two independent single-chain runs.
 *
 *   1. 200 random pairs (same base prefix, different 12th words):
 *      keyblock_fast_x2(A,B) == { keyblock_fast(A), keyblock_fast(B) }
 *   2. Target phrase through the PAIR path == documented seed
 *      (pair: target + a neighbor candidate from its base's valid list)
 *
 * Build: cl /O2 /EHsc test_e5b_x2.cpp
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

static uint64_t rng_state = 0x5DEECE66D1234ULL;
static uint64_t rng_next(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

int main(void) {
    int fails = 0;

    unsigned char vocab[WLBP_VOCAB_BYTES];
    wlbp_thread_vocab_init(vocab, WLBP_BLOB);
    uint64_t table[2048];
    wlbp_build_repr_table(vocab, table);

    /* ---- check 1: x2 == 2x single (200 random pairs, same base) ---- */
    for (int t = 0; t < 200; t++) {
        uint16_t base11[11];
        for (int w = 0; w < 11; w++) base11[w] = (uint16_t)(rng_next() % 2048);

        uint64_t prefix_w[16], cur;
        uint32_t bits = 0, prefix_len = 0;
        int wi = 0;
        wlbp_base_prefix_build(table, base11, prefix_w, &cur, &bits, &wi, &prefix_len);

        uint16_t wA = (uint16_t)(rng_next() % 2048);
        uint16_t wB = (uint16_t)(rng_next() % 2048);

        uint64_t wbA = table[wA], wbB = table[wB];
        uint32_t lenA = (uint32_t)wlbp_wb_len(wbA), lenB = (uint32_t)wlbp_wb_len(wbB);
        uint64_t keyA[16], keyB[16];
        wlbp_keyblock_tail_swap(prefix_w, cur, bits, wi, wbA, lenA, keyA);
        wlbp_keyblock_tail_swap(prefix_w, cur, bits, wi, wbB, lenB, keyB);

        uint8_t sxA[64], sxB[64], sA[64], sB[64];
        pbkdf2_sha512_keyblock_fast_x2(keyA, prefix_len + lenA, keyB, prefix_len + lenB, 2048, sxA, sxB);
        pbkdf2_sha512_keyblock_fast(keyA, prefix_len + lenA, 2048, sA);
        pbkdf2_sha512_keyblock_fast(keyB, prefix_len + lenB, 2048, sB);

        if (memcmp(sxA, sA, 64) != 0 || memcmp(sxB, sB, 64) != 0) {
            printf("check1 FAIL t=%d (wA=%u wB=%u)\n", t, wA, wB);
            fails++;
            break;
        }
    }
    printf("check1 x2 interleaved == 2x single (200 pairs, byte-exact): %s\n",
           fails == 0 ? "PASS" : "FAIL");

    /* ---- check 2: target phrase through the PAIR path == documented seed ---- */
    {
        uint16_t base11[11] = {759, 1078, 213, 623, 521, 319, 416, 302, 566, 1104, 191};
        uint16_t wA = 1666; /* the known target candidate */

        uint64_t prefix_w[16], cur;
        uint32_t bits = 0, prefix_len = 0;
        int wi = 0;
        wlbp_base_prefix_build(table, base11, prefix_w, &cur, &bits, &wi, &prefix_len);

        /* neighbor candidate: next valid-ish index (any other word) */
        uint16_t wB = (wA + 1) & 2047; if (wB == 0) wB = 1;

        uint64_t wbA = table[wA], wbB = table[wB];
        uint64_t keyA[16], keyB[16];
        wlbp_keyblock_tail_swap(prefix_w, cur, bits, wi, wbA, (uint32_t)wlbp_wb_len(wbA), keyA);
        wlbp_keyblock_tail_swap(prefix_w, cur, bits, wi, wbB, (uint32_t)wlbp_wb_len(wbB), keyB);

        uint8_t sxA[64], sxB[64];
        pbkdf2_sha512_keyblock_fast_x2(keyA, prefix_len + (uint32_t)wlbp_wb_len(wbA),
                                       keyB, prefix_len + (uint32_t)wlbp_wb_len(wbB),
                                       2048, sxA, sxB);

        const char* expected_hex =
            "4cd832a5c862ef5117870c15bdf792ed558499a2c81d8bd5c68f28bf0f66671b"
            "76e6522d90fb4996cc29d1a7b37ccf0bcc8157dea21f5f065e89921841323ab8";
        uint8_t expected[64];
        for (int i = 0; i < 64; i++) { unsigned v; sscanf(expected_hex + 2*i, "%2x", &v); expected[i] = (uint8_t)v; }

        int ok = memcmp(sxA, expected, 64) == 0;
        printf("check2 target candidate through x2 pair path: %s\n", ok ? "MATCH" : "MISMATCH");
        if (!ok) fails++;
    }

    printf("RESULT: %s\n", fails == 0 ? "PASS - E5b x2 interleave is byte-exact" : "FAIL");
    return fails == 0 ? 0 : 1;
}
