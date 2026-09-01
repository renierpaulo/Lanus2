/*
 * test_e5c_x3.cpp - E5c verification: x3-interleaved PBKDF2 (compressed-state
 * interleave) must be byte-exact versus three independent single runs, and
 * the shipped 128-candidate split (42 triples + 1 pair) must reproduce the
 * documented target seed through whichever group holds candidate 1666.
 *
 * Build: cl /O2 /EHsc test_e5c_x3.cpp
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

static uint64_t rng_state = 0xABCDEF0123456789ULL;
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

    /* ---- check 1: x3 == 3x single (200 random triples, same base) ---- */
    for (int t = 0; t < 200; t++) {
        uint16_t base11[11];
        for (int w = 0; w < 11; w++) base11[w] = (uint16_t)(rng_next() % 2048);

        uint64_t prefix_w[16], cur;
        uint32_t bits = 0, prefix_len = 0;
        int wi = 0;
        wlbp_base_prefix_build(table, base11, prefix_w, &cur, &bits, &wi, &prefix_len);

        uint16_t w[3];
        uint64_t key[3][16];
        uint32_t len[3];
        for (int c = 0; c < 3; c++) {
            w[c] = (uint16_t)(rng_next() % 2048);
            uint64_t wb = table[w[c]];
            len[c] = (uint32_t)wlbp_wb_len(wb);
            wlbp_keyblock_tail_swap(prefix_w, cur, bits, wi, wb, len[c], key[c]);
        }

        uint8_t sx[3][64], ss[3][64];
        pbkdf2_sha512_keyblock_fast_x3(key[0], prefix_len + len[0],
                                       key[1], prefix_len + len[1],
                                       key[2], prefix_len + len[2],
                                       2048, sx[0], sx[1], sx[2]);
        for (int c = 0; c < 3; c++)
            pbkdf2_sha512_keyblock_fast(key[c], prefix_len + len[c], 2048, ss[c]);

        for (int c = 0; c < 3; c++) {
            if (memcmp(sx[c], ss[c], 64) != 0) {
                printf("check1 FAIL t=%d chain %d (w=%u)\n", t, c, w[c]);
                fails++;
                break;
            }
        }
        if (fails) break;
    }
    printf("check1 x3 interleaved == 3x single (200 triples, byte-exact): %s\n",
           fails == 0 ? "PASS" : "FAIL");

    /* ---- check 2: shipped split (42 triples + 1 pair) reproduces target ---- */
    {
        uint16_t base11[11] = {759, 1078, 213, 623, 521, 319, 416, 302, 566, 1104, 191};
        const uint16_t target_w12 = 1666;
        const char* expected_hex =
            "4cd832a5c862ef5117870c15bdf792ed558499a2c81d8bd5c68f28bf0f66671b"
            "76e6522d90fb4996cc29d1a7b37ccf0bcc8157dea21f5f065e89921841323ab8";
        uint8_t expected[64];
        for (int i = 0; i < 64; i++) { unsigned v; sscanf(expected_hex + 2*i, "%2x", &v); expected[i] = (uint8_t)v; }

        uint64_t prefix_w[16], cur;
        uint32_t bits = 0, prefix_len = 0;
        int wi = 0;
        wlbp_base_prefix_build(table, base11, prefix_w, &cur, &bits, &wi, &prefix_len);

        /* collect valids (invariant 128) */
        uint16_t cand[12];
        memcpy(cand, base11, 11 * sizeof(uint16_t));
        uint16_t valid_w12[256];
        int n_valid = 0;
        for (uint32_t w12 = 0; w12 < 2048; w12++) {
            cand[11] = (uint16_t)w12;
            /* reuse the shipped checksum via wlbp-free local copy: */
            extern int dummy_never; (void)dummy_never; /* placeholder, checksum below */
            /* NOTE: verify_checksum_12 lives in main.cu; replicate minimal check here */
            /* simple approach: use the key-block path and check via E1-proven
               invariant instead - enumerate with a local checksum */
            /* local checksum implementation (proven equivalent in E1) */
            {
                uint8_t entropy[16];
                uint32_t bitsv = 0;
                int bc = 0, bi = 0;
                for (int w = 0; w < 12; w++) {
                    for (int b = 10; b >= 0; b--) {
                        bitsv = (bitsv << 1) | ((cand[w] >> b) & 1);
                        if (++bc == 8) { if (bi < 16) entropy[bi++] = bitsv & 0xFF; bitsv = 0; bc = 0; }
                    }
                }
                /* sha256 first byte via sha256() on 16 bytes */
                uint8_t h[32];
                sha256(entropy, 16, h);
                if ((uint8_t)(h[0] >> 4) == (uint8_t)(cand[11] & 0x0F)) {
                    if (n_valid < 256) valid_w12[n_valid] = (uint16_t)w12;
                    n_valid++;
                }
            }
        }
        printf("check2 valid count: %d (invariant 128)\n", n_valid);
        if (n_valid != 128) fails++;

        /* run the shipped split: triples + pair; verify target seed */
        int found = 0, triples = 0, pairs = 0;
        int v = 0;
        for (; v + 2 < n_valid; v += 3) {
            triples++;
            uint16_t w[3];
            uint64_t key[3][16];
            uint32_t len[3];
            for (int c = 0; c < 3; c++) {
                w[c] = valid_w12[v + c];
                uint64_t wb = table[w[c]];
                len[c] = (uint32_t)wlbp_wb_len(wb);
                wlbp_keyblock_tail_swap(prefix_w, cur, bits, wi, wb, len[c], key[c]);
            }
            uint8_t seeds[3][64];
            pbkdf2_sha512_keyblock_fast_x3(key[0], prefix_len + len[0],
                                           key[1], prefix_len + len[1],
                                           key[2], prefix_len + len[2],
                                           2048, seeds[0], seeds[1], seeds[2]);
            for (int c = 0; c < 3; c++)
                if (w[c] == target_w12 && memcmp(seeds[c], expected, 64) == 0)
                    found = 1;
        }
        if (v + 1 < n_valid) {
            pairs++;
            uint16_t w[2];
            uint64_t key[2][16];
            uint32_t len[2];
            for (int c = 0; c < 2; c++) {
                w[c] = valid_w12[v + c];
                uint64_t wb = table[w[c]];
                len[c] = (uint32_t)wlbp_wb_len(wb);
                wlbp_keyblock_tail_swap(prefix_w, cur, bits, wi, wb, len[c], key[c]);
            }
            uint8_t seeds[2][64];
            pbkdf2_sha512_keyblock_fast_x2(key[0], prefix_len + len[0],
                                           key[1], prefix_len + len[1],
                                           2048, seeds[0], seeds[1]);
            for (int c = 0; c < 2; c++)
                if (w[c] == target_w12 && memcmp(seeds[c], expected, 64) == 0)
                    found = 1;
            v += 2;
        }
        printf("check2 split: %d triples + %d pair; target seed via shipped split: %s\n",
               triples, pairs, found ? "MATCH" : "MISMATCH");
        if (!found) fails++;
    }

    printf("RESULT: %s\n", fails == 0 ? "PASS - E5c x3 interleave verified" : "FAIL");
    return fails == 0 ? 0 : 1;
}
