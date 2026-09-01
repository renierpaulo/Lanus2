/*
 * test_etapa_e2.cpp - E2 verification: base prefix built ONCE + per-candidate
 * tail swap must produce byte-exact key blocks equal to full re-assembly.
 *
 *   tail_swap(prefix_build(base11), word12) == assemble_key12_rt(base11+word12)
 *
 * Build: cl /O2 /EHsc test_etapa_e2.cpp
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

static uint64_t rng_state = 0x123456789ABCDEF1ULL;
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

    /* ---- 200k random (base11, word12) pairs: tail swap == full assembly ---- */
    for (int t = 0; t < 200000; t++) {
        uint16_t base11[11];
        for (int w = 0; w < 11; w++) base11[w] = (uint16_t)(rng_next() % 2048);
        uint16_t w12 = (uint16_t)(rng_next() % 2048);

        /* prefix ONCE */
        uint64_t prefix_w[16], cur;
        uint32_t bits = 0, prefix_len = 0;
        int wi = 0;
        wlbp_base_prefix_build(table, base11, prefix_w, &cur, &bits, &wi, &prefix_len);

        /* per-candidate tail swap */
        uint64_t wb12 = table[w12];
        uint32_t len12 = (uint32_t)wlbp_wb_len(wb12);
        uint64_t key_tail[16];
        wlbp_keyblock_tail_swap(prefix_w, cur, bits, wi, wb12, len12, key_tail);

        /* reference: full 12-word assembly */
        uint16_t full12[12];
        for (int w = 0; w < 11; w++) full12[w] = base11[w];
        full12[11] = w12;
        uint64_t key_ref[16];
        uint32_t ref_len = 0;
        wlbp_assemble_key12_rt(table, full12, key_ref, &ref_len);

        if (memcmp(key_tail, key_ref, 128) != 0) {
            printf("FAIL t=%d (base %u,%u,%u... w12=%u) prefix_len=%u ref_len=%u\n",
                   t, base11[0], base11[1], base11[2], w12, prefix_len, ref_len);
            for (int i = 0; i < 16; i++)
                printf("  w[%2d] tail=%016llx ref=%016llx %s\n", i,
                       (unsigned long long)key_tail[i], (unsigned long long)key_ref[i],
                       key_tail[i] == key_ref[i] ? "" : "  <-- MISMATCH");
            fails++;
            break;
        }
        if (prefix_len + len12 != ref_len) {
            printf("FAIL t=%d: prefix_len %u + len12 %u != ref_len %u\n", t, prefix_len, len12, ref_len);
            fails++;
            break;
        }
    }
    printf("check1 tail swap == full assembly (200k pairs, byte-exact + len): %s\n",
           fails == 0 ? "PASS" : "FAIL");

    /* ---- target phrase: base = first 11 words, candidate = 1666 ---- */
    {
        uint16_t base11[11] = {759, 1078, 213, 623, 521, 319, 416, 302, 566, 1104, 191};
        uint16_t w12 = 1666;

        uint64_t prefix_w[16], cur;
        uint32_t bits = 0, prefix_len = 0;
        int wi = 0;
        wlbp_base_prefix_build(table, base11, prefix_w, &cur, &bits, &wi, &prefix_len);

        uint64_t wb12 = table[w12];
        uint64_t key_tail[16];
        wlbp_keyblock_tail_swap(prefix_w, cur, bits, wi, wb12, (uint32_t)wlbp_wb_len(wb12), key_tail);

        uint16_t full12[12] = {759, 1078, 213, 623, 521, 319, 416, 302, 566, 1104, 191, 1666};
        uint64_t key_ref[16];
        uint32_t ref_len = 0;
        wlbp_assemble_key12_rt(table, full12, key_ref, &ref_len);

        int ok = memcmp(key_tail, key_ref, 128) == 0;
        printf("check2 target phrase tail swap == assembly: %s\n", ok ? "PASS" : "FAIL");
        if (!ok) fails++;

        /* and the full PBKDF2 through the tail-swap block hits the documented seed */
        pbkdf2_sha512_keyblock_fast(key_tail, ref_len, 2048, (uint8_t*)key_ref /*reuse*/);
        uint8_t seed_fast[64];
        pbkdf2_sha512_keyblock_fast(key_tail, ref_len, 2048, seed_fast);
        const char* expected_hex =
            "4cd832a5c862ef5117870c15bdf792ed558499a2c81d8bd5c68f28bf0f66671b"
            "76e6522d90fb4996cc29d1a7b37ccf0bcc8157dea21f5f065e89921841323ab8";
        uint8_t expected[64];
        for (int i = 0; i < 64; i++) { unsigned v; sscanf(expected_hex + 2*i, "%2x", &v); expected[i] = (uint8_t)v; }
        int e2e = memcmp(seed_fast, expected, 64) == 0;
        printf("check3 PBKDF2 via tail-swap block == documented target seed: %s\n", e2e ? "MATCH" : "MISMATCH");
        if (!e2e) fails++;
    }

    printf("RESULT: %s\n", fails == 0 ? "PASS - E2 base prefix + tail swap verified" : "FAIL");
    return fails == 0 ? 0 : 1;
}
