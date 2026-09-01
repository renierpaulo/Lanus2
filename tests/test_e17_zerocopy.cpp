/*
 * test_e17_zerocopy.cpp - E17 verification: the zero-copy PBKDF2 chain
 * (outer_work doubles as U — no separate chain array) must be byte-exact
 * versus pbkdf2_sha512_keyblock_fast.
 *
 * Build: cl /O2 /EHsc test_e17_zerocopy.cpp /Fe:t17.exe
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

static uint64_t rng_state = 0x2E7C0EC0DE000001ULL;
static uint64_t rng_next(void) {
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7; rng_state ^= rng_state << 17;
    return rng_state;
}

int main(void) {
    int fails = 0;

    /* ---- check 1: zero-copy == keyblock_fast (300 random keys) ---- */
    for (int t = 0; t < 300; t++) {
        uint64_t key[16];
        for (int i = 0; i < 16; i++) key[i] = rng_next();
        uint32_t klen = 60 + (uint32_t)(rng_next() % 48);

        uint8_t s_ref[64], s_zc[64];
        pbkdf2_sha512_keyblock_fast(key, klen, 2048, s_ref);

        /* build the pre-states (same as keyblock_fast does) */
        uint32_t i0 = (klen + 7u) / 8u;
        if (i0 > 16u) i0 = 16u;
        uint64_t inpre[8], outpre[8];
        {
            uint64_t ip[16], op[16];
            for (int i = 0; i < 16; i++) {
                ip[i] = key[i] ^ 0x3636363636363636ULL;
                op[i] = key[i] ^ 0x5c5c5c5c5c5c5c5cULL;
            }
            SHA512State_t s;
            sha512_init_state_opt(&s);
            sha512_compress16_rot_opt(&s, ip, i0, d_KP36);
            for (int j = 0; j < 8; j++) inpre[j] = s.h[j];
            sha512_init_state_opt(&s);
            sha512_compress16_rot_opt(&s, op, i0, d_KP5c);
            for (int j = 0; j < 8; j++) outpre[j] = s.h[j];
        }
        pbkdf2_chain_zero_copy(inpre, outpre, 2048, s_zc);

        if (memcmp(s_ref, s_zc, 64) != 0) {
            printf("check1 FAIL t=%d klen=%u\n", t, klen);
            fails++;
            break;
        }
    }
    printf("check1 zero-copy chain == keyblock_fast (300 keys): %s\n", fails == 0 ? "PASS" : "FAIL");

    /* ---- check 2: documented target seed through the zero-copy path ---- */
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

        uint32_t i0 = (prefix_len + (uint32_t)wlbp_wb_len(wb) + 7u) / 8u;
        if (i0 > 16u) i0 = 16u;
        uint64_t inpre[8], outpre[8];
        {
            uint64_t ip[16], op[16];
            for (int i = 0; i < 16; i++) {
                ip[i] = key[i] ^ 0x3636363636363636ULL;
                op[i] = key[i] ^ 0x5c5c5c5c5c5c5c5cULL;
            }
            SHA512State_t s;
            sha512_init_state_opt(&s);
            sha512_compress16_rot_opt(&s, ip, i0, d_KP36);
            for (int j = 0; j < 8; j++) inpre[j] = s.h[j];
            sha512_init_state_opt(&s);
            sha512_compress16_rot_opt(&s, op, i0, d_KP5c);
            for (int j = 0; j < 8; j++) outpre[j] = s.h[j];
        }
        uint8_t seed[64];
        pbkdf2_chain_zero_copy(inpre, outpre, 2048, seed);

        const char* expected_hex =
            "4cd832a5c862ef5117870c15bdf792ed558499a2c81d8bd5c68f28bf0f66671b"
            "76e6522d90fb4996cc29d1a7b37ccf0bcc8157dea21f5f065e89921841323ab8";
        uint8_t expected[64];
        for (int i = 0; i < 64; i++) { unsigned v; sscanf(expected_hex + 2*i, "%2x", &v); expected[i] = (uint8_t)v; }

        int ok = memcmp(seed, expected, 64) == 0;
        printf("check2 target seed via zero-copy: %s\n", ok ? "MATCH" : "MISMATCH");
        if (!ok) fails++;
    }

    printf("RESULT: %s\n", fails == 0 ? "PASS - E17 zero-copy verified" : "FAIL");
    return fails == 0 ? 0 : 1;
}
