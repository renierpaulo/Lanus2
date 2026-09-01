/*
 * test_pbkdf2_equiv.c - MODEL 1 verification: PBKDF2-from-keyblock (WLBP-PR
 * register path) must be byte-exact versus the classic mnemonic-string path.
 *
 * Check 1: target phrase "galaxy man boy evil donkey child cross chair egg
 *          meat blood space" (indices 759,1078,213,623,521,319,416,302,566,
 *          1104,191,1666) must reproduce the documented expected seed
 *          4cd832a5...3ab8 on BOTH paths.
 * Check 2: 200 random phrases - new path == classic path (64-byte seeds).
 *
 * Host shim makes the __device__ code compile as plain C.
 * Build:  cl /O2 test_pbkdf2_equiv.c /Fe:test_pbkdf2_equiv.exe  (vcvars env)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ---- host compilation shim (must precede the CUDA headers) ---- */
#define __device__
#define __constant__ static
#define __forceinline__ static
#define __align__(x)

#include "src/pbkdf2_opt.cuh"
#include "src/wlbp.cuh"

static uint64_t rng_state = 0xA0761D6478BD642FULL;
static uint64_t rng_next(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

static void hex2bin(const char* hex, uint8_t* out, int n) {
    for (int i = 0; i < n; i++) {
        unsigned v;
        sscanf(hex + 2 * i, "%2x", &v);
        out[i] = (uint8_t)v;
    }
}

/* classic path: mnemonic string -> pbkdf2_sha512_mnemonic */
static void classic_seed(const unsigned char* vocab, const uint16_t* idx, uint8_t* seed64) {
    uint8_t mnemonic[256];
    int mn_len = 0;
    unsigned char word[9];
    for (int w = 0; w < 12; w++) {
        int len = wlbp_decode_word(vocab, idx[w], word);
        if (w > 0) mnemonic[mn_len++] = ' ';
        memcpy(mnemonic + mn_len, word, len);
        mn_len += len;
    }
    const uint8_t* salt = (const uint8_t*)"mnemonic";
    pbkdf2_sha512_mnemonic(mnemonic, mn_len, salt, 8, 2048, seed64);
}

/* new path: register-assembled key block -> pbkdf2_sha512_keyblock */
static void keyblock_seed(const unsigned char* vocab, const uint16_t* idx, uint8_t* seed64) {
    uint64_t key_w[16];
    wlbp_assemble_key12(vocab, idx, key_w);
    const uint8_t* salt = (const uint8_t*)"mnemonic";
    pbkdf2_sha512_keyblock(key_w, salt, 8, 2048, seed64);
}

int main(void) {
    /* the pattern: thread receives full vocabulary */
    unsigned char vocab[WLBP_VOCAB_BYTES];
    wlbp_thread_vocab_init(vocab, WLBP_BLOB);

    /* ---- Check 1: target phrase, both paths vs documented expected seed ---- */
    uint16_t target[12] = {759, 1078, 213, 623, 521, 319, 416, 302, 566, 1104, 191, 1666};
    const char* expected_hex =
        "4cd832a5c862ef5117870c15bdf792ed558499a2c81d8bd5c68f28bf0f66671b"
        "76e6522d90fb4996cc29d1a7b37ccf0bcc8157dea21f5f065e89921841323ab8";
    uint8_t expected[64], seedA[64], seedB[64];
    hex2bin(expected_hex, expected, 64);

    classic_seed(vocab, target, seedA);
    keyblock_seed(vocab, target, seedB);

    int fails = 0;
    printf("check1 target phrase:\n");
    printf("  classic  path vs expected seed: %s\n",
           memcmp(seedA, expected, 64) == 0 ? "MATCH" : "MISMATCH");
    printf("  keyblock path vs expected seed: %s\n",
           memcmp(seedB, expected, 64) == 0 ? "MATCH" : "MISMATCH");
    if (memcmp(seedA, expected, 64) != 0 || memcmp(seedB, expected, 64) != 0) fails++;

    /* ---- Check 2: 200 random phrases, classic vs keyblock ---- */
    for (int t = 0; t < 200; t++) {
        uint16_t idx[12];
        for (int k = 0; k < 12; k++) idx[k] = (uint16_t)(rng_next() % 2048);
        classic_seed(vocab, idx, seedA);
        keyblock_seed(vocab, idx, seedB);
        if (memcmp(seedA, seedB, 64) != 0) {
            printf("check2 FAIL phrase %d (idx %u %u %u ...)\n", t, idx[0], idx[1], idx[2]);
            fails++;
            break;
        }
    }
    printf("check2 (200 random phrases): %s\n", fails == 0 ? "PASS - seeds byte-exact" : "FAIL");
    printf("RESULT: %s\n", fails == 0 ? "PASS - Model 1 process change is byte-exact" : "FAIL");
    return fails == 0 ? 0 : 1;
}
