/*
 * test_wlbp_keygen.c - ETAPA A verification: the redesigned in-thread process
 * (vocabulary as compute table + register-only key assembly) must produce
 * byte-exact results versus the classic word-by-word ASCII string path.
 *
 * Check 1: all 2048 words extract identically via packed path and string path.
 * Check 2: 200,000 random phrases - wlbp_assemble_key12 (registers) equals
 *          the classic mnemonic-string-into-128B-buffer HMAC key block.
 *
 * Build:  cl /O2 test_wlbp_keygen.c /Fe:test_wlbp_keygen.exe  (vcvars env)
 * Run:    test_wlbp_keygen.exe wordlist.txt
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WLBP_HOST_TEST 1
#include "src/wlbp.cuh"

static uint64_t rng_state = 0x9E3779B97F4A7C15ULL;
static uint64_t rng_next(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

/* classic path: decode words to ASCII string, zero-pad into 128B, read BE u64s */
static void classic_key_block(const unsigned char* vocab, const uint16_t* idx, uint64_t w[16]) {
    unsigned char key[128];
    memset(key, 0, 128);
    int pos = 0;
    unsigned char word[9];
    for (int k = 0; k < 12; k++) {
        int len = wlbp_decode_word(vocab, idx[k], word);
        if (k > 0) key[pos++] = ' ';
        memcpy(key + pos, word, len);
        pos += len;
    }
    for (int i = 0; i < 16; i++) {
        uint64_t v = 0;
        for (int b = 0; b < 8; b++) v = (v << 8) | key[i * 8 + b];
        w[i] = v;
    }
}

int main(int argc, char** argv) {
    if (argc < 2) { printf("usage: %s wordlist.txt\n", argv[0]); return 1; }

    /* the pattern: thread receives full vocabulary */
    unsigned char vocab[WLBP_VOCAB_BYTES];
    wlbp_thread_vocab_init(vocab, WLBP_BLOB);

    /* ---- Check 1: all 2048 words, packed path == string path == file ---- */
    FILE* f = fopen(argv[1], "r");
    if (!f) { printf("FAIL: cannot open %s\n", argv[1]); return 1; }
    char expected[64];
    int fails = 0, count = 0;
    while (fgets(expected, sizeof(expected), f) && count < 2048) {
        expected[strcspn(expected, "\r\n")] = 0;
        if (!expected[0]) continue;

        unsigned char s[9];
        int len_s = wlbp_decode_word(vocab, (uint32_t)count, s);
        uint64_t packed;
        int len_p = wlbp_word_bytes(vocab, (uint32_t)count, &packed);

        uint64_t ref = 0;
        for (int c = 0; c < len_s; c++) ref = (ref << 8) | (uint64_t)s[c];

        if (len_s != (int)strlen(expected) || memcmp(s, expected, len_s + 1) != 0 ||
            len_p != len_s || packed != ref) {
            printf("FAIL idx %d: file='%s' str='%s' packed=%016llx ref=%016llx\n",
                   count, expected, s, (unsigned long long)packed, (unsigned long long)ref);
            fails++;
        }
        count++;
    }
    fclose(f);
    printf("check1 (2048-word extraction): %s (%d words, %d fails)\n",
           (count == 2048 && fails == 0) ? "PASS" : "FAIL", count, fails);
    if (count != 2048 || fails) return 1;

    /* ---- Check 2: 200k random phrases, register assembly == classic ---- */
    const int N = 200000;
    for (int t = 0; t < N; t++) {
        uint16_t idx[12];
        for (int k = 0; k < 12; k++) idx[k] = (uint16_t)(rng_next() % 2048);

        uint64_t w_new[16], w_ref[16];
        wlbp_assemble_key12(vocab, idx, w_new);
        classic_key_block(vocab, idx, w_ref);

        if (memcmp(w_new, w_ref, 128) != 0) {
            printf("FAIL phrase %d (idx %u %u %u ...):\n", t, idx[0], idx[1], idx[2]);
            for (int i = 0; i < 16; i++)
                printf("  w[%2d] new=%016llx ref=%016llx %s\n", i,
                       (unsigned long long)w_new[i], (unsigned long long)w_ref[i],
                       w_new[i] == w_ref[i] ? "" : "  <-- MISMATCH");
            fails++;
            break;
        }
    }
    printf("check2 (200k phrases, register vs classic): %s\n", fails == 0 ? "PASS" : "FAIL");
    printf("RESULT: %s\n", fails == 0 ? "PASS - process redesign is byte-exact" : "FAIL");
    return fails == 0 ? 0 : 1;
}
