/*
 * test_wlbp_host.c - host verification of the EXACT decoder logic that runs
 * on the GPU (wlbp.cuh compiled without __CUDACC__). Proves the pattern:
 * thread receives the full 2048-word vocabulary and decodes every word
 * bit-exactly from its private copy.
 *
 * Build:  gcc -O2 -o test_wlbp_host test_wlbp_host.c
 * Run:    ./test_wlbp_host cuda-scanner/wordlist.txt
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WLBP_HOST_TEST 1
#include "src/wlbp.cuh"

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("usage: %s wordlist.txt\n", argv[0]);
        return 1;
    }

    /* Simulate the per-thread pattern on the host: private vocab copy */
    unsigned char vocab[WLBP_VOCAB_BYTES];
    wlbp_thread_vocab_init(vocab, WLBP_BLOB);

    FILE* f = fopen(argv[1], "r");
    if (!f) { printf("FAIL: cannot open %s\n", argv[1]); return 1; }

    char expected[64];
    int fails = 0, count = 0;
    while (fgets(expected, sizeof(expected), f) && count < 2048) {
        expected[strcspn(expected, "\r\n")] = 0;
        if (!expected[0]) continue;

        unsigned char word[9];
        int len = wlbp_decode_word(vocab, (uint32_t)count, word);

        if (len != (int)strlen(expected) || memcmp(word, expected, len + 1) != 0) {
            printf("FAIL idx %d: expected '%s' got '%s' (len %d)\n",
                   count, expected, word, len);
            fails++;
        }
        count++;
    }
    fclose(f);

    printf("decoded %d words from thread-private vocab (%d bytes)\n", count, WLBP_VOCAB_BYTES);
    printf("RESULT: %s (%d failures)\n", (count == 2048 && fails == 0) ? "PASS" : "FAIL", fails);
    return (count == 2048 && fails == 0) ? 0 : 1;
}
