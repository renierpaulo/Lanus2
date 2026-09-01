/*
 * test_etapa_e1.cpp - E1 verification: the thread's vocabulary iteration.
 * The thread holds a base of 11 words and enumerates ALL 2048 vocabulary
 * words as position 12, checksum-filtering internally.
 *
 * Mathematical invariant: for a fixed 11-word base (121 bits), the 12th word
 * carries 7 entropy bits + 4 checksum bits -> EXACTLY 2048/16 = 128 of the
 * 2048 candidates yield a valid BIP39 checksum.
 *
 * Checks:
 *   1. Enumeration yields EXACTLY 128 valid candidates (for several bases).
 *   2. The known target phrase (all 12 words valid) is found by the
 *      enumeration of its own 11-word base.
 *   3. Independent re-check of every accepted candidate with a second
 *      packing implementation.
 *
 * Build: cl /O2 /EHsc test_etapa_e1.cpp
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>

#define __device__
#define __constant__ static
#define __forceinline__ static
#define __align__(x)

#include "src/sha256.cuh"

/* ---- checksum path replicated from main.cu (verify_checksum_12 + helper) ---- */
__device__ __forceinline__ uint32_t rotr32(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

__device__ void sha256_checksum_only(const uint8_t* entropy, int ent_bytes, uint8_t* first_byte) {
    uint32_t h0 = 0x6a09e667, h1 = 0xbb67ae85, h2 = 0x3c6ef372, h3 = 0xa54ff53a;
    uint32_t h4 = 0x510e527f, h5 = 0x9b05688c, h6 = 0x1f83d9ab, h7 = 0x5be0cd19;
    uint32_t w[64];

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        if (i < (ent_bytes + 3) / 4) {
            int base = i * 4;
            w[i] = 0;
            if (base < ent_bytes) w[i] |= (uint32_t)entropy[base] << 24;
            if (base + 1 < ent_bytes) w[i] |= (uint32_t)entropy[base + 1] << 16;
            if (base + 2 < ent_bytes) w[i] |= (uint32_t)entropy[base + 2] << 8;
            if (base + 3 < ent_bytes) w[i] |= (uint32_t)entropy[base + 3];
        } else if (i == ent_bytes / 4) {
            int pos = ent_bytes % 4;
            w[i] = 0x80000000 >> (pos * 8);
        } else {
            w[i] = 0;
        }
    }
    w[15] = ent_bytes * 8;

    #pragma unroll
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = rotr32(w[i-15], 7) ^ rotr32(w[i-15], 18) ^ (w[i-15] >> 3);
        uint32_t s1 = rotr32(w[i-2], 17) ^ rotr32(w[i-2], 19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }

    uint32_t a = h0, b = h1, c = h2, d = h3;
    uint32_t e = h4, f = h5, g = h6, h = h7;
    #pragma unroll
    for (int i = 0; i < 64; i++) {
        uint32_t S1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t temp1 = h + S1 + ch + K256[i] + w[i];
        uint32_t S0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = S0 + maj;
        h = g; g = f; f = e; e = d + temp1;
        d = c; c = b; b = a; a = temp1 + temp2;
    }
    h0 += a;
    *first_byte = (h0 >> 24) & 0xFF;
}

__device__ bool verify_checksum_12(const uint16_t* indices) {
    uint8_t entropy[16];
    uint32_t bits = 0;
    int bit_count = 0;
    int byte_idx = 0;
    #pragma unroll
    for (int w = 0; w < 12; w++) {
        uint16_t idx = indices[w];
        for (int b = 10; b >= 0; b--) {
            bits = (bits << 1) | ((idx >> b) & 1);
            bit_count++;
            if (bit_count == 8) {
                if (byte_idx < 16) entropy[byte_idx++] = bits & 0xFF;
                bits = 0; bit_count = 0;
            }
        }
    }
    uint8_t expected_cs;
    sha256_checksum_only(entropy, 16, &expected_cs);
    expected_cs = expected_cs >> 4;
    uint8_t actual_cs = indices[11] & 0x0F;
    return expected_cs == actual_cs;
}

/* ---- INDEPENDENT re-check: different packing style, same SHA-256 ---- */
static bool verify_checksum_12_ref(const uint16_t* idx) {
    /* 132-bit stream -> 16 bytes entropy (last word contributes 7 bits) + 4 bits cs */
    uint8_t entropy[16] = {0};
    uint32_t bitpos = 0;
    for (int w = 0; w < 12; w++) {
        for (int b = 10; b >= 0; b--) {
            uint32_t bit = (idx[w] >> b) & 1;
            if (bitpos < 128) {
                if (bit) entropy[bitpos >> 3] |= (uint8_t)(0x80u >> (bitpos & 7));
            } else {
                break; /* bits 128..131 = checksum (handled below) */
            }
            bitpos++;
        }
    }
    /* checksum = last 4 bits of the stream */
    uint8_t cs = 0;
    for (int i = 0; i < 4; i++) {
        int p = 128 + i;                    /* absolute stream position */
        int word_idx = p / 11;
        int bit_in_word = 10 - (p % 11);
        cs = (uint8_t)((cs << 1) | ((idx[word_idx] >> bit_in_word) & 1));
    }
    uint8_t h;
    sha256_checksum_only(entropy, 16, &h);
    return (uint8_t)(h >> 4) == cs;
}

/* ---- E1: the thread's vocabulary iteration (the shipped design) ---- */
static int enumerate_vocabulary_12(const uint16_t* base11, uint16_t* out_valid /* >=128 */) {
    uint16_t cand[12];
    for (int w = 0; w < 11; w++) cand[w] = base11[w];
    int n_valid = 0;
    for (uint32_t w12 = 0; w12 < 2048; w12++) {
        cand[11] = (uint16_t)w12;
        if (verify_checksum_12(cand)) {
            out_valid[n_valid++] = (uint16_t)w12;
        }
    }
    return n_valid;
}

static uint64_t rng_state = 0xDEADBEEFCAFEF00DULL;
static uint64_t rng_next(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

int main(void) {
    int fails = 0;

    /* ---- check 1: EXACTLY 128 valid candidates for random bases ---- */
    for (int t = 0; t < 50; t++) {
        uint16_t base11[11];
        for (int w = 0; w < 11; w++) base11[w] = (uint16_t)(rng_next() % 2048);
        uint16_t valid[256];
        int n = enumerate_vocabulary_12(base11, valid);
        if (n != 128) {
            printf("check1 FAIL base %d: got %d valid (expected 128)\n", t, n);
            fails++;
            break;
        }
        /* independent re-check of every accepted candidate */
        uint16_t full[12];
        for (int w = 0; w < 11; w++) full[w] = base11[w];
        for (int v = 0; v < n; v++) {
            full[11] = valid[v];
            if (!verify_checksum_12_ref(full)) {
                printf("check1 FAIL base %d: candidate %u fails independent re-check\n", t, valid[v]);
                fails++;
                break;
            }
        }
        if (fails) break;
    }
    printf("check1 enumeration = exactly 128 valid + independent re-check (50 bases): %s\n",
           fails == 0 ? "PASS" : "FAIL");

    /* ---- check 2: the known target phrase is found by its own base ---- */
    {
        uint16_t target[12] = {759, 1078, 213, 623, 521, 319, 416, 302, 566, 1104, 191, 1666};
        uint16_t valid[256];
        int n = enumerate_vocabulary_12(target, valid);
        int found = 0;
        for (int v = 0; v < n; v++) if (valid[v] == target[11]) found = 1;
        if (n != 128 || !found) {
            printf("check2 FAIL: n=%d target_found=%d\n", n, found);
            fails++;
        }
        printf("check2 target phrase recovered by its base enumeration: %s\n",
               (n == 128 && found) ? "PASS" : "FAIL");
    }

    printf("RESULT: %s\n", fails == 0 ? "PASS - E1 vocabulary iteration verified" : "FAIL");
    return fails == 0 ? 0 : 1;
}
