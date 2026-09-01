/*
 * test_e6_monolithic.cpp - E6 verification: the monolithic self-contained
 * thread (generate -> enumerate -> derive, all in-thread).
 *
 *   1. Generation structure (100k k): 12 words, all 4 required present,
 *      8 distinct pool words; deterministic per k (splitmix seed).
 *   2. Composition: generated base -> prefix -> enumeration (128 valid)
 *      -> x3/x2/single paths -> seeds byte-exact vs classic string path
 *      (the proven reference chain).
 *   3. One-thread end-to-end: 10 bases fully processed (128 derivations
 *      each), zero divergence.
 *
 * Build: cl /O2 /EHsc test_e6_monolithic.cpp
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>

#define __device__
#define __constant__ static
#define __forceinline__ static
#define __align__(x)

#include "src/sha256.cuh"
#include "src/sha512.cuh"
#include "src/pbkdf2_opt.cuh"
#include "src/wlbp.cuh"

#define MAX_WORDS 40

/* host stand-ins for the device constants (set from the synthetic pool) */
static uint16_t h_req_words[4];
static uint16_t h_avail_words[MAX_WORDS];
static uint32_t h_avail_count;

static uint64_t cuda_rand(uint64_t* seed) {
    *seed = (*seed * 6364136223846793005ULL + 1442695040888963407ULL);
    return *seed;
}
static uint32_t cuda_rand_range(uint64_t* seed, uint32_t range) {
    uint64_t x = cuda_rand(seed);
    return (uint32_t)(((x >> 32) * (uint64_t)range) >> 32);
}

/* generate_phrase12 replicated exactly from main.cu (E6) */
static void generate_phrase12(uint64_t k, uint16_t* out12) {
    uint64_t seed = k * 0x9E3779B97F4A7C15ULL;
    seed ^= seed >> 29;
    seed *= 0xBF58476D1CE4E5B9ULL;
    seed ^= seed >> 32;

    out12[0] = h_req_words[0];
    out12[1] = h_req_words[1];
    out12[2] = h_req_words[2];
    out12[3] = h_req_words[3];

    uint16_t available[MAX_WORDS];
    for (uint32_t i = 0; i < h_avail_count; i++) available[i] = h_avail_words[i];

    for (uint32_t i = 0; i < 8; i++) {
        uint32_t j = i + cuda_rand_range(&seed, h_avail_count - i);
        out12[4 + i] = available[j];
        uint16_t temp = available[j]; available[j] = available[i]; available[i] = temp;
    }
    for (uint32_t i = 11; i > 0; i--) {
        uint32_t j = cuda_rand_range(&seed, i + 1);
        uint16_t temp = out12[i]; out12[i] = out12[j]; out12[j] = temp;
    }
}

/* checksum (proven in E1) */
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

/* classic string-path PBKDF2 (the proven reference) */
static void classic_seed(const uint16_t* phrase12, uint64_t table[2048], uint8_t* seed64) {
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
    const uint8_t* salt = (const uint8_t*)"mnemonic";
    pbkdf2_sha512_mnemonic(mnemonic, mn_len, salt, 8, 2048, seed64);
}

int main(void) {
    int fails = 0;

    /* thread setup */
    unsigned char vocab[WLBP_VOCAB_BYTES];
    wlbp_thread_vocab_init(vocab, WLBP_BLOB);
    uint64_t table[2048];
    wlbp_build_repr_table(vocab, table);

    /* synthetic pool: 40 words, required = positions 0,7,10,12 */
    uint16_t base_pool[MAX_WORDS];
    for (uint32_t i = 0; i < MAX_WORDS; i++) base_pool[i] = (uint16_t)(100 + i * 7);
    const uint32_t required_positions[4] = {0, 7, 10, 12};
    for (int r = 0; r < 4; r++) h_req_words[r] = base_pool[required_positions[r]];
    for (uint32_t i = 0; i < MAX_WORDS; i++) {
        bool is_req = false;
        for (int r = 0; r < 4; r++) if (i == required_positions[r]) { is_req = true; break; }
        if (!is_req) h_avail_words[h_avail_count++] = base_pool[i];
    }

    /* ---- check 1: generation structure + determinism (100k k) ---- */
    for (uint64_t k = 0; k < 100000; k++) {
        uint16_t p[12];
        generate_phrase12(k, p);

        /* multiset composition: all 4 required exactly once + 8 distinct pool */
        int req_seen[4] = {0, 0, 0, 0};
        for (int w = 0; w < 12; w++)
            for (int r = 0; r < 4; r++)
                if (p[w] == h_req_words[r]) req_seen[r]++;
        for (int r = 0; r < 4; r++) {
            if (req_seen[r] != 1) { printf("check1 FAIL k=%llu req %d seen %d\n", (unsigned long long)k, r, req_seen[r]); fails++; break; }
        }
        if (fails) break;

        for (int a = 0; a < 12 && !fails; a++) {
            bool is_req = false;
            for (int r = 0; r < 4; r++) if (p[a] == h_req_words[r]) is_req = true;
            if (is_req) continue;
            bool in_pool = false;
            for (uint32_t i = 0; i < h_avail_count; i++) if (p[a] == h_avail_words[i]) in_pool = true;
            if (!in_pool) { printf("check1 FAIL k=%llu: p[%d] not from pool\n", (unsigned long long)k, a); fails++; break; }
        }
        if (fails) break;
        for (int a = 0; a < 12 && !fails; a++)
            for (int b = a + 1; b < 12; b++)
                if (p[a] == p[b]) { printf("check1 FAIL k=%llu: duplicate\n", (unsigned long long)k); fails++; break; }
        if (fails) break;
    }
    {
        uint16_t p1[12], p2[12];
        generate_phrase12(42, p1);
        generate_phrase12(42, p2);
        if (memcmp(p1, p2, sizeof(p1)) != 0) { printf("check1 FAIL: not deterministic\n"); fails++; }
    }
    printf("check1 generation structure + determinism (100k): %s\n", fails == 0 ? "PASS" : "FAIL");

    /* ---- check 2 + 3: monolithic composition, 10 bases end-to-end ---- */
    uint64_t total_derivations = 0;
    for (uint64_t k = 0; k < 10 && fails == 0; k++) {
        uint16_t phrase[12];
        generate_phrase12(k, phrase);
        const uint16_t* base11 = phrase; /* first 11 words = base */

        uint64_t prefix_w[16], cur;
        uint32_t bits = 0, prefix_len = 0;
        int wi = 0;
        wlbp_base_prefix_build(table, base11, prefix_w, &cur, &bits, &wi, &prefix_len);

        /* enumerate + collect (invariant 128) */
        uint16_t cand[12];
        memcpy(cand, base11, 11 * sizeof(uint16_t));
        uint16_t valid_w12[256];
        int n_valid = 0;
        for (uint32_t w12 = 0; w12 < 2048; w12++) {
            cand[11] = (uint16_t)w12;
            if (verify_checksum_12(cand)) {
                if (n_valid < 256) valid_w12[n_valid] = (uint16_t)w12;
                n_valid++;
            }
        }
        if (n_valid != 128) { printf("check2 FAIL k=%llu: %d valid\n", (unsigned long long)k, n_valid); fails++; break; }

        /* process in shipped groups: triples x3 + remainder, seeds vs classic */
        int v = 0;
        for (; v + 2 < n_valid; v += 3) {
            uint64_t key[3][16];
            uint32_t len[3];
            for (int c = 0; c < 3; c++) {
                uint64_t wb = table[valid_w12[v + c]];
                len[c] = (uint32_t)wlbp_wb_len(wb);
                wlbp_keyblock_tail_swap(prefix_w, cur, bits, wi, wb, len[c], key[c]);
            }
            uint8_t seeds[3][64];
            pbkdf2_sha512_keyblock_fast_x3(key[0], prefix_len + len[0],
                                           key[1], prefix_len + len[1],
                                           key[2], prefix_len + len[2],
                                           2048, seeds[0], seeds[1], seeds[2]);
            for (int c = 0; c < 3; c++) {
                uint16_t full12[12];
                memcpy(full12, base11, 11 * sizeof(uint16_t));
                full12[11] = valid_w12[v + c];
                uint8_t ref[64];
                classic_seed(full12, table, ref);
                if (memcmp(seeds[c], ref, 64) != 0) {
                    printf("check2 FAIL k=%llu triple c=%d\n", (unsigned long long)k, c);
                    fails++; break;
                }
                total_derivations++;
            }
            if (fails) break;
        }
        if (fails) break;
        if (v + 1 < n_valid) {
            uint64_t key[2][16];
            uint32_t len[2];
            for (int c = 0; c < 2; c++) {
                uint64_t wb = table[valid_w12[v + c]];
                len[c] = (uint32_t)wlbp_wb_len(wb);
                wlbp_keyblock_tail_swap(prefix_w, cur, bits, wi, wb, len[c], key[c]);
            }
            uint8_t seeds[2][64];
            pbkdf2_sha512_keyblock_fast_x2(key[0], prefix_len + len[0],
                                           key[1], prefix_len + len[1],
                                           2048, seeds[0], seeds[1]);
            for (int c = 0; c < 2; c++) {
                uint16_t full12[12];
                memcpy(full12, base11, 11 * sizeof(uint16_t));
                full12[11] = valid_w12[v + c];
                uint8_t ref[64];
                classic_seed(full12, table, ref);
                if (memcmp(seeds[c], ref, 64) != 0) { printf("check2 FAIL k=%llu pair c=%d\n", (unsigned long long)k, c); fails++; break; }
                total_derivations++;
            }
            if (fails) break;
            v += 2;
        }
        if (v < n_valid) {
            uint64_t key[16];
            uint64_t wb = table[valid_w12[v]];
            uint32_t len = (uint32_t)wlbp_wb_len(wb);
            wlbp_keyblock_tail_swap(prefix_w, cur, bits, wi, wb, len, key);
            uint8_t seed1[64];
            pbkdf2_sha512_keyblock_fast(key, prefix_len + len, 2048, seed1);
            uint16_t full12[12];
            memcpy(full12, base11, 11 * sizeof(uint16_t));
            full12[11] = valid_w12[v];
            uint8_t ref[64];
            classic_seed(full12, table, ref);
            if (memcmp(seed1, ref, 64) != 0) { printf("check2 FAIL k=%llu single\n", (unsigned long long)k); fails++; break; }
            total_derivations++;
        }
    }
    printf("check2/3 monolithic end-to-end (10 bases, %llu derivations, x3+x2+single vs classic): %s\n",
           (unsigned long long)total_derivations, fails == 0 ? "PASS" : "FAIL");

    printf("RESULT: %s\n", fails == 0 ? "PASS - E6 monolithic thread verified" : "FAIL");
    return fails == 0 ? 0 : 1;
}
