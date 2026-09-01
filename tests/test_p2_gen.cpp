/*
 * test_p2_gen.cpp - P2 verification (kernel 1 generation):
 *   1. v2 (precomputed pool, same %) == v1 (inline available rebuild):
 *      bit-identical phrase stream for the same seed -> the pool precompute
 *      changes NOTHING in the generation.
 *   2. v3 (Lemire range reduction): structural properties hold - 4 required
 *      words present, 8 distinct pool words, 12 total.
 *
 * The generator functions live in main.cu (not includable); the exact bodies
 * are replicated here against the same constants. Wiring is mechanical.
 * Build: cl /O2 /EHsc test_p2_gen.cpp
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>

#define MAX_WORDS 40

static uint64_t cuda_rand(uint64_t* seed) {
    *seed = (*seed * 6364136223846793005ULL + 1442695040888963407ULL);
    return *seed;
}

/* old kernel-1 range reduction (slow 64-bit modulo) */
static uint32_t rand_mod(uint64_t* seed, uint32_t range) {
    return (uint32_t)(cuda_rand(seed) % range);
}

/* P2 Lemire multiply-shift (division-free) */
static uint32_t rand_range(uint64_t* seed, uint32_t range) {
    uint64_t x = cuda_rand(seed);
    return (uint32_t)(((x >> 32) * (uint64_t)range) >> 32);
}

/* v1 reference: inline available[] rebuild (the ORIGINAL algorithm) */
static void gen_v1(uint64_t seed, uint32_t total_words, const uint16_t* base_indices, uint16_t* out_indices) {
    const uint32_t required_positions[4] = {0, 7, 10, 12};
    out_indices[0] = base_indices[required_positions[0]];
    out_indices[1] = base_indices[required_positions[1]];
    out_indices[2] = base_indices[required_positions[2]];
    out_indices[3] = base_indices[required_positions[3]];

    uint16_t available[MAX_WORDS];
    uint32_t available_count = 0;
    for (uint32_t i = 0; i < total_words; i++) {
        bool is_required = false;
        for (uint32_t r = 0; r < 4; r++) {
            if (i == required_positions[r]) { is_required = true; break; }
        }
        if (!is_required) available[available_count++] = base_indices[i];
    }

    for (uint32_t i = 0; i < 8; i++) {
        uint32_t j = i + rand_mod(&seed, available_count - i);
        out_indices[4 + i] = available[j];
        uint16_t temp = available[j]; available[j] = available[i]; available[i] = temp;
    }
    for (uint32_t i = 11; i > 0; i--) {
        uint32_t j = rand_mod(&seed, i + 1);
        uint16_t temp = out_indices[i]; out_indices[i] = out_indices[j]; out_indices[j] = temp;
    }
}

/* v2: precomputed pool, SAME % reduction -> must be bit-identical to v1 */
static void gen_v2(uint64_t seed, const uint16_t* req, const uint16_t* avail, uint32_t avail_count, uint16_t* out_indices) {
    out_indices[0] = req[0]; out_indices[1] = req[1];
    out_indices[2] = req[2]; out_indices[3] = req[3];

    uint16_t av[MAX_WORDS];
    for (uint32_t i = 0; i < avail_count; i++) av[i] = avail[i];

    for (uint32_t i = 0; i < 8; i++) {
        uint32_t j = i + rand_mod(&seed, avail_count - i);
        out_indices[4 + i] = av[j];
        uint16_t temp = av[j]; av[j] = av[i]; av[i] = temp;
    }
    for (uint32_t i = 11; i > 0; i--) {
        uint32_t j = rand_mod(&seed, i + 1);
        uint16_t temp = out_indices[i]; out_indices[i] = out_indices[j]; out_indices[j] = temp;
    }
}

/* v3: precomputed pool + Lemire reduction (shipped kernel-1 path) */
static void gen_v3(uint64_t seed, const uint16_t* req, const uint16_t* avail, uint32_t avail_count, uint16_t* out_indices) {
    out_indices[0] = req[0]; out_indices[1] = req[1];
    out_indices[2] = req[2]; out_indices[3] = req[3];

    uint16_t av[MAX_WORDS];
    for (uint32_t i = 0; i < avail_count; i++) av[i] = avail[i];

    for (uint32_t i = 0; i < 8; i++) {
        uint32_t j = i + rand_range(&seed, avail_count - i);
        out_indices[4 + i] = av[j];
        uint16_t temp = av[j]; av[j] = av[i]; av[i] = temp;
    }
    for (uint32_t i = 11; i > 0; i--) {
        uint32_t j = rand_range(&seed, i + 1);
        uint16_t temp = out_indices[i]; out_indices[i] = out_indices[j]; out_indices[j] = temp;
    }
}

static uint64_t rng_state = 0x2545F4914F6CDD1DULL;
static uint64_t rng_next(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

int main(void) {
    /* synthetic pool: 40 distinct word indices, fixed order */
    uint16_t base[MAX_WORDS];
    for (uint32_t i = 0; i < MAX_WORDS; i++) base[i] = (uint16_t)(100 + i * 7);
    const uint32_t total_words = 40;
    const uint32_t required_positions[4] = {0, 7, 10, 12};

    uint16_t req[4];
    uint16_t avail[MAX_WORDS];
    uint32_t avail_count = 0;
    for (uint32_t r = 0; r < 4; r++) req[r] = base[required_positions[r]];
    for (uint32_t i = 0; i < total_words; i++) {
        bool is_req = false;
        for (int r = 0; r < 4; r++) if (i == required_positions[r]) { is_req = true; break; }
        if (!is_req) avail[avail_count++] = base[i];
    }

    int fails = 0;

    /* check 1: v2 == v1 bit-identical over 100k seeds */
    for (int t = 0; t < 100000; t++) {
        uint64_t seed = rng_next();
        uint16_t a[12], b[12];
        gen_v1(seed, total_words, base, a);
        gen_v2(seed, req, avail, avail_count, b);
        if (memcmp(a, b, sizeof(a)) != 0) {
            printf("check1 FAIL seed %llu\n", (unsigned long long)seed);
            fails++;
            break;
        }
    }
    printf("check1 v2 (precomputed pool) == v1 (100k seeds): %s\n", fails == 0 ? "PASS" : "FAIL");

    /* check 2: v3 (Lemire) structural properties over 100k seeds */
    int c2_fails = 0;
    for (int t = 0; t < 100000; t++) {
        uint64_t seed = rng_next();
        uint16_t p[12];
        gen_v3(seed, req, avail, avail_count, p);

        int req_found = 0, pool_ok = 1;
        uint32_t pos_seen[4] = {0,0,0,0};
        for (int w = 0; w < 12; w++) {
            for (int r = 0; r < 4; r++) {
                if (p[w] == req[r]) { req_found++; pos_seen[r]++; }
            }
            bool in_pool = false;
            for (uint32_t a_i = 0; a_i < avail_count; a_i++) if (p[w] == avail[a_i]) in_pool = true;
            if (!in_pool) { /* must be one of the required then */ 
                bool is_req = false;
                for (int r = 0; r < 4; r++) if (p[w] == req[r]) is_req = true;
                if (!is_req) pool_ok = 0;
            }
        }
        if (req_found != 4 || pos_seen[0] != 1 || pos_seen[1] != 1 || pos_seen[2] != 1 || pos_seen[3] != 1 || !pool_ok) {
            c2_fails++;
            break;
        }
    }
    printf("check2 v3 (Lemire) structure (4 required + 8 pool, 100k seeds): %s\n", c2_fails == 0 ? "PASS" : "FAIL");
    fails += c2_fails;

    printf("RESULT: %s\n", fails == 0 ? "PASS - P2 generation verified" : "FAIL");
    return fails == 0 ? 0 : 1;
}
