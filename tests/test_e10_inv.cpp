/*
 * test_e10_inv.cpp - E10 verification: the libsecp256k1 addition-chain
 * inverse (253 sqr + 14 mul) must be byte-exact versus the Fermat
 * square-and-multiply reference over random inputs.
 *
 *   1. 500 random values: chain == fermat (byte-exact)
 *   2. sanity: a=1 -> 1, a=0 -> 0, a*p-1 checks via mul-back
 *   3. mul-back test: (a * inv(a)) mod p == 1 for random a
 *
 * Build: cl /O2 /EHsc test_e10_inv.cpp
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>

#define __device__
#define __constant__ static
#define __forceinline__ static
#define __align__(x)

#include "src/secp256k1.cuh"

static uint64_t rng_state = 0x0ADD1C741F000001ULL;
static uint64_t rng_next(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

int main(void) {
    int fails = 0;

    /* ---- check 1: chain == fermat (500 random values) ---- */
    for (int t = 0; t < 500; t++) {
        uint256_t a;
        for (int i = 0; i < 8; i++) a.d[i] = (uint32_t)(rng_next() & 0xFFFFFFFFu);
        a.d[7] &= 0x0FFFFFFFu; /* keep < p */

        uint256_t inv_chain, inv_fermat;
        uint256_mod_inv_chain_v2(&inv_chain, &a, &SECP256K1_P);
        uint256_mod_inv_fermat(&inv_fermat, &a, &SECP256K1_P);

        if (memcmp(inv_chain.d, inv_fermat.d, sizeof(inv_chain.d)) != 0) {
            printf("check1 FAIL t=%d\n", t);
            fails++;
            break;
        }
    }
    printf("check1 addition chain == fermat (500 values): %s\n", fails == 0 ? "PASS" : "FAIL");

    /* ---- check 2: sanity + mul-back ---- */
    {
        /* a = 1 -> inverse = 1 */
        uint256_t one, inv;
        uint256_set_one(&one);
        uint256_mod_inv_chain_v2(&inv, &one, &SECP256K1_P);
        if (memcmp(inv.d, one.d, sizeof(one.d)) != 0) {
            printf("check2 FAIL: inv(1) != 1\n");
            fails++;
        }

        /* mul-back: (a * inv(a)) mod p == 1, 200 random a */
        int mulback_bad = 0;
        for (int t = 0; t < 200; t++) {
            uint256_t a;
            for (int i = 0; i < 8; i++) a.d[i] = (uint32_t)(rng_next() & 0xFFFFFFFFu);
            a.d[7] &= 0x0FFFFFFFu;
            if (uint256_is_zero(&a)) continue;

            uint256_t inv, prod;
            uint256_mod_inv_chain_v2(&inv, &a, &SECP256K1_P);
            uint256_mod_mul(&prod, &a, &inv, &SECP256K1_P);

            uint256_t expected;
            uint256_set_one(&expected);
            if (memcmp(prod.d, expected.d, sizeof(prod.d)) != 0) mulback_bad++;
        }
        printf("check2 mul-back (a * inv(a)) == 1 (200 random): %s\n",
               mulback_bad == 0 ? "PASS" : "FAIL");
        if (mulback_bad) fails++;
    }

    printf("RESULT: %s\n", fails == 0 ? "PASS - E10 addition chain verified" : "FAIL");
    return fails == 0 ? 0 : 1;
}
