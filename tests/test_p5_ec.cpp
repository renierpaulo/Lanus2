/*
 * test_p5_ec.cpp - P5 verification: fixed-base windowed scalar mult must
 * produce byte-exact results versus the original double-and-add path.
 *
 *   1. Table build sanity: window entry (w,1) == (2^(4w)*G) computed
 *      independently by the ORIGINAL double-and-add scalar mult.
 *   2. 300 random scalars: scalar_mult_G_window == scalar_mult (original),
 *      both normalized to affine -> identical 33-byte compressed pubkeys.
 *   3. Edge: k=0 -> infinity encoding preserved.
 *
 * Build: cl /O2 /EHsc test_p5_ec.cpp
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>

/* host shim */
#define __device__
#define __constant__ static
#define __forceinline__ static
#define __align__(x)

#include "src/secp256k1.cuh"

static uint64_t rng_state = 0x9E3779B97F4A7C15ULL;
static uint64_t rng_next(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

/* normalize a Jacobian point to affine bytes (reference conversion) */
static void to_affine_pubkey(const point_t* P, uint8_t* out33) {
    if (P->infinity) {
        out33[0] = 0x02;
        memset(out33 + 1, 0, 32);
        return;
    }
    uint256_t zinv, z2, z3, ax, ay;
    uint256_mod_inv(&zinv, &P->z, &SECP256K1_P);
    uint256_mod_mul(&z2, &zinv, &zinv, &SECP256K1_P);
    uint256_mod_mul(&z3, &z2, &zinv, &SECP256K1_P);
    uint256_mod_mul(&ax, &P->x, &z2, &SECP256K1_P);
    uint256_mod_mul(&ay, &P->y, &z3, &SECP256K1_P);
    out33[0] = (ay.d[0] & 1) ? 0x03 : 0x02;
    uint256_to_bytes(out33 + 1, &ax);
}

int main(void) {
    int fails = 0;

    /* build the window table (the same code the CUDA uploader runs) */
    static uint256_t xs[SECP_GWIN_ENTRIES], ys[SECP_GWIN_ENTRIES];
    secp256k1_gwin_build(xs, ys);
    /* host-test equivalent of the CUDA uploader's cudaMemcpyToSymbol */
    memcpy(d_Gwin_x, xs, sizeof(xs));
    memcpy(d_Gwin_y, ys, sizeof(ys));

    /* ---- check 1: table entries vs original double-and-add ---- */
    for (int w = 0; w < 64; w += 9) { /* sample windows */
        for (int d = 1; d <= 15; d += 2) { /* sample entries */
            /* scalar = d * 2^(4w) */
            uint256_t k;
            uint256_clear(&k);
            k.d[w >> 3] |= (uint32_t)d << ((w & 7) * 4);

            point_t ref;
            point_t G;
            G.x = SECP256K1_GX; G.y = SECP256K1_GY;
            uint256_set_one(&G.z); G.infinity = false;
            scalar_mult(&ref, &k, &G);

            uint8_t ref33[33], win33[33];
            to_affine_pubkey(&ref, ref33);

            /* window path through the shipped function */
            point_t wp;
            scalar_mult_G_window(&wp, &k);
            to_affine_pubkey(&wp, win33);

            if (memcmp(ref33, win33, 33) != 0) {
                printf("check1 FAIL w=%d d=%d\n", w, d);
                fails++;
                break;
            }
        }
        if (fails) break;
    }
    printf("check1 table entries vs double-and-add (sampled): %s\n", fails == 0 ? "PASS" : "FAIL");

    /* ---- check 2: 300 random scalars, window == original ---- */
    int c2_fails = 0;
    for (int t = 0; t < 300; t++) {
        uint8_t priv[32];
        for (int i = 0; i < 32; i++) priv[i] = (uint8_t)(rng_next() & 0xFF);
        priv[0] &= 0x0F; /* keep k < n with overwhelming margin */

        uint8_t ref33[33];
        {
            uint256_t k;
            bytes_to_uint256(&k, priv);
            point_t G;
            G.x = SECP256K1_GX; G.y = SECP256K1_GY;
            uint256_set_one(&G.z); G.infinity = false;
            point_t P;
            scalar_mult(&P, &k, &G);
            to_affine_pubkey(&P, ref33);
        }

        uint8_t win33[33];
        secp256k1_get_pubkey_compressed(priv, win33);

        if (memcmp(ref33, win33, 33) != 0) {
            printf("check2 FAIL scalar %d\n", t);
            c2_fails++;
            break;
        }
    }
    printf("check2 windowed pubkey == double-and-add (300 scalars): %s\n", c2_fails == 0 ? "PASS" : "FAIL");
    fails += c2_fails;

    /* ---- check 3: k = 0 -> infinity encoding ---- */
    uint8_t zero_priv[32] = {0};
    uint8_t pk[33];
    secp256k1_get_pubkey_compressed(zero_priv, pk);
    uint8_t expect[33] = {0x02, 0};
    memset(expect + 1, 0, 32);
    int c3_ok = memcmp(pk, expect, 33) == 0;
    printf("check3 k=0 infinity encoding: %s\n", c3_ok ? "PASS" : "FAIL");
    if (!c3_ok) fails++;

    printf("RESULT: %s\n", fails == 0 ? "PASS - P5 EC windowed path is byte-exact" : "FAIL");
    return fails == 0 ? 0 : 1;
}
