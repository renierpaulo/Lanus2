/*
 * gen_kps.cpp - computes the compile-time folded round constants for ETAPA C.
 * Reads K512 DIRECTLY from src/sha512.cuh (single source of truth - never
 * hand-type crypto constants).
 *
 * Emits:
 *   d_KP36[16] = K512[i]   + 0x3636363636363636          (key-block ipad tail)
 *   d_KP5c[16] = K512[i]   + 0x5c5c5c5c5c5c5c5c          (key-block opad tail)
 *   d_KPP[8]   = K512[8+j] + pad[j]                      (U-chain blocks, rounds 8..15)
 *             pad = {0x8000000000000000,0,0,0,0,0,0,0x600}
 *   d_KPS[80]  = K512[i]   + W_salt[i] (FULL schedule)   (fixed "mnemonic"+INT(1) block)
 * Build: cl /O2 /EHsc gen_kps.cpp  (vcvars env)
 */
#include <stdio.h>
#include <stdint.h>

/* host shim - reuse the REAL K512 and helpers from sha512.cuh */
#include <string.h>
#define __device__
#define __constant__ static
#define __forceinline__ static
#define __align__(x)
#include "src/sha512.cuh"

int main(void) {
    /* sanity: first/last constants of the REAL table */
    printf("// generated from src/sha512.cuh K512[0]=%016llx K512[79]=%016llx\n\n",
           (unsigned long long)K512[0], (unsigned long long)K512[79]);

    printf("__constant__ uint64_t d_KP36[16] = {\n");
    for (int i = 0; i < 16; i++)
        printf("    0x%016llxULL%s\n", (unsigned long long)(K512[i] + 0x3636363636363636ULL), i<15?",":"");
    printf("};\n\n");

    printf("__constant__ uint64_t d_KP5c[16] = {\n");
    for (int i = 0; i < 16; i++)
        printf("    0x%016llxULL%s\n", (unsigned long long)(K512[i] + 0x5c5c5c5c5c5c5c5cULL), i<15?",":"");
    printf("};\n\n");

    printf("__constant__ uint64_t d_KPP[8] = {\n");
    const uint64_t pad[8] = {0x8000000000000000ULL,0,0,0,0,0,0,0x600ULL};
    for (int j = 0; j < 8; j++)
        printf("    0x%016llxULL%s\n", (unsigned long long)(K512[8+j] + pad[j]), j<7?",":"");
    printf("};\n\n");

    /* fixed salt block: "mnemonic" || INT(1) || padding (first-iteration inner) */
    uint64_t W[80];
    W[0] = 0x6d6e656d6f6e6963ULL;
    W[1] = 0x0000000180000000ULL;
    for (int i = 2; i < 15; i++) W[i] = 0;
    W[15] = 0x460ULL; /* (128+12)*8 = 1120 bits */
    for (int i = 16; i < 80; i++)
        W[i] = gamma1_512(W[i-2]) + W[i-7] + gamma0_512(W[i-15]) + W[i-16];

    printf("__constant__ uint64_t d_KPS[80] = {\n");
    for (int i = 0; i < 80; i++)
        printf("    0x%016llxULL%s\n", (unsigned long long)(K512[i] + W[i]), i<79?",":"");
    printf("};\n");
    return 0;
}
