/*
 * gen_master_pre.cpp - computes the CONSTANT midstates of HMAC-SHA512 with
 * the fixed key "Bitcoin seed" (the BIP32 master key HMAC):
 *   d_min_pre[8]  = SHA512_compress(IV, k_ipad_block)   (key^0x36 padded)
 *   d_mout_pre[8] = SHA512_compress(IV, k_opad_block)   (key^0x5c padded)
 * These are universal constants (key never changes) -> compile-time
 * __constant__ arrays; per candidate the master HMAC costs 2 compressions
 * instead of 4 + pad-building byte work.
 * Reads K512 from the REAL sha512.cuh (never hand-typed).
 * Build: cl /O2 /EHsc gen_master_pre.cpp
 */
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#define __device__
#define __constant__ static
#define __forceinline__ static
#define __align__(x)
#include "src/pbkdf2_opt.cuh"

int main(void) {
    /* build the padded ipad/opad blocks for key "Bitcoin seed" (12 bytes) */
    uint8_t key[12] = {'B','i','t','c','o','i','n',' ','s','e','e','d'};
    uint8_t ip[128], op[128];
    memset(ip, 0x36, 128);
    memset(op, 0x5c, 128);
    for (int i = 0; i < 12; i++) { ip[i] ^= key[i]; op[i] ^= key[i]; }

    /* compress from IV */
    SHA512State_t a, b;
    sha512_init_state_opt(&a);
    sha512_transform_block_raw_opt(&a, ip);
    sha512_init_state_opt(&b);
    sha512_transform_block_raw_opt(&b, op);

    printf("__constant__ uint64_t d_min_pre[8] = {\n");
    for (int i = 0; i < 8; i++)
        printf("    0x%016llxULL%s\n", (unsigned long long)a.h[i], i<7?",":"");
    printf("};\n\n");
    printf("__constant__ uint64_t d_mout_pre[8] = {\n");
    for (int i = 0; i < 8; i++)
        printf("    0x%016llxULL%s\n", (unsigned long long)b.h[i], i<7?",":"");
    printf("};\n");
    return 0;
}
