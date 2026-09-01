#ifndef PBKDF2_VARIANTS_CU
#define PBKDF2_VARIANTS_CU

#include <stdint.h>
#include "sha512.cuh"
#include "pbkdf2_opt.cuh"

// ---- SOLO: PBKDF2-SHA512 keyblock fast (single chain) ----
__device__ void pbkdf2_sha512_keyblock_fast(
    const uint64_t key_w[16], uint32_t key_len,
    uint32_t iterations,
    uint8_t* output_64bytes
) {
    uint32_t i0 = (key_len + 7u) / 8u;
    if (i0 > 16u) i0 = 16u;

    uint64_t ipad_w[16], opad_w[16];

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        ipad_w[i] = key_w[i] ^ 0x3636363636363636ULL;
        opad_w[i] = key_w[i] ^ 0x5c5c5c5c5c5c5c5cULL;
    }

    SHA512State_t ctx_inner_pre, ctx_outer_pre;
    sha512_init_state_opt(&ctx_inner_pre);
    sha512_compress16_rot_opt(&ctx_inner_pre, ipad_w, i0, d_KP36);

    sha512_init_state_opt(&ctx_outer_pre);
    sha512_compress16_rot_opt(&ctx_outer_pre, opad_w, i0, d_KP5c);

    uint64_t U[8], T[8];

    // First iteration: inner over the fused salt block, outer over the
    // inner hash words directly (they ARE the big-endian message words)
    {
        SHA512State_t ctx = ctx_inner_pre;
        sha512_transform_saltblock_folded_opt(&ctx);

        SHA512State_t ctx_out = ctx_outer_pre;
        sha512_finish64msg_fast(&ctx_out, ctx.h);

        #pragma unroll
        for (int j = 0; j < 8; j++) { T[j] = ctx_out.h[j]; U[j] = ctx_out.h[j]; }
    }

    for (uint32_t iter = 1; iter < iterations; iter++) {
        SHA512State_t ctx = ctx_inner_pre;
        sha512_finish64msg_fast(&ctx, U);

        SHA512State_t ctx_out = ctx_outer_pre;
        sha512_finish64msg_fast(&ctx_out, ctx.h);

        #pragma unroll
        for (int j = 0; j < 8; j++) { U[j] = ctx_out.h[j]; T[j] ^= U[j]; }
    }

    #pragma unroll
    for (int j = 0; j < 8; j++) {
        uint64_t x = T[j];
        output_64bytes[j*8+0] = (uint8_t)(x >> 56);
        output_64bytes[j*8+1] = (uint8_t)(x >> 48);
        output_64bytes[j*8+2] = (uint8_t)(x >> 40);
        output_64bytes[j*8+3] = (uint8_t)(x >> 32);
        output_64bytes[j*8+4] = (uint8_t)(x >> 24);
        output_64bytes[j*8+5] = (uint8_t)(x >> 16);
        output_64bytes[j*8+6] = (uint8_t)(x >> 8);
        output_64bytes[j*8+7] = (uint8_t)(x);
    }
}

#ifndef DEV_BUILD
// ---- x2: 2-chain interleaved compressors ----

// Key-block compressor, 2 chains. foldA/foldB may differ by 1 (tail length).
__device__ void sha512_compress16_rot_x2_opt(
    SHA512State_t* sA, const uint64_t mA[16],
    SHA512State_t* sB, const uint64_t mB[16],
    uint32_t foldA, uint32_t foldB, const uint64_t* KP
) {
    uint64_t aA=sA->h[0], bA=sA->h[1], cA=sA->h[2], dA=sA->h[3];
    uint64_t eA=sA->h[4], fA=sA->h[5], gA=sA->h[6], hA=sA->h[7];
    uint64_t aB=sB->h[0], bB=sB->h[1], cB=sB->h[2], dB=sB->h[3];
    uint64_t eB=sB->h[4], fB=sB->h[5], gB=sB->h[6], hB=sB->h[7];
    uint64_t t1A, t2A, t1B, t2B;
    uint64_t WA[16], WB[16];

    #pragma unroll
    for (int t = 0; t < 16; t++) { WA[t] = mA[t]; WB[t] = mB[t]; }

    #pragma unroll
    for (int t = 0; t < 16; t++) {
        uint64_t KwA = (t >= (int)foldA) ? KP[t] : (K512[t] + WA[t]);
        uint64_t KwB = (t >= (int)foldB) ? KP[t] : (K512[t] + WB[t]);
        t1A = hA + sigma1_512(eA) + ch64(eA, fA, gA) + KwA;
        t2A = sigma0_512(aA) + maj64(aA, bA, cA);
        t1B = hB + sigma1_512(eB) + ch64(eB, fB, gB) + KwB;
        t2B = sigma0_512(aB) + maj64(aB, bB, cB);
        hA=gA; gA=fA; fA=eA; eA=dA+t1A; dA=cA; cA=bA; bA=aA; aA=t1A+t2A;
        hB=gB; gB=fB; fB=eB; eB=dB+t1B; dB=cB; cB=bB; bB=aB; aB=t1B+t2B;
    }

    #pragma unroll
    for (int t = 16; t < 80; t++) {
        WA[t & 15] += gamma1_512(WA[(t - 2) & 15]) + WA[(t - 7) & 15] + gamma0_512(WA[(t - 15) & 15]);
        WB[t & 15] += gamma1_512(WB[(t - 2) & 15]) + WB[(t - 7) & 15] + gamma0_512(WB[(t - 15) & 15]);
        t1A = hA + sigma1_512(eA) + ch64(eA, fA, gA) + K512[t] + WA[t & 15];
        t2A = sigma0_512(aA) + maj64(aA, bA, cA);
        t1B = hB + sigma1_512(eB) + ch64(eB, fB, gB) + K512[t] + WB[t & 15];
        t2B = sigma0_512(aB) + maj64(aB, bB, cB);
        hA=gA; gA=fA; fA=eA; eA=dA+t1A; dA=cA; cA=bA; bA=aA; aA=t1A+t2A;
        hB=gB; gB=fB; fB=eB; eB=dB+t1B; dB=cB; cB=bB; bB=aB; aB=t1B+t2B;
    }

    sA->h[0]+=aA; sA->h[1]+=bA; sA->h[2]+=cA; sA->h[3]+=dA;
    sA->h[4]+=eA; sA->h[5]+=fA; sA->h[6]+=gA; sA->h[7]+=hA;
    sB->h[0]+=aB; sB->h[1]+=bB; sB->h[2]+=cB; sB->h[3]+=dB;
    sB->h[4]+=eB; sB->h[5]+=fB; sB->h[6]+=gB; sB->h[7]+=hB;
}

// U-chain hot block, 2 chains: rounds 0-7 message, 8-15 folded d_KPP, 16-79 rot.
__device__ void sha512_finish64msg_x2_fast(
    SHA512State_t* sA, const uint64_t uA[8],
    SHA512State_t* sB, const uint64_t uB[8]
) {
    uint64_t aA=sA->h[0], bA=sA->h[1], cA=sA->h[2], dA=sA->h[3];
    uint64_t eA=sA->h[4], fA=sA->h[5], gA=sA->h[6], hA=sA->h[7];
    uint64_t aB=sB->h[0], bB=sB->h[1], cB=sB->h[2], dB=sB->h[3];
    uint64_t eB=sB->h[4], fB=sB->h[5], gB=sB->h[6], hB=sB->h[7];
    uint64_t t1A, t2A, t1B, t2B;
    uint64_t WA[16], WB[16];

    #pragma unroll
    for (int t = 0; t < 8; t++) { WA[t] = uA[t]; WB[t] = uB[t]; }

    #pragma unroll
    for (int t = 0; t < 8; t++) {
        t1A = hA + sigma1_512(eA) + ch64(eA, fA, gA) + K512[t] + WA[t];
        t2A = sigma0_512(aA) + maj64(aA, bA, cA);
        t1B = hB + sigma1_512(eB) + ch64(eB, fB, gB) + K512[t] + WB[t];
        t2B = sigma0_512(aB) + maj64(aB, bB, cB);
        hA=gA; gA=fA; fA=eA; eA=dA+t1A; dA=cA; cA=bA; bA=aA; aA=t1A+t2A;
        hB=gB; gB=fB; fB=eB; eB=dB+t1B; dB=cB; cB=bB; bB=aB; aB=t1B+t2B;
    }

    WA[8]  = 0x8000000000000000ULL; WB[8]  = 0x8000000000000000ULL;
    WA[9]  = 0; WA[10] = 0; WA[11] = 0; WA[12] = 0; WA[13] = 0; WA[14] = 0; WA[15] = 0x600ULL;
    WB[9]  = 0; WB[10] = 0; WB[11] = 0; WB[12] = 0; WB[13] = 0; WB[14] = 0; WB[15] = 0x600ULL;

    #pragma unroll 4
    for (int t = 8; t < 16; t++) {
        t1A = hA + sigma1_512(eA) + ch64(eA, fA, gA) + d_KPP[t - 8];
        t2A = sigma0_512(aA) + maj64(aA, bA, cA);
        t1B = hB + sigma1_512(eB) + ch64(eB, fB, gB) + d_KPP[t - 8];
        t2B = sigma0_512(aB) + maj64(aB, bB, cB);
        hA=gA; gA=fA; fA=eA; eA=dA+t1A; dA=cA; cA=bA; bA=aA; aA=t1A+t2A;
        hB=gB; gB=fB; fB=eB; eB=dB+t1B; dB=cB; cB=bB; bB=aB; aB=t1B+t2B;
    }

    #pragma unroll 4
    for (int t = 16; t < 80; t++) {
        WA[t & 15] += gamma1_512(WA[(t - 2) & 15]) + WA[(t - 7) & 15] + gamma0_512(WA[(t - 15) & 15]);
        WB[t & 15] += gamma1_512(WB[(t - 2) & 15]) + WB[(t - 7) & 15] + gamma0_512(WB[(t - 15) & 15]);
        t1A = hA + sigma1_512(eA) + ch64(eA, fA, gA) + K512[t] + WA[t & 15];
        t2A = sigma0_512(aA) + maj64(aA, bA, cA);
        t1B = hB + sigma1_512(eB) + ch64(eB, fB, gB) + K512[t] + WB[t & 15];
        t2B = sigma0_512(aB) + maj64(aB, bB, cB);
        hA=gA; gA=fA; fA=eA; eA=dA+t1A; dA=cA; cA=bA; bA=aA; aA=t1A+t2A;
        hB=gB; gB=fB; fB=eB; eB=dB+t1B; dB=cB; cB=bB; bB=aB; aB=t1B+t2B;
    }

    sA->h[0]+=aA; sA->h[1]+=bA; sA->h[2]+=cA; sA->h[3]+=dA;
    sA->h[4]+=eA; sA->h[5]+=fA; sA->h[6]+=gA; sA->h[7]+=hA;
    sB->h[0]+=aB; sB->h[1]+=bB; sB->h[2]+=cB; sB->h[3]+=dB;
    sB->h[4]+=eB; sB->h[5]+=fB; sB->h[6]+=gB; sB->h[7]+=hB;
}

// Fixed salt block, 2 chains (same folded constants, two states).
__device__ void sha512_transform_saltblock_folded_x2_opt(
    SHA512State_t* sA, SHA512State_t* sB
) {
    uint64_t aA=sA->h[0], bA=sA->h[1], cA=sA->h[2], dA=sA->h[3];
    uint64_t eA=sA->h[4], fA=sA->h[5], gA=sA->h[6], hA=sA->h[7];
    uint64_t aB=sB->h[0], bB=sB->h[1], cB=sB->h[2], dB=sB->h[3];
    uint64_t eB=sB->h[4], fB=sB->h[5], gB=sB->h[6], hB=sB->h[7];
    uint64_t t1A, t2A, t1B, t2B;

    #pragma unroll 4
    for (int i = 0; i < 80; i++) {
        t1A = hA + sigma1_512(eA) + ch64(eA, fA, gA) + d_KPS[i];
        t2A = sigma0_512(aA) + maj64(aA, bA, cA);
        t1B = hB + sigma1_512(eB) + ch64(eB, fB, gB) + d_KPS[i];
        t2B = sigma0_512(aB) + maj64(aB, bB, cB);
        hA=gA; gA=fA; fA=eA; eA=dA+t1A; dA=cA; cA=bA; bA=aA; aA=t1A+t2A;
        hB=gB; gB=fB; fB=eB; eB=dB+t1B; dB=cB; cB=bB; bB=aB; aB=t1B+t2B;
    }

    sA->h[0]+=aA; sA->h[1]+=bA; sA->h[2]+=cA; sA->h[3]+=dA;
    sA->h[4]+=eA; sA->h[5]+=fA; sA->h[6]+=gA; sA->h[7]+=hA;
    sB->h[0]+=aB; sB->h[1]+=bB; sB->h[2]+=cB; sB->h[3]+=dB;
    sB->h[4]+=eB; sB->h[5]+=fB; sB->h[6]+=gB; sB->h[7]+=hB;
}

// E5b: full x2 PBKDF2 - two candidates of the same base, chains interleaved.
// Byte-exact equivalent of two separate pbkdf2_sha512_keyblock_fast calls.
__device__ void pbkdf2_sha512_keyblock_fast_x2(
    const uint64_t keyA[16], uint32_t lenA,
    const uint64_t keyB[16], uint32_t lenB,
    uint32_t iterations,
    uint8_t* outA, uint8_t* outB
) {
    uint32_t i0A = (lenA + 7u) / 8u; if (i0A > 16u) i0A = 16u;
    uint32_t i0B = (lenB + 7u) / 8u; if (i0B > 16u) i0B = 16u;

    uint64_t ipadA[16], opadA[16], ipadB[16], opadB[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        ipadA[i] = keyA[i] ^ 0x3636363636363636ULL;
        opadA[i] = keyA[i] ^ 0x5c5c5c5c5c5c5c5cULL;
        ipadB[i] = keyB[i] ^ 0x3636363636363636ULL;
        opadB[i] = keyB[i] ^ 0x5c5c5c5c5c5c5c5cULL;
    }

    SHA512State_t inA, oA, inB, oB;
    sha512_init_state_opt(&inA);
    sha512_init_state_opt(&oA);
    sha512_init_state_opt(&inB);
    sha512_init_state_opt(&oB);
    sha512_compress16_rot_x2_opt(&inA, ipadA, &inB, ipadB, i0A, i0B, d_KP36);
    sha512_compress16_rot_x2_opt(&oA, opadA, &oB, opadB, i0A, i0B, d_KP5c);

    uint64_t UA[8], TA[8], UB[8], TB[8];

    { // first iteration: salt x2, outer finish x2 (outer pre-states COPIED)
        SHA512State_t cA = inA, cB = inB;
        sha512_transform_saltblock_folded_x2_opt(&cA, &cB);
        SHA512State_t wA2 = oA, wB2 = oB;
        sha512_finish64msg_x2_fast(&wA2, cA.h, &wB2, cB.h);
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            TA[j] = wA2.h[j]; UA[j] = wA2.h[j];
            TB[j] = wB2.h[j]; UB[j] = wB2.h[j];
        }
    }

    for (uint32_t iter = 1; iter < iterations; iter++) {
        SHA512State_t cA = inA, cB = inB;
        sha512_finish64msg_x2_fast(&cA, UA, &cB, UB);
        SHA512State_t wA2 = oA, wB2 = oB;
        sha512_finish64msg_x2_fast(&wA2, cA.h, &wB2, cB.h);
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            UA[j] = wA2.h[j]; TA[j] ^= UA[j];
            UB[j] = wB2.h[j]; TB[j] ^= UB[j];
        }
    }

    #pragma unroll
    for (int j = 0; j < 8; j++) {
        uint64_t x = TA[j];
        outA[j*8+0]=(uint8_t)(x>>56); outA[j*8+1]=(uint8_t)(x>>48);
        outA[j*8+2]=(uint8_t)(x>>40); outA[j*8+3]=(uint8_t)(x>>32);
        outA[j*8+4]=(uint8_t)(x>>24); outA[j*8+5]=(uint8_t)(x>>16);
        outA[j*8+6]=(uint8_t)(x>>8);  outA[j*8+7]=(uint8_t)(x);
        x = TB[j];
        outB[j*8+0]=(uint8_t)(x>>56); outB[j*8+1]=(uint8_t)(x>>48);
        outB[j*8+2]=(uint8_t)(x>>40); outB[j*8+3]=(uint8_t)(x>>32);
        outB[j*8+4]=(uint8_t)(x>>24); outB[j*8+5]=(uint8_t)(x>>16);
        outB[j*8+6]=(uint8_t)(x>>8);  outB[j*8+7]=(uint8_t)(x);
    }
}

#endif // !DEV_BUILD

#ifndef DEV_BUILD
// ---- x3: 3-chain interleaved compressors ----

__device__ void sha512_compress16_rot_x3_opt(
    SHA512State_t* sA, const uint64_t mA[16],
    SHA512State_t* sB, const uint64_t mB[16],
    SHA512State_t* sC, const uint64_t mC[16],
    uint32_t foldA, uint32_t foldB, uint32_t foldC, const uint64_t* KP
) {
    uint64_t aA=sA->h[0], bA=sA->h[1], cA=sA->h[2], dA=sA->h[3];
    uint64_t eA=sA->h[4], fA=sA->h[5], gA=sA->h[6], hA=sA->h[7];
    uint64_t aB=sB->h[0], bB=sB->h[1], cB=sB->h[2], dB=sB->h[3];
    uint64_t eB=sB->h[4], fB=sB->h[5], gB=sB->h[6], hB=sB->h[7];
    uint64_t aC=sC->h[0], bC=sC->h[1], cC=sC->h[2], dC=sC->h[3];
    uint64_t eC=sC->h[4], fC=sC->h[5], gC=sC->h[6], hC=sC->h[7];
    uint64_t t1A, t2A, t1B, t2B, t1C, t2C;
    uint64_t WA[16], WB[16], WC[16];

    #pragma unroll
    for (int t = 0; t < 16; t++) { WA[t] = mA[t]; WB[t] = mB[t]; WC[t] = mC[t]; }

    #pragma unroll
    for (int t = 0; t < 16; t++) {
        uint64_t KwA = (t >= (int)foldA) ? KP[t] : (K512[t] + WA[t]);
        uint64_t KwB = (t >= (int)foldB) ? KP[t] : (K512[t] + WB[t]);
        uint64_t KwC = (t >= (int)foldC) ? KP[t] : (K512[t] + WC[t]);
        t1A = hA + sigma1_512(eA) + ch64(eA, fA, gA) + KwA;
        t2A = sigma0_512(aA) + maj64(aA, bA, cA);
        t1B = hB + sigma1_512(eB) + ch64(eB, fB, gB) + KwB;
        t2B = sigma0_512(aB) + maj64(aB, bB, cB);
        t1C = hC + sigma1_512(eC) + ch64(eC, fC, gC) + KwC;
        t2C = sigma0_512(aC) + maj64(aC, bC, cC);
        hA=gA; gA=fA; fA=eA; eA=dA+t1A; dA=cA; cA=bA; bA=aA; aA=t1A+t2A;
        hB=gB; gB=fB; fB=eB; eB=dB+t1B; dB=cB; cB=bB; bB=aB; aB=t1B+t2B;
        hC=gC; gC=fC; fC=eC; eC=dC+t1C; dC=cC; cC=bC; bC=aC; aC=t1C+t2C;
    }

    #pragma unroll
    for (int t = 16; t < 80; t++) {
        WA[t & 15] += gamma1_512(WA[(t - 2) & 15]) + WA[(t - 7) & 15] + gamma0_512(WA[(t - 15) & 15]);
        WB[t & 15] += gamma1_512(WB[(t - 2) & 15]) + WB[(t - 7) & 15] + gamma0_512(WB[(t - 15) & 15]);
        WC[t & 15] += gamma1_512(WC[(t - 2) & 15]) + WC[(t - 7) & 15] + gamma0_512(WC[(t - 15) & 15]);
        t1A = hA + sigma1_512(eA) + ch64(eA, fA, gA) + K512[t] + WA[t & 15];
        t2A = sigma0_512(aA) + maj64(aA, bA, cA);
        t1B = hB + sigma1_512(eB) + ch64(eB, fB, gB) + K512[t] + WB[t & 15];
        t2B = sigma0_512(aB) + maj64(aB, bB, cB);
        t1C = hC + sigma1_512(eC) + ch64(eC, fC, gC) + K512[t] + WC[t & 15];
        t2C = sigma0_512(aC) + maj64(aC, bC, cC);
        hA=gA; gA=fA; fA=eA; eA=dA+t1A; dA=cA; cA=bA; bA=aA; aA=t1A+t2A;
        hB=gB; gB=fB; fB=eB; eB=dB+t1B; dB=cB; cB=bB; bB=aB; aB=t1B+t2B;
        hC=gC; gC=fC; fC=eC; eC=dC+t1C; dC=cC; cC=bC; bC=aC; aC=t1C+t2C;
    }

    sA->h[0]+=aA; sA->h[1]+=bA; sA->h[2]+=cA; sA->h[3]+=dA;
    sA->h[4]+=eA; sA->h[5]+=fA; sA->h[6]+=gA; sA->h[7]+=hA;
    sB->h[0]+=aB; sB->h[1]+=bB; sB->h[2]+=cB; sB->h[3]+=dB;
    sB->h[4]+=eB; sB->h[5]+=fB; sB->h[6]+=gB; sB->h[7]+=hB;
    sC->h[0]+=aC; sC->h[1]+=bC; sC->h[2]+=cC; sC->h[3]+=dC;
    sC->h[4]+=eC; sC->h[5]+=fC; sC->h[6]+=gC; sC->h[7]+=hC;
}

__device__ void sha512_finish64msg_x3_fast(
    SHA512State_t* sA, const uint64_t uA[8],
    SHA512State_t* sB, const uint64_t uB[8],
    SHA512State_t* sC, const uint64_t uC[8]
) {
    uint64_t aA=sA->h[0], bA=sA->h[1], cA=sA->h[2], dA=sA->h[3];
    uint64_t eA=sA->h[4], fA=sA->h[5], gA=sA->h[6], hA=sA->h[7];
    uint64_t aB=sB->h[0], bB=sB->h[1], cB=sB->h[2], dB=sB->h[3];
    uint64_t eB=sB->h[4], fB=sB->h[5], gB=sB->h[6], hB=sB->h[7];
    uint64_t aC=sC->h[0], bC=sC->h[1], cC=sC->h[2], dC=sC->h[3];
    uint64_t eC=sC->h[4], fC=sC->h[5], gC=sC->h[6], hC=sC->h[7];
    uint64_t t1A, t2A, t1B, t2B, t1C, t2C;
    uint64_t WA[16], WB[16], WC[16];

    #pragma unroll
    for (int t = 0; t < 8; t++) { WA[t] = uA[t]; WB[t] = uB[t]; WC[t] = uC[t]; }

    #pragma unroll
    for (int t = 0; t < 8; t++) {
        t1A = hA + sigma1_512(eA) + ch64(eA, fA, gA) + K512[t] + WA[t];
        t2A = sigma0_512(aA) + maj64(aA, bA, cA);
        t1B = hB + sigma1_512(eB) + ch64(eB, fB, gB) + K512[t] + WB[t];
        t2B = sigma0_512(aB) + maj64(aB, bB, cB);
        t1C = hC + sigma1_512(eC) + ch64(eC, fC, gC) + K512[t] + WC[t];
        t2C = sigma0_512(aC) + maj64(aC, bC, cC);
        hA=gA; gA=fA; fA=eA; eA=dA+t1A; dA=cA; cA=bA; bA=aA; aA=t1A+t2A;
        hB=gB; gB=fB; fB=eB; eB=dB+t1B; dB=cB; cB=bB; bB=aB; aB=t1B+t2B;
        hC=gC; gC=fC; fC=eC; eC=dC+t1C; dC=cC; cC=bC; bC=aC; aC=t1C+t2C;
    }

    WA[8]  = 0x8000000000000000ULL; WB[8]  = 0x8000000000000000ULL; WC[8]  = 0x8000000000000000ULL;
    WA[9]  = 0; WA[10] = 0; WA[11] = 0; WA[12] = 0; WA[13] = 0; WA[14] = 0; WA[15] = 0x600ULL;
    WB[9]  = 0; WB[10] = 0; WB[11] = 0; WB[12] = 0; WB[13] = 0; WB[14] = 0; WB[15] = 0x600ULL;
    WC[9]  = 0; WC[10] = 0; WC[11] = 0; WC[12] = 0; WC[13] = 0; WC[14] = 0; WC[15] = 0x600ULL;

    #pragma unroll 4
    for (int t = 8; t < 16; t++) {
        t1A = hA + sigma1_512(eA) + ch64(eA, fA, gA) + d_KPP[t - 8];
        t2A = sigma0_512(aA) + maj64(aA, bA, cA);
        t1B = hB + sigma1_512(eB) + ch64(eB, fB, gB) + d_KPP[t - 8];
        t2B = sigma0_512(aB) + maj64(aB, bB, cB);
        t1C = hC + sigma1_512(eC) + ch64(eC, fC, gC) + d_KPP[t - 8];
        t2C = sigma0_512(aC) + maj64(aC, bC, cC);
        hA=gA; gA=fA; fA=eA; eA=dA+t1A; dA=cA; cA=bA; bA=aA; aA=t1A+t2A;
        hB=gB; gB=fB; fB=eB; eB=dB+t1B; dB=cB; cB=bB; bB=aB; aB=t1B+t2B;
        hC=gC; gC=fC; fC=eC; eC=dC+t1C; dC=cC; cC=bC; bC=aC; aC=t1C+t2C;
    }

    #pragma unroll 4
    for (int t = 16; t < 80; t++) {
        WA[t & 15] += gamma1_512(WA[(t - 2) & 15]) + WA[(t - 7) & 15] + gamma0_512(WA[(t - 15) & 15]);
        WB[t & 15] += gamma1_512(WB[(t - 2) & 15]) + WB[(t - 7) & 15] + gamma0_512(WB[(t - 15) & 15]);
        WC[t & 15] += gamma1_512(WC[(t - 2) & 15]) + WC[(t - 7) & 15] + gamma0_512(WC[(t - 15) & 15]);
        t1A = hA + sigma1_512(eA) + ch64(eA, fA, gA) + K512[t] + WA[t & 15];
        t2A = sigma0_512(aA) + maj64(aA, bA, cA);
        t1B = hB + sigma1_512(eB) + ch64(eB, fB, gB) + K512[t] + WB[t & 15];
        t2B = sigma0_512(aB) + maj64(aB, bB, cB);
        t1C = hC + sigma1_512(eC) + ch64(eC, fC, gC) + K512[t] + WC[t & 15];
        t2C = sigma0_512(aC) + maj64(aC, bC, cC);
        hA=gA; gA=fA; fA=eA; eA=dA+t1A; dA=cA; cA=bA; bA=aA; aA=t1A+t2A;
        hB=gB; gB=fB; fB=eB; eB=dB+t1B; dB=cB; cB=bB; bB=aB; aB=t1B+t2B;
        hC=gC; gC=fC; fC=eC; eC=dC+t1C; dC=cC; cC=bC; bC=aC; aC=t1C+t2C;
    }

    sA->h[0]+=aA; sA->h[1]+=bA; sA->h[2]+=cA; sA->h[3]+=dA;
    sA->h[4]+=eA; sA->h[5]+=fA; sA->h[6]+=gA; sA->h[7]+=hA;
    sB->h[0]+=aB; sB->h[1]+=bB; sB->h[2]+=cB; sB->h[3]+=dB;
    sB->h[4]+=eB; sB->h[5]+=fB; sB->h[6]+=gB; sB->h[7]+=hB;
    sC->h[0]+=aC; sC->h[1]+=bC; sC->h[2]+=cC; sC->h[3]+=dC;
    sC->h[4]+=eC; sC->h[5]+=fC; sC->h[6]+=gC; sC->h[7]+=hC;
}

__device__ void sha512_transform_saltblock_folded_x3_opt(
    SHA512State_t* sA, SHA512State_t* sB, SHA512State_t* sC
) {
    uint64_t aA=sA->h[0], bA=sA->h[1], cA=sA->h[2], dA=sA->h[3];
    uint64_t eA=sA->h[4], fA=sA->h[5], gA=sA->h[6], hA=sA->h[7];
    uint64_t aB=sB->h[0], bB=sB->h[1], cB=sB->h[2], dB=sB->h[3];
    uint64_t eB=sB->h[4], fB=sB->h[5], gB=sB->h[6], hB=sB->h[7];
    uint64_t aC=sC->h[0], bC=sC->h[1], cC=sC->h[2], dC=sC->h[3];
    uint64_t eC=sC->h[4], fC=sC->h[5], gC=sC->h[6], hC=sC->h[7];
    uint64_t t1A, t2A, t1B, t2B, t1C, t2C;

    #pragma unroll 4
    for (int i = 0; i < 80; i++) {
        t1A = hA + sigma1_512(eA) + ch64(eA, fA, gA) + d_KPS[i];
        t2A = sigma0_512(aA) + maj64(aA, bA, cA);
        t1B = hB + sigma1_512(eB) + ch64(eB, fB, gB) + d_KPS[i];
        t2B = sigma0_512(aB) + maj64(aB, bB, cB);
        t1C = hC + sigma1_512(eC) + ch64(eC, fC, gC) + d_KPS[i];
        t2C = sigma0_512(aC) + maj64(aC, bC, cC);
        hA=gA; gA=fA; fA=eA; eA=dA+t1A; dA=cA; cA=bA; bA=aA; aA=t1A+t2A;
        hB=gB; gB=fB; fB=eB; eB=dB+t1B; dB=cB; cB=bB; bB=aB; aB=t1B+t2B;
        hC=gC; gC=fC; fC=eC; eC=dC+t1C; dC=cC; cC=bC; bC=aC; aC=t1C+t2C;
    }

    sA->h[0]+=aA; sA->h[1]+=bA; sA->h[2]+=cA; sA->h[3]+=dA;
    sA->h[4]+=eA; sA->h[5]+=fA; sA->h[6]+=gA; sA->h[7]+=hA;
    sB->h[0]+=aB; sB->h[1]+=bB; sB->h[2]+=cB; sB->h[3]+=dB;
    sB->h[4]+=eB; sB->h[5]+=fB; sB->h[6]+=gB; sB->h[7]+=hB;
    sC->h[0]+=aC; sC->h[1]+=bC; sC->h[2]+=cC; sC->h[3]+=dC;
    sC->h[4]+=eC; sC->h[5]+=fC; sC->h[6]+=gC; sC->h[7]+=hC;
}

// E5c: full x3 PBKDF2 - three candidates of the same base, chains interleaved.
// Byte-exact equivalent of three pbkdf2_sha512_keyblock_fast calls.
__device__ void pbkdf2_sha512_keyblock_fast_x3(
    const uint64_t keyA[16], uint32_t lenA,
    const uint64_t keyB[16], uint32_t lenB,
    const uint64_t keyC[16], uint32_t lenC,
    uint32_t iterations,
    uint8_t* outA, uint8_t* outB, uint8_t* outC
) {
    uint32_t i0A = (lenA + 7u) / 8u; if (i0A > 16u) i0A = 16u;
    uint32_t i0B = (lenB + 7u) / 8u; if (i0B > 16u) i0B = 16u;
    uint32_t i0C = (lenC + 7u) / 8u; if (i0C > 16u) i0C = 16u;

    uint64_t ipadA[16], opadA[16], ipadB[16], opadB[16], ipadC[16], opadC[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        ipadA[i] = keyA[i] ^ 0x3636363636363636ULL;
        opadA[i] = keyA[i] ^ 0x5c5c5c5c5c5c5c5cULL;
        ipadB[i] = keyB[i] ^ 0x3636363636363636ULL;
        opadB[i] = keyB[i] ^ 0x5c5c5c5c5c5c5c5cULL;
        ipadC[i] = keyC[i] ^ 0x3636363636363636ULL;
        opadC[i] = keyC[i] ^ 0x5c5c5c5c5c5c5c5cULL;
    }

    /* compressed-state design: pre-states live in local arrays (L1), copied
     * into working registers per iteration - frees 32 regs per chain.
     * VOLATILE is deliberate: it forbids ptxas from promoting the arrays
     * back into registers (which would blow the budget to ~336 and spill). */
    volatile uint64_t inpreA[8], outpreA[8], inpreB[8], outpreB[8], inpreC[8], outpreC[8];
    SHA512State_t tA, tB, tC;
    sha512_init_state_opt(&tA);
    sha512_init_state_opt(&tB);
    sha512_init_state_opt(&tC);
    sha512_compress16_rot_x3_opt(&tA, ipadA, &tB, ipadB, &tC, ipadC, i0A, i0B, i0C, d_KP36);
    #pragma unroll
    for (int j = 0; j < 8; j++) { inpreA[j] = tA.h[j]; inpreB[j] = tB.h[j]; inpreC[j] = tC.h[j]; }
    sha512_init_state_opt(&tA);
    sha512_init_state_opt(&tB);
    sha512_init_state_opt(&tC);
    sha512_compress16_rot_x3_opt(&tA, opadA, &tB, opadB, &tC, opadC, i0A, i0B, i0C, d_KP5c);
    #pragma unroll
    for (int j = 0; j < 8; j++) { outpreA[j] = tA.h[j]; outpreB[j] = tB.h[j]; outpreC[j] = tC.h[j]; }

    uint64_t UA[8], TA[8], UB[8], TB[8], UC[8], TC[8];

    { // first iteration
        SHA512State_t cA, cB, cC;
        #pragma unroll
        for (int j = 0; j < 8; j++) { cA.h[j] = inpreA[j]; cB.h[j] = inpreB[j]; cC.h[j] = inpreC[j]; }
        sha512_transform_saltblock_folded_x3_opt(&cA, &cB, &cC);
        SHA512State_t wA, wB, wC;
        #pragma unroll
        for (int j = 0; j < 8; j++) { wA.h[j] = outpreA[j]; wB.h[j] = outpreB[j]; wC.h[j] = outpreC[j]; }
        sha512_finish64msg_x3_fast(&wA, cA.h, &wB, cB.h, &wC, cC.h);
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            TA[j] = wA.h[j]; UA[j] = wA.h[j];
            TB[j] = wB.h[j]; UB[j] = wB.h[j];
            TC[j] = wC.h[j]; UC[j] = wC.h[j];
        }
    }

    for (uint32_t iter = 1; iter < iterations; iter++) {
        SHA512State_t cA, cB, cC;
        #pragma unroll
        for (int j = 0; j < 8; j++) { cA.h[j] = inpreA[j]; cB.h[j] = inpreB[j]; cC.h[j] = inpreC[j]; }
        sha512_finish64msg_x3_fast(&cA, UA, &cB, UB, &cC, UC);
        SHA512State_t wA, wB, wC;
        #pragma unroll
        for (int j = 0; j < 8; j++) { wA.h[j] = outpreA[j]; wB.h[j] = outpreB[j]; wC.h[j] = outpreC[j]; }
        sha512_finish64msg_x3_fast(&wA, cA.h, &wB, cB.h, &wC, cC.h);
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            UA[j] = wA.h[j]; TA[j] ^= UA[j];
            UB[j] = wB.h[j]; TB[j] ^= UB[j];
            UC[j] = wC.h[j]; TC[j] ^= UC[j];
        }
    }

    #pragma unroll
    for (int j = 0; j < 8; j++) {
        uint64_t x = TA[j];
        outA[j*8+0]=(uint8_t)(x>>56); outA[j*8+1]=(uint8_t)(x>>48);
        outA[j*8+2]=(uint8_t)(x>>40); outA[j*8+3]=(uint8_t)(x>>32);
        outA[j*8+4]=(uint8_t)(x>>24); outA[j*8+5]=(uint8_t)(x>>16);
        outA[j*8+6]=(uint8_t)(x>>8);  outA[j*8+7]=(uint8_t)(x);
        x = TB[j];
        outB[j*8+0]=(uint8_t)(x>>56); outB[j*8+1]=(uint8_t)(x>>48);
        outB[j*8+2]=(uint8_t)(x>>40); outB[j*8+3]=(uint8_t)(x>>32);
        outB[j*8+4]=(uint8_t)(x>>24); outB[j*8+5]=(uint8_t)(x>>16);
        outB[j*8+6]=(uint8_t)(x>>8);  outB[j*8+7]=(uint8_t)(x);
        x = TC[j];
        outC[j*8+0]=(uint8_t)(x>>56); outC[j*8+1]=(uint8_t)(x>>48);
        outC[j*8+2]=(uint8_t)(x>>40); outC[j*8+3]=(uint8_t)(x>>32);
        outC[j*8+4]=(uint8_t)(x>>24); outC[j*8+5]=(uint8_t)(x>>16);
        outC[j*8+6]=(uint8_t)(x>>8);  outC[j*8+7]=(uint8_t)(x);
    }
}

#endif // !DEV_BUILD

#ifndef DEV_BUILD
// ---- x4: 4-chain interleaved compressors ----

// U-chain hot block, 4 chains: rounds 0-7 message, 8-15 folded d_KPP, 16-79 rot.
__device__ void sha512_finish64msg_x4_fast(
    SHA512State_t* sA, const uint64_t uA[8],
    SHA512State_t* sB, const uint64_t uB[8],
    SHA512State_t* sC, const uint64_t uC[8],
    SHA512State_t* sD, const uint64_t uD[8]
) {
    uint64_t aA=sA->h[0], bA=sA->h[1], cA=sA->h[2], dA=sA->h[3];
    uint64_t eA=sA->h[4], fA=sA->h[5], gA=sA->h[6], hA=sA->h[7];
    uint64_t aB=sB->h[0], bB=sB->h[1], cB=sB->h[2], dB=sB->h[3];
    uint64_t eB=sB->h[4], fB=sB->h[5], gB=sB->h[6], hB=sB->h[7];
    uint64_t aC=sC->h[0], bC=sC->h[1], cC=sC->h[2], dC=sC->h[3];
    uint64_t eC=sC->h[4], fC=sC->h[5], gC=sC->h[6], hC=sC->h[7];
    uint64_t aD=sD->h[0], bD=sD->h[1], cD=sD->h[2], dD=sD->h[3];
    uint64_t eD=sD->h[4], fD=sD->h[5], gD=sD->h[6], hD=sD->h[7];
    uint64_t t1A, t2A, t1B, t2B, t1C, t2C, t1D, t2D;
    uint64_t WA[16], WB[16], WC[16], WD[16];

    #pragma unroll
    for (int t = 0; t < 8; t++) { WA[t] = uA[t]; WB[t] = uB[t]; WC[t] = uC[t]; WD[t] = uD[t]; }

    #pragma unroll
    for (int t = 0; t < 8; t++) {
        t1A = hA + sigma1_512(eA) + ch64(eA, fA, gA) + K512[t] + WA[t];
        t2A = sigma0_512(aA) + maj64(aA, bA, cA);
        t1B = hB + sigma1_512(eB) + ch64(eB, fB, gB) + K512[t] + WB[t];
        t2B = sigma0_512(aB) + maj64(aB, bB, cB);
        t1C = hC + sigma1_512(eC) + ch64(eC, fC, gC) + K512[t] + WC[t];
        t2C = sigma0_512(aC) + maj64(aC, bC, cC);
        t1D = hD + sigma1_512(eD) + ch64(eD, fD, gD) + K512[t] + WD[t];
        t2D = sigma0_512(aD) + maj64(aD, bD, cD);
        hA=gA; gA=fA; fA=eA; eA=dA+t1A; dA=cA; cA=bA; bA=aA; aA=t1A+t2A;
        hB=gB; gB=fB; fB=eB; eB=dB+t1B; dB=cB; cB=bB; bB=aB; aB=t1B+t2B;
        hC=gC; gC=fC; fC=eC; eC=dC+t1C; dC=cC; cC=bC; bC=aC; aC=t1C+t2C;
        hD=gD; gD=fD; fD=eD; eD=dD+t1D; dD=cD; cD=bD; bD=aD; aD=t1D+t2D;
    }

    WA[8]  = 0x8000000000000000ULL; WB[8]  = 0x8000000000000000ULL;
    WC[8]  = 0x8000000000000000ULL; WD[8]  = 0x8000000000000000ULL;
    WA[9]  = 0; WA[10] = 0; WA[11] = 0; WA[12] = 0; WA[13] = 0; WA[14] = 0; WA[15] = 0x600ULL;
    WB[9]  = 0; WB[10] = 0; WB[11] = 0; WB[12] = 0; WB[13] = 0; WB[14] = 0; WB[15] = 0x600ULL;
    WC[9]  = 0; WC[10] = 0; WC[11] = 0; WC[12] = 0; WC[13] = 0; WC[14] = 0; WC[15] = 0x600ULL;
    WD[9]  = 0; WD[10] = 0; WD[11] = 0; WD[12] = 0; WD[13] = 0; WD[14] = 0; WD[15] = 0x600ULL;

    #pragma unroll
    for (int t = 8; t < 16; t++) {
        t1A = hA + sigma1_512(eA) + ch64(eA, fA, gA) + d_KPP[t - 8];
        t2A = sigma0_512(aA) + maj64(aA, bA, cA);
        t1B = hB + sigma1_512(eB) + ch64(eB, fB, gB) + d_KPP[t - 8];
        t2B = sigma0_512(aB) + maj64(aB, bB, cB);
        t1C = hC + sigma1_512(eC) + ch64(eC, fC, gC) + d_KPP[t - 8];
        t2C = sigma0_512(aC) + maj64(aC, bC, cC);
        t1D = hD + sigma1_512(eD) + ch64(eD, fD, gD) + d_KPP[t - 8];
        t2D = sigma0_512(aD) + maj64(aD, bD, cD);
        hA=gA; gA=fA; fA=eA; eA=dA+t1A; dA=cA; cA=bA; bA=aA; aA=t1A+t2A;
        hB=gB; gB=fB; fB=eB; eB=dB+t1B; dB=cB; cB=bB; bB=aB; aB=t1B+t2B;
        hC=gC; gC=fC; fC=eC; eC=dC+t1C; dC=cC; cC=bC; bC=aC; aC=t1C+t2C;
        hD=gD; gD=fD; fD=eD; eD=dD+t1D; dD=cD; cD=bD; bD=aD; aD=t1D+t2D;
    }

    #pragma unroll 4
    for (int t = 16; t < 80; t++) {
        WA[t & 15] += gamma1_512(WA[(t - 2) & 15]) + WA[(t - 7) & 15] + gamma0_512(WA[(t - 15) & 15]);
        WB[t & 15] += gamma1_512(WB[(t - 2) & 15]) + WB[(t - 7) & 15] + gamma0_512(WB[(t - 15) & 15]);
        WC[t & 15] += gamma1_512(WC[(t - 2) & 15]) + WC[(t - 7) & 15] + gamma0_512(WC[(t - 15) & 15]);
        WD[t & 15] += gamma1_512(WD[(t - 2) & 15]) + WD[(t - 7) & 15] + gamma0_512(WD[(t - 15) & 15]);
        t1A = hA + sigma1_512(eA) + ch64(eA, fA, gA) + K512[t] + WA[t & 15];
        t2A = sigma0_512(aA) + maj64(aA, bA, cA);
        t1B = hB + sigma1_512(eB) + ch64(eB, fB, gB) + K512[t] + WB[t & 15];
        t2B = sigma0_512(aB) + maj64(aB, bB, cB);
        t1C = hC + sigma1_512(eC) + ch64(eC, fC, gC) + K512[t] + WC[t & 15];
        t2C = sigma0_512(aC) + maj64(aC, bC, cC);
        t1D = hD + sigma1_512(eD) + ch64(eD, fD, gD) + K512[t] + WD[t & 15];
        t2D = sigma0_512(aD) + maj64(aD, bD, cD);
        hA=gA; gA=fA; fA=eA; eA=dA+t1A; dA=cA; cA=bA; bA=aA; aA=t1A+t2A;
        hB=gB; gB=fB; fB=eB; eB=dB+t1B; dB=cB; cB=bB; bB=aB; aB=t1B+t2B;
        hC=gC; gC=fC; fC=eC; eC=dC+t1C; dC=cC; cC=bC; bC=aC; aC=t1C+t2C;
        hD=gD; gD=fD; fD=eD; eD=dD+t1D; dD=cD; cD=bD; bD=aD; aD=t1D+t2D;
    }

    sA->h[0]+=aA; sA->h[1]+=bA; sA->h[2]+=cA; sA->h[3]+=dA;
    sA->h[4]+=eA; sA->h[5]+=fA; sA->h[6]+=gA; sA->h[7]+=hA;
    sB->h[0]+=aB; sB->h[1]+=bB; sB->h[2]+=cB; sB->h[3]+=dB;
    sB->h[4]+=eB; sB->h[5]+=fB; sB->h[6]+=gB; sB->h[7]+=hB;
    sC->h[0]+=aC; sC->h[1]+=bC; sC->h[2]+=cC; sC->h[3]+=dC;
    sC->h[4]+=eC; sC->h[5]+=fC; sC->h[6]+=gC; sC->h[7]+=hC;
    sD->h[0]+=aD; sD->h[1]+=bD; sD->h[2]+=cD; sD->h[3]+=dD;
    sD->h[4]+=eD; sD->h[5]+=fD; sD->h[6]+=gD; sD->h[7]+=hD;
}

#endif // !DEV_BUILD

// ---- E9: master HMAC with the FIXED key "Bitcoin seed" (BIP32 root). ----
// Byte-exact equivalent of hmac_sha512("Bitcoin seed", 12, seed, 64, I).
__device__ void master_hmac_fast(const uint8_t* seed64, uint8_t* I64) {
    uint64_t sw[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        uint64_t v = 0;
        #pragma unroll
        for (int b = 0; b < 8; b++) v = (v << 8) | (uint64_t)seed64[j*8+b];
        sw[j] = v;
    }
    SHA512State_t c;
    #pragma unroll
    for (int j = 0; j < 8; j++) c.h[j] = d_min_pre[j];
    sha512_finish64msg_fast(&c, sw);            /* inner: pre + seed block */
    SHA512State_t o;
    #pragma unroll
    for (int j = 0; j < 8; j++) o.h[j] = d_mout_pre[j];
    sha512_finish64msg_fast(&o, c.h);           /* outer: pre + inner block */
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        uint64_t x = o.h[j];
        I64[j*8+0]=(uint8_t)(x>>56); I64[j*8+1]=(uint8_t)(x>>48);
        I64[j*8+2]=(uint8_t)(x>>40); I64[j*8+3]=(uint8_t)(x>>32);
        I64[j*8+4]=(uint8_t)(x>>24); I64[j*8+5]=(uint8_t)(x>>16);
        I64[j*8+6]=(uint8_t)(x>>8);  I64[j*8+7]=(uint8_t)(x);
    }
}

#ifndef DEV_BUILD
// E7: full x4 PBKDF2 - four candidates of the same base. ALL chain state
// (pre-states, U, T) in volatile local (L1); registers hold only the hot
// working sets. Byte-exact equivalent of four keyblock_fast calls.
__device__ void pbkdf2_sha512_keyblock_fast_x4(
    const uint64_t keyA[16], uint32_t lenA,
    const uint64_t keyB[16], uint32_t lenB,
    const uint64_t keyC[16], uint32_t lenC,
    const uint64_t keyD[16], uint32_t lenD,
    uint32_t iterations,
    uint8_t* outA, uint8_t* outB, uint8_t* outC, uint8_t* outD
) {
    const uint64_t* keys[4] = {keyA, keyB, keyC, keyD};
    uint32_t lens[4] = {lenA, lenB, lenC, lenD};

    /* compressed state: everything cold-per-iteration lives here (L1) */
    volatile uint64_t inpre[4][8], outpre[4][8], U[4][8], T[4][8];

    /* setup: sequential per chain (2 compressions each - negligible share).
     * No interleave needed here; registers stay free for the hot loop. */
    for (int c = 0; c < 4; c++) {
        uint32_t i0 = (lens[c] + 7u) / 8u; if (i0 > 16u) i0 = 16u;
        uint64_t ip[16], op[16];
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            ip[i] = keys[c][i] ^ 0x3636363636363636ULL;
            op[i] = keys[c][i] ^ 0x5c5c5c5c5c5c5c5cULL;
        }
        SHA512State_t s;
        sha512_init_state_opt(&s);
        sha512_compress16_rot_opt(&s, ip, i0, d_KP36);
        #pragma unroll
        for (int j = 0; j < 8; j++) inpre[c][j] = s.h[j];
        sha512_init_state_opt(&s);
        sha512_compress16_rot_opt(&s, op, i0, d_KP5c);
        #pragma unroll
        for (int j = 0; j < 8; j++) outpre[c][j] = s.h[j];
    }

    { /* first iteration: sequential salt blocks (1 compression each), then
         the four outers interleave in one x4 call */
        SHA512State_t c[4];
        for (int ch = 0; ch < 4; ch++) {
            #pragma unroll
            for (int j = 0; j < 8; j++) c[ch].h[j] = inpre[ch][j];
            sha512_transform_saltblock_folded_opt(&c[ch]);
        }
        SHA512State_t wA, wB, wC, wD;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            wA.h[j] = outpre[0][j]; wB.h[j] = outpre[1][j];
            wC.h[j] = outpre[2][j]; wD.h[j] = outpre[3][j];
        }
        sha512_finish64msg_x4_fast(&wA, c[0].h, &wB, c[1].h, &wC, c[2].h, &wD, c[3].h);
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            U[0][j] = wA.h[j]; T[0][j] = wA.h[j];
            U[1][j] = wB.h[j]; T[1][j] = wB.h[j];
            U[2][j] = wC.h[j]; T[2][j] = wC.h[j];
            U[3][j] = wD.h[j]; T[3][j] = wD.h[j];
        }
    }

    for (uint32_t iter = 1; iter < iterations; iter++) {
        SHA512State_t cA, cB, cC, cD;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            cA.h[j] = inpre[0][j]; cB.h[j] = inpre[1][j];
            cC.h[j] = inpre[2][j]; cD.h[j] = inpre[3][j];
        }
        sha512_finish64msg_x4_fast(&cA, (const uint64_t*)U[0], &cB, (const uint64_t*)U[1],
                                   &cC, (const uint64_t*)U[2], &cD, (const uint64_t*)U[3]);
        SHA512State_t wA, wB, wC, wD;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            wA.h[j] = outpre[0][j]; wB.h[j] = outpre[1][j];
            wC.h[j] = outpre[2][j]; wD.h[j] = outpre[3][j];
        }
        sha512_finish64msg_x4_fast(&wA, cA.h, &wB, cB.h, &wC, cC.h, &wD, cD.h);
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            U[0][j] = wA.h[j]; T[0][j] ^= wA.h[j];
            U[1][j] = wB.h[j]; T[1][j] ^= wB.h[j];
            U[2][j] = wC.h[j]; T[2][j] ^= wC.h[j];
            U[3][j] = wD.h[j]; T[3][j] ^= wD.h[j];
        }
    }

    uint8_t* outs[4] = {outA, outB, outC, outD};
    for (int c = 0; c < 4; c++) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            uint64_t x = T[c][j];
            outs[c][j*8+0]=(uint8_t)(x>>56); outs[c][j*8+1]=(uint8_t)(x>>48);
            outs[c][j*8+2]=(uint8_t)(x>>40); outs[c][j*8+3]=(uint8_t)(x>>32);
            outs[c][j*8+4]=(uint8_t)(x>>24); outs[c][j*8+5]=(uint8_t)(x>>16);
            outs[c][j*8+6]=(uint8_t)(x>>8);  outs[c][j*8+7]=(uint8_t)(x);
        }
    }
}

#endif // !DEV_BUILD

#endif
