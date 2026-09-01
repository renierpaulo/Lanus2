/*
 * kernels.cu - Extracted GPU kernels and process_base functions
 * Extracted from main.cu for modularity.
 */

#include <stdint.h>
#include <string.h>
#include <cuda_runtime.h>

#include "sha256.cuh"
#include "sha512.cuh"
#include "ripemd160.cuh"
#include "secp256k1.cuh"
#include "base58.cuh"
#include "bip39.cuh"
#include "pbkdf2_opt.cuh"
#include "wlbp.cuh"
#ifndef DEV_BUILD
#include "bip32_fast.cuh"
#endif
#include "checksum_fast.cuh"
#include "byte_elim.cuh"

// ============================================================================
// Configuration (must match main.cu)
// ============================================================================
#define PBKDF2_ITERATIONS 2048
#define MAX_WORDS 40

// NOTE (single-TU build): d_word_indices / d_word_count / d_factorials /
// d_req_words / d_avail_words / d_avail_count are DEFINED in main.cu BEFORE
// this file is #included — same for the lanus CLI parity constants
// (d_required_count / d_wild_count / d_pin / d_pin_count / d_perm_free /
// d_binom / d_exhaustive) and byte_elim's d_coin / d_use_bloom / d_bloom_*.
// No declarations needed here — CUDA forbids re-declaring __constant__ vars
// extern after definition.

// ============================================================================
// BIP32 Child Key Derivation
// ============================================================================
__device__ void derive_child_key(
    const uint8_t* parent_key,
    const uint8_t* parent_chaincode,
    uint32_t index,
    uint8_t* child_key,
    uint8_t* child_chaincode,
    bool hardened,
    bool debug = false
) {
    uint8_t data[37];
    uint8_t I[64];
    
    if (hardened) {
        index |= 0x80000000;
        data[0] = 0x00;
        memcpy(data + 1, parent_key, 32);
    } else {
        uint8_t pubkey[33];
        secp256k1_get_pubkey_compressed(parent_key, pubkey);
        memcpy(data, pubkey, 33);
    }
    
    data[33] = (index >> 24) & 0xFF;
    data[34] = (index >> 16) & 0xFF;
    data[35] = (index >> 8) & 0xFF;
    data[36] = index & 0xFF;
    
    if (debug) {
        printf("DEBUG data: ");
        for(int k=0; k<37; k++) printf("%02x", data[k]);
        printf("\n");
    }
    
    hmac_sha512(parent_chaincode, 32, data, 37, I);
    
    if (debug) {
        printf("DEBUG IL: ");
        for(int k=0; k<32; k++) printf("%02x", I[k]);
        printf("\n");
    }
    
    // Add parent key to derived key (mod n)
    secp256k1_scalar_add(I, parent_key, child_key);
    memcpy(child_chaincode, I + 32, 32);
}


// ============================================================================
// Ultra-fast SHA-256 for checksum (device)
// ============================================================================
__device__ __forceinline__ uint32_t rotr32(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

#ifndef DEV_BUILD
__device__ void sha256_checksum_only(const uint8_t* entropy, int ent_bytes, uint8_t* first_byte) {
    // Initialize hash state
    uint32_t h0 = 0x6a09e667, h1 = 0xbb67ae85, h2 = 0x3c6ef372, h3 = 0xa54ff53a;
    uint32_t h4 = 0x510e527f, h5 = 0x9b05688c, h6 = 0x1f83d9ab, h7 = 0x5be0cd19;
    
    // Prepare message (entropy + padding)
    uint32_t w[64];
    
    // Pack entropy into words (big-endian)
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
            // Padding starts here
            int pos = ent_bytes % 4;
            w[i] = 0x80000000 >> (pos * 8);
        } else {
            w[i] = 0;
        }
    }
    
    // Length in bits at the end
    w[15] = ent_bytes * 8;
    
    // Extend
    #pragma unroll
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = rotr32(w[i-15], 7) ^ rotr32(w[i-15], 18) ^ (w[i-15] >> 3);
        uint32_t s1 = rotr32(w[i-2], 17) ^ rotr32(w[i-2], 19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    
    // Compress
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

// ============================================================================
// Simple LCG random number generator for CUDA
// ============================================================================
#endif // !DEV_BUILD

__device__ uint64_t cuda_rand(uint64_t* seed) {
    *seed = (*seed * 6364136223846793005ULL + 1442695040888963407ULL);
    return *seed;
}

// P2: division-free range reduction (Lemire multiply-shift).
// 64-bit % compiles to ~40+ serial instructions on GPU; this is 2 ops.
__device__ __forceinline__ uint32_t cuda_rand_range(uint64_t* seed, uint32_t range) {
    uint64_t x = cuda_rand(seed);
    return (uint32_t)(((x >> 32) * (uint64_t)range) >> 32);
}

// ============================================================================
// P2: random 12-word phrase, 4 REQUIRED + 8 drawn from the precomputed pool.
// Pool/required live in constant memory (host-built once) - the per-thread
// O(n*4) available[] rebuild is gone. Range reduction is division-free.
// ============================================================================
#ifndef DEV_BUILD
__device__ void random_12word_required_fast(uint64_t seed, uint16_t* out_indices) {
    out_indices[0] = d_req_words[0];
    out_indices[1] = d_req_words[1];
    out_indices[2] = d_req_words[2];
    out_indices[3] = d_req_words[3];

    uint16_t available[MAX_WORDS];
    #pragma unroll
    for (uint32_t i = 0; i < d_avail_count; i++) available[i] = d_avail_words[i];

    // select 8 distinct words (Fisher-Yates partial)
    for (uint32_t i = 0; i < 8; i++) {
        uint32_t j = i + cuda_rand_range(&seed, d_avail_count - i);
        out_indices[4 + i] = available[j];
        uint16_t temp = available[j];
        available[j] = available[i];
        available[i] = temp;
    }

    // shuffle all 12 positions
    for (uint32_t i = 11; i > 0; i--) {
        uint32_t j = cuda_rand_range(&seed, i + 1);
        uint16_t temp = out_indices[i];
        out_indices[i] = out_indices[j];
        out_indices[j] = temp;
    }
}

// ============================================================================
// Validate BIP39 checksum - 12 words
// Returns true if valid
// ============================================================================
__device__ bool verify_checksum_12(const uint16_t* indices) {
    uint8_t entropy[16];
    
    // Pack 128 bits (first 11 words + 7 bits of 12th)
    uint32_t bits = 0;
    int bit_count = 0;
    int byte_idx = 0;
    
    #pragma unroll
    for (int w = 0; w < 12; w++) {
        uint16_t idx = indices[w];
        // Add 11 bits
        for (int b = 10; b >= 0; b--) {
            bits = (bits << 1) | ((idx >> b) & 1);
            bit_count++;
            if (bit_count == 8) {
                if (byte_idx < 16) entropy[byte_idx++] = bits & 0xFF;
                bits = 0;
                bit_count = 0;
            }
        }
    }
    
    // Now entropy has 16 bytes, and we have 4 bits checksum in the last word
    uint8_t expected_cs;
    sha256_checksum_only(entropy, 16, &expected_cs);
    expected_cs = expected_cs >> 4; // First 4 bits
    
    uint8_t actual_cs = indices[11] & 0x0F; // Last 4 bits of 12th word
    
    return expected_cs == actual_cs;
}
#endif // !DEV_BUILD

// ============================================================================
// Validate BIP39 checksum - 24 words
// ============================================================================
// ============================================================================
// E6: IN-THREAD GENERATION - lanus CLI parity (ported from lanus original)
//   -exh  : k -> frase BIJETIVO (unrank combinatorial, cobertura garantida)
//   random: sorteio com reposicao (mix64 + LCG), igual ao original
// Ambos respeitam -req N, -wild N e -pin POS:PALAVRA.
// A posicao 12 (indice 11) e ENUMERADA pelo caller (solo kernel): o gerador
// produz a base de 11 palavras (out12[0..10]).
// ============================================================================
__device__ __forceinline__ uint64_t lg_mix64(uint64_t z) {
    z += 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

__device__ uint64_t lg_cuda_rand(uint64_t* seed) {
    *seed = (*seed * 6364136223846793005ULL + 1442695040888963407ULL);
    return lg_mix64(*seed);
}

// random 12-word phrase honoring req/wild/pin (ported from lanus original)
__device__ void random_12word_with_required(uint64_t seed, uint32_t total_words,
                                            const uint16_t* base_indices,
                                            uint16_t* out_indices) {
    const uint32_t nreq  = d_required_count;
    const uint32_t nwild = d_wild_count;

    for (uint32_t i = 0; i < nreq; i++) out_indices[i] = base_indices[i];

    for (uint32_t i = 0; i < nwild; i++)
        out_indices[nreq + i] = (uint16_t)(lg_cuda_rand(&seed) % 2048u);

    uint16_t pool[MAX_WORDS];
    uint32_t navail = total_words - nreq;
    for (uint32_t i = 0; i < navail; i++) pool[i] = base_indices[nreq + i];

    const uint32_t nchoose = 12u - nreq - nwild;
    for (uint32_t i = 0; i < nchoose; i++) {
        uint32_t j = (uint32_t)(lg_cuda_rand(&seed) % navail);
        out_indices[nreq + nwild + i] = pool[j];
        pool[j] = pool[navail - 1];
        navail--;
    }

    if (d_pin_count == 0) {
        for (int i = 11; i > 0; i--) {
            uint32_t j = (uint32_t)(lg_cuda_rand(&seed) % (uint32_t)(i + 1));
            uint16_t t = out_indices[i];
            out_indices[i] = out_indices[j];
            out_indices[j] = t;
        }
    } else {
        // embaralha so as palavras livres; as fixas vao direto na posicao
        uint16_t doze[12];
        #pragma unroll
        for (int i = 0; i < 12; i++) doze[i] = out_indices[i];

        bool used[12];
        #pragma unroll
        for (int i = 0; i < 12; i++) used[i] = false;
        for (int p = 0; p < 12; p++) {
            uint16_t w = d_pin[p];
            if (w == 0xFFFFu) continue;
            for (int i = 0; i < 12; i++)
                if (!used[i] && doze[i] == w) { used[i] = true; break; }
        }

        uint16_t fr[12];
        int nf = 0;
        for (int i = 0; i < 12; i++) if (!used[i]) fr[nf++] = doze[i];
        for (int i = nf - 1; i > 0; i--) {
            uint32_t j = (uint32_t)(lg_cuda_rand(&seed) % (uint32_t)(i + 1));
            uint16_t t = fr[i]; fr[i] = fr[j]; fr[j] = t;
        }

        int t2 = 0;
        for (int p = 0; p < 12; p++)
            out_indices[p] = (d_pin[p] != 0xFFFFu) ? d_pin[p] : fr[t2++];
    }
}

// --- exhaustive unranking (ported verbatim from lanus original) ---
__device__ void unrank_combo(uint64_t idx, uint32_t nn, uint32_t kk, uint8_t* out) {
    uint64_t x = d_binom[nn][kk] - 1 - idx;
    int a = (int)nn, b = (int)kk;
    for (uint32_t i = 0; i < kk; i++) {
        a--;
        while (a > 0 && d_binom[a][b] > x) a--;
        x -= d_binom[a][b];
        out[i] = (uint8_t)((int)nn - 1 - a);
        b--;
    }
}

__device__ void unrank_perm11(uint64_t idx, const uint16_t* items, uint16_t* out) {
    uint16_t pool[12];
    #pragma unroll
    for (int i = 0; i < 11; i++) pool[i] = items[i];
    int navail = 11;
    for (int i = 11; i > 0; i--) {
        uint64_t f = d_factorials[i - 1];
        uint32_t j = (uint32_t)(idx / f);
        idx -= (uint64_t)j * f;
        out[11 - i] = pool[j];
        for (int m = j; m < navail - 1; m++) pool[m] = pool[m + 1];
        navail--;
    }
}

// Coloca as 11 palavras da base nas posicoes 0..10, respeitando -pin.
// Pin na posicao 12 e tratado pelo caller (restricao da enumeracao).
// Bijetivo sobre (11 - pins_na_base)!.
__device__ void place_with_pins_11(uint64_t idx, const uint16_t* items, uint16_t* out) {
    int pin_base = 0;
    for (int p = 0; p < 11; p++) if (d_pin[p] != 0xFFFFu) pin_base++;
    if (pin_base == 0) { unrank_perm11(idx, items, out); return; }

    bool used[12];
    #pragma unroll
    for (int i = 0; i < 12; i++) used[i] = false;
    for (int p = 0; p < 11; p++) {
        uint16_t w = d_pin[p];
        if (w == 0xFFFFu) continue;
        for (int i = 0; i < 11; i++)
            if (!used[i] && items[i] == w) { used[i] = true; break; }
    }

    uint16_t pool[12];
    int nfree = 0;
    for (int i = 0; i < 11; i++) if (!used[i]) pool[nfree++] = items[i];

    uint16_t perm[12];
    int navail = nfree;
    for (int i = nfree; i > 0; i--) {
        uint64_t f = d_factorials[i - 1];
        uint32_t j = (uint32_t)(idx / f);
        idx -= (uint64_t)j * f;
        perm[nfree - i] = pool[j];
        for (int m = j; m < navail - 1; m++) pool[m] = pool[m + 1];
        navail--;
    }

    int t = 0;
    for (int p = 0; p < 11; p++)
        out[p] = (d_pin[p] != 0xFFFFu) ? d_pin[p] : perm[t++];
}

// k-esima BASE (11 palavras) do espaco de busca. Posicao 12 enumerada a parte.
__device__ void generate_phrase12(uint64_t k, uint16_t* out12) {
    if (d_exhaustive) {
        const uint32_t nreq  = d_required_count;
        const uint32_t nwild = d_wild_count;
        const uint32_t Kbase = 11u - nreq - nwild;
        const uint32_t Psz   = d_word_count - nreq;

        uint64_t rest      = k / d_perm_free;     // (combo, curingas)
        uint64_t perm_idx  = k - rest * d_perm_free;

        uint16_t wild[3];
        for (uint32_t i = 0; i < nwild; i++) {
            wild[i] = (uint16_t)(rest % 2048ULL);
            rest /= 2048ULL;
        }
        uint64_t combo_idx = rest;

        uint8_t sel[12];
        unrank_combo(combo_idx, Psz, Kbase, sel);

        uint16_t doze[12];
        for (uint32_t i = 0; i < nreq; i++)  doze[i] = d_word_indices[i];
        for (uint32_t i = 0; i < nwild; i++) doze[nreq + i] = wild[i];
        for (uint32_t i = 0; i < Kbase; i++) doze[nreq + nwild + i] = d_word_indices[nreq + sel[i]];

        place_with_pins_11(perm_idx, doze, out12);   // out12[0..10]
    } else {
        uint64_t seed = lg_mix64(k * 0x9E3779B97F4A7C15ULL + 0x123456789ABCDEFULL);
        random_12word_with_required(seed, d_word_count, d_word_indices, out12);
    }
    out12[11] = 0;   // 12th word: enumerated by the caller
}

// BIP32 child derivation (hardened + non-hardened)
// Full derivation + target compare for one candidate seed
__device__ void derive_and_match(
    const uint8_t* seed,
    const uint16_t* indices12,
    uint32_t* found_count,
    uint8_t* found_privkeys,
    uint16_t* found_indices
) {
    /* E20: delegate to the zero-byte-round-trip version from byte_elim.cuh
     * The seed bytes are big-endian — repack into BE u64 words for the
     * u64-native pipeline (the old byte round-trip is eliminated). */
    uint64_t seed_w[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        seed_w[j] = ((uint64_t)seed[j*8+0] << 56) | ((uint64_t)seed[j*8+1] << 48) |
                    ((uint64_t)seed[j*8+2] << 40) | ((uint64_t)seed[j*8+3] << 32) |
                    ((uint64_t)seed[j*8+4] << 24) | ((uint64_t)seed[j*8+5] << 16) |
                    ((uint64_t)seed[j*8+6] << 8)  | ((uint64_t)seed[j*8+7]);
    }
    derive_and_match_u64(seed_w, indices12,
                         found_count, found_privkeys, found_indices);
}

#ifndef DEV_BUILD
// E11: x2 process function - pairs, register-resident hot loop
__device__ void process_base_vocabulary_x2(
    const uint16_t* base11,
    uint32_t* found_count,
    uint8_t* found_privkeys,
    uint16_t* found_indices
) {
    uint16_t indices[12];
    #pragma unroll
    for (int i = 0; i < 11; i++) indices[i] = base11[i];

    // E2: base prefix assembled ONCE per base
    uint64_t prefix_w[16], cur;
    uint32_t bits = 0, prefix_len = 0;
    int wi = 0;
    wlbp_base_prefix_build(d_repr_table, base11, prefix_w, &cur, &bits, &wi, &prefix_len);

    // collect valid candidates first (mathematical invariant: exactly 128)
    uint16_t valid_w12[128];
    int n_valid = 0;
    for (uint32_t w12 = 0; w12 < 2048; w12++) {
        indices[11] = (uint16_t)w12;
        if (verify_checksum_12(indices)) {
            if (n_valid < 128) valid_w12[n_valid] = (uint16_t)w12;
            n_valid++;
        }
    }

    // process in PAIRS with x2-interleaved PBKDF2 (zero-traffic hot loop)
    int v = 0;
    for (; v + 1 < n_valid; v += 2) {
        uint16_t wA = valid_w12[v], wB = valid_w12[v + 1];
        indices[11] = wA;
        uint64_t wbA = WLBP_LOAD(&d_repr_table[wA]);
        uint32_t lenA = (uint32_t)wlbp_wb_len(wbA);
        uint64_t keyA[16];
        wlbp_keyblock_tail_swap(prefix_w, cur, bits, wi, wbA, lenA, keyA);

        indices[11] = wB;
        uint64_t wbB = WLBP_LOAD(&d_repr_table[wB]);
        uint32_t lenB = (uint32_t)wlbp_wb_len(wbB);
        uint64_t keyB[16];
        wlbp_keyblock_tail_swap(prefix_w, cur, bits, wi, wbB, lenB, keyB);

        uint8_t seedA[64], seedB[64];
        pbkdf2_sha512_keyblock_fast_x2(keyA, prefix_len + lenA, keyB, prefix_len + lenB,
                                       PBKDF2_ITERATIONS, seedA, seedB);

        indices[11] = wA;
        derive_and_match(seedA, indices, found_count, found_privkeys, found_indices);
        indices[11] = wB;
        derive_and_match(seedB, indices, found_count, found_privkeys, found_indices);
    }
    if (v < n_valid) {
        uint16_t wA = valid_w12[v];
        indices[11] = wA;
        uint64_t wbA = WLBP_LOAD(&d_repr_table[wA]);
        uint32_t lenA = (uint32_t)wlbp_wb_len(wbA);
        uint64_t keyA[16];
        wlbp_keyblock_tail_swap(prefix_w, cur, bits, wi, wbA, lenA, keyA);
        uint8_t seedA[64];
        pbkdf2_sha512_keyblock_fast(keyA, prefix_len + lenA, PBKDF2_ITERATIONS, seedA);
        derive_and_match(seedA, indices, found_count, found_privkeys, found_indices);
    }
}

// E12: x3 process function - triples, one more chain in flight
__device__ void process_base_vocabulary_x3(
    const uint16_t* base11,
    uint32_t* found_count,
    uint8_t* found_privkeys,
    uint16_t* found_indices
) {
    uint16_t indices[12];
    #pragma unroll
    for (int i = 0; i < 11; i++) indices[i] = base11[i];

    uint64_t prefix_w[16], cur;
    uint32_t bits = 0, prefix_len = 0;
    int wi = 0;
    wlbp_base_prefix_build(d_repr_table, base11, prefix_w, &cur, &bits, &wi, &prefix_len);

    uint16_t valid_w12[128];
    int n_valid = 0;
    for (uint32_t w12 = 0; w12 < 2048; w12++) {
        indices[11] = (uint16_t)w12;
        if (verify_checksum_12(indices)) {
            if (n_valid < 128) valid_w12[n_valid] = (uint16_t)w12;
            n_valid++;
        }
    }

    int v = 0;
    for (; v + 2 < n_valid; v += 3) {
        uint16_t w[3];
        uint64_t key[3][16];
        uint32_t len[3];
        #pragma unroll
        for (int c = 0; c < 3; c++) {
            w[c] = valid_w12[v + c];
            indices[11] = w[c];
            uint64_t wb = WLBP_LOAD(&d_repr_table[w[c]]);
            len[c] = (uint32_t)wlbp_wb_len(wb);
            wlbp_keyblock_tail_swap(prefix_w, cur, bits, wi, wb, len[c], key[c]);
        }
        uint8_t seeds[3][64];
        pbkdf2_sha512_keyblock_fast_x3(key[0], prefix_len + len[0],
                                       key[1], prefix_len + len[1],
                                       key[2], prefix_len + len[2],
                                       PBKDF2_ITERATIONS, seeds[0], seeds[1], seeds[2]);
        #pragma unroll
        for (int c = 0; c < 3; c++) {
            indices[11] = w[c];
            derive_and_match(seeds[c], indices, found_count, found_privkeys, found_indices);
        }
    }
    if (v + 1 < n_valid) {
        uint16_t wA = valid_w12[v], wB = valid_w12[v + 1];
        indices[11] = wA;
        uint64_t wbA = WLBP_LOAD(&d_repr_table[wA]);
        uint32_t lenA = (uint32_t)wlbp_wb_len(wbA);
        uint64_t keyA[16];
        wlbp_keyblock_tail_swap(prefix_w, cur, bits, wi, wbA, lenA, keyA);
        indices[11] = wB;
        uint64_t wbB = WLBP_LOAD(&d_repr_table[wB]);
        uint32_t lenB = (uint32_t)wlbp_wb_len(wbB);
        uint64_t keyB[16];
        wlbp_keyblock_tail_swap(prefix_w, cur, bits, wi, wbB, lenB, keyB);
        uint8_t seedA[64], seedB[64];
        pbkdf2_sha512_keyblock_fast_x2(keyA, prefix_len + lenA, keyB, prefix_len + lenB,
                                       PBKDF2_ITERATIONS, seedA, seedB);
        indices[11] = wA;
        derive_and_match(seedA, indices, found_count, found_privkeys, found_indices);
        indices[11] = wB;
        derive_and_match(seedB, indices, found_count, found_privkeys, found_indices);
        v += 2;
    }
    if (v < n_valid) {
        uint16_t wA = valid_w12[v];
        indices[11] = wA;
        uint64_t wbA = WLBP_LOAD(&d_repr_table[wA]);
        uint32_t lenA = (uint32_t)wlbp_wb_len(wbA);
        uint64_t keyA[16];
        wlbp_keyblock_tail_swap(prefix_w, cur, bits, wi, wbA, lenA, keyA);
        uint8_t seedA[64];
        pbkdf2_sha512_keyblock_fast(keyA, prefix_len + lenA, PBKDF2_ITERATIONS, seedA);
        derive_and_match(seedA, indices, found_count, found_privkeys, found_indices);
    }
}

#endif // !DEV_BUILD

#ifndef DEV_BUILD
// ---------------------------------------------------------------------------
// E8 RESTORED (user priority): THE WIDE ENGINE - all 128 PBKDF2 chains
// RESIDENT in this thread simultaneously. Every chain's full state
// (in_pre, out_pre, U, T = 32 uint64) lives in thread-private local memory
// (32KB). ONE pass: iteration 1..2048, ALL 128 chains advance together
// (32 x4 sweeps per iteration). Registers hold only 4 hot working sets.
// The hardest step - porting all 128 PBKDF2s inside one thread without
// breaking anything - is this function, proven byte-exact on host
// (test_e8_wide.cpp: 640 seeds vs classic, zero divergence).
// ---------------------------------------------------------------------------
__device__ void process_base_vocabulary_wide(
    const uint16_t* base11,
    uint32_t* found_count,
    uint8_t* found_privkeys,
    uint16_t* found_indices
) {
    uint16_t indices[12];
    #pragma unroll
    for (int i = 0; i < 11; i++) indices[i] = base11[i];

    // E2: base prefix assembled ONCE per base
    uint64_t prefix_w[16], cur;
    uint32_t bits = 0, prefix_len = 0;
    int wi = 0;
    wlbp_base_prefix_build(d_repr_table, base11, prefix_w, &cur, &bits, &wi, &prefix_len);

    // collect valid candidates (mathematical invariant: exactly 128)
    uint16_t valid_w12[128];
    int n_valid = 0;
    for (uint32_t w12 = 0; w12 < 2048; w12++) {
        indices[11] = (uint16_t)w12;
        if (verify_checksum_12(indices)) {
            if (n_valid < 128) valid_w12[n_valid] = (uint16_t)w12;
            n_valid++;
        }
    }

    // ALL 128 chains resident: full PBKDF2 state in thread-private local
    volatile uint64_t inpre[128][8], outpre[128][8], U[128][8], T[128][8];

    // upfront: per-chain tail swap + key block + ipad/opad pre-states (1x each)
    for (int c = 0; c < n_valid; c++) {
        indices[11] = valid_w12[c];
        uint64_t wb = WLBP_LOAD(&d_repr_table[valid_w12[c]]);
        uint32_t len = (uint32_t)wlbp_wb_len(wb);
        uint64_t key[16];
        wlbp_keyblock_tail_swap(prefix_w, cur, bits, wi, wb, len, key);

        uint32_t i0 = (prefix_len + len + 7u) / 8u;
        if (i0 > 16u) i0 = 16u;
        uint64_t ip[16], op[16];
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            ip[i] = key[i] ^ 0x3636363636363636ULL;
            op[i] = key[i] ^ 0x5c5c5c5c5c5c5c5cULL;
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

    // iteration 1: salt block per chain (constant block, 1x each)
    for (int c = 0; c < n_valid; c++) {
        SHA512State_t s;
        #pragma unroll
        for (int j = 0; j < 8; j++) s.h[j] = inpre[c][j];
        sha512_transform_saltblock_folded_opt(&s);
        SHA512State_t w;
        #pragma unroll
        for (int j = 0; j < 8; j++) w.h[j] = outpre[c][j];
        sha512_finish64msg_fast(&w, s.h);
        #pragma unroll
        for (int j = 0; j < 8; j++) { U[c][j] = w.h[j]; T[c][j] = w.h[j]; }
    }

    // THE ONE PASS: iterations 2..2048 - ALL chains advance TOGETHER
    for (uint32_t iter = 1; iter < PBKDF2_ITERATIONS; iter++) {
        int g = 0;
        for (; g + 3 < n_valid; g += 4) {
            SHA512State_t cA, cB, cC, cD;
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                cA.h[j] = inpre[g][j];     cB.h[j] = inpre[g + 1][j];
                cC.h[j] = inpre[g + 2][j]; cD.h[j] = inpre[g + 3][j];
            }
            sha512_finish64msg_x4_fast(&cA, (const uint64_t*)U[g], &cB, (const uint64_t*)U[g + 1],
                                       &cC, (const uint64_t*)U[g + 2], &cD, (const uint64_t*)U[g + 3]);
            SHA512State_t wA, wB, wC, wD;
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                wA.h[j] = outpre[g][j];     wB.h[j] = outpre[g + 1][j];
                wC.h[j] = outpre[g + 2][j]; wD.h[j] = outpre[g + 3][j];
            }
            sha512_finish64msg_x4_fast(&wA, cA.h, &wB, cB.h, &wC, cC.h, &wD, cD.h);
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                U[g][j] = wA.h[j];     T[g][j] ^= wA.h[j];
                U[g + 1][j] = wB.h[j]; T[g + 1][j] ^= wB.h[j];
                U[g + 2][j] = wC.h[j]; T[g + 2][j] ^= wC.h[j];
                U[g + 3][j] = wD.h[j]; T[g + 3][j] ^= wD.h[j];
            }
        }
        // remainder (cannot happen with the 128 invariant - kept safe)
        for (int c = g; c < n_valid; c++) {
            SHA512State_t s;
            #pragma unroll
            for (int j = 0; j < 8; j++) s.h[j] = inpre[c][j];
            sha512_finish64msg_fast(&s, (const uint64_t*)U[c]);
            SHA512State_t w;
            #pragma unroll
            for (int j = 0; j < 8; j++) w.h[j] = outpre[c][j];
            sha512_finish64msg_fast(&w, s.h);
            #pragma unroll
            for (int j = 0; j < 8; j++) { U[c][j] = w.h[j]; T[c][j] ^= w.h[j]; }
        }
    }

    // harvest: seeds from T accumulators -> derive + match
    for (int c = 0; c < n_valid; c++) {
        uint8_t seed[64];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            uint64_t x = T[c][j];
            seed[j*8+0] = (uint8_t)(x >> 56); seed[j*8+1] = (uint8_t)(x >> 48);
            seed[j*8+2] = (uint8_t)(x >> 40); seed[j*8+3] = (uint8_t)(x >> 32);
            seed[j*8+4] = (uint8_t)(x >> 24); seed[j*8+5] = (uint8_t)(x >> 16);
            seed[j*8+6] = (uint8_t)(x >> 8);  seed[j*8+7] = (uint8_t)(x);
        }
        indices[11] = valid_w12[c];
        derive_and_match(seed, indices, found_count, found_privkeys, found_indices);
    }
}

#endif // !DEV_BUILD

// ---------------------------------------------------------------------------
// E14: SOLO process function - one candidate at a time, minimal registers.
// Uses pbkdf2_sha512_keyblock_fast (single-chain, all-in-registers).
// The GPU's warp scheduler hides SHA-512 latency across many lightweight
// threads instead of interleaving within one heavy thread.
// Target: __launch_bounds__(128, 3) = 3 blocks x 128 = 384 threads/SM.
// Gives ~170 regs/thread (vs 128 at minBlocks=4), avoiding register spills.
// ---------------------------------------------------------------------------
__device__ void process_base_vocabulary_solo(
    const uint16_t* base11,
    uint32_t* found_count,
    uint8_t* found_privkeys,
    uint16_t* found_indices
) {
    uint16_t indices[12];
    #pragma unroll
    for (int i = 0; i < 11; i++) indices[i] = base11[i];

    // E20: checksum fast init (rounds 0-2 computed once per base)
    ChecksumFastState_t cst;
    checksum_fast_init(base11, &cst);

    // prefix built ONCE per base (persistent across all 128 candidates)
    uint64_t prefix_w[16], cur;
    uint32_t bits = 0, prefix_len = 0;
    int wi = 0;
    wlbp_base_prefix_build(d_repr_table, base11, prefix_w, &cur, &bits, &wi, &prefix_len);

    // E20r: enumerate the 128 VALID candidates directly: for each entropy
    // class v, the single valid 12th word is (v<<4) | cs4(v) - computed, not
    // probed (the old v<<4 test only ever matched checksum==0, scanning just
    // 1/16 of the valid space). -pin 12:word restricts the entropy class.
    uint32_t v_start = 0, v_end = 128;
    if (d_pin[11] != 0xFFFFu) { v_start = d_pin[11] >> 4; v_end = v_start + 1; }
    for (uint32_t v = v_start; v < v_end; v++) {
        uint16_t w12 = (uint16_t)((v << 4) | checksum_fast_cs4(&cst, (uint16_t)(v << 4)));
        indices[11] = w12;

        // lanus parity: sem curingas, a palavra 12 vem do arquivo (-words);
        // com -wild N as curingas abrem o espaco para as 2048 palavras.
        if (d_wild_count == 0) {
            bool in_words = false;
            for (uint32_t iw = 0; iw < d_word_count; iw++) {
                if (d_word_indices[iw] == w12) { in_words = true; break; }
            }
            if (!in_words) continue;
        }

        uint64_t wb = WLBP_LOAD(&d_repr_table[w12]);
        uint32_t len = (uint32_t)wlbp_wb_len(wb);
        uint64_t key[16];
        wlbp_keyblock_tail_swap(prefix_w, cur, bits, wi, wb, len, key);

        // prefix_w/cur/bits/wi/key are DEAD here — compiler reuses those
        // registers for the compression a..h and W[16] working set.

        uint8_t seed[64];
        pbkdf2_sha512_keyblock_fast(key, prefix_len + len, PBKDF2_ITERATIONS, seed);

        // key and setup temps DEAD — registers available for derive

        derive_and_match(seed, indices, found_count, found_privkeys, found_indices);
    }
}

// ============================================================================
// Kernels
// ============================================================================

// E13: SPLIT kernels - each depth compiles with its own register budget,
// so the x2 path is not inflated by the x3 path (register quantization:
// 128-thread blocks x ~158 regs = 3 blocks/SM = 384 threads/SM).
// E14: SOLO kernel - max occupancy (512 threads/SM via 128 regs cap)
__global__ void __launch_bounds__(128, 3) kernel_scanner_solo(
    uint64_t start_k,
    uint32_t bases_per_thread,
    uint32_t* found_count,
    uint8_t* found_privkeys,
    uint16_t* found_indices
) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;

    for (uint32_t b = 0; b < bases_per_thread; b++) {
        uint64_t k = start_k + b * stride + tid;
        uint16_t phrase[12];
        generate_phrase12(k, phrase);
        process_base_vocabulary_solo(phrase, found_count, found_privkeys, found_indices);
    }
}

#ifndef DEV_BUILD
__global__ void __launch_bounds__(128) kernel_scanner_x2(
    uint64_t start_k,
    uint32_t bases_per_thread,
    uint32_t* found_count,
    uint8_t* found_privkeys,
    uint16_t* found_indices
) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;

    for (uint32_t b = 0; b < bases_per_thread; b++) {
        uint64_t k = start_k + b * stride + tid;
        uint16_t phrase[12];
        generate_phrase12(k, phrase);
        process_base_vocabulary_x2(phrase, found_count, found_privkeys, found_indices);
    }
}

__global__ void __launch_bounds__(128) kernel_scanner_x3(
    uint64_t start_k,
    uint32_t bases_per_thread,
    uint32_t* found_count,
    uint8_t* found_privkeys,
    uint16_t* found_indices
) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;

    for (uint32_t b = 0; b < bases_per_thread; b++) {
        uint64_t k = start_k + b * stride + tid;
        uint16_t phrase[12];
        generate_phrase12(k, phrase);
        process_base_vocabulary_x3(phrase, found_count, found_privkeys, found_indices);
    }
}

// ============================================================================
// SELFTEST KERNEL: feeds the known target base directly through the full
// monolithic pipeline (enumeration -> wide PBKDF2 -> BIP32 -> EC -> hash160)
// on real silicon. Expected: FOUND with the documented private key.
// ============================================================================
__global__ void __launch_bounds__(256) kernel_scanner_wide(
    uint64_t start_k,
    uint32_t bases_per_thread,
    uint32_t* found_count,
    uint8_t* found_privkeys,
    uint16_t* found_indices
) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;

    for (uint32_t b = 0; b < bases_per_thread; b++) {
        uint64_t k = start_k + b * stride + tid;
        uint16_t phrase[12];
        generate_phrase12(k, phrase);
        process_base_vocabulary_wide(phrase, found_count, found_privkeys, found_indices);
    }
}

#endif // !DEV_BUILD

__global__ void kernel_selftest(
    uint32_t* found_count,
    uint8_t* found_privkeys,
    uint16_t* found_indices
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        uint16_t base[11] = {759, 1078, 213, 623, 521, 319, 416, 302, 566, 1104, 191};
#ifdef DEV_BUILD
        process_base_vocabulary_solo(base, found_count, found_privkeys, found_indices);
#else
        process_base_vocabulary_wide(base, found_count, found_privkeys, found_indices);
#endif

        /* stage-by-stage diagnostics for candidate 1666 */
        printf("\n[DIAG] device repr[1666] = %016llx\n",
               (unsigned long long)WLBP_LOAD(&d_repr_table[1666]));
        printf("[DIAG] device Gwin_x[0].d[0] = %08x (expect 16f81798)\n",
               (unsigned)d_Gwin_x[0].d[0]);

        uint64_t prefix_w[16], cur;
        uint32_t bits = 0, prefix_len = 0;
        int wi = 0;
        wlbp_base_prefix_build(d_repr_table, base, prefix_w, &cur, &bits, &wi, &prefix_len);
        printf("[DIAG] prefix_len = %u\n", prefix_len);

        uint64_t wb = WLBP_LOAD(&d_repr_table[1666]);
        uint64_t key_w[16];
        wlbp_keyblock_tail_swap(prefix_w, cur, bits, wi, wb, (uint32_t)wlbp_wb_len(wb), key_w);
        printf("[DIAG] key_w[0] = %016llx\n", (unsigned long long)key_w[0]);

        uint8_t seed[64];
        pbkdf2_sha512_keyblock_fast(key_w, prefix_len + (uint32_t)wlbp_wb_len(wb), PBKDF2_ITERATIONS, seed);
        printf("[DIAG] seed[0..7] = ");
        for (int i = 0; i < 8; i++) printf("%02x", seed[i]);
        printf(" (expect 4cd832a5c862ef51)\n");

        uint8_t I[64];
        master_hmac_fast(seed, I);
        printf("[DIAG] master[0..7] = ");
        for (int i = 0; i < 8; i++) printf("%02x", I[i]);
        printf(" (expect 7441876923f2e91c)\n");

        /* BIP32 chain on device */
        uint8_t master_key[32], master_chaincode[32];
        memcpy(master_key, I, 32);
        memcpy(master_chaincode, I + 32, 32);
        uint8_t key[32], chaincode[32], temp_key[32], temp_chaincode[32];
        derive_child_key(master_key, master_chaincode, 44, key, chaincode, true, false);
        derive_child_key(key, chaincode, 0, temp_key, temp_chaincode, true, false);
        memcpy(key, temp_key, 32); memcpy(chaincode, temp_chaincode, 32);
        derive_child_key(key, chaincode, 0, temp_key, temp_chaincode, true, false);
        memcpy(key, temp_key, 32); memcpy(chaincode, temp_chaincode, 32);
        derive_child_key(key, chaincode, 0, temp_key, temp_chaincode, false, false);
        memcpy(key, temp_key, 32); memcpy(chaincode, temp_chaincode, 32);
        uint8_t private_key[32];
        derive_child_key(key, chaincode, 0, private_key, temp_chaincode, false, false);
        printf("[DIAG] privkey[0..7] = ");
        for (int i = 0; i < 8; i++) printf("%02x", private_key[i]);
        printf(" (expect 20bbcda671e9f66c)\n");

        uint8_t pubkey[33];
        secp256k1_get_pubkey_compressed(private_key, pubkey);
        printf("[DIAG] pubkey[0..7] = ");
        for (int i = 0; i < 8; i++) printf("%02x", pubkey[i]);
        printf(" (expect 026397423d347ccb)\n");

        uint8_t sha_hash[32], h160[20];
        sha256(pubkey, 33, sha_hash);
        ripemd160(sha_hash, 32, h160);
        printf("[DIAG] hash160[0..7] = ");
        for (int i = 0; i < 8; i++) printf("%02x", h160[i]);
        printf(" (expect 232fb8a4bb0b8be8)\n");

        printf("[DIAG] d_num_targets = %u\n", d_num_targets);
        if (d_num_targets > 0) {
            printf("[DIAG] target[0] = ");
            for (int i = 0; i < 8; i++) printf("%02x", d_target_hashes_ptr[i]);
            printf("...\n");
        }
    }
}
