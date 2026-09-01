/*
 * test_etapa_e3.cpp - E3 verification: the COMPLETE thread body of the new
 * vocabulary-processor kernel, executed on the host.
 *
 * Pipeline inside one thread:
 *   base(11 words) -> enumerate 2048 candidates -> checksum filter (~128)
 *   -> for each valid: tail swap -> PBKDF2 fast -> BIP32 m/44'/0'/0'/0/0
 *   -> pubkey (windowed) -> hash160 -> target compare
 *
 * Anchor: for the base of the known target phrase, candidate 1666 must
 * produce the documented hash160 232fb8a4bb0b8be8daeb78d9022d126006309c5c.
 *
 * Build: cl /O2 /EHsc test_etapa_e3.cpp
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>

#define __device__
#define __host__
#define __constant__ static
#define __forceinline__ static
#define __align__(x)

#include "src/sha256.cuh"
#include "src/sha512.cuh"
#include "src/ripemd160.cuh"
#include "src/secp256k1.cuh"
#include "src/pbkdf2_opt.cuh"
#include "src/wlbp.cuh"

/* ---- checksum path (replicated from main.cu, proven in E1) ---- */
__forceinline__ uint32_t rotr32(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

static void sha256_checksum_only(const uint8_t* entropy, int ent_bytes, uint8_t* first_byte) {
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
            w[i] = 0x80000000 >> ((ent_bytes % 4) * 8);
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

/* ---- BIP32 child derivation (replicated from main.cu) ---- */
static void derive_child_key(const uint8_t* parent_key, const uint8_t* parent_chaincode,
                             uint32_t index, uint8_t* child_key, uint8_t* child_chaincode,
                             bool hardened) {
    uint8_t data[37];
    uint8_t I[64];
    if (hardened) {
        index |= 0x80000000u;
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
    hmac_sha512(parent_chaincode, 32, data, 37, I);
    secp256k1_scalar_add(I, parent_key, child_key);
    memcpy(child_chaincode, I + 32, 32);
}

int main(void) {
    int fails = 0;

    /* thread setup: vocabulary -> representative table */
    unsigned char vocab[WLBP_VOCAB_BYTES];
    wlbp_thread_vocab_init(vocab, WLBP_BLOB);
    uint64_t table[2048];
    wlbp_build_repr_table(vocab, table);

    /* host-test equivalent of secp256k1_gwin_upload(): build + install table */
    static uint256_t gxs[SECP_GWIN_ENTRIES], gys[SECP_GWIN_ENTRIES];
    secp256k1_gwin_build(gxs, gys);
    memcpy(d_Gwin_x, gxs, sizeof(gxs));
    memcpy(d_Gwin_y, gys, sizeof(gys));

    /* the thread's base: first 11 words of the known target phrase */
    uint16_t base[11] = {759, 1078, 213, 623, 521, 319, 416, 302, 566, 1104, 191};
    uint16_t indices[12];
    memcpy(indices, base, sizeof(base));

    /* E2: prefix built once per thread */
    uint64_t prefix_w[16], cur;
    uint32_t bits = 0, prefix_len = 0;
    int wi = 0;
    wlbp_base_prefix_build(table, base, prefix_w, &cur, &bits, &wi, &prefix_len);

    /* the documented expected hash160 for the full target phrase */
    const uint8_t expect_hash160[20] = {
        0x23, 0x2f, 0xb8, 0xa4, 0xbb, 0x0b, 0x8b, 0xe8, 0xda, 0xeb,
        0x78, 0xd9, 0x02, 0x2d, 0x12, 0x60, 0x06, 0x30, 0x9c, 0x5c
    };

    /* ---- the E3 thread loop ---- */
    int n_valid = 0, found = 0;
    for (uint32_t w12 = 0; w12 < 2048; w12++) {
        indices[11] = (uint16_t)w12;
        if (!verify_checksum_12(indices)) continue;
        n_valid++;

        /* E2 tail swap */
        uint64_t wb12 = WLBP_LOAD(&table[w12]);
        uint32_t len12 = (uint32_t)wlbp_wb_len(wb12);
        uint64_t key_w[16];
        wlbp_keyblock_tail_swap(prefix_w, cur, bits, wi, wb12, len12, key_w);

        /* P1 PBKDF2 fast */
        uint8_t seed[64];
        pbkdf2_sha512_keyblock_fast(key_w, prefix_len + len12, 2048, seed);

        /* master key */
        uint8_t master_key[32], master_chaincode[32];
        {
            uint8_t I[64];
            hmac_sha512((const uint8_t*)"Bitcoin seed", 12, seed, 64, I);
            memcpy(master_key, I, 32);
            memcpy(master_chaincode, I + 32, 32);
        }

        /* BIP32 m/44'/0'/0'/0/0 */
        uint8_t key[32], chaincode[32], tk[32], tc[32];
        derive_child_key(master_key, master_chaincode, 44, key, chaincode, true);
        derive_child_key(key, chaincode, 0, tk, tc, true);
        memcpy(key, tk, 32); memcpy(chaincode, tc, 32);
        derive_child_key(key, chaincode, 0, tk, tc, true);
        memcpy(key, tk, 32); memcpy(chaincode, tc, 32);
        derive_child_key(key, chaincode, 0, tk, tc, false);
        memcpy(key, tk, 32); memcpy(chaincode, tc, 32);
        uint8_t private_key[32];
        derive_child_key(key, chaincode, 0, private_key, tc, false);

        /* pubkey + hash160 */
        uint8_t pubkey[33];
        secp256k1_get_pubkey_compressed(private_key, pubkey);
        uint8_t sha_hash[32], hash160[20];
        sha256(pubkey, 33, sha_hash);
        ripemd160(sha_hash, 32, hash160);

        if (w12 == 1666) {
            printf("[debug w12=1666]\nseed: ");
            for (int i = 0; i < 64; i++) printf("%02x", seed[i]);
            printf("\nmaster_key: ");
            for (int i = 0; i < 32; i++) printf("%02x", master_key[i]);
            printf("\nprivate_key: ");
            for (int i = 0; i < 32; i++) printf("%02x", private_key[i]);
            printf("\npubkey: ");
            for (int i = 0; i < 33; i++) printf("%02x", pubkey[i]);
            printf("\nhash160: ");
            for (int i = 0; i < 20; i++) printf("%02x", hash160[i]);
            printf("\nexpected: 232fb8a4bb0b8be8daeb78d9022d126006309c5c\n");
        }

        if (memcmp(hash160, expect_hash160, 20) == 0) {
            found = 1;
            printf("FOUND target at candidate w12=%u (as expected: 1666)\n", w12);
        }
    }

    printf("candidates enumerated: 2048, valid (checksum): %d (invariant: 128)\n", n_valid);
    if (n_valid != 128) { printf("FAIL: valid count != 128\n"); fails++; }
    if (!found) { printf("FAIL: target hash160 not found by the thread loop\n"); fails++; }

    printf("RESULT: %s\n", fails == 0 ? "PASS - E3 thread pipeline complete and byte-exact" : "FAIL");
    return fails == 0 ? 0 : 1;
}
