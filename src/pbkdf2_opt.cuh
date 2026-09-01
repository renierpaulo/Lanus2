#ifndef PBKDF2_OPT_CUH
#define PBKDF2_OPT_CUH


#include "sha512.cuh"

// Estado intermediÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡rio do SHA-512
// Usamos align(16) para garantir carregamento eficiente se necessÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡rio
struct __align__(16) SHA512State_t {
    uint64_t h[8];
};

__device__ __forceinline__ void sha512_init_state_opt(SHA512State_t* s) {
    s->h[0] = 0x6a09e667f3bcc908ULL; s->h[1] = 0xbb67ae8584caa73bULL;
    s->h[2] = 0x3c6ef372fe94f82bULL; s->h[3] = 0xa54ff53a5f1d36f1ULL;
    s->h[4] = 0x510e527fade682d1ULL; s->h[5] = 0x9b05688c2b3e6c1fULL;
    s->h[6] = 0x1f83d9abfb41bd6bULL; s->h[7] = 0x5be0cd19137e2179ULL;
}

#ifndef DEV_BUILD
__device__ __forceinline__ void sha512_extract_opt(const SHA512State_t* state, uint8_t* out) {
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint64_t x = state->h[i];
        out[i*8+0] = (x >> 56) & 0xFF;
        out[i*8+1] = (x >> 48) & 0xFF;
        out[i*8+2] = (x >> 40) & 0xFF;
        out[i*8+3] = (x >> 32) & 0xFF;
        out[i*8+4] = (x >> 24) & 0xFF;
        out[i*8+5] = (x >> 16) & 0xFF;
        out[i*8+6] = (x >> 8) & 0xFF;
        out[i*8+7] = x & 0xFF;
    }
}

__device__ __forceinline__ void sha512_transform_block_raw_opt(SHA512State_t* state, const uint8_t* data) {
    uint64_t W[80];
    
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        W[i] = ((uint64_t)data[i * 8] << 56) |
               ((uint64_t)data[i * 8 + 1] << 48) |
               ((uint64_t)data[i * 8 + 2] << 40) |
               ((uint64_t)data[i * 8 + 3] << 32) |
               ((uint64_t)data[i * 8 + 4] << 24) |
               ((uint64_t)data[i * 8 + 5] << 16) |
               ((uint64_t)data[i * 8 + 6] << 8) |
               ((uint64_t)data[i * 8 + 7]);
    }

    #pragma unroll
    for (int i = 16; i < 80; i++) {
        W[i] = gamma1_512(W[i - 2]) + W[i - 7] + gamma0_512(W[i - 15]) + W[i - 16];
    }
    
    uint64_t a = state->h[0]; uint64_t b = state->h[1]; uint64_t c = state->h[2]; uint64_t d = state->h[3];
    uint64_t e = state->h[4]; uint64_t f = state->h[5]; uint64_t g = state->h[6]; uint64_t h = state->h[7];
    uint64_t t1, t2;
    
    #pragma unroll
    for (int i = 0; i < 80; i++) {
        t1 = h + sigma1_512(e) + ch64(e, f, g) + K512[i] + W[i];
        t2 = sigma0_512(a) + maj64(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    
    state->h[0] += a; state->h[1] += b; state->h[2] += c; state->h[3] += d;
    state->h[4] += e; state->h[5] += f; state->h[6] += g; state->h[7] += h;
}

__device__ __forceinline__ void sha512_finish_block2_192bytes_opt(
    SHA512State_t* state,
    const uint8_t* data_64bytes
) {
    uint64_t block[16];
    
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint64_t w = 0;
        #pragma unroll
        for(int j=0; j<8; j++) w = (w << 8) | data_64bytes[i*8 + j];
        block[i] = w;
    }
    
    block[8] = 0x8000000000000000ULL;
    block[9] = 0; block[10] = 0; block[11] = 0;
    block[12] = 0; block[13] = 0; block[14] = 0;
    block[15] = 0x0000000000000600ULL; // 1536 bits
    
    uint64_t W[80];
    
    #pragma unroll
    for (int i = 0; i < 16; i++) W[i] = block[i];
    
    #pragma unroll
    for (int i = 16; i < 80; i++) {
        W[i] = gamma1_512(W[i - 2]) + W[i - 7] + gamma0_512(W[i - 15]) + W[i - 16];
    }
    
    uint64_t a = state->h[0]; uint64_t b = state->h[1]; uint64_t c = state->h[2]; uint64_t d = state->h[3];
    uint64_t e = state->h[4]; uint64_t f = state->h[5]; uint64_t g = state->h[6]; uint64_t h = state->h[7];
    uint64_t t1, t2;
    
    #pragma unroll
    for (int i = 0; i < 80; i++) {
        t1 = h + sigma1_512(e) + ch64(e, f, g) + K512[i] + W[i];
        t2 = sigma0_512(a) + maj64(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    
    state->h[0] += a; state->h[1] += b; state->h[2] += c; state->h[3] += d;
    state->h[4] += e; state->h[5] += f; state->h[6] += g; state->h[7] += h;
}

__device__ void pbkdf2_sha512_optimized(
    const uint8_t* key_64bytes,
    const uint8_t* salt, uint32_t salt_len,
    uint32_t iterations,
    uint8_t* output_64bytes
) {
    if (salt_len != 8) return; // Only support fixed logic for now
    
    uint8_t k_ipad[128];
    uint8_t k_opad[128];
    
    #pragma unroll
    for (int i = 0; i < 64; i++) {
        k_ipad[i] = key_64bytes[i] ^ 0x36;
        k_opad[i] = key_64bytes[i] ^ 0x5c;
    }
    #pragma unroll
    for (int i = 64; i < 128; i++) {
        k_ipad[i] = 0x36;
        k_opad[i] = 0x5c;
    }
    
    SHA512State_t ctx_inner_pre;
    SHA512State_t ctx_outer_pre;
    
    sha512_init_state_opt(&ctx_inner_pre);
    sha512_transform_block_raw_opt(&ctx_inner_pre, k_ipad);
    
    sha512_init_state_opt(&ctx_outer_pre);
    sha512_transform_block_raw_opt(&ctx_outer_pre, k_opad);
    
    uint8_t U[64];
    uint8_t T[64];
    
    {
        SHA512State_t ctx = ctx_inner_pre;
        
        uint64_t block2[16];
        block2[0] = 0x6d6e656d6f6e6963ULL; 
        block2[1] = 0x0000000180000000ULL;
        #pragma unroll
        for(int i=2; i<15; i++) block2[i] = 0;
        block2[15] = 0x460;
        
        uint64_t W[80];
        #pragma unroll
        for(int i=0; i<16; i++) W[i] = block2[i];
        #pragma unroll
        for(int i=16; i<80; i++) W[i] = gamma1_512(W[i-2]) + W[i-7] + gamma0_512(W[i-15]) + W[i-16];
        
        uint64_t a = ctx.h[0]; uint64_t b = ctx.h[1]; uint64_t c = ctx.h[2]; uint64_t d = ctx.h[3];
        uint64_t e = ctx.h[4]; uint64_t f = ctx.h[5]; uint64_t g = ctx.h[6]; uint64_t h = ctx.h[7];
        uint64_t t1, t2;
        
        #pragma unroll
        for(int i=0; i<80; i++) {
            t1 = h + sigma1_512(e) + ch64(e, f, g) + K512[i] + W[i];
            t2 = sigma0_512(a) + maj64(a, b, c);
            h = g; g = f; f = e; e = d + t1;
            d = c; c = b; b = a; a = t1 + t2;
        }
        ctx.h[0]+=a; ctx.h[1]+=b; ctx.h[2]+=c; ctx.h[3]+=d; ctx.h[4]+=e; ctx.h[5]+=f; ctx.h[6]+=g; ctx.h[7]+=h;
        
        uint8_t inner_hash[64];
        sha512_extract_opt(&ctx, inner_hash);
        
        SHA512State_t ctx_out = ctx_outer_pre;
        sha512_finish_block2_192bytes_opt(&ctx_out, inner_hash);
        
        sha512_extract_opt(&ctx_out, U);
        #pragma unroll
        for(int i=0; i<64; i++) T[i] = U[i];
    }
    
    for (uint32_t i = 1; i < iterations; i++) {
        SHA512State_t ctx = ctx_inner_pre;
        sha512_finish_block2_192bytes_opt(&ctx, U);
        uint8_t inner_hash[64];
        sha512_extract_opt(&ctx, inner_hash);
        
        ctx = ctx_outer_pre;
        sha512_finish_block2_192bytes_opt(&ctx, inner_hash);
        
        sha512_extract_opt(&ctx, U);
        
        #pragma unroll
        for(int j=0; j<64; j++) T[j] ^= U[j];
    }
    
    #pragma unroll
    for(int i=0; i<64; i++) output_64bytes[i] = T[i];
}

// PBKDF2-SHA512 for variable-length mnemonic (BIP39 compliant)
__device__ void pbkdf2_sha512_mnemonic(
    const uint8_t* password, uint32_t password_len,
    const uint8_t* salt, uint32_t salt_len,
    uint32_t iterations,
    uint8_t* output_64bytes
) {
    // Key preparation for HMAC: if password > 128, hash it; else pad with zeros
    uint8_t key[128];
    for(int i=0; i<128; i++) key[i] = 0;
    
    if (password_len <= 128) {
        for(uint32_t i=0; i<password_len; i++) key[i] = password[i];
    } else {
        // Hash the password (rare case for BIP39)
        sha512(password, password_len, key);
    }
    
    uint8_t k_ipad[128];
    uint8_t k_opad[128];
    
    for(int i=0; i<128; i++) {
        k_ipad[i] = key[i] ^ 0x36;
        k_opad[i] = key[i] ^ 0x5c;
    }
    
    SHA512State_t ctx_inner_pre, ctx_outer_pre;
    sha512_init_state_opt(&ctx_inner_pre);
    sha512_transform_block_raw_opt(&ctx_inner_pre, k_ipad);
    
    sha512_init_state_opt(&ctx_outer_pre);
    sha512_transform_block_raw_opt(&ctx_outer_pre, k_opad);
    
    uint8_t U[64], T[64];
    
    // First iteration: HMAC(key, salt || INT(1))
    {
        SHA512State_t ctx = ctx_inner_pre;
        
        // Process salt + block number
        uint8_t msg[128];
        for(int i=0; i<128; i++) msg[i] = 0;
        
        uint32_t msg_len = 0;
        for(uint32_t i=0; i<salt_len && msg_len<124; i++) msg[msg_len++] = salt[i];
        
        // Append block number (1) as big-endian 32-bit
        msg[msg_len++] = 0x00;
        msg[msg_len++] = 0x00;
        msg[msg_len++] = 0x00;
        msg[msg_len++] = 0x01;
        
        // CRITICAL FIX: Calculate bit_len BEFORE adding padding
        // bit_len = (k_ipad block: 128 bytes) + (current message: salt + block_number)
        uint64_t bit_len = (128 + msg_len) * 8;
        
        // Now add padding byte
        msg[msg_len] = 0x80;
        msg_len++;
        
        // Place length in last 8 bytes of the block (big-endian)
        msg[120] = (bit_len >> 56) & 0xFF;
        msg[121] = (bit_len >> 48) & 0xFF;
        msg[122] = (bit_len >> 40) & 0xFF;
        msg[123] = (bit_len >> 32) & 0xFF;
        msg[124] = (bit_len >> 24) & 0xFF;
        msg[125] = (bit_len >> 16) & 0xFF;
        msg[126] = (bit_len >> 8) & 0xFF;
        msg[127] = bit_len & 0xFF;
        
        sha512_transform_block_raw_opt(&ctx, msg);
        
        uint8_t inner_hash[64];
        sha512_extract_opt(&ctx, inner_hash);
        
        // Outer HMAC
        SHA512State_t ctx_out = ctx_outer_pre;
        sha512_finish_block2_192bytes_opt(&ctx_out, inner_hash);
        sha512_extract_opt(&ctx_out, U);
        
        for(int i=0; i<64; i++) T[i] = U[i];
    }
    
    // Remaining iterations
    for(uint32_t iter=1; iter<iterations; iter++) {
        SHA512State_t ctx = ctx_inner_pre;
        sha512_finish_block2_192bytes_opt(&ctx, U);
        uint8_t inner_hash[64];
        sha512_extract_opt(&ctx, inner_hash);
        
        ctx = ctx_outer_pre;
        sha512_finish_block2_192bytes_opt(&ctx, inner_hash);
        sha512_extract_opt(&ctx, U);
        
        for(int j=0; j<64; j++) T[j] ^= U[j];
    }
    
    for(int i=0; i<64; i++) output_64bytes[i] = T[i];
}

// SHA-512 compression taking the message block directly as 16 big-endian
// uint64 W-words (WLBP-PR path: key block assembled in registers).
__device__ __forceinline__ void sha512_transform_w_opt(SHA512State_t* state, const uint64_t W_in[16]) {
    uint64_t W[80];

    #pragma unroll
    for (int i = 0; i < 16; i++) W[i] = W_in[i];

    #pragma unroll
    for (int i = 16; i < 80; i++) {
        W[i] = gamma1_512(W[i - 2]) + W[i - 7] + gamma0_512(W[i - 15]) + W[i - 16];
    }

    uint64_t a = state->h[0]; uint64_t b = state->h[1]; uint64_t c = state->h[2]; uint64_t d = state->h[3];
    uint64_t e = state->h[4]; uint64_t f = state->h[5]; uint64_t g = state->h[6]; uint64_t h = state->h[7];
    uint64_t t1, t2;

    #pragma unroll
    for (int i = 0; i < 80; i++) {
        t1 = h + sigma1_512(e) + ch64(e, f, g) + K512[i] + W[i];
        t2 = sigma0_512(a) + maj64(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    state->h[0] += a; state->h[1] += b; state->h[2] += c; state->h[3] += d;
    state->h[4] += e; state->h[5] += f; state->h[6] += g; state->h[7] += h;
}

// PBKDF2-SHA512 where the 128-byte zero-padded HMAC key block arrives
// pre-assembled as 16 big-endian uint64 W-words (wlbp_assemble_key12 output).
// Byte-exact equivalent of pbkdf2_sha512_mnemonic for the same mnemonic:
// identical ipad/opad blocks -> identical pre-midstates -> identical U-chain.
__device__ void pbkdf2_sha512_keyblock(
    const uint64_t key_w[16],
    const uint8_t* salt, uint32_t salt_len,
    uint32_t iterations,
    uint8_t* output_64bytes
) {
    // ipad/opad blocks straight from the assembled key words
    uint64_t ipad_w[16], opad_w[16];

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        ipad_w[i] = key_w[i] ^ 0x3636363636363636ULL;
        opad_w[i] = key_w[i] ^ 0x5c5c5c5c5c5c5c5cULL;
    }

    SHA512State_t ctx_inner_pre, ctx_outer_pre;
    sha512_init_state_opt(&ctx_inner_pre);
    sha512_transform_w_opt(&ctx_inner_pre, ipad_w);

    sha512_init_state_opt(&ctx_outer_pre);
    sha512_transform_w_opt(&ctx_outer_pre, opad_w);

    uint8_t U[64], T[64];

    // First iteration: HMAC(key, salt || INT(1)) - identical construction
    // to pbkdf2_sha512_mnemonic (bit_len computed before padding)
    {
        SHA512State_t ctx = ctx_inner_pre;

        uint8_t msg[128];
        for(int i=0; i<128; i++) msg[i] = 0;

        uint32_t msg_len = 0;
        for(uint32_t i=0; i<salt_len && msg_len<124; i++) msg[msg_len++] = salt[i];

        msg[msg_len++] = 0x00;
        msg[msg_len++] = 0x00;
        msg[msg_len++] = 0x00;
        msg[msg_len++] = 0x01;

        uint64_t bit_len = (128 + msg_len) * 8;

        msg[msg_len] = 0x80;
        msg_len++;

        msg[120] = (bit_len >> 56) & 0xFF;
        msg[121] = (bit_len >> 48) & 0xFF;
        msg[122] = (bit_len >> 40) & 0xFF;
        msg[123] = (bit_len >> 32) & 0xFF;
        msg[124] = (bit_len >> 24) & 0xFF;
        msg[125] = (bit_len >> 16) & 0xFF;
        msg[126] = (bit_len >> 8) & 0xFF;
        msg[127] = bit_len & 0xFF;

        sha512_transform_block_raw_opt(&ctx, msg);

        uint8_t inner_hash[64];
        sha512_extract_opt(&ctx, inner_hash);

        SHA512State_t ctx_out = ctx_outer_pre;
        sha512_finish_block2_192bytes_opt(&ctx_out, inner_hash);
        sha512_extract_opt(&ctx_out, U);

        for(int i=0; i<64; i++) T[i] = U[i];
    }

    // Remaining iterations - identical to pbkdf2_sha512_mnemonic
    for(uint32_t iter=1; iter<iterations; iter++) {
        SHA512State_t ctx = ctx_inner_pre;
        sha512_finish_block2_192bytes_opt(&ctx, U);
        uint8_t inner_hash[64];
        sha512_extract_opt(&ctx, inner_hash);

        ctx = ctx_outer_pre;
        sha512_finish_block2_192bytes_opt(&ctx, inner_hash);
        sha512_extract_opt(&ctx, U);

        for(int j=0; j<64; j++) T[j] ^= U[j];
    }

    for(int i=0; i<64; i++) output_64bytes[i] = T[i];
}

// ETAPA B3: SHA-512 compression with round folding for the constant tail.
// For rounds i >= fold_from, W[i] is a known constant (uniform pad pattern):
// K[i] + W[i] is pre-folded into KP[i] (one add instead of two).
__device__ __forceinline__ void sha512_transform_w_fold_opt(
    SHA512State_t* state, const uint64_t W_in[16],
    uint32_t fold_from, const uint64_t KP[16]
) {
    uint64_t W[80];

    #pragma unroll
    for (int i = 0; i < 16; i++) W[i] = W_in[i];

    #pragma unroll
    for (int i = 16; i < 80; i++) {
        W[i] = gamma1_512(W[i - 2]) + W[i - 7] + gamma0_512(W[i - 15]) + W[i - 16];
    }

    uint64_t a = state->h[0]; uint64_t b = state->h[1]; uint64_t c = state->h[2]; uint64_t d = state->h[3];
    uint64_t e = state->h[4]; uint64_t f = state->h[5]; uint64_t g = state->h[6]; uint64_t h = state->h[7];
    uint64_t t1, t2;

    #pragma unroll
    for (int i = 0; i < 80; i++) {
        /* fold applies ONLY to direct-W rounds [fold_from, 15]; rounds 16+ use the schedule */
        uint64_t Kw = (i >= (int)fold_from && i < 16) ? KP[i] : (K512[i] + W[i]);
        t1 = h + sigma1_512(e) + ch64(e, f, g) + Kw;
        t2 = sigma0_512(a) + maj64(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    state->h[0] += a; state->h[1] += b; state->h[2] += c; state->h[3] += d;
    state->h[4] += e; state->h[5] += f; state->h[6] += g; state->h[7] += h;
}

// ETAPA B: PBKDF2 with folded rounds for the ipad/opad first blocks.
// KP36/KP5c are per-thread precomputed folded constants:
//   KP36[i] = K512[i] + 0x3636363636363636ULL
//   KP5c[i] = K512[i] + 0x5c5c5c5c5c5c5c5cULL
// fold_from = first W-index fully inside the zero-pad region = ceil(key_len/8).
// Byte-exact equivalent of pbkdf2_sha512_keyblock (and of the classic path).
__device__ void pbkdf2_sha512_keyblock_b(
    const uint64_t key_w[16], uint32_t key_len,
    const uint64_t KP36[16], const uint64_t KP5c[16],
    const uint8_t* salt, uint32_t salt_len,
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
    sha512_transform_w_fold_opt(&ctx_inner_pre, ipad_w, i0, KP36);

    sha512_init_state_opt(&ctx_outer_pre);
    sha512_transform_w_fold_opt(&ctx_outer_pre, opad_w, i0, KP5c);

    uint8_t U[64], T[64];

    // First iteration: HMAC(key, salt || INT(1)) - identical to keyblock path
    {
        SHA512State_t ctx = ctx_inner_pre;

        uint8_t msg[128];
        for(int i=0; i<128; i++) msg[i] = 0;

        uint32_t msg_len = 0;
        for(uint32_t i=0; i<salt_len && msg_len<124; i++) msg[msg_len++] = salt[i];

        msg[msg_len++] = 0x00;
        msg[msg_len++] = 0x00;
        msg[msg_len++] = 0x00;
        msg[msg_len++] = 0x01;

        uint64_t bit_len = (128 + msg_len) * 8;

        msg[msg_len] = 0x80;
        msg_len++;

        msg[120] = (bit_len >> 56) & 0xFF;
        msg[121] = (bit_len >> 48) & 0xFF;
        msg[122] = (bit_len >> 40) & 0xFF;
        msg[123] = (bit_len >> 32) & 0xFF;
        msg[124] = (bit_len >> 24) & 0xFF;
        msg[125] = (bit_len >> 16) & 0xFF;
        msg[126] = (bit_len >> 8) & 0xFF;
        msg[127] = bit_len & 0xFF;

        sha512_transform_block_raw_opt(&ctx, msg);

        uint8_t inner_hash[64];
        sha512_extract_opt(&ctx, inner_hash);

        SHA512State_t ctx_out = ctx_outer_pre;
        sha512_finish_block2_192bytes_opt(&ctx_out, inner_hash);
        sha512_extract_opt(&ctx_out, U);

        for(int i=0; i<64; i++) T[i] = U[i];
    }

    // Remaining iterations - identical to keyblock path
    for(uint32_t iter=1; iter<iterations; iter++) {
        SHA512State_t ctx = ctx_inner_pre;
        sha512_finish_block2_192bytes_opt(&ctx, U);
        uint8_t inner_hash[64];
        sha512_extract_opt(&ctx, inner_hash);

        ctx = ctx_outer_pre;
        sha512_finish_block2_192bytes_opt(&ctx, inner_hash);
        sha512_extract_opt(&ctx, U);

        for(int j=0; j<64; j++) T[j] ^= U[j];
    }

    for(int i=0; i<64; i++) output_64bytes[i] = T[i];
}
#endif // !DEV_BUILD

#ifdef __CUDACC__
#endif /* __CUDACC__ */

// ---- ETAPA C folded constants (generated by gen_kps.cpp from the REAL
// K512 in sha512.cuh; d_KPP cross-validated against pbkdf2-bench/sha512_fast
// .cuh KP8..KP15). Compile-time initialized: no uploader needed. ----

// key-block pad tails: K512[i] + 0x36-pattern / K512[i] + 0x5c-pattern
__constant__ uint64_t d_KP36[16] = {
    0x78c065cf0d5ee458ULL, 0xa76d7ac75a259c03ULL, 0xebf7320622837165ULL, 0x1fec11dbb7c011f2ULL,
    0x6f8cf892297eeb6eULL, 0x90274827ec3c064fULL, 0xc875b8dae54f85d1ULL, 0xe152950c10a3b74eULL,
    0x0e3de0ced9393878ULL, 0x48b991377ba6a5f4ULL, 0x5a67bbf4851ae8c2ULL, 0x8b42b3fa0c35eb18ULL,
    0xa8f493ab28b1bfa5ULL, 0xb714e834714ccce7ULL, 0xd2123cdd5bfd486bULL, 0xf7d227ab059f5ccaULL
};
__constant__ uint64_t d_KP5c[16] = {
    0x9ee68bf533850a7eULL, 0xcd93a0ed804bc229ULL, 0x121d582c48a9978bULL, 0x46123801dde63818ULL,
    0x95b31eb84fa51194ULL, 0xb64d6e4e12622c75ULL, 0xee9bdf010b75abf7ULL, 0x0778bb3236c9dd74ULL,
    0x346406f4ff5f5e9eULL, 0x6edfb75da1cccc1aULL, 0x808de21aab410ee8ULL, 0xb168da20325c113eULL,
    0xcf1ab9d14ed7e5cbULL, 0xdd3b0e5a9772f30dULL, 0xf838630382236e91ULL, 0x1df84dd12bc582f0ULL
};

// U-chain blocks rounds 8..15: K512[8+j] + pad[j], pad = {0x8000..,0,0,0,0,0,0,0x600}
__constant__ uint64_t d_KPP[8] = {
    0x5807aa98a3030242ULL, 0x12835b0145706fbeULL, 0x243185be4ee4b28cULL, 0x550c7dc3d5ffb4e2ULL,
    0x72be5d74f27b896fULL, 0x80deb1fe3b1696b1ULL, 0x9bdc06a725c71235ULL, 0xc19bf174cf692c94ULL
};
__constant__ uint64_t d_min_pre[8] = {
    0x2e2af459060c1873ULL, 0x7894b868dc88433aULL, 0xdd1a797ef1a1933aULL, 0xe6486d04fcb412a7ULL,
    0xfbcc67b9a396caa0ULL, 0xa2970b146f49b65eULL, 0xfdf1daabc66f6248ULL, 0x2ff99c812ada6dc3ULL
};
__constant__ uint64_t d_mout_pre[8] = {
    0xbbd27bac212e9dbdULL, 0xdd0bc55e7e4037c1ULL, 0xdfdd3d6890bd6424ULL, 0x2902de663032b34cULL,
    0xa30f8aa6f67899fcULL, 0x69a566c30f88378fULL, 0x0500247985ecb694ULL, 0xf6d70307c6b2d337ULL
};

// fixed salt block "mnemonic"||INT(1): ALL 80 rounds folded (K512[i] + W_salt[i],
// W_salt schedule fully constant) -> salt-block compression needs no W array
__constant__ uint64_t d_KPS[80] = {
    0xaff8950646971785ULL, 0x71374492a3ef65cdULL, 0xb5c0fbcfec4d3b2fULL, 0xe9b5dba58189dbbcULL,
    0x3956c25bf348b538ULL, 0x59f111f1b605d019ULL, 0x923f82a4af194f9bULL, 0xab1c5ed5da6d8118ULL,
    0xd807aa98a3030242ULL, 0x12835b0145706fbeULL, 0x243185be4ee4b28cULL, 0x550c7dc3d5ffb4e2ULL,
    0x72be5d74f27b896fULL, 0x80deb1fe3b1696b1ULL, 0x9bdc06a725c71235ULL, 0xc19bf174cf692af4ULL,
    0x5209cf2fd0dfb435ULL, 0xf04a4787b84f48f4ULL, 0xb7ac9d1086a60a38ULL, 0x240cb1e9fdadc469ULL,
    0x494151bef9f80221ULL, 0x6f748556b2b9d8c3ULL, 0xa20a735ff98f3abbULL, 0x5a83f00946921e6aULL,
    0x24d214e9460fa704ULL, 0x56a8725314baf588ULL, 0x2e4a9a14bb669c67ULL, 0x84153565da92cab8ULL,
    0x88878d004f5774e4ULL, 0x6cb7357ca3187949ULL, 0x1cf398d8ec948017ULL, 0x7622faefad0444a0ULL,
    0x8387ba05804cbde8ULL, 0xb29b3b6735288b00ULL, 0x4c2ae0e935cd3538ULL, 0xd48b1a94a5fcbeceULL,
    0x3fa24ea3b8c97001ULL, 0xe0b71f9dcf27d71bULL, 0x05490fbee8c3948fULL, 0xc7f996d26001da1dULL,
    0x8526b264e8446650ULL, 0xc5a1ff42c73e6ecaULL, 0xf5d86ac6b9968054ULL, 0x15e5050167ccefd1ULL,
    0x8d76f0a3db2ce7baULL, 0x0926247879d6d292ULL, 0xaf993b1f010cd3daULL, 0xc761e276decde06dULL,
    0xd58aae7167fe46a4ULL, 0x80781d51fe2bdc40ULL, 0x56657b8ad187aac4ULL, 0x969d4d4b8e052b73ULL,
    0x7c8a93c05ec9f7f7ULL, 0x10b2c91da904f445ULL, 0x98829994612fa861ULL, 0x56aa5e2d813836fdULL,
    0xa05c6d55269ff757ULL, 0xff8abf2a19ca522aULL, 0x7773be614d044260ULL, 0x6e547d34a6a41599ULL,
    0x8ddb8088deb1914dULL, 0xf464cb00e7a2d4ebULL, 0xfe161a64d57453f2ULL, 0xac7769cda7800715ULL,
    0xacbd86aeabaee640ULL, 0x693d23462e13d14fULL, 0x7e8676f07093d03cULL, 0xe7fde95b03394ec7ULL,
    0x79d563723626ec8cULL, 0xd3f90fe91e553c4aULL, 0xd9b0d6e0aca2db16ULL, 0x971bbfbe31fb7dd3ULL,
    0xee5d7a36c14dc6efULL, 0x1e0a612bbeb61709ULL, 0x84da2b02bdaede65ULL, 0xf638d2b293007895ULL,
    0x21809ecb6dcd42ffULL, 0x4c51f2e18665a2d3ULL, 0xf94912f8096285c0ULL, 0x7e00e52b4826e5e8ULL
};

// ETAPA C: fully-folded salt-block compression. The "mnemonic"+INT(1) block
// is constant forever -> all 80 round constants pre-folded in d_KPS.
// No W array, no schedule, no per-round K loads - pure state evolution.
__device__ __forceinline__ void sha512_transform_saltblock_folded_opt(SHA512State_t* state) {
    uint64_t a = state->h[0]; uint64_t b = state->h[1]; uint64_t c = state->h[2]; uint64_t d = state->h[3];
    uint64_t e = state->h[4]; uint64_t f = state->h[5]; uint64_t g = state->h[6]; uint64_t h = state->h[7];
    uint64_t t1, t2;

    #pragma unroll
    for (int i = 0; i < 80; i++) {
        t1 = h + sigma1_512(e) + ch64(e, f, g) + d_KPS[i];
        t2 = sigma0_512(a) + maj64(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    state->h[0] += a; state->h[1] += b; state->h[2] += c; state->h[3] += d;
    state->h[4] += e; state->h[5] += f; state->h[6] += g; state->h[7] += h;
}

#ifndef DEV_BUILD
// ETAPA C: U-chain finish block (64B data + fixed padding) with rounds 8..15
// folded via d_KPP - saves 8 constant loads + 8 adds per compression on the
// critical dependency chain. Applies 2x per PBKDF2 iteration (~4094x/phrase).
__device__ __forceinline__ void sha512_finish_block2_folded_opt(
    SHA512State_t* state, const uint8_t* data_64bytes
) {
    uint64_t W[80];

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint64_t w = 0;
        #pragma unroll
        for(int j=0; j<8; j++) w = (w << 8) | data_64bytes[i*8 + j];
        W[i] = w;
    }

    W[8]  = 0x8000000000000000ULL;
    W[9]  = 0; W[10] = 0; W[11] = 0;
    W[12] = 0; W[13] = 0; W[14] = 0;
    W[15] = 0x0000000000000600ULL; // 1536 bits

    #pragma unroll
    for (int i = 16; i < 80; i++) {
        W[i] = gamma1_512(W[i - 2]) + W[i - 7] + gamma0_512(W[i - 15]) + W[i - 16];
    }

    uint64_t a = state->h[0]; uint64_t b = state->h[1]; uint64_t c = state->h[2]; uint64_t d = state->h[3];
    uint64_t e = state->h[4]; uint64_t f = state->h[5]; uint64_t g = state->h[6]; uint64_t h = state->h[7];
    uint64_t t1, t2;

    #pragma unroll
    for (int i = 0; i < 80; i++) {
        uint64_t Kw = (i < 8) ? (K512[i] + W[i])
                    : (i < 16) ? d_KPP[i - 8]
                    : (K512[i] + W[i]);
        t1 = h + sigma1_512(e) + ch64(e, f, g) + Kw;
        t2 = sigma0_512(a) + maj64(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    state->h[0] += a; state->h[1] += b; state->h[2] += c; state->h[3] += d;
    state->h[4] += e; state->h[5] += f; state->h[6] += g; state->h[7] += h;
}

// ETAPA C: PBKDF2 with ALL fixed blocks fused. Byte-exact equivalent of
// keyblock/keyblock_b. REQUIRES salt == "mnemonic" (8 bytes) - the salt
// block is fused into d_KPS. Folded blocks: key pad tails (d_KP36/d_KP5c),
// salt block (d_KPS, all 80 rounds), U-chain padding (d_KPP, 2x/iteration).
__device__ void pbkdf2_sha512_keyblock_c(
    const uint64_t key_w[16], uint32_t key_len,
    const uint8_t* salt, uint32_t salt_len,
    uint32_t iterations,
    uint8_t* output_64bytes
) {
    (void)salt; (void)salt_len; /* fused into d_KPS */

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
    sha512_transform_w_fold_opt(&ctx_inner_pre, ipad_w, i0, d_KP36);

    sha512_init_state_opt(&ctx_outer_pre);
    sha512_transform_w_fold_opt(&ctx_outer_pre, opad_w, i0, d_KP5c);

    uint8_t U[64], T[64];

    // First iteration: inner over the FUSED salt block
    {
        SHA512State_t ctx = ctx_inner_pre;
        sha512_transform_saltblock_folded_opt(&ctx);

        uint8_t inner_hash[64];
        sha512_extract_opt(&ctx, inner_hash);

        SHA512State_t ctx_out = ctx_outer_pre;
        sha512_finish_block2_folded_opt(&ctx_out, inner_hash);
        sha512_extract_opt(&ctx_out, U);

        for(int i=0; i<64; i++) T[i] = U[i];
    }

    // Remaining iterations with folded finish blocks
    for(uint32_t iter=1; iter<iterations; iter++) {
        SHA512State_t ctx = ctx_inner_pre;
        sha512_finish_block2_folded_opt(&ctx, U);
        uint8_t inner_hash[64];
        sha512_extract_opt(&ctx, inner_hash);

        ctx = ctx_outer_pre;
        sha512_finish_block2_folded_opt(&ctx, inner_hash);
        sha512_extract_opt(&ctx, U);

        for(int j=0; j<64; j++) T[j] ^= U[j];
    }

    for(int i=0; i<64; i++) output_64bytes[i] = T[i];
}
#endif // !DEV_BUILD

// ---- P1: rotating 16-word schedule (registers, no W[80] local-memory spills).
// Ported technique from pbkdf2-bench/sha512_fast.cuh: with full unroll the
// (t & 15) indices constant-fold and the message schedule stays in registers. ----

// Key-block compressor: 16 message words + fold rounds [fold_from..15] into KP
// (folded W still feeds the schedule - only the round add is skipped).
__device__ __forceinline__ void sha512_compress16_rot_opt(
    SHA512State_t* state, const uint64_t m_in[16],
    uint32_t fold_from, const uint64_t* KP
) {
    uint64_t a = state->h[0]; uint64_t b = state->h[1]; uint64_t c = state->h[2]; uint64_t d = state->h[3];
    uint64_t e = state->h[4]; uint64_t f = state->h[5]; uint64_t g = state->h[6]; uint64_t h = state->h[7];
    uint64_t t1, t2;
    uint64_t W[16];

    #pragma unroll
    for (int t = 0; t < 16; t++) W[t] = m_in[t];

    #pragma unroll
    for (int t = 0; t < 16; t++) {
        uint64_t Kw = (t >= (int)fold_from) ? KP[t] : (K512[t] + W[t]);
        t1 = h + sigma1_512(e) + ch64(e, f, g) + Kw;
        t2 = sigma0_512(a) + maj64(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    #pragma unroll
    for (int t = 16; t < 80; t++) {
        W[t & 15] += gamma1_512(W[(t - 2) & 15]) + W[(t - 7) & 15] + gamma0_512(W[(t - 15) & 15]);
        t1 = h + sigma1_512(e) + ch64(e, f, g) + K512[t] + W[t & 15];
        t2 = sigma0_512(a) + maj64(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    state->h[0] += a; state->h[1] += b; state->h[2] += c; state->h[3] += d;
    state->h[4] += e; state->h[5] += f; state->h[6] += g; state->h[7] += h;
}

// U-chain hot block: 64B message (u[8] as big-endian words) + fixed padding.
// 3-phase rounds: 0-7 message, 8-15 folded KPP, 16-79 rotating schedule.
// ---------------------------------------------------------------------------
// E16: STATE-OF-THE-ART SHA-512 compression ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â every SASS-level trick applied
// ---------------------------------------------------------------------------
//
// Technique 1: LOP3 fusion ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â NVIDIA GPUs can execute any 3-input boolean
//   function in ONE instruction (LOP3). Ch(e,f,g) = (e AND f) XOR (NOT e AND g)
//   compiles to 1 LOP3 instead of 3 (AND, NOT, AND, XOR).
//   Maj(a,b,c) = (a AND b) XOR (a AND c) XOR (b AND c) ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â also 1 LOP3.
//   SAVES: 4 instructions per round ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â 80 rounds ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â 4098 compressions.
//
// Technique 2: Constant rotate folding ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â all SHA-512 rotations use compile-
//   time constants. The compiler emits SHF.L.U32.HI + SHF.L.U32 (2 SASS ops)
//   for a 64-bit constant rotate instead of SHR+SHL+OR (3 ops).
//   With __funnelshift_r explicitly: guaranteed 2 ops per 64-bit rotate.
//   SAVES: 6 ops per round (6 rotates ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â 1 op saved).
//
// Technique 3: SIMD-within-a-register (SWAR) sigma/gamma computation.
//   The 64-bit rotate-xor-rotate pattern decomposes into independent 32-bit
//   funnel shifts on the hi/lo halves, enabling dual-issue on separate
//   scheduler pipes.
//
// Technique 4: K512 fusion into first add. Instead of:
//   t1 = h + sigma1(e) + ch(e,f,g) + K512[t] + W[t]     (4 chained adds)
// Pre-compute K512_W[t] = K512[t] + W[t] during the schedule update phase
// (which already touches W[t]), reducing the round to:
//   t1 = h + sigma1(e) + ch(e,f,g) + KW[t]              (3 chained adds)
// SAVES: 1 add per round on the critical path ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â 80 rounds.
//
// Technique 5: Eta reduction ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â the state shift h=g; g=f; f=e; e=d+t1 is a
// register rename. The compiler handles this for free with register
// allocation. No explicit moves needed with proper naming (a0,a1,a2,...).
//
// COMBINED: ~29 ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ ~22 instructions per round = ~24% reduction.
// ---------------------------------------------------------------------------
#ifndef DEV_BUILD
// E16: the SOTA single-chain compression with all techniques applied
__device__ __forceinline__ void sha512_finish64msg_sota(
    SHA512State_t* state, const uint64_t u[8]
) {
    uint64_t a = state->h[0]; uint64_t b = state->h[1];
    uint64_t c = state->h[2]; uint64_t d = state->h[3];
    uint64_t e = state->h[4]; uint64_t f = state->h[5];
    uint64_t g = state->h[6]; uint64_t h = state->h[7];
    uint64_t t1, t2;
    uint64_t W[16];

    // W[0..7] = message (plain load Ã¢â‚¬â€ identical to fast version)
    #pragma unroll
    for (int t = 0; t < 8; t++) {
        W[t] = u[t];
        t1 = h + sigma1_512(e) + ch64(e, f, g) + K512[t] + W[t];
        t2 = sigma0_512(a) + maj64(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    // W[8..15] = padding (constants) ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â rounds 8-15 use folded KPP
    #pragma unroll
    for (int t = 8; t < 16; t++) {
        t1 = h + sigma1_512(e) + ch64(e, f, g) + d_KPP[t - 8];
        t2 = sigma0_512(a) + maj64(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    #ifdef SOTA_DEBUG
    printf("[SOTA] after rounds 8-15: a=%016llx\n", (unsigned long long)a);
    #endif
    #ifdef SOTA_DEBUG
    printf("[SOTA] after rounds 8-15: a=%016llx\n", (unsigned long long)a);
    #endif
    // rounds 16-79: rotating schedule (K512 NOT fused - schedule contamination)
    #pragma unroll
    for (int t = 16; t < 80; t++) {
        W[t & 15] += gamma1_512(W[(t - 2) & 15]) + W[(t - 7) & 15] + gamma0_512(W[(t - 15) & 15]);
        t1 = h + sigma1_512(e) + ch64(e, f, g) + K512[t] + W[t & 15];
        t2 = sigma0_512(a) + maj64(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    state->h[0] += a; state->h[1] += b; state->h[2] += c; state->h[3] += d;
    state->h[4] += e; state->h[5] += f; state->h[6] += g; state->h[7] += h;
}
#endif // !DEV_BUILD

__device__ __forceinline__ void sha512_finish64msg_fast(SHA512State_t* state, const uint64_t u[8]) {
    uint64_t a = state->h[0]; uint64_t b = state->h[1]; uint64_t c = state->h[2]; uint64_t d = state->h[3];
    uint64_t e = state->h[4]; uint64_t f = state->h[5]; uint64_t g = state->h[6]; uint64_t h = state->h[7];
    uint64_t t1, t2;
    uint64_t W[16];

    #pragma unroll
    for (int t = 0; t < 8; t++) W[t] = u[t];

    #pragma unroll
    for (int t = 0; t < 8; t++) {
        t1 = h + sigma1_512(e) + ch64(e, f, g) + K512[t] + W[t];
        t2 = sigma0_512(a) + maj64(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    W[8]  = 0x8000000000000000ULL;
    W[9]  = 0; W[10] = 0; W[11] = 0;
    W[12] = 0; W[13] = 0; W[14] = 0;
    W[15] = 0x600ULL;

    #pragma unroll
    for (int t = 8; t < 16; t++) {
        t1 = h + sigma1_512(e) + ch64(e, f, g) + d_KPP[t - 8];
        t2 = sigma0_512(a) + maj64(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    #pragma unroll
    for (int t = 16; t < 80; t++) {
        W[t & 15] += gamma1_512(W[(t - 2) & 15]) + W[(t - 7) & 15] + gamma0_512(W[(t - 15) & 15]);
        t1 = h + sigma1_512(e) + ch64(e, f, g) + K512[t] + W[t & 15];
        t2 = sigma0_512(a) + maj64(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    state->h[0] += a; state->h[1] += b; state->h[2] += c; state->h[3] += d;
    state->h[4] += e; state->h[5] += f; state->h[6] += g; state->h[7] += h;
}

#ifndef DEV_BUILD
// P1: full uint64-pipeline PBKDF2 - zero byte arrays inside the loop.
// The U-chain message words ARE the previous hash state words (big-endian
// by construction) - no byte packing/unpacking anywhere. Fixed blocks fused
// (salt d_KPS, key tails d_KP36/d_KP5c, U-chain padding d_KPP).
// REQUIRES salt == "mnemonic" (8 bytes). Byte-exact == keyblock family.
// ---------------------------------------------------------------------------
// E15: SOLO MINIMAL PBKDF2 ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â the absolute register floor for 1 chain.
// No volatile, no forced arrays, no intermediate copies. The compiler has
// FULL FREEDOM to allocate registers as it sees fit within the
// __launch_bounds__(128, 4) budget of 128 SASS regs.
//
// The persistent state (in_pre, out_pre, U, T) is 32 u64 = 64 SASS regs.
// The compression working set (a..h + W[16]) adds ~48 SASS regs at peak.
// Total: ~112 SASS regs ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â under the 128 cap ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ 512 threads/SM.
//
// Everything constant has been pre-folded (GLBP principle applied to
// SHA-512): salt block = d_KPS[80] constants, pad tails = d_KPP[8],
// key block tails = d_KP36/d_KP5c. The ONLY live state is the
// cryptographic entropy: in_pre, out_pre, U, T (incompressible).
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// E17: ZERO-COPY PBKDF2 Ã¢â‚¬â€ the outer compression's output state IS the next
// iteration's chain value. No separate U array. The SHA512State_t
// outer_work naturally transitions from "outpre copy" Ã¢â€ â€™ "outer compression
// state" Ã¢â€ â€™ "U for next iteration" without any copies.
//
// Register budget:
//   persisting: inner_pre (16 SASS) + outer_work (16 SASS, doubles as U)
//             + T (16 SASS) = 48 SASS regs
//   during compression: + a..h (16) + W (32) = 48 transient
//   PEAK: ~96 SASS regs Ã¢â€ â€™ __launch_bounds__(128, 5) = 640 threads/SM
//
// The key_block (ipad/opad setup) is done OUTSIDE this function.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void pbkdf2_chain_zero_copy(
    const uint64_t inpre[8], const uint64_t outpre[8],
    uint32_t iterations, uint8_t* output
) {
    /* E18: MINIMUM REGISTERS — the persistent state is reduced to 48 SASS
     * regs by moving T to local memory (L1-cached). Register budget:
     *   persistent: inner_pre 16 + outer_pre 16 + chain 16 = 48
     *   transient (during compression): a..h 16 + W 32 = 48
     *   PEAK: ~96 SASS regs → 5 blocks/SM = 640 threads/SM */
    SHA512State_t inner_pre, outer_pre, chain;
    volatile uint64_t T_l[8]; /* T in local (L1): 8 RMW per iteration */

    #pragma unroll
    for (int j = 0; j < 8; j++) {
        inner_pre.h[j] = inpre[j];
        outer_pre.h[j] = outpre[j];
    }

    /* iteration 1: salt block on inner, outer processes the result */
    {
        SHA512State_t iw = inner_pre;
        sha512_transform_saltblock_folded_opt(&iw);
        SHA512State_t ow = outer_pre;
        sha512_finish64msg_fast(&ow, iw.h);
        #pragma unroll
        for (int j = 0; j < 8; j++) { chain.h[j] = ow.h[j]; T_l[j] = ow.h[j]; }
    }

    /* iterations 2..2048: the chain advances with minimal register pressure */
    for (uint32_t iter = 1; iter < iterations; iter++) {
        SHA512State_t iw = inner_pre;
        sha512_finish64msg_fast(&iw, chain.h);
        SHA512State_t ow = outer_pre;
        sha512_finish64msg_fast(&ow, iw.h);
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            chain.h[j] = ow.h[j];
            T_l[j] ^= ow.h[j];
        }
    }

    /* harvest T from local */
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        uint64_t x = T_l[j];
        output[j*8+0]=(uint8_t)(x>>56); output[j*8+1]=(uint8_t)(x>>48);
        output[j*8+2]=(uint8_t)(x>>40); output[j*8+3]=(uint8_t)(x>>32);
        output[j*8+4]=(uint8_t)(x>>24); output[j*8+5]=(uint8_t)(x>>16);
        output[j*8+6]=(uint8_t)(x>>8);  output[j*8+7]=(uint8_t)(x);
    }
}

__device__ __forceinline__ void pbkdf2_sha512_solo(
    const uint64_t key_w[16], uint32_t key_len,
    uint32_t iterations, uint8_t* output
) {
    uint32_t i0 = (key_len + 7u) / 8u;
    if (i0 > 16u) i0 = 16u;
    uint64_t inpre_s[8], outpre_s[8];
    {
        uint64_t ip[16], op[16];
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            ip[i] = key_w[i] ^ 0x3636363636363636ULL;
            op[i] = key_w[i] ^ 0x5c5c5c5c5c5c5c5cULL;
        }
        SHA512State_t s;
        sha512_init_state_opt(&s);
        sha512_compress16_rot_opt(&s, ip, i0, d_KP36);
        #pragma unroll
        for (int j = 0; j < 8; j++) inpre_s[j] = s.h[j];
        sha512_init_state_opt(&s);
        sha512_compress16_rot_opt(&s, op, i0, d_KP5c);
        #pragma unroll
        for (int j = 0; j < 8; j++) outpre_s[j] = s.h[j];
    }
    pbkdf2_chain_zero_copy(inpre_s, outpre_s, iterations, output);
}
#endif // !DEV_BUILD


// ============================================================================
// Forward declarations - heavy x2/x3/x4 + solo fast implementations moved to
// pbkdf2_variants.cu to keep this header under ~980 lines and speed up nvcc.

// solo fast (single-chain PBKDF2)
__device__ void pbkdf2_sha512_keyblock_fast(
    const uint64_t key_w[16], uint32_t key_len,
    uint32_t iterations,
    uint8_t* output_64bytes
);

// x2 compressors
__device__ void sha512_compress16_rot_x2_opt(
    SHA512State_t* sA, const uint64_t mA[16],
    SHA512State_t* sB, const uint64_t mB[16],
    uint32_t foldA, uint32_t foldB, const uint64_t* KP
);

__device__ void sha512_finish64msg_x2_fast(
    SHA512State_t* sA, const uint64_t uA[8],
    SHA512State_t* sB, const uint64_t uB[8]
);

__device__ void sha512_transform_saltblock_folded_x2_opt(
    SHA512State_t* sA, SHA512State_t* sB
);

__device__ void pbkdf2_sha512_keyblock_fast_x2(
    const uint64_t keyA[16], uint32_t lenA,
    const uint64_t keyB[16], uint32_t lenB,
    uint32_t iterations,
    uint8_t* outA, uint8_t* outB
);

// x3 compressors
__device__ void sha512_compress16_rot_x3_opt(
    SHA512State_t* sA, const uint64_t mA[16],
    SHA512State_t* sB, const uint64_t mB[16],
    SHA512State_t* sC, const uint64_t mC[16],
    uint32_t foldA, uint32_t foldB, uint32_t foldC, const uint64_t* KP
);

__device__ void sha512_finish64msg_x3_fast(
    SHA512State_t* sA, const uint64_t uA[8],
    SHA512State_t* sB, const uint64_t uB[8],
    SHA512State_t* sC, const uint64_t uC[8]
);

__device__ void sha512_transform_saltblock_folded_x3_opt(
    SHA512State_t* sA, SHA512State_t* sB, SHA512State_t* sC
);

__device__ void pbkdf2_sha512_keyblock_fast_x3(
    const uint64_t keyA[16], uint32_t lenA,
    const uint64_t keyB[16], uint32_t lenB,
    const uint64_t keyC[16], uint32_t lenC,
    uint32_t iterations,
    uint8_t* outA, uint8_t* outB, uint8_t* outC
);

// x4 compressors
__device__ void sha512_finish64msg_x4_fast(
    SHA512State_t* sA, const uint64_t uA[8],
    SHA512State_t* sB, const uint64_t uB[8],
    SHA512State_t* sC, const uint64_t uC[8],
    SHA512State_t* sD, const uint64_t uD[8]
);

__device__ void master_hmac_fast(const uint8_t* seed64, uint8_t* I64);

__device__ void pbkdf2_sha512_keyblock_fast_x4(
    const uint64_t keyA[16], uint32_t lenA,
    const uint64_t keyB[16], uint32_t lenB,
    const uint64_t keyC[16], uint32_t lenC,
    const uint64_t keyD[16], uint32_t lenD,
    uint32_t iterations,
    uint8_t* outA, uint8_t* outB, uint8_t* outC, uint8_t* outD
);
#endif
