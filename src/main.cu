/*
 * BIP39 CUDA Scanner v6.0 - ULTRA SPEED (Lanus2 + lanus CLI parity)
 *
 * Architecture:
 * 1. GPU generates phrase bases (unrank bijetivo no -exh, sorteio no random)
 * 2. Position-12 enumerated directly-valid (checksum computed, not probed)
 * 3. Valid candidates go through PBKDF2 + BIP32 + EC (word-shaped pipeline)
 * 4. Compare against targets / bloom filter (BTC hash160 or ETH keccak addr)
 *
 * CLI (lanus original parity):
 *   -words F -a F [-req N] [--bloom MB] [-coin btc|eth] [-gpus N]
 *   [-wild N] [-pin POS:WORD]... [-exh] [-resume K]
 * plus Lanus2 tuning flags: [--batch N] [--k2blocks N] [--selftest] [--depth N]
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>
#include <thread>
#include <vector>
#include <atomic>
#include <mutex>
#include <algorithm>

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
// Configuration
// ============================================================================
#define BATCH_SIZE_DEFAULT (4 * 1024 * 1024)  // 4.19M candidates/batch (--batch N overrides)
#define MAX_WORDS 40
#define PBKDF2_ITERATIONS 2048
#define DEBUG_MODE 0  // Set to 1 to enable debug output

// Global counters
std::atomic<uint64_t> g_permutations_tested(0);
std::atomic<uint64_t> g_valid_checksums(0);
std::atomic<uint64_t> g_addresses_checked(0);
std::atomic<uint32_t> g_found(0);
std::atomic<bool> g_stop(false);
std::mutex g_print_mutex;

// ============================================================================
// Main-scoped device constants
// (curve / hash / PBKDF2 / WLBP / byte_elim constants now live in their own
//  headers — single-TU build, include guards prevent duplicate definitions)
// ============================================================================
__constant__ uint16_t d_word_indices[MAX_WORDS];
__constant__ uint32_t d_word_count;
__constant__ uint64_t d_factorials[25];

// P2 legacy pool (no longer used by generation; kept for layout stability)
__constant__ uint16_t d_req_words[4];
__constant__ uint16_t d_avail_words[MAX_WORDS];
__constant__ uint32_t d_avail_count;

// lanus CLI parity constants (used by kernels.cu generation + byte_elim compare)
__constant__ uint32_t d_required_count;   // -req N
__constant__ uint32_t d_wild_count;       // -wild N
__constant__ uint16_t d_pin[12];          // -pin POS:WORD (0xFFFF = free)
__constant__ uint32_t d_pin_count;
__constant__ uint64_t d_perm_free;        // (11 - pins_in_base)!
__constant__ uint64_t d_binom[41][13];    // C(n,k) for the combinatorial unrank
__constant__ uint32_t d_exhaustive;       // 1 = -exh unrank, 0 = random draw

// ============================================================================
// Kernel declarations (definitions in kernels.cu)
// ============================================================================
__global__ void kernel_scanner_solo(uint64_t start_k, uint32_t bases_per_thread,
    uint32_t* found_count, uint8_t* found_privkeys, uint16_t* found_indices);
#ifndef DEV_BUILD
__global__ void kernel_scanner_x2(uint64_t start_k, uint32_t bases_per_thread,
    uint32_t* found_count, uint8_t* found_privkeys, uint16_t* found_indices);
__global__ void kernel_scanner_x3(uint64_t start_k, uint32_t bases_per_thread,
    uint32_t* found_count, uint8_t* found_privkeys, uint16_t* found_indices);
__global__ void kernel_scanner_wide(uint64_t start_k, uint32_t bases_per_thread,
    uint32_t* found_count, uint8_t* found_privkeys, uint16_t* found_indices);
#endif // !DEV_BUILD
__global__ void kernel_selftest(uint32_t* found_count, uint8_t* found_privkeys,
    uint16_t* found_indices);

// ============================================================================
// Host functions
// ============================================================================

// Checa uma chamada CUDA e aborta dizendo onde falhou (ported from lanus).
#define CUDA_CHECK(call) do {                                                  \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
        printf("\n[X] CUDA falhou em %s:%d -> %s\n    chamada: %s\n",          \
               __FILE__, __LINE__, cudaGetErrorString(_e), #call);             \
        return 1;                                                              \
    }                                                                          \
} while (0)

void load_wordlist(const char* filename, char wordlist[2048][16]) {
    FILE* f = fopen(filename, "r");
    if (!f) {
        printf("Error: Cannot open wordlist %s\n", filename);
        exit(1);
    }

    char line[64];
    int idx = 0;
    while (fgets(line, sizeof(line), f) && idx < 2048) {
        line[strcspn(line, "\r\n")] = 0;
        memset(wordlist[idx], 0, 16);
        strncpy(wordlist[idx], line, 15);
        idx++;
    }
    fclose(f);
    printf("First word in list: '%s'\n", wordlist[0]);
    printf("Last word (idx-1): '%s'\n", wordlist[idx-1]);
    printf("Loaded %d words from wordlist\n", idx);
}

void load_target_words(const char* filename, char wordlist[2048][16], uint16_t* indices, uint32_t* count) {
    FILE* f = fopen(filename, "r");
    if (!f) {
        printf("Error: Cannot open words file %s\n", filename);
        exit(1);
    }

    char word[64];
    *count = 0;
    while (fscanf(f, "%s", word) == 1 && *count < MAX_WORDS) {
        bool found_match = false;
        for (int i = 0; i < 2048; i++) {
            if (strcmp(word, wordlist[i]) == 0) {
                indices[*count] = i;
                (*count)++;
                found_match = true;
                break;
            }
        }
        if (!found_match) {
             printf("Warning: Word '%s' not found in wordlist!\n", word);
        }
    }
    fclose(f);
    printf("Loaded %u target words\n", *count);
}

uint64_t factorial(int n) {
    uint64_t f = 1;
    for (int i = 2; i <= n; i++) f *= i;
    return f;
}

// ============================================================================
// Aviso por Telegram (opcional, configurado por variavel de ambiente)
//   TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID  -> ativa
//   TELEGRAM_FULL=1                       -> inclui frase e chave na mensagem
// ============================================================================
static bool telegram_ativo() {
    const char* t = getenv("TELEGRAM_BOT_TOKEN");
    const char* c = getenv("TELEGRAM_CHAT_ID");
    return t && c && *t && *c;
}

static bool telegram_manda(const char* texto) {
    const char* tok  = getenv("TELEGRAM_BOT_TOKEN");
    const char* chat = getenv("TELEGRAM_CHAT_ID");
    if (!tok || !chat || !*tok || !*chat) return false;

    // texto vai por arquivo: evita qualquer problema de escape no shell
    char tmp[] = "/tmp/lanus_tg_XXXXXX";
    int fd = mkstemp(tmp);
    if (fd < 0) return false;
    FILE* f = fdopen(fd, "w");
    if (!f) { close(fd); unlink(tmp); return false; }
    fputs(texto, f);
    fclose(f);

    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
             "curl -s -m 25 -o /dev/null -X POST "
             "'https://api.telegram.org/bot%s/sendMessage' "
             "-d chat_id=%s --data-urlencode text@%s",
             tok, chat, tmp);
    int rc = system(cmd);
    unlink(tmp);
    return rc == 0;
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
    printf("============================================================\n");
    printf("  BIP39 CUDA Scanner v6.0 - ULTRA SPEED MODE (Lanus2)\n");
    printf("============================================================\n");

    if (argc < 3) {
        printf("Usage: %s -words <words.txt> -a <addresses.txt> [-req N] [--bloom <MB>] [-coin btc|eth]\n", argv[0]);
        printf("  -coin S: btc (default) ou eth - moeda alvo\n");
        printf("  -req N : as N primeiras palavras do arquivo sao OBRIGATORIAS (default 4)\n");
        printf("  -gpus N: usar N GPUs (default: todas as detectadas)\n");
        printf("  -wild N: N palavras LIVRES da lista BIP39 completa (2048)\n");
        printf("  -pin POS:PALAVRA: fixa PALAVRA na posicao POS (1..12)\n");
        printf("           ex: -pin 1:hazard -pin 12:source\n");
        printf("           a palavra tem de estar entre as -req; cada pin divide o espaco\n");
        printf("           cada curinga multiplica o espaco por 2048 (max 3)\n");
        printf("  -exh   : varredura EXAUSTIVA (cobertura garantida)\n");
        printf("  -resume K : retoma a varredura exaustiva a partir de k=K\n");
        printf("  --batch N / --k2blocks N / --selftest / --depth N : tuning Lanus2\n");
        return 1;
    }

    cudaDeviceSetLimit(cudaLimitStackSize, 49152);
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1024 * 1024 * 32);

    const char* words_file = NULL;
    const char* addr_file = NULL;
    uint64_t batch_size = BATCH_SIZE_DEFAULT;
    int k2_blocks = 512;
    int selftest = 0;
    int depth = 1;
    int bloom_mb = 0;
    int required_count = 4;
    int exhaustive = 0;
    int n_gpus = 0;       // 0 = usar todas as detectadas
    int wild_count = 0;   // -wild N: N palavras livres das 2048
    uint16_t h_pin[12];   // -pin POS:PALAVRA (0xFFFF = posicao livre)
    for (int i = 0; i < 12; i++) h_pin[i] = 0xFFFFu;
    char pin_txt[12][16];
    for (int i = 0; i < 12; i++) pin_txt[i][0] = 0;
    int pin_count = 0;
    unsigned long long resume_k = 0;
    int coin_type = 0;    // 0 = BTC (default), 60 = ETH

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-words") == 0 && i + 1 < argc) words_file = argv[++i];
        else if (strcmp(argv[i], "-a") == 0 && i + 1 < argc) addr_file = argv[++i];
        else if (strcmp(argv[i], "--bloom") == 0 && i + 1 < argc) bloom_mb = atoi(argv[++i]);
        else if (strcmp(argv[i], "-req") == 0 && i + 1 < argc) required_count = atoi(argv[++i]);
        else if (strcmp(argv[i], "-exh") == 0) exhaustive = 1;
        else if (strcmp(argv[i], "-coin") == 0 && i + 1 < argc) {
            ++i;
            if (strcmp(argv[i], "btc") == 0 || strcmp(argv[i], "BTC") == 0) coin_type = 0;
            else if (strcmp(argv[i], "eth") == 0 || strcmp(argv[i], "ETH") == 0) coin_type = 60;
            else { printf("Error: -coin aceita btc ou eth\n"); return 1; }
        }
        else if (strcmp(argv[i], "-gpus") == 0 && i + 1 < argc) n_gpus = atoi(argv[++i]);
        else if (strcmp(argv[i], "-wild") == 0 && i + 1 < argc) wild_count = atoi(argv[++i]);
        else if (strcmp(argv[i], "-pin") == 0 && i + 1 < argc) {
            const char* spec = argv[++i];
            const char* dp = strchr(spec, ':');
            if (!dp) { printf("Error: -pin espera POSICAO:PALAVRA (ex: -pin 1:hazard)\n"); return 1; }
            int pos = atoi(spec);
            if (pos < 1 || pos > 12) { printf("Error: -pin posicao %d fora de 1..12\n", pos); return 1; }
            if (pin_txt[pos-1][0]) { printf("Error: -pin posicao %d definida duas vezes\n", pos); return 1; }
            snprintf(pin_txt[pos-1], 16, "%s", dp + 1);
            pin_count++;
        }
        else if (strcmp(argv[i], "-resume") == 0 && i + 1 < argc) resume_k = strtoull(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) batch_size = atoll(argv[++i]);
        else if (strcmp(argv[i], "--k2blocks") == 0 && i + 1 < argc) k2_blocks = atoi(argv[++i]);
        else if (strcmp(argv[i], "--selftest") == 0) selftest = 1;
        else if (strcmp(argv[i], "--depth") == 0 && i + 1 < argc) depth = atoi(argv[++i]);
    }

    if (!words_file || !addr_file) {
        printf("Error: Missing required arguments\n");
        return 1;
    }
    if (batch_size < 65536) batch_size = 65536;

    // ------------------------------------------------------------------
    // Load addresses: BTC = base58; ETH = hex 0x... ou 40 chars
    // ------------------------------------------------------------------
    std::vector<uint8_t> target_hashes;
    int num_targets = 0;

    FILE* f_addr = fopen(addr_file, "r");
    if (f_addr) {
        char line[128];
        while (fgets(line, sizeof(line), f_addr)) {
            line[strcspn(line, "\r\n")] = 0;
            if (strlen(line) < 20) continue;

            uint8_t hash[20];
            bool okaddr = false;
            if (coin_type == 60 && strlen(line) >= 40) {
                // formato hex 0x... ou puro (40 chars)
                const char* hp = line;
                if (hp[0]=='0' && (hp[1]=='x'||hp[1]=='X')) hp += 2;
                if (strlen(hp) >= 40) okaddr = true;
                for (int q = 0; okaddr && q < 20; q++) {
                    char c1=hp[2*q], c2=hp[2*q+1];
                    auto hv=[](char c)->uint8_t{return (uint8_t)((c<='9')?c-'0':((c|32)-'a'+10));};
                    if (!((c1>='0'&&c1<='9')||(c1|32)>='a'&&(c1|32)<='f')) {okaddr=false;break;}
                    if (!((c2>='0'&&c2<='9')||(c2|32)>='a'&&(c2|32)<='f')) {okaddr=false;break;}
                    hash[q]=(uint8_t)((hv(c1)<<4)|hv(c2));
                }
            }
            if (!okaddr && coin_type == 0 && base58_decode_address(line, hash)) {
                okaddr = true;
            }
            if (okaddr) {
                for(int q=0;q<20;q++) target_hashes.push_back(hash[q]);
            }
        }
        fclose(f_addr);
        num_targets = (int)(target_hashes.size() / 20);
        printf("Loaded %d target addresses\n", num_targets);
    } else {
        printf("Error: Cannot open address file %s\n", addr_file);
        return 1;
    }

    // Load wordlist
    char wordlist[2048][16];
    load_wordlist("wordlist.txt", wordlist);

    // Load target words
    uint16_t h_word_indices[MAX_WORDS];
    uint32_t word_count;
    load_target_words(words_file, wordlist, h_word_indices, &word_count);

    // ------------------------------------------------------------------
    // -pin: resolve texto -> indice BIP39 e valida
    // ------------------------------------------------------------------
    if (pin_count > 0) {
        for (int p = 0; p < 12; p++) {
            if (!pin_txt[p][0]) continue;
            int idx = -1;
            for (int w = 0; w < 2048; w++)
                if (strcmp(pin_txt[p], wordlist[w]) == 0) { idx = w; break; }
            if (idx < 0) {
                printf("Error: -pin %d:%s -- '%s' nao e palavra BIP39\n",
                       p + 1, pin_txt[p], pin_txt[p]);
                return 1;
            }
            // tem de estar entre as obrigatorias, senao pode nao ser sorteada
            bool obrig = false;
            for (int r = 0; r < required_count && r < (int)word_count; r++)
                if (h_word_indices[r] == (uint16_t)idx) { obrig = true; break; }
            if (!obrig) {
                printf("Error: -pin %d:%s -- essa palavra precisa estar entre as %d\n"
                       "       obrigatorias (as primeiras do -words). Aumente o -req\n"
                       "       ou mova '%s' para o inicio do arquivo.\n",
                       p + 1, pin_txt[p], required_count, pin_txt[p]);
                return 1;
            }
            h_pin[p] = (uint16_t)idx;
        }
        printf("Posicoes fixas (-pin): ");
        for (int p = 0; p < 12; p++)
            if (h_pin[p] != 0xFFFFu) printf("%d:%s ", p + 1, wordlist[h_pin[p]]);
        printf("\n");
    }

    // ------------------------------------------------------------------
    // Validacoes (ported from lanus original)
    // ------------------------------------------------------------------
    if (required_count < 0 || required_count > 12) {
        printf("Error: -req deve estar entre 0 e 12 (recebido %d)\n", required_count);
        return 1;
    }
    if (word_count < 12) {
        printf("Error: o arquivo -words precisa de pelo menos 12 palavras (tem %u)\n", word_count);
        return 1;
    }
    if ((uint32_t)required_count > word_count) {
        printf("Error: -req %d maior que o total de palavras (%u)\n", required_count, word_count);
        return 1;
    }
    if (wild_count < 0 || wild_count > 3) {
        printf("Error: -wild deve estar entre 0 e 3 (recebido %d)\n", wild_count);
        return 1;
    }
    if (required_count + wild_count > 11) {
        printf("Error: -req %d + -wild %d deixa menos de 1 palavra livre (max 11)\n",
               required_count, wild_count);
        return 1;
    }
    {
        int precisa = 11 - required_count - wild_count;   // sorteadas do pool p/ base
        int tem = (int)word_count - required_count;
        if (tem < precisa) {
            printf("Error: pool tem %d palavras, precisa de %d\n", tem, precisa);
            return 1;
        }
    }

    // (11 - pins_na_base)! — pins na posicao 12 restringem a enumeracao
    int pin_base = 0;
    for (int p = 0; p < 11; p++) if (h_pin[p] != 0xFFFFu) pin_base++;
    uint64_t h_perm_free = 1ULL;
    for (int i = 2; i <= 11 - pin_base; i++) h_perm_free *= (uint64_t)i;

    // ------------------------------------------------------------------
    // Symbol uploads
    // ------------------------------------------------------------------
    uint32_t rc_ = (uint32_t)required_count;
    uint32_t wc_ = (uint32_t)wild_count;
    uint32_t pc_ = (uint32_t)pin_count;
    uint32_t ex_ = (uint32_t)exhaustive;
    uint32_t hc_ = (uint32_t)coin_type;
    CUDA_CHECK(cudaMemcpyToSymbol(d_required_count, &rc_, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_wild_count, &wc_, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_pin, h_pin, 12 * sizeof(uint16_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_pin_count, &pc_, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_perm_free, &h_perm_free, sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_exhaustive, &ex_, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_coin, &hc_, sizeof(uint32_t)));

    // C(n,k) para o unranking combinatorio
    static uint64_t h_binom[41][13];
    for (int a = 0; a <= 40; a++) {
        for (int b = 0; b <= 12; b++) {
            if (b == 0)      h_binom[a][b] = 1;
            else if (b > a)  h_binom[a][b] = 0;
            else             h_binom[a][b] = h_binom[a-1][b-1] + h_binom[a-1][b];
        }
    }
    CUDA_CHECK(cudaMemcpyToSymbol(d_binom, h_binom, sizeof(h_binom)));

    // espaco total de BASES (posicao 12 enumera ~128 validas por base)
    uint64_t total_space = h_binom[word_count - required_count][11 - required_count - wild_count]
                           * h_perm_free;
    for (int w = 0; w < wild_count; w++) total_space *= 2048ULL;

    uint64_t total_perms = factorial(word_count);
    printf("Word count: %u\n", word_count);
    if (wild_count > 0)
        printf("Curingas: %d palavra(s) LIVRE(s) da lista completa (2048)\n", wild_count);
    printf("Obrigatorias (%d): ", required_count);
    for (int i = 0; i < required_count; i++) printf("%s ", wordlist[h_word_indices[i]]);
    printf("\n");
    printf("Espaco de bases: %llu (x ~128 validas por base)\n",
           (unsigned long long)total_space);
    printf("Total permutations (12!): %llu\n", (unsigned long long)total_perms);

    // Prepare factorials
    uint64_t h_factorials[25];
    h_factorials[0] = 1;
    for (int i = 1; i <= 24; i++) h_factorials[i] = h_factorials[i-1] * (uint64_t)i;

    CUDA_CHECK(cudaMemcpyToSymbol(d_word_indices, h_word_indices, MAX_WORDS * sizeof(uint16_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_word_count, &word_count, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_factorials, h_factorials, 25 * sizeof(uint64_t)));

    wlbp_upload_repr_table();
    secp256k1_gwin_upload();

    uint64_t threads_total = (uint64_t)k2_blocks * 256;
    uint32_t bases_per_thread = (uint32_t)(batch_size / threads_total);
    if (bases_per_thread < 1) bases_per_thread = 1;
    uint64_t bases_per_launch = threads_total * (uint64_t)bases_per_thread;

    // ------------------------------------------------------------------
    // Worker por GPU. Tudo declarado aqui e LOCAL da thread.
    // ------------------------------------------------------------------
    auto worker = [&](int dev, uint64_t k_ini, uint64_t k_fim) -> int {
        CUDA_CHECK(cudaSetDevice(dev));
        cudaDeviceSetLimit(cudaLimitStackSize, 49152);
        cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1024 * 1024 * 32);

        // simbolos __constant__ existem por dispositivo: reenviar em cada um
        CUDA_CHECK(cudaMemcpyToSymbol(d_word_indices, h_word_indices, MAX_WORDS * sizeof(uint16_t)));
        CUDA_CHECK(cudaMemcpyToSymbol(d_word_count, &word_count, sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpyToSymbol(d_factorials, h_factorials, 25 * sizeof(uint64_t)));
        {
            uint32_t rc2 = (uint32_t)required_count;
            uint32_t wc2 = (uint32_t)wild_count;
            uint32_t pc2 = (uint32_t)pin_count;
            uint32_t ex2 = (uint32_t)exhaustive;
            uint32_t hc2 = (uint32_t)coin_type;
            CUDA_CHECK(cudaMemcpyToSymbol(d_required_count, &rc2, sizeof(uint32_t)));
            CUDA_CHECK(cudaMemcpyToSymbol(d_wild_count, &wc2, sizeof(uint32_t)));
            CUDA_CHECK(cudaMemcpyToSymbol(d_pin, h_pin, 12 * sizeof(uint16_t)));
            CUDA_CHECK(cudaMemcpyToSymbol(d_pin_count, &pc2, sizeof(uint32_t)));
            CUDA_CHECK(cudaMemcpyToSymbol(d_perm_free, &h_perm_free, sizeof(uint64_t)));
            CUDA_CHECK(cudaMemcpyToSymbol(d_binom, h_binom, sizeof(h_binom)));
            CUDA_CHECK(cudaMemcpyToSymbol(d_exhaustive, &ex2, sizeof(uint32_t)));
            CUDA_CHECK(cudaMemcpyToSymbol(d_coin, &hc2, sizeof(uint32_t)));
        }

        wlbp_upload_repr_table();
        secp256k1_gwin_upload();

        uint32_t* d_found_count;
        CUDA_CHECK(cudaMalloc(&d_found_count, sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(d_found_count, 0, sizeof(uint32_t)));

        uint8_t* d_found_privkeys;
        CUDA_CHECK(cudaMalloc(&d_found_privkeys, 100 * 32));

        uint16_t* d_found_indices;
        CUDA_CHECK(cudaMalloc(&d_found_indices, 100 * MAX_WORDS * sizeof(uint16_t)));

        // Target hashes + bloom (por dispositivo)
        uint32_t use_bloom = (bloom_mb > 0 && num_targets > 0) ? 1u : 0u;
        CUDA_CHECK(cudaMemcpyToSymbol(d_use_bloom, &use_bloom, sizeof(uint32_t)));

        if (num_targets > 0) {
            uint8_t* d_targets;
            CUDA_CHECK(cudaMalloc(&d_targets, target_hashes.size()));
            CUDA_CHECK(cudaMemcpy(d_targets, target_hashes.data(), target_hashes.size(), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpyToSymbol(d_target_hashes_ptr, &d_targets, sizeof(uint8_t*)));
            CUDA_CHECK(cudaMemcpyToSymbol(d_num_targets, &num_targets, sizeof(uint32_t)));
        }

        if (use_bloom) {
            // bloom: 1 bit por linha de acao; FNV-1a sobre os 5 words do digest
            uint64_t m_bits = (uint64_t)bloom_mb * 1024ULL * 1024ULL * 8ULL;
            uint32_t bk = 8;
            size_t nbytes = (size_t)(m_bits / 8);
            uint8_t* h_bits = (uint8_t*)calloc(nbytes, 1);
            if (!h_bits) { printf("[X] bloom alloc falhou\n"); return 1; }
            for (int t = 0; t < num_targets; t++) {
                const uint8_t* th = target_hashes.data() + t * 20;
                uint64_t h1 = 1469598103934665603ULL, h2 = 1469598103934665603ULL;
                for (int w = 0; w < 5; w++) {
                    uint32_t tw = ((uint32_t)th[w*4+0] << 24) | ((uint32_t)th[w*4+1] << 16) |
                                  ((uint32_t)th[w*4+2] << 8)  |  (uint32_t)th[w*4+3];
                    for (int b = 0; b < 4; b++) {
                        h1 ^= (tw >> 24) & 0xFFu; h1 *= 1099511628211ULL; tw <<= 8;
                    }
                }
                for (int w = 4; w >= 0; w--) {
                    uint32_t tw = ((uint32_t)th[w*4+0] << 24) | ((uint32_t)th[w*4+1] << 16) |
                                  ((uint32_t)th[w*4+2] << 8)  |  (uint32_t)th[w*4+3];
                    for (int b = 0; b < 4; b++) {
                        h2 ^= (tw >> 24) & 0xFFu; h2 *= 1099511628211ULL; tw <<= 8;
                    }
                }
                h2 ^= 0x5A5A5A5A5A5A5A5AULL;
                for (uint32_t j = 0; j < bk; j++) {
                    uint64_t combined = h1 + j * h2;
                    uint64_t bit_index = combined % m_bits;
                    h_bits[bit_index >> 3] |= (uint8_t)(1u << (bit_index & 7u));
                }
            }
            uint8_t* d_bits;
            CUDA_CHECK(cudaMalloc(&d_bits, nbytes));
            CUDA_CHECK(cudaMemcpy(d_bits, h_bits, nbytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpyToSymbol(d_bloom_bits, &d_bits, sizeof(uint8_t*)));
            CUDA_CHECK(cudaMemcpyToSymbol(d_bloom_m_bits, &m_bits, sizeof(uint64_t)));
            CUDA_CHECK(cudaMemcpyToSymbol(d_bloom_k, &bk, sizeof(uint32_t)));
            free(h_bits);
            printf("[GPU %d] Bloom: %d MB, %d hashes, %d alvos\n", dev, bloom_mb, bk, num_targets);
        }

        // tempo de RELOGIO (steady_clock: clock() mede CPU e infla com threads)
        auto start_time = std::chrono::steady_clock::now();

        // Single-kernel loop
        uint64_t k = k_ini;
        while (true) {
            if (g_stop.load()) break;                 // outra GPU achou
            if (exhaustive && k >= k_fim) {
                printf("\n[GPU %d] faixa varrida por completo.\n", dev);
                break;
            }
            uint64_t remaining = (exhaustive && k_fim > k) ? (k_fim - k) : ~0ULL;
            uint32_t bpt = bases_per_thread;
            if (remaining < threads_total * (uint64_t)bpt) {
                bpt = (uint32_t)(remaining / threads_total);
                if (bpt < 1) break;
            }
            uint64_t bpl = threads_total * (uint64_t)bpt;

            cudaMemset(d_found_count, 0, sizeof(uint32_t));

#ifdef DEV_BUILD
            // DEV build: solo kernel only (fast compile; --depth ignored)
            kernel_scanner_solo<<<k2_blocks, 128>>>(
                k, bpt,
                d_found_count, d_found_privkeys, d_found_indices
            );
#else
            if (depth == 1)
                kernel_scanner_solo<<<k2_blocks, 128>>>(
                    k, bpt,
                    d_found_count, d_found_privkeys, d_found_indices
                );
            else if (depth == 0)
                kernel_scanner_wide<<<k2_blocks, 256>>>(
                    k, bpt,
                    d_found_count, d_found_privkeys, d_found_indices
                );
            else if (depth == 3)
                kernel_scanner_x3<<<k2_blocks, 128>>>(
                    k, bpt,
                    d_found_count, d_found_privkeys, d_found_indices
                );
            else
                kernel_scanner_x2<<<k2_blocks, 128>>>(
                    k, bpt,
                    d_found_count, d_found_privkeys, d_found_indices
                );
#endif // DEV_BUILD
            {
                cudaError_t ke = cudaDeviceSynchronize();
                if (ke == cudaSuccess) ke = cudaGetLastError();
                if (ke != cudaSuccess) {
                    printf("\n[X] ERRO no kernel: %s\n", cudaGetErrorString(ke));
                    return 1;
                }
            }

            uint32_t found_now;
            CUDA_CHECK(cudaMemcpy(&found_now, d_found_count, sizeof(uint32_t), cudaMemcpyDeviceToHost));
            if (found_now > 0) {
                g_found.store(found_now);
                g_stop.store(true);
                printf("\n[+] FOUND %u MATCHES!\n", found_now);

                uint32_t nshow = found_now > 100 ? 100 : found_now;
                uint16_t h_fidx[100 * MAX_WORDS];
                uint8_t  h_fkey[100 * 32];
                CUDA_CHECK(cudaMemcpy(h_fidx, d_found_indices, (size_t)nshow * MAX_WORDS * sizeof(uint16_t), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_fkey, d_found_privkeys, (size_t)nshow * 32, cudaMemcpyDeviceToHost));

                FILE* ff = fopen("FOUND.txt", "a");
                for (uint32_t s = 0; s < nshow; s++) {
                    char phrase[512]; int pl = 0;
                    for (uint32_t w = 0; w < 12; w++) {
                        const char* wd = wordlist[h_fidx[s * MAX_WORDS + w]];
                        if (w) phrase[pl++] = ' ';
                        for (int c = 0; wd[c]; c++) phrase[pl++] = wd[c];
                    }
                    phrase[pl] = 0;

                    char keyhex[80];
                    for (int b = 0; b < 32; b++) sprintf(keyhex + b * 2, "%02x", h_fkey[s * 32 + b]);
                    keyhex[64] = 0;

                    printf("\n*** FRASE ENCONTRADA ***\n");
                    printf("  Frase   : %s\n", phrase);
                    printf("  Privkey : %s\n", keyhex);
                    if (coin_type == 60) printf("  Caminho : m/44'/60'/0'/0/0 (ETH)\n");
                    else printf("  Caminho : m/44'/0'/0'/0/0 (BTC)\n");
                    if (ff) fprintf(ff, "%s\t%s\n", phrase, keyhex);
                }
                if (ff) { fclose(ff); printf("\n  (tambem salvo em FOUND.txt)\n"); }

                // aviso no Telegram
                if (telegram_ativo()) {
                    const char* full = getenv("TELEGRAM_FULL");
                    char msg[1024];
                    if (full && *full == '1') {
                        char fr[512]; int fl = 0;
                        for (uint32_t w = 0; w < 12; w++) {
                            const char* wd = wordlist[h_fidx[w]];
                            if (w) fr[fl++] = ' ';
                            for (int c = 0; wd[c]; c++) fr[fl++] = wd[c];
                        }
                        fr[fl] = 0;
                        char kx[80];
                        for (int b = 0; b < 32; b++) sprintf(kx + b * 2, "%02x", h_fkey[b]);
                        kx[64] = 0;
                        snprintf(msg, sizeof(msg),
                                 "*** LANUS ACHOU ***\n\nFrase:\n%s\n\nPrivkey:\n%s\n\nCaminho: %s",
                                 fr, kx, coin_type == 60 ? "m/44'/60'/0'/0/0" : "m/44'/0'/0'/0/0");
                    } else {
                        snprintf(msg, sizeof(msg),
                                 "*** LANUS ACHOU ***\n%u resultado(s).\n"
                                 "A frase e a chave estao em FOUND.txt na maquina.\n"
                                 "(defina TELEGRAM_FULL=1 se quiser receber aqui)", nshow);
                    }
                    printf("  Telegram: %s\n", telegram_manda(msg) ? "avisado" : "FALHOU o envio");
                }
                fflush(stdout);
                break;
            }

            g_permutations_tested += bpl * 128;
            g_valid_checksums += bpl * 128;

            k += bpl;

            if (dev == 0 && (k / bases_per_launch) % 10 == 0) {
                double elapsed = std::chrono::duration<double>(
                                     std::chrono::steady_clock::now() - start_time).count();
                double rate_now = g_permutations_tested.load() / (elapsed > 0 ? elapsed : 1);
                if (exhaustive) {
                    double done = (double)(k - k_ini);
                    double span = (double)(k_fim - k_ini);
                    double pct = span > 0 ? 100.0 * done / span : 100.0;
                    double eta_h = rate_now > 0 ? (span - done) / rate_now / 3600.0 : 0;
                    printf("\n[EXAUSTIVO] k=%llu (%.4f%%) | restam ~%.1f h | %llu derivacoes | %.2f M/s\n",
                           (unsigned long long)k, pct, eta_h,
                           (unsigned long long)g_permutations_tested.load(), rate_now / 1e6);
                    FILE* pf = fopen("progress.txt", "w");
                    if (pf) { fprintf(pf, "%llu\n", (unsigned long long)k); fclose(pf); }
                } else {
                    printf("\n[E6 SCANNER] Bases: %llu | Derivacoes: %llu | %.2f M/s | Elapsed: %.1fs\n",
                           (unsigned long long)g_permutations_tested.load(),
                           (unsigned long long)g_valid_checksums.load(),
                           rate_now / 1e6, elapsed);
                }
                fflush(stdout);
            }
        }

        cudaFree(d_found_count);
        cudaFree(d_found_privkeys);
        cudaFree(d_found_indices);
        return 0;
    };  // fim do worker

    // ------------------------------------------------------------------
    // Selftest (GPU 0) antes dos workers
    // ------------------------------------------------------------------
    if (selftest) {
        CUDA_CHECK(cudaSetDevice(0));
        printf("\n[SELFTEST] target base -> full monolithic pipeline on silicon...\n");
        uint32_t* d_found_count;
        CUDA_CHECK(cudaMalloc(&d_found_count, sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(d_found_count, 0, sizeof(uint32_t)));
        uint8_t* d_found_privkeys;
        CUDA_CHECK(cudaMalloc(&d_found_privkeys, 100 * 32));
        uint16_t* d_found_indices;
        CUDA_CHECK(cudaMalloc(&d_found_indices, 100 * MAX_WORDS * sizeof(uint16_t)));

        kernel_selftest<<<1, 32>>>(d_found_count, d_found_privkeys, d_found_indices);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) { printf("[SELFTEST] CUDA error: %s\n", cudaGetErrorString(err)); return 1; }
        uint32_t found_now;
        CUDA_CHECK(cudaMemcpy(&found_now, d_found_count, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        if (found_now > 0) {
            uint8_t h_priv[100 * 32];
            uint16_t h_idx[100 * 12];
            CUDA_CHECK(cudaMemcpy(h_priv, d_found_privkeys, found_now * 32, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_idx, d_found_indices, found_now * 12 * sizeof(uint16_t), cudaMemcpyDeviceToHost));
            const uint8_t expected_priv[32] = {
                0x20,0xbb,0xcd,0xa6,0x71,0xe9,0xf6,0x6c,0xfe,0xde,0xdb,0x3c,0xc6,0x76,0x35,0x84,
                0xc0,0x2d,0xb5,0x53,0x0c,0xb1,0x97,0x5d,0xfa,0x82,0xd0,0x03,0xe5,0x23,0x3f,0xaf
            };
            for (uint32_t f = 0; f < found_now && f < 100; f++) {
                printf("[SELFTEST] FOUND #%u\n  privkey: ", f);
                for (int i = 0; i < 32; i++) printf("%02x", h_priv[f * 32 + i]);
                printf("\n  indices: ");
                for (int w = 0; w < 12; w++) printf("%u ", h_idx[f * 12 + w]);
                printf("\n");
                int ok = memcmp(h_priv + f * 32, expected_priv, 32) == 0;
                printf("[SELFTEST] privkey vs documented: %s\n", ok ? "MATCH" : "MISMATCH");
                if (!ok) { printf("[SELFTEST] RESULT: FAIL\n"); return 1; }
            }
            printf("[SELFTEST] RESULT: PASS - full pipeline verified on silicon\n");
        } else {
            printf("[SELFTEST] RESULT: FAIL - no match found\n");
            return 1;
        }
        cudaFree(d_found_count);
        cudaFree(d_found_privkeys);
        cudaFree(d_found_indices);
        if (exhaustive || required_count != 4 || wild_count != 0 || pin_count != 0 || coin_type != 0) {
            return 0;   // selftest-only run com flags: nao inicia o scan
        }
    }

    printf("\n============================================================\n");
    printf("E6 MONOLITHIC SCANNER - one thread does the whole process\n");
    printf("Geometry: %d blocks x 128 threads x %u bases/thread = %llu bases/launch\n",
           k2_blocks, bases_per_thread, (unsigned long long)bases_per_launch);
    if (exhaustive)
        printf("Modo EXAUSTIVO: cobertura garantida; retomar com -resume <k>\n");
    else
        printf("Modo ALEATORIO (sorteio com reposicao, sem garantia de cobertura)\n");
    if (telegram_ativo()) {
        char msg[512];
        snprintf(msg, sizeof(msg),
                 "lanus2: busca iniciada\nalvo: %s\nobrigatorias: %d  curingas: %d\n"
                 "(esta e a mensagem de teste - o aviso do achado usa o mesmo canal)",
                 addr_file, required_count, wild_count);
        printf("Telegram: %s\n", telegram_manda(msg) ? "configurado, teste enviado"
                                                      : "configurado, mas o envio FALHOU");
    } else {
        printf("Telegram: desativado (defina TELEGRAM_BOT_TOKEN e TELEGRAM_CHAT_ID)\n");
    }
    printf("Press Ctrl+C to stop\n");
    printf("============================================================\n\n");

    // ------------------------------------------------------------------
    // Dispara uma thread por GPU, fatiando o espaco entre elas.
    // ------------------------------------------------------------------
    int dev_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    if (dev_count < 1) { printf("Nenhuma GPU CUDA encontrada\n"); return 1; }
    int G = (n_gpus > 0 && n_gpus <= dev_count) ? n_gpus : dev_count;

    printf("Usando %d GPU(s) de %d disponiveis\n", G, dev_count);
    for (int d = 0; d < G; d++) {
        cudaDeviceProp p;
        if (cudaGetDeviceProperties(&p, d) == cudaSuccess)
            printf("  GPU %d: %s\n", d, p.name);
    }

    uint64_t base_k = exhaustive ? (uint64_t)resume_k : 0;
    uint64_t restante = (exhaustive && total_space > base_k) ? (total_space - base_k) : 0;
    uint64_t fatia = (G > 0) ? (restante / (uint64_t)G) : restante;

    if (exhaustive) {
        printf("Espaco fatiado entre as GPUs (%llu bases cada)\n",
               (unsigned long long)fatia);
    }
    printf("============================================================\n\n");

    std::vector<std::thread> ths;
    for (int d = 0; d < G; d++) {
        uint64_t ki, kf;
        if (exhaustive) {
            ki = base_k + fatia * (uint64_t)d;
            kf = (d == G - 1) ? total_space : (base_k + fatia * (uint64_t)(d + 1));
        } else {
            // no modo aleatorio a semente vem de k: deslocar evita caminhos iguais
            ki = base_k + (uint64_t)d * 0x1000000000ULL;
            kf = ~0ULL;
        }
        ths.emplace_back([&worker, d, ki, kf]() { worker(d, ki, kf); });
    }
    for (auto& t : ths) t.join();

    printf("\n\n============================================================\n");
    printf("SCAN COMPLETE\n");
    printf("Total permutations: %llu\n", (unsigned long long)g_permutations_tested.load());
    printf("Valid checksums: %llu\n", (unsigned long long)g_valid_checksums.load());
    printf("Found: %u\n", g_found.load());
    printf("============================================================\n");

    return 0;
}

// ============================================================================
// Single-TU: include device code directly (no -rdc=true needed)
// ============================================================================
#include "pbkdf2_variants.cu"
#include "kernels.cu"
