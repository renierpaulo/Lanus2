# Lanus2

GPU-accelerated BIP39 seed phrase recovery scanner built on a compressed register-level architecture (GLBP technique).

## Architecture

Monolithic kernel design: each GPU thread is a fully self-contained scanner that generates, filters, and derives candidate phrases without leaving the thread.

```
THREAD (autonomous):
  generate base (Lemire in-thread)
  → enumerate 2048 words (checksum in-thread) → 128 valid
  → 128 PBKDF2 chains (x2 pairs, all in registers, zero local traffic)
    → BIP32 m/44'/0'/0'/0/0 (windowed EC + addition-chain inverse)
    → hash160 → compare target → FOUND or next
```

## Key optimizations

| Technique | Detail |
|---|---|
| WLBP | 2048 BIP39 words compressed 32768→5784 bytes (5.67x), bit-packed with GLBP discipline |
| Representative table | Host-built 2048×u64 table, threads access via __ldg |
| Rotating W[16] | SHA-512 schedule in registers (W[80] eliminated everywhere) |
| Constant folding | Salt block d_KPS[80], pad tails d_KPP[8], key tails d_KP36/KP5c — all in __constant__ |
| Zero-copy chain | Outer compression output IS the next chain value (no separate U array) |
| Windowed EC | 64×15 precomputed G table on device, ~60 point_adds (no doubles) |
| Addition-chain inverse | 253 sqr + 14 mul (vs ~510 square-and-multiply) |
| Solo minimal | Per-candidate transient prefix, minimum register state (~96 SASS regs peak) |
| 128-thread blocks | Register quantization: 3 blocks/SM = 384 threads/SM |

## Build

```bash
make clean && make ARCH=sm_120 NVCC_FLAGS="-O3 -arch=sm_120 --use_fast_math -Xcompiler -O3"
```

## Run

```bash
# Selftest (validates full pipeline against documented target)
./build/bip39_scanner -words my_words_fixed.txt -a target.txt --selftest

# Scan (full occupancy)
./build/bip39_scanner -words my_words_fixed.txt -a target.txt --depth 1 --batch 348160 --k2blocks 1360
```

## CLI options

| Flag | Description |
|---|---|
| `-words FILE` | BIP39 word candidates (required) |
| `-a FILE` | Target address(es) in base58 (required) |
| `--depth N` | 1=solo (min regs, max occupancy), 2=x2 pairs, 3=x3, 0=wide (128 simultaneous) |
| `--batch N` | Total bases per launch (default 4.19M) |
| `--k2blocks N` | Number of thread blocks (default 512) |
| `--selftest` | Run known-target validation and exit |

## Tests

Host-side test suite (13 files) validates every pipeline stage byte-exact against the classic BIP39 implementation and the documented target seed `4cd832a5...` / hash160 `232fb8a4...`.

```bash
# On MSVC:
cl /nologo /O2 /EHsc tests/test_e17_zerocopy.cpp /Fe:test.exe
test.exe wordlist.txt
```

## Measured performance (RTX 5090)

| Metric | Value |
|---|---|
| Bases/s | 67.8M |
| Full derivations/s | 4.32M |
| Candidate checksums/s | 138.5M |
| SHA-512 compressions/s | ~4.1 billion |
