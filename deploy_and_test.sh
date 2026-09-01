#!/bin/bash
# ONE-SHOT DEPLOY + TEST for the lanus E11 scanner
# Usage (from your machine): ssh <instance> "bash" < deploy_and_test.sh
set -e
cd /root/lanus 2>/dev/null || { mkdir -p /root/lanus && cd /root/lanus && tar xzf /root/lanus_deploy.tar.gz; }

# build only if binary missing or sources newer
if [ ! -f build/bip39_scanner ] || [ -n "$(find src -newer build/bip39_scanner 2>/dev/null | head -1)" ]; then
    echo "[BUILD] compiling..."
    make clean >/dev/null 2>&1
    make ARCH=sm_120 NVCC_FLAGS="-O3 -arch=sm_120 --use_fast_math -Xcompiler -O3" >/root/build.log 2>&1
    echo "[BUILD] done rc=$?"
fi

echo "=== SELFTEST ==="
./build/bip39_scanner -words my_words_fixed.txt -a target.txt --selftest 2>&1 | grep -E 'FOUND|MATCH|RESULT'

echo "=== THROUGHPUT (full occupancy) ==="
timeout 75 stdbuf -o0 ./build/bip39_scanner -words my_words_fixed.txt -a target.txt --depth 2 --batch 348160 --k2blocks 1360 2>&1 | grep -E 'E6|FOUND'
echo "=== reference points: old real max ~0.4-0.6M deriv/s | E8-wide ~1-2.5M | target 4x = ~2.4-2.6M ==="
