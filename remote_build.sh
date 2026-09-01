#!/bin/bash
cd /root/lanus
make clean >/dev/null 2>&1
make ARCH=sm_120 NVCC_FLAGS="-O3 -arch=sm_120 --use_fast_math -Xcompiler -O3" > /root/build.log 2>&1
echo "BUILD_DONE rc=$?" >> /root/build.log
