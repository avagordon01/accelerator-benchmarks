#!/usr/bin/env bash

set -o errexit

export CXX=/usr/bin/clang++

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

prime-run \
    ./build/main \
    --benchmark_out=output.json \
    --benchmark_out_format=json
python plot.py -f output.json --logx --logy
