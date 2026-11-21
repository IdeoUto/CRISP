#!/bin/sh
set -eu
TASK_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cmake -S "$TASK_ROOT/../CRISP" -B "$TASK_ROOT/build-fss" -DCMAKE_BUILD_TYPE=Release -DOPENSSL_ROOT_DIR=/opt/homebrew/opt/openssl
cmake --build "$TASK_ROOT/build-fss" -j4
cmake -S "$TASK_ROOT/../CKKS-master" -B "$TASK_ROOT/build-ckks" -DCMAKE_BUILD_TYPE=Release -DSEAL_DIR=/usr/local/lib/cmake/SEAL-3.6 -DOPENSSL_ROOT_DIR=/opt/homebrew/opt/openssl
cmake --build "$TASK_ROOT/build-ckks" -j4
ctest --test-dir "$TASK_ROOT/build-fss" --output-on-failure
ctest --test-dir "$TASK_ROOT/build-ckks" --output-on-failure
mkdir -p "$TASK_ROOT/../CRISP/bin" "$TASK_ROOT/../CKKS-master/bin"
for task_exe in crisp_relu crisp_pool test_fss; do
 cp "$TASK_ROOT/build-fss/$task_exe" "$TASK_ROOT/../CRISP/bin/$task_exe"
done
for task_exe in crisp_linear simc_ring32 test_ckks; do
 cp "$TASK_ROOT/build-ckks/$task_exe" "$TASK_ROOT/../CKKS-master/bin/$task_exe"
done
