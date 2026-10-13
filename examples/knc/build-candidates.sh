#!/usr/bin/env bash
# Run inside the isolated MPSS build environment. Never installs a runtime.
set -euo pipefail
if [[ $# != 3 ]]; then
    echo "Usage: $0 SOURCE BUILD NEW_OUTPUT" >&2
    exit 1
fi
source=$(realpath "$1")
build=$(realpath -m "$2")
output=$(realpath -m "$3")
test ! -e "$output"
mkdir -p "$output"
cc=/usr/linux-k1om-4.7/bin/x86_64-k1om-linux-gcc
readelf=/usr/linux-k1om-4.7/bin/x86_64-k1om-linux-readelf
"$cc" --version > "$output/compiler.txt"
for profile in 3120A 5110P 7120P; do
    case $profile in 3120A) l1=4; l2=16;; *) l1=8; l2=32;; esac
    dest=$output/$profile
    mkdir -p "$dest/bin" "$dest/lib"
    cmake -S "$source" -B "$build" \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE="$source/cmake/k1om.cmake" \
        -DGGML_KNC=ON -DGGML_KNC_PROFILE="$profile" \
        -DGGML_KNC_PREFETCH_L1="$l1" -DGGML_KNC_PREFETCH_L2="$l2" \
        -DGGML_KNC_PREFETCH_STEP=64 -DGGML_KNC_PREFETCH_Q8=ON \
        -DGGML_NATIVE=OFF -DGGML_CPU_REPACK=OFF -DGGML_LLAMAFILE=OFF \
        -DGGML_BACKEND_DL=OFF -DBUILD_SHARED_LIBS=OFF \
        -DLLAMA_BUILD_COMMON=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_APP=OFF \
        -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_KNC_TESTS=ON \
        -DLLAMA_BUILD_MTMD=ON -DLLAMA_BUILD_KNC_RUNNER=ON -DLLAMA_KNC_HOST_REFERENCE=OFF \
        -DMTMD_VIDEO=OFF -DLLAMA_SUBPROCESS=OFF > "$dest/configure.log" 2>&1
    cmake --build "$build" --target llama-knc-runner test-knc-mtmd test-knc-dot test-knc-mq \
        test-backend-ops test-quantize-fns test-sampling -j "${BUILD_JOBS:-4}" > "$dest/build.log" 2>&1
    for name in llama-knc-runner test-knc-mtmd test-knc-dot test-knc-mq test-backend-ops test-quantize-fns test-sampling; do
        cp "$build/bin/$name" "$dest/bin/"
    done
    cp -L "$("$cc" -print-file-name=libgomp.so.1)" "$dest/lib/libgomp.so.1"
    cp "$build/CMakeCache.txt" "$dest/"
    cp "$build/ggml/src/knc/candidate-config.h" "$dest/"
    for binary in "$dest"/bin/* "$dest"/lib/*; do
        "$readelf" -hW -dW -VW "$binary" > "$binary.abi.txt"
    done
    (cd "$dest"; sha256sum bin/* lib/* > SHA256SUMS)
    echo "BUILT $profile (native qualification pending)"
done
