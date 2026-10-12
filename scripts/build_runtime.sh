#!/usr/bin/env bash
# Build the llama.cpp shared libraries Mica's direct readout binds to (tag b11010, CUDA, sm_86 = RTX 3090).
# Change CMAKE_CUDA_ARCHITECTURES for other GPUs (e.g. 89 for RTX 40xx, 90 for H100). Output: ./runtime/lib*.so
set -euo pipefail
CUDA_ARCH=${CUDA_ARCH:-86}
git clone -q --depth 1 --branch b11010 https://github.com/ggml-org/llama.cpp llama.cpp-b11010
cmake -S llama.cpp-b11010 -B llama.cpp-b11010/build -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=ON \
      -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF
cmake --build llama.cpp-b11010/build --target llama -j "$(nproc)"
mkdir -p runtime && find llama.cpp-b11010/build -name 'lib*.so*' -exec cp -P {} runtime/ \;
ls runtime
