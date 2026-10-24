#!/usr/bin/env bash
# Build the llama.cpp shared libraries Mica's direct readout binds to (tag b11700: chunked CUDA / HIP kernel for the
# model's linear-attention layers). Output: ./runtime/lib*.so (or .dylib)
#   BACKEND=cuda   (default) NVIDIA; CUDA_ARCH=86 (RTX 30xx), 89 (RTX 40xx, L40S), 90 (H100), several as "86;89"
#   BACKEND=vulkan AMD, Intel or NVIDIA through Vulkan (needs the Vulkan SDK / glslc)
#   BACKEND=hip    AMD through ROCm; GPU_TARGETS=gfx1100 (RX 7900) etc.
#   BACKEND=metal  Apple silicon
set -euo pipefail
TAG=${LLAMA_TAG:-b11700}
BACKEND=${BACKEND:-cuda}
case "$BACKEND" in
  cuda)   FLAGS=(-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH:-86}") ;;
  vulkan) FLAGS=(-DGGML_VULKAN=ON) ;;
  hip)    FLAGS=(-DGGML_HIP=ON -DGPU_TARGETS="${GPU_TARGETS:-gfx1100}") ;;
  metal)  FLAGS=(-DGGML_METAL=ON) ;;
  *) echo "BACKEND must be cuda, vulkan, hip or metal" >&2; exit 1 ;;
esac
git clone -q --depth 1 --branch "$TAG" https://github.com/ggml-org/llama.cpp "llama.cpp-$TAG"
PATCHES="$(cd "$(dirname "$0")" && pwd)/patches"
# every backend: long-input mode (llama_set_upper_stale), Mica's faster path for prompts over 1k tokens
if [ -f "$PATCHES/long-input-$TAG.patch" ]; then git -C "llama.cpp-$TAG" apply "$PATCHES/long-input-$TAG.patch"; fi
# Vulkan at b11700: tiled concat for the linear-attention conv input, and a wider int8 matmul tile on RX 6000 cards
# with AMD's driver (GGML_VK_RDNA2_TUNE=0 turns the tile off at run time)
if [ "$BACKEND" = vulkan ] && [ -f "$PATCHES/vulkan-$TAG.patch" ]; then git -C "llama.cpp-$TAG" apply "$PATCHES/vulkan-$TAG.patch"; fi
cmake -S "llama.cpp-$TAG" -B "llama.cpp-$TAG/build" "${FLAGS[@]}" -DBUILD_SHARED_LIBS=ON \
      -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF
cmake --build "llama.cpp-$TAG/build" --target llama -j "$(getconf _NPROCESSORS_ONLN)"
mkdir -p runtime && find "llama.cpp-$TAG/build" \( -name 'lib*.so*' -o -name 'lib*.dylib' \) -exec cp -P {} runtime/ \;
ls runtime
