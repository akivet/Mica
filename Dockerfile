# Mica v0.5 4B - /v1/systemone server, one image.
#   docker build -t mica-v0.5-4b .
#   docker run --gpus all -p 8010:8010 -v mica-weights:/weights mica-v0.5-4b
# Weights are pulled from Hugging Face at MICA_REVISION on first start (cached in the /weights volume).
# Other quantizations: -e MICA_GGUF=mica-v0.5-4b-{BF16,Q8_0,IQ4_XS}.gguf
# CUDA_ARCH: 86 = RTX 3090/A6000, 89 = RTX 40xx/L40S, 90 = H100/H200. Several: "86;89;90".
ARG CUDA_ARCH="86;89;90"

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS runtime-build
ARG CUDA_ARCH
RUN apt-get update && apt-get install -y --no-install-recommends git cmake build-essential ca-certificates && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch b11010 https://github.com/ggml-org/llama.cpp /src \
 && cmake -S /src -B /src/build -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
          -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF \
 && cmake --build /src/build --target llama -j "$(nproc)" \
 && mkdir -p /runtime && find /src/build -name 'lib*.so*' -exec cp -P {} /runtime/ \;

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip curl ca-certificates && rm -rf /var/lib/apt/lists/*
# Python only renders prompts and tokenizes; inference runs in llama.cpp on the GPU, so the CPU torch wheel is enough.
RUN pip3 install --no-cache-dir torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip3 install --no-cache-dir "transformers==5.5.0" safetensors "huggingface_hub>=1.0"
COPY --from=runtime-build /runtime /app/runtime
COPY mica /app/mica
COPY calibration.json scripts/entrypoint.sh /app/
WORKDIR /app
ENV MICA_REPO=sky7350/Mica-v0.5-4B \
    MICA_REVISION=main \
    MICA_GGUF=mica-v0.5-4b-Q8_0.gguf \
    MICA_NAME=mica-v0.5-4b \
    PORT=8010
EXPOSE 8010
ENTRYPOINT ["bash", "/app/entrypoint.sh"]
