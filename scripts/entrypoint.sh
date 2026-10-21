#!/usr/bin/env bash
# Download the weights once (GGUF + tokenizer files) and serve /v1/systemone on $PORT.
set -euo pipefail
MICA_REPO=${MICA_REPO:-sky7350/Mica-v0.5-4B}
MICA_REVISION=${MICA_REVISION:-main}
MICA_GGUF=${MICA_GGUF:-mica-v0.5-4b-Q8_0.gguf}
MICA_NAME=${MICA_NAME:-mica-v0.5-4b}
PORT=${PORT:-8010}
W=${WEIGHTS_DIR:-/weights}
if [ ! -f "$W/$MICA_GGUF" ]; then
  hf download "$MICA_REPO" "$MICA_GGUF" tokenizer.json tokenizer_config.json chat_template.jinja config.json \
     --revision "$MICA_REVISION" --local-dir "$W"
fi
exec python3 -m mica.typesafe_server --gguf "$W/$MICA_GGUF" --hf "$W" --runtime /app/runtime \
     --calibration /app/calibration.json --flash-attn -1 --host 0.0.0.0 --port "$PORT" --name "$MICA_NAME"
