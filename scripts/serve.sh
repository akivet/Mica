#!/usr/bin/env bash
# Serve Mica v0.5 4B on the TypeSafe /v1/systemone wire format.
# usage: bash scripts/serve.sh [GGUF] [PORT]   (default: the Q8_0 GGUF, port 8010)
# Weights:
#   hf download sky7350/Mica-v0.5-4B mica-v0.5-4b-Q8_0.gguf tokenizer.json tokenizer_config.json \
#      chat_template.jinja config.json --revision "${MICA_REVISION:-main}" --local-dir weights
# Files: mica-v0.5-4b-{BF16,Q8_0,IQ4_XS}.gguf
set -euo pipefail
GGUF=${1:-weights/mica-v0.5-4b-Q8_0.gguf}
PORT=${2:-8010}
python -m mica.typesafe_server --gguf "$GGUF" --hf weights --runtime runtime \
  --calibration calibration.json --flash-attn -1 --port "$PORT" --name mica-v0.5-4b
