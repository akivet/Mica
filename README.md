<img src="assets/banner.png" alt="Mica" width="100%">

<p align="center">
  <a href="https://huggingface.co/sky7350/Mica-v0.5-4B"><b>Model card</b></a> &nbsp;·&nbsp;
  <a href="#speed"><b>Speed</b></a> &nbsp;·&nbsp;
  <a href="#quick-start"><b>Quick start</b></a> &nbsp;·&nbsp;
  <a href="#api"><b>API</b></a> &nbsp;·&nbsp;
  <a href="#choose-a-file"><b>Choose a file</b></a> &nbsp;·&nbsp;
  <a href="#configuration"><b>Configuration</b></a> &nbsp;·&nbsp;
  <a href="https://ko-fi.com/micakiv"><b>Ko-fi</b></a>
</p>

<p align="center">Looking forward to v1.0? A small donation on <a href="https://ko-fi.com/micakiv">Ko-fi</a> would mean a lot. I'm just a broke student, and I need your help.</p>

Mica is a 4B model for making decisions inside software. Send it the material a decision depends on, the questions
you need answered and the answers you allow; it returns a probability for every answer. No text is generated, so the
cost of a decision is one read of the input.

This repository is the HTTP server for **Mica v0.5 4B**. It runs the INT8 weights on the CUDA runtime, or the GGUF weights on llama.cpp, on one GPU.

- **Option order does not move it.** Shuffle the allowed answers and it picks the same one 97.3% of the time.
- **It can hold back.** Set `abstain_threshold` and it flags the cases it is not sure about instead of guessing.
- **Score questions take 2–255 levels**, choice questions up to 255 options, inputs up to 32k tokens.

| 4B | 32,768 | 2–255 | 0 | EN · KO · ZH | Apache-2.0 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| parameters | input tokens | options or score levels | tokens generated | main languages | license |

## Results

<img src="assets/results_jdi_all.png" alt="JDI 0.3" width="100%">

<img src="assets/results_jevbench_all.png" alt="JevBench Hard" width="100%">

<img src="assets/results_jdi_4b.png" alt="JDI 0.3 · 4B-class models" width="100%">

<img src="assets/results_jevbench_4b.png" alt="JevBench Hard · 4B-class models" width="100%">

| Model | Size | JDI 0.3 | JevBench Hard |
|---|:---:|:---:|:---:|
| PPLX Decider Flash | 4B | 60.1 | — |
| Jev | ~100B | 59.1 | 80/111 |
| **Mica v0.5** | 4B | **58.8** | **84/111** |
| CoCo Decision | 4B | 55.4 | — |
| d3-mini | 4B | 54.9 | 68/111 |
| OneJev 1.5 | 4B | 54.9 | — |
| MS Decision-1 | 9B | 54.7 | 91/111 |
| Clef | 9B | 53.0 | 72/111 |
| RSI-Jev 6.1 | 4B | 51.0 | — |
| d1 | 3B | 49.0 | 60/111 |
| Kev | 4B | 39.1 | 60/111 |
| Mica v0.1 | 4B | 37.2 | 72/111 |
| Solar Decide | 35B | 36.7 | 82/111 |
| Solar Decide Flash | 35B | 23.1 | 68/111 |
| d3-flash | 9B | — | 70/111 |
| Imajev | 4B | — | 80/111 |

Same items and scoring for every model; other models are our runs.

## Speed

<img src="assets/latency.png" alt="Latency by input length" width="100%">

<img src="assets/latency_pplx27.png" alt="Mica v0.5 and Perplexity Decider 27B" width="100%">

<img src="assets/latency_api.png" alt="Mica v0.5 and API models" width="100%">

Measured on one NVIDIA RTX 6000 Ada with the INT8 CUDA runtime. Per-task latency and more options are on the
[model card](https://huggingface.co/sky7350/Mica-v0.5-4B#speed).

## Quick start

### CUDA runtime (INT8, NVIDIA GPUs with 8 GB or more)

Linux (or WSL2 on Windows), CUDA 12, Python 3.10 to 3.13:

```bash
git clone https://github.com/akivet/Mica && cd Mica
python3 -m venv .venv && . .venv/bin/activate
pip install -r v0.5/serving/requirements.txt
pip install --no-build-isolation causal-conv1d==1.7.0
hf download sky7350/Mica-v0.5-4B --include "int8/*" --local-dir weights
python v0.5/serving/serve.py --backend hf --model weights/int8 --name mica-v0.5-4b --port 8010
```

Windows (experimental), Python 3.10 to 3.13, an NVIDIA driver for CUDA 12.8, in PowerShell:

```powershell
git clone https://github.com/akivet/Mica; cd Mica
py -3.12 -m venv .venv; .venv\Scripts\Activate.ps1
pip install -r v0.5/serving/requirements-windows.txt
hf download sky7350/Mica-v0.5-4B --include "int8/*" --local-dir weights
python v0.5/serving/serve.py --backend hf --model weights/int8 --name mica-v0.5-4b --port 8010
```

PyTorch's Windows builds have no FlashAttention kernel, so the Windows runtime brings its own precompiled attention
kernel; on a Linux GPU set up the same way it stays within 1-12% of the Linux latency (256 to 32k tokens) with the
same accuracy. It has not run on a Windows PC yet: if the start line says `bfloat16` or requests are slow, run the
Linux commands above in WSL2.

INT8 runs on compute capability 8.0 and newer (RTX 30/40/50 series, RTX A-series and Ada workstation cards, L4, L40S,
A100, H100, H200, B200); other GPUs fall back to bfloat16 by themselves. The first start compiles and tunes kernels and
captures CUDA graphs, printing each stage, and saves the results in `--state-dir`. On an RTX 6000 Ada the first start
took about 3 minutes and later starts with the same `--state-dir` about 1 minute, so keep `--state-dir` on a
persistent disk. This work is CPU-bound: on a rented GPU whose host CPU was shared with other tenants and fully
loaded, the first start took over an hour, so use a machine with its own CPU cores. The longest input is
set from the GPU's memory (the GPU table lists it). For bfloat16 from the full weights, download
`model.safetensors*`, `*.json`, `*.jinja` and `*.txt` and run with `--model weights --w8a8 off`.

### Docker (GGUF, llama.cpp)

```bash
git clone https://github.com/akivet/Mica && cd Mica
docker build -t mica .
docker run --gpus all -p 8010:8010 -v mica-weights:/weights mica
```

The first start downloads `mica-v0.5-4b-Q8_0.gguf` and the tokenizer into the `mica-weights` volume. For a smaller GPU
pick a file from [Choose a file](#choose-a-file) and pass it as `-e MICA_GGUF=...`. The image is built for CUDA
architectures 86, 89 and 90 (RTX 30xx, RTX 40xx and L40S, H100); pass `--build-arg CUDA_ARCH=...` for others.

### Without Docker (GGUF, llama.cpp)

Linux, CUDA 12, Python 3.10 or newer:

```bash
pip install -r requirements.txt
bash scripts/build_runtime.sh        # builds llama.cpp into ./runtime
hf download sky7350/Mica-v0.5-4B mica-v0.5-4b-Q8_0.gguf \
   tokenizer.json tokenizer_config.json chat_template.jinja config.json --local-dir weights
bash scripts/serve.sh                # http://127.0.0.1:8010/v1/systemone
```

## API

`POST /v1/systemone`

```bash
curl -s localhost:8010/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": {"order": 4471, "paid": 129.00, "refund_requested": 129.00, "delivered": false,
            "carrier_note": "Package returned to sender: address not found"},
  "questions": {
    "refund":   {"type": "noul", "instructions": "Should the full refund be approved automatically?"},
    "priority": {"type": "score", "instructions": "How urgent is this ticket?",
                 "criteria": ["Can wait", "Today", "Within the hour"]}
  }}'
```

```json
{
  "model": "mica-v0.5-4b",
  "answers": {
    "refund":   {"type": "noul", "noul": 0.91},
    "priority": {"type": "score", "score": 1.12, "level": 1, "expected_level": 1.12,
                 "probabilities": {"0": 0.12, "1": 0.64, "2": 0.24},
                 "legend": {"0": "Can wait", "1": "Today", "2": "Within the hour"},
                 "confidence": 0.71, "margin": 0.40}
  },
  "usage": {"input_tokens": 168, "output_tokens": 0},
  "latency_ms": 41.7
}
```

The values show the response shape only.

| Type | `criteria` | Result |
|---|---|---|
| `noul` | Optional `{"false": "...", "true": "..."}` | `noul`: probability of true |
| `choice` | `{"id": "description", ...}`, 2–255 options | `choice`, `probabilities`, `confidence` |
| `score` | `["lowest level", ..., "highest level"]`, 2–255 levels | `level`, `expected_level`, `probabilities` |

Optional per question: `"multi": true` (choice with several correct options), `"allow_cannot_grade": true` (score),
`"abstain_threshold": <0..1>` (any type). Inputs over 32,768 tokens and malformed requests get HTTP 400 with a message;
nothing is truncated.

`GET /health` returns `{"status": "ok", "model": ...}`.

For how to write the state, the rule and the answers, and recipes for security, engineering, moderation,
search and routing decisions, see the [prompting guide](https://huggingface.co/sky7350/Mica-v0.5-4B#prompting-guide).

## Choose a file

| GPU memory | For example | Recommended file | Longest input | Accuracy vs BF16 |
|---|---|---|:---:|:---:|
| 24 GB and up | RTX 3090, 4090, 5090 | INT8 · CUDA runtime | 32k | same |
| 16 GB | RTX 4060 Ti 16 GB, 5060 Ti | INT8 · CUDA runtime | 27k | same |
| 12 GB | RTX 3060 12 GB, 4070 | INT8 · CUDA runtime | 12k | same |
| 8 GB | RTX 3070, 4060 | INT8 · CUDA runtime | 6k | same |
| AMD, Apple, CPU | RX 7900, M-series | `mica-v0.5-4b-Q8_0.gguf` | — | -0.2 pt |

All files are in [sky7350/Mica-v0.5-4B](https://huggingface.co/sky7350/Mica-v0.5-4B/tree/main).

## Configuration

Docker environment variables:

| Variable | Default | |
|---|---|---|
| `MICA_GGUF` | `mica-v0.5-4b-Q8_0.gguf` | Weights file to download and serve |
| `MICA_REPO` | `sky7350/Mica-v0.5-4B` | Hugging Face repository |
| `MICA_REVISION` | `main` | Repository revision |
| `PORT` | `8010` | Listening port |

Server flags (`python -m mica.server --help`):

| Flag | Default | |
|---|---|---|
| `--n-ctx` | `32768` | Longest accepted input, in tokens |
| `--seqs` | `8` | Questions decoded side by side on one input; more are handled in groups of this size |
| `--flash-attn` | `-1` | `-1` auto, `0` off (faster on some Vulkan GPUs), `1` on |
| `--calibration` | `calibration.json` | Temperature applied to the probabilities |

## Tests

```bash
python -m pytest tests
```

The tests check request parsing, prompt rendering and response shapes against fixed cases. They need the tokenizer
files but no GPU.

## Earlier versions

The v0.1 4B server is kept at tag [`v0.1.2`](https://github.com/akivet/Mica/tree/v0.1.2), with its weights at
[sky7350/Mica-v0.1-4B](https://huggingface.co/sky7350/Mica-v0.1-4B).

## License

Apache-2.0. The model is a fine-tune of Qwen/Qwen3.5-4B (Apache-2.0). llama.cpp (MIT) is built locally and not
redistributed. See `NOTICE`.
