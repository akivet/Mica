"""HTTP endpoint for Mica on the /v1/systemone wire format.

python -m mica.server --gguf model.gguf --hf <tokenizer dir> --runtime <llama.cpp build> [--calibration calib.json]

POST /v1/systemone (alias /v1/decide)
  request : {"model": "...", "state": <text or JSON value>,
             "questions": {"<name>": {"type": "noul" | "choice" | "score", "instructions": <text or JSON value>,
                                      "criteria": ...,                  # see below
                                      "multi": bool,                    # choice only
                                      "allow_cannot_grade": bool,       # score only
                                      "abstain_threshold": 0..1}}}
    noul   criteria: {"false": ..., "true": ...} in any order (optional; defaults to No / Yes)
    choice criteria: {"<id>": <text or JSON value>, ...} with 2..255 entries
    score  criteria: [<level text or JSON value>, ...] lowest first, 2..255 levels (254 with allow_cannot_grade)
  response: {"model": "...", "answers": {"<name>": {...}}, "usage": {"input_tokens": N, "output_tokens": 0},
             "latency_ms": ms}
    noul           -> {"type": "noul", "noul": p_true}
    choice         -> {"type": "choice", "choice": id, "probabilities": {id: p}, "confidence": c}
    choice (multi) -> {"type": "choice", "multi": true, "inclusion": {id: p}, "selected": [id, ...],
                       "count_distribution": [...], "expected_count": x}
    score          -> {"type": "score", "score": expected level, "level": mode, "probabilities": {"0".."n-1": p},
                       "legend": {...}, "confidence": c, "expected_level": x, "margin": m
                       [, "cannot_grade_probability": p]}
    with abstain_threshold, an answer also carries "confidence_pmax" and "abstain".
Questions on the same state share the rendered prefix: the state is prefilled once and each question's suffix is
decoded as an independent sequence. ``usage.input_tokens`` counts that shared prefix once. Invalid requests and
inputs longer than the context return HTTP 400; input is never truncated.
"""
import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .contract import answer_request

JUDGE = None
MODEL_NAME = 'mica-v0.5-4b'
MAX_BODY_BYTES = 64 * 1024 * 1024


def input_tokens(rows):
    """Prefill tokens actually needed: the shared prefix once, plus each question's own suffix."""
    toks = getattr(JUDGE, 'last_prompts', None)
    if toks is None or len(toks) != len(rows):   # the judge keeps the token ids of its last call
        from .native import prompt_for
        toks = [JUDGE.tokenize(prompt_for(r, JUDGE.tokenizer)) for r in rows]
    common = 0
    if len(toks) > 1:
        while all(common < len(t) for t in toks) and len({t[common] for t in toks}) == 1:
            common += 1
    return sum(len(t) for t in toks) - common * (len(toks) - 1)


def answer(body):
    """Answer one parsed request; ValueError means the request is invalid or too long."""
    judged = []

    def predict(rows):
        judged.extend(rows)
        return JUDGE.judge(rows, share_prefix=len(rows) > 1)

    response = answer_request(body, predict, model=MODEL_NAME)
    response['usage'] = {'input_tokens': input_tokens(judged) if judged else 0, 'output_tokens': 0}
    return response


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'   # keep-alive: every response carries Content-Length

    def _send(self, code, obj):
        data = json.dumps(obj, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        if code != 200:   # an error may leave an unread body on the socket: do not reuse the connection
            self.send_header('Connection', 'close')
            self.close_connection = True
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.rstrip('/') in ('/health', '/v1/models'):
            return self._send(200, {'status': 'ok', 'model': MODEL_NAME})
        self._send(404, {'error': 'not found'})

    def do_POST(self):
        if self.path.rstrip('/') not in ('/v1/systemone', '/v1/decide'):
            return self._send(404, {'error': 'not found'})
        t0 = time.time()
        try:
            length = int(self.headers.get('Content-Length', 0))
            if length < 0 or length > MAX_BODY_BYTES:
                raise ValueError('request body is too large')
            body = json.loads(self.rfile.read(length) or b'{}')
            response = answer(body)
        except ValueError as e:  # invalid JSON, contract violations, inputs over the context (never truncated)
            return self._send(400, {'error': str(e)})
        except RecursionError:
            return self._send(400, {'error': 'request is nested too deeply'})
        except Exception as e:  # runtime failure
            return self._send(500, {'error': f'{type(e).__name__}: {e}'})
        response['latency_ms'] = round((time.time() - t0) * 1000, 1)
        self._send(200, response)

    def log_message(self, *a):
        pass


def main():
    global JUDGE, MODEL_NAME
    ap = argparse.ArgumentParser()
    ap.add_argument('--gguf', required=True)
    ap.add_argument('--hf', required=True, help='directory with tokenizer.json, tokenizer_config.json, chat_template.jinja')
    ap.add_argument('--calibration')
    ap.add_argument('--runtime', required=True, help='directory with the llama.cpp shared libraries')
    ap.add_argument('--port', type=int, default=8010)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--n-ctx', type=int, default=32768, help='context per question (tokens); longer inputs get HTTP 400')
    # Each sequence holds a full --n-ctx of attention KV (about 1 GiB at 32k for this model at F16), so 8 sequences
    # plus BF16 weights fit a 24 GB GPU. More questions than --seqs are answered in windows of --seqs.
    ap.add_argument('--seqs', type=int, default=8, help='parallel sequences (questions decoded together)')
    ap.add_argument('--flash-attn', type=int, default=-1, help='-1 auto (off on Vulkan builds), 0 off, 1 on')
    ap.add_argument('--n-ubatch', type=int, default=512, help='tokens per prefill step on the GPU')
    ap.add_argument('--state-cache', choices=('on', 'off'), default='on',
                    help='keep the most recent long state on the GPU (one more sequence of --n-ctx) so a later request '
                         'with the same state, or a grown one, decodes only the new tokens')
    ap.add_argument('--long-input', choices=('on', 'off'), default='on',
                    help='faster long-input mode for prompts over 1k tokens (builds that support it; others run exactly)')
    ap.add_argument('--name', default='mica-v0.5-4b')
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from .direct_backend import DirectJudge
    tok = AutoTokenizer.from_pretrained(a.hf)
    kw = dict(runtime_dir=a.runtime, n_ctx=a.n_ctx, n_seq_max=a.seqs, max_length=a.n_ctx, flash_attn=a.flash_attn, n_ubatch=a.n_ubatch,
              state_cache=a.state_cache == 'on', long_input=a.long_input == 'on')
    JUDGE = (DirectJudge.from_calibration(a.gguf, tok, a.calibration, **kw) if a.calibration else DirectJudge(a.gguf, tok, **kw))
    MODEL_NAME = a.name
    print(json.dumps({'serving': f'http://{a.host}:{a.port}/v1/systemone', 'model': MODEL_NAME,
                      'long_input': JUDGE.long_input}), flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
