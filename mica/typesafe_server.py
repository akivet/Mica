"""Jev/TypeSafe-compatible HTTP endpoint for Mica (TypeSafe /v1/systemone wire format).

python -m mica.typesafe_server --gguf model.gguf --hf <tokenizer dir> [--calibration calib.json] [--port 8010]
POST /v1/systemone  (same path as the TypeSafe API and Kev's local server)
  request : {"model": "...", "state": "<text or object>", "questions": {"<name>": {"type": "noul"|"choice"|"score",
             "instructions": "...", "criteria": {id: text} (noul: keys false/true; choice: option ids) | [level texts] (score)}}}
  response: {"model": "mica-v0.1-4b", "answers": {"<name>": {...}}, "usage": {"input_tokens": N, "output_tokens": 0}, "latency_ms": ms}
    noul   -> {"type": "noul", "noul": p_true, "answer": bool, "confidence": max(p)}
    choice -> {"type": "choice", "choice": id, "probabilities": {id: p}, "answer": id, "confidence": max(p)}
    score  -> {"type": "score", "score": int, "probabilities": {"0".."n-1": p}, "answer": int, "confidence": max(p)}
Questions on the same state share the rendered prefix (one prefill of the state, suffixes decoded as independent
sequences), which is exactly the multi-question pattern of the wire format. Invalid requests return HTTP 400; an input longer than the context gets HTTP 422 (input_too_long) and is never truncated.
"""
import argparse, json, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .direct_backend import InputTooLongError

JUDGE = None
MODEL_NAME = 'mica-v0.1-4b'


def rows_from_request(body):
    state = body.get('state')
    if not isinstance(state, str):
        state = json.dumps(state, ensure_ascii=False, indent=1)
    qs = body.get('questions')
    if not isinstance(qs, dict) or not qs:
        raise ValueError('questions must be a non-empty object')
    rows = []
    for name, q in qs.items():
        kind = str(q.get('type', '')).lower()
        crit = q.get('criteria')
        if kind == 'noul':
            if isinstance(crit, dict):
                f = crit.get('false') or crit.get('no') or 'false'
                t = crit.get('true') or crit.get('yes') or 'true'
            else:
                f, t = 'false', 'true'
            cands = [{'id': 'false', 'text': str(f)}, {'id': 'true', 'text': str(t)}]
        elif kind == 'choice':
            if not isinstance(crit, dict) or len(crit) < 2:
                raise ValueError(f'{name}: choice needs criteria {{id: text}} with >= 2 options')
            cands = [{'id': str(k), 'text': str(v)} for k, v in crit.items()]
        elif kind == 'score':
            if not isinstance(crit, list) or not 2 <= len(crit) <= 10:
                raise ValueError(f'{name}: score needs 2-10 level texts')
            cands = [{'id': str(i), 'text': str(v)} for i, v in enumerate(crit)]
        else:
            raise ValueError(f'{name}: unknown type {kind!r}')
        rows.append({'id': name, 'kind': kind, 'state': state, 'question': str(q.get('instructions', '')), 'candidates': cands})
    return rows


def answer(row, p):
    # JevBench's `typesafe` adapter requires "type" on every answer and "choice" on choice answers
    # (TypeSafe's own shape); "answer" is kept for older clients.
    conf = max(p)
    if row['kind'] == 'noul':
        return {'type': 'noul', 'noul': p[1], 'answer': p[1] >= 0.5, 'confidence': conf}
    ids = [c['id'] for c in row['candidates']]
    k = max(range(len(p)), key=p.__getitem__)
    out = {'type': row['kind'], 'probabilities': dict(zip(ids, p)), 'answer': ids[k], 'confidence': conf}
    if row['kind'] == 'choice':
        out['choice'] = ids[k]
    else:
        out['score'] = k
        out['answer'] = k
    return out


def input_tokens(rows):
    """Prefill tokens actually computed: the shared state prefix once, plus each question's own suffix."""
    from .native import prompt_for
    toks = [JUDGE.tokenize(prompt_for(r, JUDGE.tokenizer)) for r in rows]
    common = 0
    if len(toks) > 1:
        while all(common < len(t) for t in toks) and len({t[common] for t in toks}) == 1:
            common += 1
    return sum(len(t) for t in toks) - common * (len(toks) - 1)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        data = json.dumps(obj, ensure_ascii=False).encode('utf-8')
        self.send_response(code); self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(data)))
        self.end_headers(); self.wfile.write(data)

    def do_GET(self):
        if self.path.rstrip('/') in ('/health', '/v1/models'):
            return self._send(200, {'status': 'ok', 'model': MODEL_NAME})
        self._send(404, {'error': 'not found'})

    def do_POST(self):
        if self.path.rstrip('/') not in ('/v1/systemone', '/v1/decide'):
            return self._send(404, {'error': 'not found'})
        t0 = time.time()
        try:
            body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))) or b'{}')
            rows = rows_from_request(body)
        except (ValueError, json.JSONDecodeError) as e:
            return self._send(400, {'error': str(e)})
        try:
            probs = JUDGE.judge(rows, share_prefix=len(rows) > 1)
        except InputTooLongError as e:  # refused, not truncated; a question error, not a bad request
            return self._send(422, {'error': {'type': 'question_error', 'code': 'input_too_long', 'message': str(e)}})
        except ValueError as e:  # e.g. a label that is not one token in this runtime
            return self._send(400, {'error': str(e)})
        self._send(200, {'model': MODEL_NAME, 'answers': {r['id']: answer(r, p) for r, p in zip(rows, probs)},
                         'usage': {'input_tokens': input_tokens(rows), 'output_tokens': 0},
                         'latency_ms': round((time.time() - t0) * 1000, 1)})

    def log_message(self, *a):
        pass


def main():
    global JUDGE, MODEL_NAME
    ap = argparse.ArgumentParser()
    ap.add_argument('--gguf', required=True); ap.add_argument('--hf', required=True)
    ap.add_argument('--calibration'); ap.add_argument('--runtime', required=True)
    ap.add_argument('--port', type=int, default=8010); ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--n-ctx', type=int, default=8192); ap.add_argument('--seqs', type=int, default=8)
    ap.add_argument('--flash-attn', type=int, default=-1, help='-1 auto, 0 off (Vulkan/RX6600XT), 1 on')
    ap.add_argument('--name', default='mica-v0.1-4b')
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from .direct_backend import DirectJudge
    tok = AutoTokenizer.from_pretrained(a.hf)
    kw = dict(runtime_dir=a.runtime, n_ctx=a.n_ctx, n_seq_max=a.seqs, max_length=a.n_ctx, flash_attn=a.flash_attn, n_ubatch=512)
    JUDGE = (DirectJudge.from_calibration(a.gguf, tok, a.calibration, **kw) if a.calibration else DirectJudge(a.gguf, tok, **kw))
    MODEL_NAME = a.name
    print(json.dumps({'serving': f'http://{a.host}:{a.port}/v1/systemone', 'model': MODEL_NAME}), flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
