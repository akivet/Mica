import argparse
import json
import sys
import threading
import time
from pathlib import Path
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
import m81mym3jhph2kl5gy7i8b as fast_path
JUDGE = None
MODEL_NAME = 'systemone'
DEBUG_TRACE = False
QUESTION_SERVER_FIELDS = ('_v6gd7oavvss5c4jnyhxhc',)
CONTRACT_FIELDS = ('model', 'state', 'questions', 'images')

def contract_request(body):
    if not isinstance(body, dict):
        raise ValueError('Request must be an object')
    request = {k: v for k, v in body.items() if k in CONTRACT_FIELDS}
    questions = request.get('questions')
    experts = {}
    if isinstance(questions, dict):
        clean = {}
        for name, q in questions.items():
            if isinstance(q, dict):
                q = dict(q)
                for field in QUESTION_SERVER_FIELDS:
                    if q.get(field):
                        experts[name] = str(q[field])
                    q.pop(field, None)
                crit = q.get('criteria')
                if str(q.get('type', '')).lower() == 'noul' and isinstance(crit, dict) and ({str(k).lower() for k in crit} == {'no', 'yes'}):
                    q['criteria'] = {'false' if str(k).lower() == 'no' else 'true': v for k, v in crit.items()}
            clean[name] = q
        request['questions'] = clean
    return (request, experts)

def _with_experts(rows, experts):
    if not experts:
        return rows
    return [dict(r, _v6gd7oavvss5c4jnyhxhc=experts[r.get('multi_of', r['id'])]) if r.get('multi_of', r['id']) in experts else r for r in rows]

def rows_from_request(body):
    from m6lir2n4n221f2rp5z93i import rows_from_request as contract_rows
    request, experts = contract_request(body)
    return _with_experts(contract_rows(request), experts)
REASONING = {'default': 'off', 'budget_ms': 500.0, 'profile': None, 'config': None, 'gate': None}

def reasoning_options(body):
    mode = body.get('reasoning', REASONING['default'])
    if mode not in ('off', 'on', 'auto'):
        raise ValueError('reasoning must be "off", "on" or "auto"')
    budget = body.get('budget_ms', REASONING['budget_ms'])
    if isinstance(budget, bool) or not isinstance(budget, (int, float)) or (not 0 < budget <= 60000):
        raise ValueError('budget_ms must be a positive number of milliseconds')
    return (mode, float(budget))

def debug_timing_flag(body):
    flag = body.get('debug_timing', False)
    if not isinstance(flag, bool):
        raise ValueError('debug_timing must be a boolean')
    return flag

def begin_timing(debug, t_start, t_parsed):
    if not debug:
        return (None, None)
    tracer = fast_path.LatencyTracer(t0=t_start)
    tracer.add('parse', (t_parsed - t_start) * 1000.0)
    return (tracer, fast_path.start(tracer)[1])

def attach_timing(payload, tracer, token, *, input_tokens=None):
    if tracer is None:
        return payload
    fast_path.finish(token)
    t = time.perf_counter()
    json.dumps(payload, ensure_ascii=False)
    tracer.add('respond', (time.perf_counter() - t) * 1000.0)
    report = tracer.report()
    for name in fast_path.STAGES:
        report['stages_ms'].setdefault(name, 0.0)
    if REASONING.get('strict_timing', False):
        report['stages_ms']['other'] = report['other_ms']
        payload['latency_ms'] = round(report['total_ms'], 1)
    else:
        report['total_ms'] = round(tracer.elapsed_ms(), 3)
    stats = getattr(JUDGE, 'cache_stats', None)
    if stats is not None:
        report['caches'] = stats()
    if input_tokens is not None:
        report['input_tokens'] = input_tokens
    if DEBUG_TRACE:
        payload['timing'] = report
    return payload

def reasoned_answer(row, result):
    out = answer(row, result['probs'])
    out['prediction_set'] = list(result['prediction_set'])
    out['abstain'] = result['abstain']
    if DEBUG_TRACE:
        out['trace_meta'] = result['trace_meta']
    return out

def legacy_fields(out):
    out = dict(out)
    if out.get('type') == 'noul' and 'noul' in out:
        out.setdefault('answer', out['noul'] > 0.5)
        out.setdefault('confidence', max(out['noul'], 1 - out['noul']))
    elif out.get('type') == 'choice' and 'choice' in out:
        out.setdefault('answer', out['choice'])
    elif out.get('type') == 'score' and 'level' in out:
        out.setdefault('answer', out['level'])
    return out

def answer(row, p):
    from m6lir2n4n221f2rp5z93i import response_envelope
    p = list(p)
    out = response_envelope([row], [p], model=MODEL_NAME or 'systemone')['answers'][row['id']]
    return legacy_fields(out)

def input_tokens(rows):
    from m6lir2n4n221f2rp5z93i import prompt_for
    toks = [JUDGE.tokenize(prompt_for(r, JUDGE.tokenizer)) for r in rows]
    common = 0
    if len(toks) > 1:
        while all((common < len(t) for t in toks)) and len({t[common] for t in toks}) == 1:
            common += 1
    return sum((len(t) for t in toks)) - common * (len(toks) - 1)

def reasoned_input_tokens(rows, results):
    metas = [r.get('trace_meta') or {} for r in results]
    if metas and all(('prompt_tokens' in m for m in metas)):
        return sum((m['prompt_tokens'] for m in metas)) - metas[0].get('prompt_shared_prefix', 0) * (len(metas) - 1)
    if REASONING.get('strict_timing', False):
        raise ValueError('Strict timing requires computed prompt token counts')
    return input_tokens(rows)

class Handler(BaseHTTPRequestHandler):
    disable_nagle_algorithm = True
    wbufsize = 1 << 16

    def _send(self, code, obj):
        if 'error' in obj:
            obj = {'error': 'Invalid request' if code == 400 else 'Request failed' if code >= 500 else 'not found'}
        elif not DEBUG_TRACE:
            obj = dict(obj)
            obj.pop('timing', None)
            latency = obj.pop('latency_ms', None)
            if latency is not None:
                obj['latency_band_ms'] = '0-50' if latency < 50 else '50-200' if latency < 200 else '200-500' if latency < 500 else '500+'
        data = json.dumps(obj, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.rstrip('/') in ('/health', '/v1/models'):
            return self._send(200, {'status': 'ok', 'model': MODEL_NAME})
        self._send(404, {'error': 'not found'})

    def do_POST(self):
        if self.path.rstrip('/') not in ('/v1/systemone', '/v1/decide'):
            return self._send(404, {'error': 'not found'})
        t_start = time.perf_counter()
        try:
            body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))) or b'{}')
            if not isinstance(body, dict):
                raise ValueError('Request must be an object')
            request, experts = contract_request(body)
            rows = rows_from_request(body)
            mode, budget = reasoning_options(body)
            debug = debug_timing_flag(body)
            if mode != 'off' and any((r.get('multi') for r in rows)):
                raise ValueError('multi is served with reasoning "off" only')
        except (ValueError, json.JSONDecodeError) as e:
            return self._send(400, {'error': str(e)})
        tracer, token = begin_timing(DEBUG_TRACE and debug or REASONING.get('strict_timing', False), t_start, time.perf_counter())
        try:
            self._answer(rows, mode, budget, t_start, tracer, token, request=request, experts=experts)
        except Exception as exc:
            self._send(500, {'error': 'Request failed'})
            print(json.dumps({'request_failed': type(exc).__name__, 'detail': str(exc)[:300]}), file=sys.stderr, flush=True)
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        finally:
            fast_path.finish(token)

    def _answer(self, rows, mode, budget, t_start, tracer, token, request=None, experts=None):
        if mode != 'off':
            try:
                results = JUDGE.judge_reasoned(rows, reasoning=mode, budget_ms=budget, profile=REASONING['profile'], config=REASONING['config'], gate=REASONING['gate'], t_start=t_start)
            except ValueError as e:
                return self._send(400, {'error': str(e)})
            decoded = sum((r['trace_meta']['tokens']['decoded'] for r in results))
            answers = {r['id']: reasoned_answer(r, res) for r, res in zip(rows, results)}
            latency = round((time.perf_counter() - t_start) * 1000, 1)
            try:
                usage = {'input_tokens': reasoned_input_tokens(rows, results), 'output_tokens': decoded}
            except ValueError as e:
                return self._send(400, {'error': str(e)})
            return self._send(200, attach_timing({'model': MODEL_NAME, 'answers': answers, 'usage': usage, 'latency_ms': latency}, tracer, token, input_tokens=usage['input_tokens']))
        report, served = ({}, {'rows': rows})

        def predict(batch):
            batch = _with_experts(batch, experts)
            served['rows'] = batch
            return JUDGE.judge(batch, share_prefix=len(batch) > 1, report=report)
        from m6lir2n4n221f2rp5z93i import answer_request
        try:
            result = answer_request(request, predict, model=MODEL_NAME)
        except ValueError as e:
            return self._send(400, {'error': str(e)})
        answers = {k: legacy_fields(v) for k, v in result['answers'].items()}
        rows = served['rows']
        latency = round((time.perf_counter() - t_start) * 1000, 1)
        if REASONING.get('strict_timing', False) and 'input_tokens' not in report:
            return self._send(400, {'error': 'Strict timing requires computed prompt token counts'})
        usage = {'input_tokens': report['input_tokens'] if 'input_tokens' in report else input_tokens(rows), 'output_tokens': 0}
        self._send(200, attach_timing({'model': MODEL_NAME, 'answers': answers, 'usage': usage, 'latency_ms': latency}, tracer, token, input_tokens=usage['input_tokens']))

    def log_message(self, *a):
        pass
_v11in3z4qruotk1862nps = ('noul', 'choice', 'score')
WARM = {'last_request': None, 'lock': threading.Lock()}

def _warm_row(kind, state, index=0):
    if kind == 'noul':
        cands = [{'id': 'false', 'text': 'false'}, {'id': 'true', 'text': 'true'}]
    elif kind == 'score':
        cands = [{'id': str(i), 'text': f'level {i}'} for i in range(11)]
    else:
        cands = [{'id': f'option{i}', 'text': f'Option {i}'} for i in range(4)]
    return {'id': f'warm-{kind}-{index}', 'kind': kind, 'state': state, 'question': f'Warm-up question {index}?', 'candidates': cands}

def _v2lbiu05abgrrc954oo6t(tokenizer, length, kind, *, questions=1):
    from m6lir2n4n221f2rp5z93i import prompt_for

    def size(n):
        rows = [_warm_row(kind, 'warm ' * n, i) for i in range(questions)]
        return (len(tokenizer.encode(prompt_for(rows[0], tokenizer), add_special_tokens=False)), rows)
    base, rows = size(0)
    if base >= length:
        return rows
    lo, hi = (0, max(1, length - base))
    while size(hi)[0] < length:
        hi *= 2
    while lo < hi:
        mid = (lo + hi) // 2
        if size(mid)[0] < length:
            lo = mid + 1
        else:
            hi = mid
    return size(lo)[1]

def _v5fk4i78hn5w2galet9gy(judge, buckets, *, max_questions=3, kinds=_v11in3z4qruotk1862nps, log=None):
    limit = min(getattr(judge, 'max_length', 1 << 30), getattr(judge, 'n_ctx_seq', 1 << 30))
    seqs = max(1, min(max_questions, getattr(judge, 'n_seq_max', max_questions)))
    report = []
    for length in buckets:
        if length >= limit:
            report.append({'bucket': length, 'skipped': 'over max_length'})
            continue
        for kind in kinds:
            rows = _v2lbiu05abgrrc954oo6t(judge.tokenizer, length, kind)
            t = time.perf_counter()
            judge.judge(rows, share_prefix=False)
            report.append({'bucket': length, 'kind': kind, 'questions': 1, 'ms': round((time.perf_counter() - t) * 1000.0, 1)})
        if seqs > 1:
            rows = _v2lbiu05abgrrc954oo6t(judge.tokenizer, length, 'choice', questions=seqs)
            t = time.perf_counter()
            judge.judge(rows, share_prefix=True)
            report.append({'bucket': length, 'kind': 'choice', 'questions': seqs, 'ms': round((time.perf_counter() - t) * 1000.0, 1)})
    if log is not None:
        print(json.dumps({'_vc2bm7ib2g84yw3czx30t': report}), file=log, flush=True)
    return report

def desktop_handler(a):
    if not (a.keep_alive or a.keep_warm_seconds > 0):
        return Handler

    class DesktopHandler(Handler):
        if a.keep_alive:
            protocol_version = 'HTTP/1.1'
            timeout = a.keep_alive_timeout

            def _send(self, code, obj):
                self._close_after = code >= 400
                super()._send(code, obj)

            def end_headers(self):
                if getattr(self, '_close_after', False):
                    self.send_header('Connection', 'close')
                super().end_headers()

        def do_POST(self):
            WARM['last_request'] = time.monotonic()
            if a.keep_alive and self.path.rstrip('/') not in ('/v1/systemone', '/v1/decide'):
                self.rfile.read(int(self.headers.get('Content-Length', 0) or 0))
            try:
                return super().do_POST()
            finally:
                WARM['last_request'] = time.monotonic()
    return DesktopHandler