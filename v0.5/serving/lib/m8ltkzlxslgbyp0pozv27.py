from __future__ import annotations
import json
import os
import statistics
import time
from pathlib import Path
TUNE_VERSION = 1
FILE = 'route_profile.json'
GRAPH_BUCKET = 64
EXACT_FLOOR, EXACT_CEIL = (1024, 2048)
WASTE_SHARE = 0.05
GRAPH_BUDGET = 0.06
FAST_CANDIDATES = (2048, 3072, 4096, 6144, 8192)
DEFAULTS = {'int8': dict(exact_max=1536, graph_max=1536, fast_buckets=[1664, 1792, 1920, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096], graph_a=6.45, graph_b=0.0368, fork_a=162.0, fork_b=3.1, bf16_from=0), 'bf16': dict(exact_max=1024, graph_max=512, fast_buckets=[1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048], graph_a=9.4, graph_b=0.0565, fork_a=170.0, fork_b=4.0, bf16_from=0)}

def default(precision):
    out = dict(DEFAULTS['int8' if precision == 'int8' else 'bf16'])
    out['fast_buckets'] = list(out['fast_buckets'])
    out['source'] = 'default'
    return out

def device_key(name, cc, total_bytes, precision, dual=False):
    return '%s|sm%d%d|%.0fGiB|%s|dual%d|v%d' % (name, cc[0], cc[1], total_bytes / 2 ** 30, precision, int(bool(dual)), TUNE_VERSION)

def load(directory, key):
    path = Path(directory) / FILE
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    hit = data.get(key) if isinstance(data, dict) else None
    return dict(hit, source='saved') if isinstance(hit, dict) and 'exact_max' in hit else None

def save(directory, key, profile):
    path = Path(directory) / FILE
    try:
        data = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
    except (OSError, ValueError):
        data = {}
    data[key] = {k: v for k, v in profile.items() if k != 'source'}
    os.makedirs(directory, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True, indent=1), encoding='utf-8')

def measure(fn, clock=time.perf_counter, sync=lambda: None, reps=3, warm=1):
    for _ in range(warm):
        fn()
        sync()
    out = []
    for _ in range(reps):
        sync()
        t = clock()
        fn()
        sync()
        out.append((clock() - t) * 1000.0)
    return statistics.median(out)

def plan(precision, graph_cap, dual=False):
    cases = [('graph', 512, 1), ('graph', graph_cap, 1)]
    cases += [('exact', n, 1) for n in (EXACT_FLOOR + 512, EXACT_CEIL)]
    for n in FAST_CANDIDATES:
        cases += [('fast', n, 1), ('fast_graph', n, 1)]
    cases += [('fast', EXACT_FLOOR + 512, 1), ('fast_graph', EXACT_FLOOR + 512, 1)]
    cases += [('fork', 1024, 2), ('fork', 1024, 8)]
    if dual and precision == 'int8':
        cases += [('fast_bf16', n, 1) for n in (2048, 8192, 32000)] + [('fast', 32000, 1)]
    return cases

def calibrate(run, precision, graph_cap, clock=time.perf_counter, sync=lambda: None, mem=None, total=None, dual=False, reps=3, budget_s=30.0):
    t0 = clock()
    timings, pools = ({}, {})
    for kind, n, q in plan(precision, graph_cap, dual):
        if clock() - t0 > budget_s:
            break
        before = mem() if mem else None
        try:
            ms = measure(lambda: run(kind, n, q), clock, sync, reps)
        except Exception:
            continue
        timings['%s@%d@%d' % (kind, n, q)] = round(ms, 3)
        if mem and kind == 'fast_graph':
            pools[n] = max(0, mem() - before)
    prof = choose(timings, precision, graph_cap, pools=pools, total=total, dual=dual)
    prof['calibration_s'] = round(clock() - t0, 1)
    return prof

def _get(t, kind, n, q=1):
    return t.get('%s@%d@%d' % (kind, n, q))

def choose(t, precision, graph_cap, pools=None, total=None, dual=False):
    prof = default(precision)
    prof['source'] = 'calibrated'
    prof['timings'] = dict(t)
    g1, g2 = (_get(t, 'graph', 512), _get(t, 'graph', graph_cap))
    if g1 and g2 and (graph_cap > 512):
        b = max(0.0, (g2 - g1) / (graph_cap - 512))
        prof['graph_a'], prof['graph_b'] = (round(g1 - b * 512, 3), round(b, 5))
    graph_ms = lambda n: prof['graph_a'] + prof['graph_b'] * n
    fast_best = lambda n: min((x for x in (_get(t, 'fast', n), _get(t, 'fast_graph', n), float('inf')) if x))
    exact = prof['exact_max']
    for n in (EXACT_FLOOR + 512, EXACT_CEIL):
        if n <= exact or fast_best(n) == float('inf'):
            continue
        ex = graph_ms(n) if n <= graph_cap else _get(t, 'exact', n)
        if ex is not None and ex <= fast_best(n):
            exact = n
        else:
            break
    prof['exact_max'] = exact
    budget = (total or 0) * GRAPH_BUDGET
    graph = exact // GRAPH_BUCKET * GRAPH_BUCKET
    prof['graph_max'] = min(graph, graph_cap // GRAPH_BUCKET * GRAPH_BUCKET) if graph_cap else graph
    pts = sorted((n for n in FAST_CANDIDATES if _get(t, 'fast_graph', n) and _get(t, 'fast', n)))
    if pts:
        top, used = (0, 0)
        for n in pts:
            eager, gr = (_get(t, 'fast', n), _get(t, 'fast_graph', n))
            per_tok = gr / n
            step = spacing(gr, per_tok)
            waste = step / 2 * per_tok
            pool = (pools or {}).get(n, 0)
            if budget and used + pool > budget:
                break
            if eager - gr > waste:
                top, used = (n, used + pool)
        prof['fast_buckets'] = fast_buckets(exact, top, t)
    _v42qit0g7jmeue13rbjti, f8 = (_get(t, 'fork', 1024, 2), _get(t, 'fork', 1024, 8))
    if _v42qit0g7jmeue13rbjti and f8:
        b = max(0.0, (f8 - _v42qit0g7jmeue13rbjti) / 6)
        prof['fork_a'], prof['fork_b'] = (round(_v42qit0g7jmeue13rbjti - 2 * b, 3), round(b, 4))
    prof['bf16_from'] = 0
    if dual and precision == 'int8':
        lens = [n for n in (2048, 8192, 32000) if _get(t, 'fast_bf16', n) and _get(t, 'fast', n)]
        faster = [n for n in lens if _get(t, 'fast_bf16', n) < _get(t, 'fast', n)]
        for n in lens:
            if all((m in faster for m in lens if m >= n)):
                prof['bf16_from'] = n
                break
    return prof

def spacing(call_ms, per_token_ms, low=128, high=512):
    s = int(2 * WASTE_SHARE * call_ms / max(per_token_ms, 1e-09)) // 128 * 128
    return max(low, min(high, s))

def fast_buckets(exact_max, top, t=None):
    out, n = ([], exact_max)
    while top and n < top:
        probe = next((p for p in FAST_CANDIDATES if p > n), FAST_CANDIDATES[-1])
        gr = _get(t or {}, 'fast_graph', probe)
        step = spacing(gr, gr / probe) if gr else 128 if n < 2048 else 256
        n = min(top, (n // step + 1) * step)
        out.append(n)
    return out