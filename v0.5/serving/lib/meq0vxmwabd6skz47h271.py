import argparse
import collections
import json
import os
import queue
import sys
import threading
import time
import types
from pathlib import Path
import torch
GRAPH_MAX = 512
EXACT_MAX = 1024
GRAPH_BUCKET = 64
WARM_BUCKETS = (64, 256, 512, 1024, 2048, 4096)
BATCH_ROWS = 64
BATCH_TOKENS = 32768
LONG_EVERY = 1
FORK_MIN = 32
EXACT_MAX_INT8 = 1536
GRAPH_MAX_INT8 = 1536
BATCH_GRAPH_SIZES = (16, 8)
BATCH_GRAPH_TOKENS = 4096
FAST_GRAPH_WINDOWS = (256, 512)
FAST_GRAPH_MAX_WINDOW = 1024
_ENCODE_LOCK = threading.Lock()
DESKTOP_FLAGS = ('--cache-ram', '0', '--no-cache-idle-slots', '--ctx-checkpoints', '0', '--sleep-idle-seconds', '-1')

def route_limits(precision, exact_max=0, graph_max=0):
    exact = int(exact_max) or (EXACT_MAX_INT8 if precision == 'int8' else EXACT_MAX)
    graph = int(graph_max) or (GRAPH_MAX_INT8 if precision == 'int8' else GRAPH_MAX)
    if exact <= 0 or graph <= 0 or graph % GRAPH_BUCKET:
        raise ValueError('exact-max must be positive and graph-max a positive multiple of %d' % GRAPH_BUCKET)
    return (exact, graph)

def install_glue(body):
    import m7ml5d1vtymnzohdbris7 as fork
    import m4042euhfp7ynv7osi045 as _vba9w2x4ebr0mfzzp69di
    fork.SERVE['conv_cl'] = True
    fork.SERVE['gqa_flash'] = True
    return {'norms': fork.install_norms(body), 'conv_channel_last': True, 'gqa_flash': True, 'flash_built': _vba9w2x4ebr0mfzzp69di.FLASH_BUILT, 'aot_attention': _vba9w2x4ebr0mfzzp69di.install_aot_attention()}

def _labels():
    from m6lir2n4n221f2rp5z93i import OPTION_LABELS
    return ['No', 'Yes'] + list(OPTION_LABELS)
_OPAQUE_ID = None
_TRUTH_IDS = {'true': 'c1001', 'false': 'c1002', 'unknown': 'c1003'}

def serving_row(row):
    global _OPAQUE_ID
    import re
    if str(row.get('kind', '')).lower() != 'choice' or not row.get('candidates'):
        return row
    if _OPAQUE_ID is None:
        _OPAQUE_ID = re.compile('(?:o|opt|option)[_ ]?(\\d+)', re.IGNORECASE)
    ids = [str(c.get('id', '')) for c in row['candidates']]
    mapped = []
    for cid in ids:
        m = _OPAQUE_ID.fullmatch(cid)
        mapped.append(f'c{m.group(1)}' if m else _TRUTH_IDS.get(cid, cid))
    if mapped == ids or len(set(mapped)) != len(mapped):
        return row
    return dict(row, candidates=[dict(c, id=new) for c, new in zip(row['candidates'], mapped)])

def encode(tokenizer, row):
    from m6lir2n4n221f2rp5z93i import escape_special, prompt_for
    text = prompt_for(row, tokenizer)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    state = row['state'] if isinstance(row['state'], str) else json.dumps(row['state'], ensure_ascii=False)
    start = text.index('<state>\n') + len('<state>\n')
    char = start + len(escape_special(state)) + len('\n</state>\n')
    boundary = next((i for i, (_, e) in enumerate(enc['offset_mapping']) if e > char), None)
    return (enc['input_ids'], boundary)
HEADER_PROBES = ('x', '\nx', '\n\nx', '\r\nx', '\rx', ' x', '\tx', '{"a": 1}', '[1]', chr(12288) + 'x', chr(8203) + 'x', '<x')

def stable_header(tokenizer, header):
    from m6lir2n4n221f2rp5z93i import prompt_for
    header = tuple(header)
    keep = len(header)
    for first in HEADER_PROBES:
        row = {'id': 'h', 'kind': 'noul', 'state': first, 'question': 'q?', 'candidates': [{'id': 'false', 'text': 'No'}, {'id': 'true', 'text': 'Yes'}]}
        ids = tokenizer(prompt_for(row, tokenizer), add_special_tokens=False)['input_ids']
        n = 0
        while n < keep and n < len(ids) and (ids[n] == header[n]):
            n += 1
        keep = n
    return header[:keep]

class HFJudge:

    def __init__(self, folder, device='cuda', max_length=32768, recipe=True, w8a8='off', exact_max=0, graph_max=0, quant_backend='aot', precision_route=False, memory_limit=0, state_dir=None):
        from transformers import AutoTokenizer
        from transformers import Qwen3_5ForCausalLM
        from transformers import Qwen3_5TextConfig
        import m8rhab841mmga1gnl3mp8 as fb
        import m59i8a56wgcx69l9bd9nz as w8_runtime
        folder = Path(folder)
        cuda = torch.device(device).type == 'cuda' and torch.cuda.is_available()
        if cuda and memory_limit:
            w8_runtime.limit_memory(device, memory_limit)
        t_load = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
        self.quant = {'precision': 'bf16', 'request': w8a8, 'reason': 'off'}
        want = recipe and w8a8 != 'off'
        pre = (folder / w8_runtime.FILE).is_file()
        has_bf16 = any(((folder / n).is_file() for n in ('model.safetensors', 'model.safetensors.index.json')))
        test = None
        shapes = None
        if want and quant_backend == 'aot' and cuda:
            shapes = w8_runtime.linear_shapes(folder)
            test = w8_runtime.selftest_cached(state_dir, shapes, device) or w8_runtime.selftest_start(shapes, device)
        model = None
        if pre and (want or not has_bf16):
            model, self.quant = w8_runtime.load_file(folder, device, w8a8 if want else 'off', backend=quant_backend, selftest=test, bf16_ok=has_bf16)
            test = None
            if precision_route and self.quant['precision'] == 'int8':
                self.quant['dual_refused'] = 'pre-converted weights'
        if model is None:
            config = Qwen3_5TextConfig.from_pretrained(folder, local_files_only=True)
            model = Qwen3_5ForCausalLM.from_pretrained(folder, config=config, dtype=torch.bfloat16, local_files_only=True)
            model = model.to(device).eval().requires_grad_(False)
            if want and (not pre):
                extra = {'dual': True} if precision_route else {}
                if test is not None:
                    extra['selftest'] = test
                self.quant = w8_runtime.apply(model.model, device, w8a8, backend=quant_backend, **extra)
            else:
                w8_runtime.selftest_cancel(test)
        self.dual = bool(self.quant.get('dual'))
        if shapes is not None:
            try:
                w8_runtime.selftest_remember(state_dir, shapes, device, self.quant.get('selftest'))
            except OSError:
                pass
        self.load_seconds = round(time.perf_counter() - t_load, 1)
        self.load_peak = torch.cuda.max_memory_reserved(torch.device(device)) if cuda else None
        self.glue = install_glue(model.model) if recipe else {}
        self.context_cap = None
        self.memory_profile = 'standard'
        if cuda:
            total = w8_runtime.total_memory(device)
            self.memory_profile, self.context_cap = w8_runtime.memory_plan(total, self.quant['precision'], self.dual, 'file' if self.quant.get('source') == 'file' else 'convert')
            if self.context_cap == 0:
                raise ValueError('this GPU has too little memory for the HF backend (use the GGUF backend)')
            max_length = min(int(max_length), self.context_cap)
        lean = self.memory_profile == 'lean'
        if lean and (not graph_max):
            graph_max = w8_runtime.LEAN_GRAPH_MAX
        self.batch_tokens = min(BATCH_TOKENS, max_length) if lean else BATCH_TOKENS
        ids = [self.tokenizer.encode(label, add_special_tokens=False) for label in _labels()]
        if any((len(i) != 1 for i in ids)):
            raise ValueError('invalid_model')
        index = torch.tensor([i[0] for i in ids], device=device)
        weight = model.get_output_embeddings().weight.detach().index_select(0, index).float().clone()

        def _v67ub3xu71brsc1w0l52j(x):
            with torch.autocast(device_type=x.device.type, enabled=False):
                return torch.nn.functional.linear(x.float(), weight).float()
        self.bench = fb.Bench(model.model, _v67ub3xu71brsc1w0l52j, self.tokenizer, header=fb.system_header(self.tokenizer))
        self.max_length, self.n_seq_max, self.recipe = (int(max_length), 1, bool(recipe))
        cuda = self.bench.device.type == 'cuda'
        self.graph = recipe and cuda
        self.exact_max, self.graph_max = route_limits(self.quant['precision'], exact_max, graph_max)
        self._v5wz3n1fab22ol5uhwgzb = fb._v83m8rh6inetzovw1w89s(self.bench, stable_header(self.tokenizer, self.bench.header), graph=self.graph, bucket=GRAPH_BUCKET, graph_max=self.graph_max)
        self.fast = None
        if recipe:
            import m5k7yldwzx2gowr44m6pk as _v6ur7q369jbjyz1zwxrpd
            import me8tcb8ro755mvb8keg6p as _v42qit0g7jmeue13rbjti
            import m4042euhfp7ynv7osi045 as frt
            self.fast = frt.FastRuntime(_v42qit0g7jmeue13rbjti.runtime(self.bench, _v6ur7q369jbjyz1zwxrpd.CUT, _v42qit0g7jmeue13rbjti.EXT), **frt.runtime_options())
        self.routes = {'graph': 0, 'exact': 0, 'long': 0}
        import m8ltkzlxslgbyp0pozv27 as route_tune
        self.profile = route_tune.default(self.quant['precision'])
        self.profile['exact_max'], self.profile['graph_max'] = (self.exact_max, self.graph_max)
        self.fixed_limits = bool(exact_max or graph_max)
        self.bf16_from = 0
        self.batch_bucket = GRAPH_BUCKET
        self.batch_graphs = {}

    def apply_profile(self, prof):
        prof = dict(prof)
        if self.fixed_limits:
            prof['exact_max'], prof['graph_max'] = (self.exact_max, self.graph_max)
        self.exact_max = int(prof['exact_max'])
        self.graph_max = min(int(prof['graph_max']), self.exact_max) // GRAPH_BUCKET * GRAPH_BUCKET
        self._v5wz3n1fab22ol5uhwgzb.graph_max = self.graph_max
        self.bf16_from = int(prof.get('bf16_from') or 0) if self.dual else 0
        if getattr(self, 'memory_profile', 'standard') == 'lean':
            prof['fast_buckets'] = []
        buckets = sorted((b for b in prof.get('fast_buckets') or () if b > self.exact_max and b <= self.max_length))
        prof['fast_buckets'] = buckets
        if self.fast is not None and self.graph:
            import m4042euhfp7ynv7osi045 as frt
            old = self.fast.graph
            self.fast.graph = frt.F1cGraph(self.fast.rt, buckets, lean=self.fast.lean, max_window=FAST_GRAPH_MAX_WINDOW) if buckets else None
            self.fast.use_graph = self.fast.graph is not None
            del old
        self.profile = prof
        return prof

    def graph_ms(self, width):
        return self.profile['graph_a'] + self.profile['graph_b'] * width

    def eager_ms(self, rows, shared=False):
        p = self.profile
        return p['fork_a'] * (1.0 if shared else 0.5) + p['fork_b'] * rows

    def replay_widths(self, enc):
        widths = []
        for e in enc:
            ids = e[3]
            start = self._v5wz3n1fab22ol5uhwgzb.H if self._v5wz3n1fab22ol5uhwgzb.H and len(ids) > self._v5wz3n1fab22ol5uhwgzb.H and (tuple(ids[:self._v5wz3n1fab22ol5uhwgzb.H]) == self._v5wz3n1fab22ol5uhwgzb.header_ids) else 0
            w = len(ids) - start
            if not (self.graph and w <= self._v5wz3n1fab22ol5uhwgzb.graph_max):
                return None
            widths.append(w)
        return widths

    def replay_cheaper(self, enc, shared):
        widths = self.replay_widths(enc)
        return widths is not None and sum((self.graph_ms(-(-w // GRAPH_BUCKET) * GRAPH_BUCKET) for w in widths)) < self.eager_ms(len(enc), shared)

    def long_logits(self, ids, s):
        x = torch.tensor([ids], device=self.bench.device)
        bf16_from = getattr(self, 'bf16_from', 0)
        if bf16_from and len(ids) >= bf16_from:
            import m59i8a56wgcx69l9bd9nz as w8_runtime
            graph, self.fast.graph = (self.fast.graph, None)
            w8_runtime.PRECISION['mode'] = 'bf16'
            try:
                z = self.fast.fast(x, s, ids, '_v9qzrgapy1ah7g1jj5alh', False)
            finally:
                w8_runtime.PRECISION['mode'] = 'int8'
                self.fast.graph = graph
            self.routes['long_bf16'] = self.routes.get('long_bf16', 0) + 1
        else:
            graph = getattr(self.fast, 'graph', None)
            if graph is not None and graph.plan(x, s, ids) is not None:
                self.routes['long_graph'] = self.routes.get('long_graph', 0) + 1
            z = self.fast.fast(x, s, ids, '_v9qzrgapy1ah7g1jj5alh', False)
        self.routes['long'] += 1
        return z.float().reshape(-1).clone()

    def tokenize(self, text, *, add_special=False):
        return self.tokenizer(text, add_special_tokens=add_special)['input_ids']

    def logits(self, row):
        ids, s = encode(self.tokenizer, row)
        if len(ids) > self.max_length:
            raise ValueError('Input exceeds max_length; evidence was not truncated')
        x = torch.tensor([ids], device=self.bench.device)
        if self.fast is not None and len(ids) > getattr(self, 'exact_max', EXACT_MAX) and s:
            z = self.long_logits(ids, s)
        else:
            self.routes['graph' if self._v5wz3n1fab22ol5uhwgzb.plan(x, ids)[2] else 'exact'] += 1
            z = self._v5wz3n1fab22ol5uhwgzb(x, ids)
        z = z.float().reshape(-1).clone()
        return (z[:2] if row['kind'].lower() == 'noul' else z[2:2 + len(row['candidates'])], len(ids))

    @torch.inference_mode()
    def judge(self, rows, *, share_prefix=False, report=None, reasoning='off', **options):
        if reasoning != 'off' or options:
            raise ValueError('only reasoning="off" is served')
        res = self.judge_many([rows])[0]
        if isinstance(res, Exception):
            raise res
        out, lengths, computed = res
        if report is not None:
            report['prompt_tokens'] = lengths
            report['input_tokens'] = computed
        return out

    @torch.inference_mode()
    def encode_request(self, rows):
        try:
            with _ENCODE_LOCK:
                out = []
                for row in rows:
                    if row.get('_v6gd7oavvss5c4jnyhxhc') is not None or row.get('images'):
                        raise ValueError('This release evaluates text only')
                    ids, s = encode(self.tokenizer, serving_row(row))
                    if len(ids) > self.max_length:
                        raise ValueError('Input exceeds max_length; evidence was not truncated')
                    out.append((ids, s))
                return out
        except ValueError as exc:
            return exc

    def is_long(self, encoded):
        limit = getattr(self, 'exact_max', EXACT_MAX)
        return not isinstance(encoded, Exception) and any((len(ids) > limit for ids, _ in encoded))

    @torch.inference_mode()
    def judge_many(self, requests, on_result=None, encoded=None):
        import m7ml5d1vtymnzohdbris7 as fork
        _v5wz3n1fab22ol5uhwgzb, dev = (self._v5wz3n1fab22ol5uhwgzb, self.bench.device)
        exact_max = getattr(self, 'exact_max', EXACT_MAX)
        out = [None] * len(requests)
        z, lengths, computed = ({}, {}, {})
        singles, forks, longs = ([], [], [])
        for i, rows in enumerate(requests):
            pre = encoded[i] if encoded is not None and encoded[i] is not None else self.encode_request(rows)
            if isinstance(pre, Exception):
                out[i] = pre
                continue
            enc = []
            for j, (row, (ids, s)) in enumerate(zip(rows, pre)):
                enc.append((i, j, row, ids, s))
                lengths[i, j] = len(ids)
            if len(enc) > 1:
                p = fork.common_prefix([e[3] for e in enc])
                start = _v5wz3n1fab22ol5uhwgzb.H if _v5wz3n1fab22ol5uhwgzb.H and p > _v5wz3n1fab22ol5uhwgzb.H and (tuple(enc[0][3][:_v5wz3n1fab22ol5uhwgzb.H]) == _v5wz3n1fab22ol5uhwgzb.header_ids) else 0
                if p - start >= FORK_MIN and (not (getattr(self, 'profile', None) and self.replay_cheaper(enc, True))):
                    forks.append((p, start, enc))
                    continue
                if p - start >= FORK_MIN:
                    self.routes['fork_graph'] = self.routes.get('fork_graph', 0) + len(enc)
                    singles.extend(((i, j, row, ids, s, 'replay') for i, j, row, ids, s in enc))
                    continue
            for e in enc:
                (longs if self.fast is not None and len(e[3]) > exact_max and e[4] else singles).append(e)

        def put(i, j, row, logits):
            z[i, j] = logits[:2] if row['kind'].lower() == 'noul' else logits[2:2 + len(row['candidates'])]
            computed[i] = computed.get(i, 0)

        def finish(i):
            if out[i] is None:
                rows = requests[i]
                out[i] = ([z[i, j].softmax(-1).tolist() for j in range(len(rows))], [lengths[i, j] for j in range(len(rows))], computed.get(i, 0))
            if on_result is not None:
                on_result(i, out[i])
        long_req = {i for i, rows in enumerate(requests) if out[i] is None and any((lengths[i, j] > exact_max for j in range(len(rows))))}
        for i in range(len(requests)):
            if out[i] is not None:
                finish(i)
        for phase in (False, True):
            ph_longs = [e for e in longs if (e[0] in long_req) is phase]
            ph_forks = [f for f in forks if (f[2][0][0] in long_req) is phase]
            ph_singles = [e for e in singles if (e[0] in long_req) is phase]
            self._run_phase(ph_longs, ph_forks, ph_singles, put, computed)
            for i in range(len(requests)):
                if out[i] is None and (i in long_req) is phase:
                    finish(i)
        return out

    def _run_phase(self, longs, forks, singles, put, computed):
        _v5wz3n1fab22ol5uhwgzb, dev = (self._v5wz3n1fab22ol5uhwgzb, self.bench.device)
        for i, j, row, ids, s in longs:
            put(i, j, row, self.long_logits(ids, s))
            computed[i] += len(ids)
        _vdnc499tgp6sg4zoz40ju = getattr(self, '_vdnc499tgp6sg4zoz40ju', None)
        if _vdnc499tgp6sg4zoz40ju is not None and forks:
            reqs = [dict(questions=[e[3] for e in short], boundary=min((e[4] or p for e in short))) for p, _, short in forks]
            for (p, _, short), zs in zip(forks, _vdnc499tgp6sg4zoz40ju(reqs)):
                i = short[0][0]
                for (_, j, row, _, _), zz in zip(short, zs):
                    put(i, j, row, zz.float().reshape(-1))
                computed[i] += p + sum((len(e[3]) - p for e in short))
            self.routes['_vdnc499tgp6sg4zoz40ju'] = self.routes.get('_vdnc499tgp6sg4zoz40ju', 0) + sum((len(short) for _, _, short in forks))
            forks = []
        cap = getattr(self, 'batch_tokens', BATCH_TOKENS)
        for p, start, short in forks:
            i = short[0][0]
            order = sorted(range(len(short)), key=lambda k: len(short[k][3]) - p)
            chunks, cur = ([], [])
            for k in order:
                w = len(short[k][3]) - p
                if cur and (len(cur) >= BATCH_ROWS or w * (len(cur) + 1) > cap):
                    chunks.append(cur)
                    cur = []
                cur.append(k)
            if cur:
                chunks.append(cur)
            for ks in chunks:
                states = _v5wz3n1fab22ol5uhwgzb.prefix_states(torch.tensor([short[0][3][start:p]], device=dev), start)
                tails = [short[k][3][p:] for k in ks]
                x = torch.zeros((len(tails), max(map(len, tails))), dtype=torch.long, device=dev)
                for n, t in enumerate(tails):
                    x[n, :len(t)] = torch.tensor(t, device=dev)
                zz = _v5wz3n1fab22ol5uhwgzb.run_batch(x, p, torch.tensor([p + len(t) for t in tails], device=dev), states).float()
                for n, k in enumerate(ks):
                    _, j, row, _, _ = short[k]
                    put(i, j, row, zz[n])
            self.routes['fork'] = self.routes.get('fork', 0) + len(short)
            computed[i] += p * len(chunks) + sum((len(e[3]) - p for e in short))
        groups, cur = ([], [])
        replay = [e[:5] for e in singles if len(e) > 5]
        singles = [e for e in singles if len(e) == 5]
        for e in sorted(singles, key=lambda e: len(e[3])):
            start = _v5wz3n1fab22ol5uhwgzb.H if _v5wz3n1fab22ol5uhwgzb.H and len(e[3]) > _v5wz3n1fab22ol5uhwgzb.H and (tuple(e[3][:_v5wz3n1fab22ol5uhwgzb.H]) == _v5wz3n1fab22ol5uhwgzb.header_ids) else 0
            width = len(e[3]) - start
            if cur and (cur[0][0] != start or len(cur) >= BATCH_ROWS or width * (len(cur) + 1) > getattr(self, 'batch_tokens', BATCH_TOKENS)):
                groups.append(cur)
                cur = []
            cur.append((start, width, e))
        if cur:
            groups.append(cur)
        split = []
        for g in groups:
            if len(g) > 1 and self.batch_graph_key(g) is None and getattr(self, 'profile', None) and self.replay_cheaper([e for _, _, e in g], False):
                self.routes['batch_graph'] = self.routes.get('batch_graph', 0) + len(g)
                split.extend(([x] for x in g))
            else:
                split.append(g)
        groups = split + [[(0, 0, e)] for e in replay]
        for g in groups:
            if len(g) == 1:
                _, _, (i, j, row, ids, _) = g[0]
                x = torch.tensor([ids], device=dev)
                self.routes['graph' if _v5wz3n1fab22ol5uhwgzb.plan(x, ids)[2] else 'exact'] += 1
                put(i, j, row, _v5wz3n1fab22ol5uhwgzb(x, ids).float().reshape(-1).clone())
            else:
                start = g[0][0]
                bucket = int(getattr(self, 'batch_bucket', 0) or 0) or 1
                wb = -(-max((w for _, w, _ in g)) // bucket) * bucket
                key = self.batch_graph_key(g)
                _v19ilmimnz2cxgcmg5d4x = self.batch_graphs.get(key) if key is not None else None
                x = _v19ilmimnz2cxgcmg5d4x.ids if _v19ilmimnz2cxgcmg5d4x is not None else torch.zeros((len(g), wb), dtype=torch.long, device=dev)
                if _v19ilmimnz2cxgcmg5d4x is not None:
                    x.zero_()
                for k, (_, w, e) in enumerate(g):
                    x[k, :w] = torch.tensor(e[3][start:], device=dev)
                limit = torch.tensor([start + w for _, w, _ in g], device=dev)
                if _v19ilmimnz2cxgcmg5d4x is not None:
                    _v19ilmimnz2cxgcmg5d4x.limit.copy_(limit)
                    _v19ilmimnz2cxgcmg5d4x.graph.replay()
                    zz = _v19ilmimnz2cxgcmg5d4x.out.float().clone()
                    self.routes['batch_graphed'] = self.routes.get('batch_graphed', 0) + len(g)
                else:
                    zz = _v5wz3n1fab22ol5uhwgzb.run_batch(x, start, limit).float()
                for k, (_, _, (i, j, row, _, _)) in enumerate(g):
                    put(i, j, row, zz[k])
                self.routes['batch'] = self.routes.get('batch', 0) + len(g)
            for _, _, (i, _, _, ids, _) in g:
                computed[i] += len(ids)

    def cache_stats(self):
        return {'routes': dict(self.routes)}

    def batch_graph_key(self, g):
        graphs = getattr(self, 'batch_graphs', None)
        if not graphs or g[0][0] != self._v5wz3n1fab22ol5uhwgzb.H:
            return None
        bucket = int(getattr(self, 'batch_bucket', 0) or 0) or 1
        key = (len(g), -(-max((w for _, w, _ in g)) // bucket) * bucket)
        return key if key in graphs else None

    def capture_batch(self, b, wb):
        _v5wz3n1fab22ol5uhwgzb, dev, start = (self._v5wz3n1fab22ol5uhwgzb, self.bench.device, self._v5wz3n1fab22ol5uhwgzb.H)
        ids = torch.zeros((b, wb), dtype=torch.long, device=dev)
        limit = torch.full((b,), start + wb, dtype=torch.long, device=dev)
        g = types.SimpleNamespace(ids=ids, limit=limit, graph=None, out=None)
        fn = lambda: _v5wz3n1fab22ol5uhwgzb.run_batch(g.ids, start, g.limit)
        if _v5wz3n1fab22ol5uhwgzb.pool is None:
            _v5wz3n1fab22ol5uhwgzb.pool = torch.cuda.graph_pool_handle()
        import m8rhab841mmga1gnl3mp8 as fb
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(fb._v6ff1myegbtwfltdvya89):
                fn()
        torch.cuda.current_stream().wait_stream(side)
        g.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g.graph, pool=_v5wz3n1fab22ol5uhwgzb.pool):
            g.out = fn()
        self.batch_graphs[b, wb] = g
        return g

    def defer_graphs(self):
        if not self.graph:
            return 0
        start = self._v5wz3n1fab22ol5uhwgzb.H
        plan = [('_v5wz3n1fab22ol5uhwgzb', (start, wb)) for wb in range(GRAPH_BUCKET, self._v5wz3n1fab22ol5uhwgzb.graph_max + 1, GRAPH_BUCKET)]
        g = self.fast.graph if self.fast is not None else None
        if g is not None:
            hs = self.fast.rt.H
            plan += [('fast', ('_v9qzrgapy1ah7g1jj5alh', hs, nb, min(wb, nb))) for nb in g.buckets for wb in FAST_GRAPH_WINDOWS if nb > hs]
            g.defer = True
        if self._v5wz3n1fab22ol5uhwgzb.H and self.recipe and (getattr(self, 'memory_profile', 'standard') != 'lean'):
            plan += [('batch', bw) for bw in self.batch_shapes(BATCH_GRAPH_SIZES, graphs=True)]
        self._v5wz3n1fab22ol5uhwgzb.defer = True
        self._v7o47h2cyt6c1gl8a4cof = collections.deque(plan)
        return len(plan)

    def save_new_tuning(self, every=60.0):
        directory = getattr(self, 'state_dir', None)
        if not directory or self.bench.device.type != 'cuda':
            return False
        now = time.perf_counter()
        if now - self.__dict__.get('_tuning_saved_at', 0.0) < every:
            return False
        self._tuning_saved_at = now
        import m92dxpxu92qra09wgs5nj as kernel_state
        state = kernel_state.capture(self.bench.device)
        n = sum((len(k['entries']) for k in (state or {}).get('kernels', {}).values()))
        if not state or n <= self.__dict__.get('_tuning_keys', 0):
            return False
        path = Path(directory) / KERNEL_PROFILE
        os.makedirs(directory, exist_ok=True)
        tmp = str(path) + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(state, f, sort_keys=True, indent=1, allow_nan=False)
        os.replace(tmp, path)
        self._tuning_keys = n
        return True

    def warm_state(self):
        _v5wz3n1fab22ol5uhwgzb, g = (self._v5wz3n1fab22ol5uhwgzb, self.fast.graph if self.fast is not None else None)
        ready = len(_v5wz3n1fab22ol5uhwgzb.graphs) + (len(g.graphs) if g is not None else 0) + len(self.batch_graphs)
        queued = len(self.__dict__.get('_v7o47h2cyt6c1gl8a4cof') or ()) + len(_v5wz3n1fab22ol5uhwgzb.pending) + (len(g.pending) if g is not None else 0)
        return {'graphs_ready': ready, 'graphs_queued': queued, 'warming': queued > 0}

    def capture_step(self):
        _v5wz3n1fab22ol5uhwgzb, g = (self._v5wz3n1fab22ol5uhwgzb, self.fast.graph if self.fast is not None else None)
        plan = self.__dict__.setdefault('_v7o47h2cyt6c1gl8a4cof', collections.deque())
        while True:
            if _v5wz3n1fab22ol5uhwgzb.pending:
                kind, key = ('_v5wz3n1fab22ol5uhwgzb', _v5wz3n1fab22ol5uhwgzb.pending.pop(0))
            elif g is not None and g.pending:
                kind, key = ('fast', g.pending.pop(0))
            elif plan:
                kind, key = plan.popleft()
            else:
                return False
            with torch.inference_mode():
                if kind == '_v5wz3n1fab22ol5uhwgzb' and key not in _v5wz3n1fab22ol5uhwgzb.graphs:
                    _v5wz3n1fab22ol5uhwgzb.capture(*key)
                elif kind == 'fast' and g is not None and (key not in g.graphs) and (len(g.graphs) < g.max_graphs):
                    g.capture(key, bounds=True)
                elif kind == 'batch' and key not in self.batch_graphs:
                    self.capture_batch(*key)
                else:
                    continue
            self.graphs_ready = len(_v5wz3n1fab22ol5uhwgzb.graphs) + (len(g.graphs) if g is not None else 0) + len(self.batch_graphs)
            return True

    def batch_shapes(self, sizes, graphs=False):
        top = min(self.exact_max, self.max_length) - self._v5wz3n1fab22ol5uhwgzb.H
        out = []
        for b in sizes:
            for wb in range(GRAPH_BUCKET, top // GRAPH_BUCKET * GRAPH_BUCKET + 1, GRAPH_BUCKET):
                if b * wb > getattr(self, 'batch_tokens', BATCH_TOKENS) or (graphs and b * wb > BATCH_GRAPH_TOKENS):
                    continue
                out.append((b, wb))
        return out

    def prewarm_batches(self, sizes=(16, 8, 4, 2)):
        _v5wz3n1fab22ol5uhwgzb, dev = (self._v5wz3n1fab22ol5uhwgzb, self.bench.device)
        n = 0
        with torch.inference_mode():
            for b, wb in self.batch_shapes(sizes):
                if (b, wb) in self.batch_graphs:
                    continue
                x = torch.zeros((b, wb), dtype=torch.long, device=dev)
                _v5wz3n1fab22ol5uhwgzb.run_batch(x, _v5wz3n1fab22ol5uhwgzb.H, torch.full((b,), _v5wz3n1fab22ol5uhwgzb.H + wb, dtype=torch.long, device=dev))
                n += 1
        return n

    def precapture(self):
        if not self.graph:
            return 0
        start = self._v5wz3n1fab22ol5uhwgzb.H
        with torch.inference_mode():
            for wb in range(GRAPH_BUCKET, self._v5wz3n1fab22ol5uhwgzb.graph_max + 1, GRAPH_BUCKET):
                if (start, wb) not in self._v5wz3n1fab22ol5uhwgzb.graphs:
                    self._v5wz3n1fab22ol5uhwgzb.capture(start, wb)
            g = self.fast.graph if self.fast is not None else None
            if g is not None:
                hs = self.fast.rt.H
                for nb in g.buckets:
                    for wb in FAST_GRAPH_WINDOWS:
                        key = ('_v9qzrgapy1ah7g1jj5alh', hs, nb, min(wb, nb))
                        if nb > hs and key not in g.graphs and (len(g.graphs) < g.max_graphs):
                            g.s_len.fill_(max(hs, nb - min(wb, nb)))
                            g.n_len.fill_(nb)
                            g.capture(key)
            if self._v5wz3n1fab22ol5uhwgzb.H and self.recipe and (getattr(self, 'memory_profile', 'standard') != 'lean'):
                for b, wb in self.batch_shapes(BATCH_GRAPH_SIZES, graphs=True):
                    if (b, wb) not in self.batch_graphs:
                        self.capture_batch(b, wb)
        return len(self._v5wz3n1fab22ol5uhwgzb.graphs) + (len(g.graphs) if g is not None else 0) + len(self.batch_graphs)

    def calibration_run(self, kind, n, q=1):
        import random
        import m59i8a56wgcx69l9bd9nz as w8_runtime
        _v5wz3n1fab22ol5uhwgzb, dev, fast = (self._v5wz3n1fab22ol5uhwgzb, self.bench.device, self.fast)
        if n > self.max_length:
            raise ValueError('over the served context')
        rng = random.Random(n * 31 + q)
        head = list(_v5wz3n1fab22ol5uhwgzb.header_ids)
        ids = head + [rng.randrange(1000, 60000) for _ in range(n - len(head))]
        x = torch.tensor([ids], device=dev)
        s = max(len(head), n - 160)
        if kind == 'graph':
            start, width, graphed = _v5wz3n1fab22ol5uhwgzb.plan(x, ids)
            if not graphed:
                raise ValueError('no graph at this length')
            return _v5wz3n1fab22ol5uhwgzb(x, ids)
        if kind == 'exact':
            start, width, _ = _v5wz3n1fab22ol5uhwgzb.plan(x, ids)
            return _v5wz3n1fab22ol5uhwgzb.run(x[:, start:], start, width)
        if kind in ('fast', 'fast_bf16'):
            graph, fast.graph = (fast.graph, None)
            w8_runtime.PRECISION['mode'] = 'bf16' if kind == 'fast_bf16' else 'int8'
            try:
                return fast.fast(x, s, ids, '_v9qzrgapy1ah7g1jj5alh', False)
            finally:
                w8_runtime.PRECISION['mode'] = 'int8'
                fast.graph = graph
        if kind == 'fast_graph':
            import m4042euhfp7ynv7osi045 as frt
            import m8ltkzlxslgbyp0pozv27 as route_tune
            g = self._probe_graph = getattr(self, '_probe_graph', None) or frt.F1cGraph(fast.rt, [route_tune.EXACT_FLOOR + 512] + list(route_tune.FAST_CANDIDATES), lean=fast.lean, max_graphs=16)
            out = g(x, s, ids, '_v9qzrgapy1ah7g1jj5alh')
            if out is None:
                raise ValueError('no fast graph')
            return out
        if kind == 'fork':
            start = _v5wz3n1fab22ol5uhwgzb.H
            tails = [[rng.randrange(1000, 60000) for _ in range(40)] for _ in range(q)]
            states = _v5wz3n1fab22ol5uhwgzb.prefix_states(x[:, start:], start)
            t = torch.tensor(tails, device=dev)
            return _v5wz3n1fab22ol5uhwgzb.run_batch(t, n, torch.full((q,), n + 40, device=dev), states)
        raise ValueError(kind)

    def tune(self, state_dir, mode='auto'):
        import m8ltkzlxslgbyp0pozv27 as route_tune
        prec = self.quant['precision']
        if getattr(self, 'memory_profile', 'standard') == 'lean':
            mode = 'off'
        report = {'mode': mode}
        prof = None
        if self.recipe and self.bench.device.type == 'cuda' and (mode != 'off'):
            dev = self.bench.device
            props = torch.cuda.get_device_properties(dev)
            key = route_tune.device_key(props.name, torch.cuda.get_device_capability(dev), props.total_memory, prec, self.dual)
            report['key'] = key
            prof = route_tune.load(state_dir, key)
            if prof is None:
                try:
                    with torch.inference_mode():
                        prof = route_tune.calibrate(self.calibration_run, prec, self.graph_max, sync=torch.cuda.synchronize, mem=lambda: torch.cuda.memory_reserved(dev), total=props.total_memory, dual=self.dual)
                    route_tune.save(state_dir, key, prof)
                except Exception as exc:
                    report['failed'] = type(exc).__name__
                    prof = None
                finally:
                    self._probe_graph = None
                    torch.cuda.empty_cache()
        if prof is None:
            prof = route_tune.default(prec)
            prof['exact_max'], prof['graph_max'] = (self.exact_max, self.graph_max)
        prof = self.apply_profile(prof)
        report.update({k: prof.get(k) for k in ('source', 'exact_max', 'graph_max', 'fast_buckets', 'bf16_from', 'calibration_s')})
        report['precision_route'] = 'on (BF16 from %d tokens)' % self.bf16_from if self.bf16_from else 'on (INT8 everywhere: BF16 not faster)' if self.dual else 'off'
        return report
BATCH_SIZES = (16, 8)
BATCH_SIZES_EAGER = (16, 8, 4, 2)
PAD_MAX = 0.5

class Batcher:

    def __init__(self, judge, max_rows=BATCH_ROWS * 4, policy='auto'):
        if policy not in ('auto', 'shaped', 'greedy'):
            raise ValueError('policy must be auto, shaped or greedy')
        self._judge, self._max, self._policy = (judge, max_rows, policy)
        self._queue = queue.Queue()
        self.lanes = {'short_cycles': 0, 'long_units': 0}
        self.shapes = {}
        threading.Thread(target=self._loop, daemon=True, name='batcher').start()

    def __getattr__(self, name):
        return getattr(self._judge, name)

    def cache_stats(self):
        out = self._judge.cache_stats()
        out['lanes'] = dict(getattr(self, 'lanes', None) or {})
        out['short_units'] = dict(self.__dict__.get('shapes') or {})
        return out

    def judge(self, rows, *, share_prefix=False, report=None, reasoning='off', **options):
        if reasoning != 'off' or options:
            raise ValueError('only reasoning="off" is served')
        item = {'rows': rows, 'done': threading.Event(), 'result': None}
        pre = getattr(self._judge, 'encode_request', None)
        if pre is not None:
            item['enc'] = pre(rows)
            item['long'] = bool(self._judge.is_long(item['enc']))
        self._queue.put(item)
        item['done'].wait()
        res = item['result']
        if isinstance(res, BaseException):
            raise res
        out, lengths, computed = res
        if report is not None:
            report['prompt_tokens'] = lengths
            report['input_tokens'] = computed
        return out

    def _run(self, items):

        def answer(k, res):
            items[k]['result'] = res
            items[k]['done'].set()
        kwargs = {}
        if any(('enc' in it for it in items)):
            kwargs['encoded'] = [it.get('enc') for it in items]
        try:
            results = self._judge.judge_many([it['rows'] for it in items], on_result=answer, **kwargs)
        except BaseException as exc:
            results = [exc] * len(items)
        for it, res in zip(items, results):
            if not it['done'].is_set():
                it['result'] = res
                it['done'].set()

    def _mode(self):
        policy = self.__dict__.get('_policy', 'auto')
        if policy != 'auto':
            return policy
        judge = self._judge
        if not hasattr(judge, 'encode_request'):
            return 'greedy'
        device = getattr(getattr(judge, 'bench', None), 'device', None)
        return 'shaped' if getattr(device, 'type', None) == 'cuda' else 'greedy'

    def _loop(self):
        if self.__dict__.get('lanes') is None:
            self.lanes = {'short_cycles': 0, 'long_units': 0}
        return self._loop_shaped() if self._mode() == 'shaped' else self._loop_greedy()

    def _wait(self):
        step = getattr(self._judge, 'capture_step', None)
        while step is not None:
            try:
                return self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                more = step()
            except Exception as exc:
                print(json.dumps({'graph_capture_failed': type(exc).__name__, 'detail': str(exc)[:200]}), file=sys.stderr, flush=True)
                more = True
            if not more:
                if not self.__dict__.get('_graphs_logged'):
                    self._graphs_logged = True
                    print(json.dumps({'graphs_ready': getattr(self._judge, 'graphs_ready', 0)}), flush=True)
                break
        save = getattr(self._judge, 'save_new_tuning', None)
        if save is not None:
            try:
                save()
            except Exception as exc:
                print(json.dumps({'kernel_profile_save_failed': type(exc).__name__}), file=sys.stderr, flush=True)
        return self._queue.get()

    def _shape(self, item):
        enc = item.get('enc')
        if enc is None or isinstance(enc, BaseException) or len(enc) != 1:
            return None
        ids = enc[0][0]
        _v5wz3n1fab22ol5uhwgzb = getattr(self._judge, '_v5wz3n1fab22ol5uhwgzb', None)
        h = int(getattr(_v5wz3n1fab22ol5uhwgzb, 'H', 0) or 0)
        start = h if h and len(ids) > h and (tuple(ids[:h]) == tuple(_v5wz3n1fab22ol5uhwgzb.header_ids)) else 0
        return (start, len(ids) - start)

    def _units(self, waiting):
        judge = self._judge
        _v5wz3n1fab22ol5uhwgzb = getattr(judge, '_v5wz3n1fab22ol5uhwgzb', None)
        graph_max = int(getattr(_v5wz3n1fab22ol5uhwgzb, 'graph_max', 0) or 0) if getattr(judge, 'graph', False) else 0
        bucket = int(getattr(judge, 'batch_bucket', 0) or 0)
        shape = {id(it): self._shape(it) for it in waiting}
        units, left = ([], list(waiting))
        while left:
            head, unit = (left[0], [left[0]])
            hs = shape[id(head)]
            if hs is not None:
                peers = sorted((it for it in left if shape[id(it)] is not None and shape[id(it)][0] == hs[0]), key=lambda it: shape[id(it)][1])
                k = next((n for n, it in enumerate(peers) if it is head))
                sizes = BATCH_SIZES_EAGER if hs[1] > graph_max else BATCH_SIZES if bucket else ()
                for size in sizes:
                    best = None
                    for a in range(max(0, k - size + 1), min(k, len(peers) - size) + 1):
                        ws = [shape[id(it)][1] for it in peers[a:a + size]]
                        width = -(-ws[-1] // bucket) * bucket if bucket else ws[-1]
                        real, pad = (sum(ws), size * width - sum(ws))
                        if size <= BATCH_ROWS and size * ws[-1] <= BATCH_TOKENS and (pad <= PAD_MAX * real) and (best is None or pad < best[0]):
                            best = (pad, a)
                    if best is not None:
                        unit = peers[best[1]:best[1] + size]
                        break
            taken = {id(it) for it in unit}
            left = [it for it in left if id(it) not in taken]
            units.append(unit)
        return units

    def _loop_shaped(self):
        lanes, shapes = (self.lanes, self.__dict__.setdefault('shapes', {}))
        pending, waiting = (collections.deque(), [])
        owed, fresh = (set(), False)
        while True:
            new = [] if waiting or pending else [self._wait()]
            rows = sum((len(it['rows']) for it in waiting + new))
            while rows < self._max:
                try:
                    new.append(self._queue.get_nowait())
                except queue.Empty:
                    break
                rows += len(new[-1]['rows'])
            idle = not pending
            for it in new:
                (pending if it.get('long') else waiting).append(it)
            if fresh or (pending and idle):
                owed, fresh = ({id(it) for it in waiting}, False)
            if pending and (not waiting or not owed):
                self._run([pending.popleft()])
                lanes['long_units'] += 1
                fresh = True
                continue
            done = []
            for unit in self._units(waiting):
                self._run(unit)
                shapes[len(unit)] = shapes.get(len(unit), 0) + 1
                done.extend(unit)
                if len(unit) == 1:
                    break
            gone = {id(it) for it in done}
            waiting = [it for it in waiting if id(it) not in gone]
            owed -= gone
            lanes['short_cycles'] += 1

    def _loop_greedy(self):
        lanes = self.lanes
        pending = collections.deque()
        carry, streak = ([], 0)
        while True:
            items, carry = (carry, [])
            rows = sum((len(it['rows']) for it in items))
            if not items and (not pending):
                items.append(self._wait())
                rows = len(items[0]['rows'])
            while rows < self._max:
                try:
                    items.append(self._queue.get_nowait())
                except queue.Empty:
                    break
                rows += len(items[-1]['rows'])
            if not pending:
                streak = 0
            short = [it for it in items if not it.get('long')]
            pending.extend((it for it in items if it.get('long')))
            if pending and (streak >= LONG_EVERY or not short):
                carry = short
                self._run([pending.popleft()])
                lanes['long_units'] += 1
                streak = 0
            elif short:
                self._run(short)
                lanes['short_cycles'] += 1
                streak += 1
KERNEL_PROFILE = 'kernel_profile_s2.json'
_v8avhrah2rilxek288ji1 = 'profiles'

def _ve2xl32kur2wbir4pc0oq():
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kernels', _v8avhrah2rilxek288ji1)
    return sorted((os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.z'))) if os.path.isdir(folder) else []

def read_profile(path):
    import zlib
    with open(path, 'rb') as f:
        data = f.read()
    return json.loads(zlib.decompress(data) if path.endswith('.z') else data)

def kernel_profile(judge, directory, warm, prewarm=False):
    import m92dxpxu92qra09wgs5nj as kernel_state
    device = judge.bench.device
    path = Path(directory) / KERNEL_PROFILE
    report = {'restored': False, 'saved': False}
    if device.type != 'cuda':
        warm()
        return report
    for candidate in ([str(path)] if path.is_file() else []) + _ve2xl32kur2wbir4pc0oq():
        try:
            kernel_state.restore(read_profile(candidate), device)
            report['restored'] = True
            report['source'] = 'state-dir' if candidate == str(path) else 'shipped'
            break
        except (ValueError, KeyError, OSError) as exc:
            report.setdefault('restore_refused', []).append(type(exc).__name__)
    warm()
    if not report['restored']:
        runner = getattr(judge, 'prewarm_batches', None)
        if prewarm and runner is not None:
            print(json.dumps({'starting': 'batch shapes'}), flush=True)
            t = time.perf_counter()
            report['batch_shapes'] = runner()
            report['batch_prewarm_s'] = round(time.perf_counter() - t, 1)
        state = kernel_state.capture(device)
        if state is not None:
            try:
                os.makedirs(directory, exist_ok=True)
                path.write_text(json.dumps(state, sort_keys=True, indent=1, allow_nan=False), encoding='utf-8')
                report['saved'] = True
            except OSError:
                pass
    return report

def gguf_judge(gguf, tokenizer_dir, runtime_dir, n_ctx=8192, seqs=4, flash_attn=-1, recipe=True):
    from transformers import AutoTokenizer
    from m1lb4jzkaeky3louxngpd import DirectJudge

    class Judge(DirectJudge):

        def judge(self, rows, *args, **kwargs):
            return super().judge([serving_row(r) for r in rows], *args, **kwargs)

        def select_experts(self, rows):
            if any((row.get('_v6gd7oavvss5c4jnyhxhc') for row in rows)):
                raise ValueError('This release serves the common model only')
            return [{'_v6gd7oavvss5c4jnyhxhc': None, 'source': 'no_experts', 'confidence': None} for _ in rows]
    tok = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    kw = dict(runtime_dir=runtime_dir, n_ctx=n_ctx, n_seq_max=seqs, max_length=n_ctx, flash_attn=flash_attn, n_ubatch=512)
    if recipe:
        kw.update(device_prefix_cache=True, device_prefix_slots=1, token_cache_entries=256)
    return Judge(gguf, tok, **kw)
WARM_FORK_QUESTIONS = 4

def warm_up(judge, buckets=WARM_BUCKETS):
    import m9qy9tkk89d48t0lxgr4e as ts
    buckets = [b for b in buckets if b < judge.max_length]
    report = ts._v5fk4i78hn5w2galet9gy(judge, buckets, max_questions=judge.n_seq_max)
    if hasattr(judge, 'judge_many'):
        for length in buckets:
            rows = ts._v2lbiu05abgrrc954oo6t(judge.tokenizer, length, 'choice', questions=WARM_FORK_QUESTIONS)
            t = time.perf_counter()
            judge.judge(rows, share_prefix=True)
            report.append({'bucket': length, 'kind': 'choice', 'questions': len(rows), 'route': 'fork', 'ms': round((time.perf_counter() - t) * 1000.0, 1)})
            pair = [ts._v2lbiu05abgrrc954oo6t(judge.tokenizer, length, kind) for kind in ('noul', 'choice')]
            t = time.perf_counter()
            judge.judge_many(pair)
            report.append({'bucket': length, 'requests': len(pair), 'route': 'batch', 'ms': round((time.perf_counter() - t) * 1000.0, 1)})
    return report

def stage(name, t0):
    print(json.dumps({'starting': name, 'elapsed_s': round(time.perf_counter() - t0, 1)}), flush=True)

def build(a):
    recipe = a.mode == 'recipe'
    report = {'mode': a.mode, 'backend': a.backend}
    t = time.perf_counter()
    stage('load', t)
    if a.backend == 'hf':
        judge = HFJudge(a.model, device=a.device, max_length=a.max_length, recipe=recipe, w8a8=getattr(a, 'w8a8', 'off') if recipe else 'off', exact_max=getattr(a, 'exact_max', 0), graph_max=getattr(a, 'graph_max', 0), precision_route=recipe and getattr(a, 'precision_route', 'off') == 'on', memory_limit=int(float(getattr(a, 'memory_limit', 0) or 0) * 2 ** 30), state_dir=getattr(a, 'state_dir', None))
        judge.state_dir = a.state_dir
        report['precision'] = {k: judge.quant[k] for k in ('precision', 'request', 'reason', 'dual', 'dual_refused', 'source', 'selftest') if k in judge.quant}
        report['max_length'], report['context_cap'] = (judge.max_length, judge.context_cap)
        report['memory_profile'], report['load_seconds'] = (judge.memory_profile, judge.load_seconds)
        report['glue'] = judge.glue
        if recipe:
            graphs = getattr(a, 'graphs', 'background')
            tune = getattr(a, 'route_tune', 'off')
            if graphs == 'start' or tune == 'auto':
                stage('cuda graphs', t)
                judge.precapture()
            t2 = time.perf_counter()
            stage('routes', t)
            report['routes'] = judge.tune(a.state_dir, tune)
            report['routes']['tune_seconds'] = round(time.perf_counter() - t2, 1)
            if graphs == 'start':
                report['graphs'] = judge.precapture()
            else:
                report['graphs'] = {'mode': 'background', 'queued': judge.defer_graphs()}
            stage('kernel warm-up (first start on this GPU tunes kernels; later starts reuse --state-dir)', t)
            report['kernels'] = kernel_profile(judge, a.state_dir, lambda: warm_up(judge), prewarm=getattr(a, 'prewarm_batches', 'off') == 'on')
        report['exact_max'], report['graph_max'] = (judge.exact_max, judge.graph_max)
        if judge.bench.device.type == 'cuda':
            torch.cuda.empty_cache()
        if judge.load_peak is not None:
            dev = judge.bench.device
            report['memory'] = {'load_peak_gib': round(judge.load_peak / 2 ** 30, 2), 'start_peak_gib': round(torch.cuda.max_memory_reserved(dev) / 2 ** 30, 2), 'reserved_gib': round(torch.cuda.memory_reserved(dev) / 2 ** 30, 2), 'allocated_gib': round(torch.cuda.memory_allocated(dev) / 2 ** 30, 2)}
    else:
        judge = gguf_judge(a.model, a.tokenizer, a.runtime, n_ctx=a.n_ctx, seqs=a.seqs, flash_attn=a.flash_attn, recipe=recipe)
        if recipe:
            warm_up(judge)
    report['start_seconds'] = round(time.perf_counter() - t, 1)
    return (judge, report)

def parser():
    ap = argparse.ArgumentParser(prog='serve')
    ap.add_argument('--backend', choices=('hf', 'gguf'), default='hf')
    ap.add_argument('--mode', choices=('recipe', 'plain'), default='recipe')
    ap.add_argument('--model', help='merged HF folder (hf) or .gguf file (gguf); hf default: the INT8 weights, downloaded once into <state-dir>/weights')
    ap.add_argument('--tokenizer', help='HF tokenizer folder (gguf)')
    ap.add_argument('--runtime', help='folder with the llama.cpp shared library (gguf)')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--max-length', type=int, default=32768)
    ap.add_argument('--n-ctx', type=int, default=8192)
    ap.add_argument('--seqs', type=int, default=4)
    ap.add_argument('--flash-attn', type=int, default=-1)
    ap.add_argument('--w8a8', choices=('auto', 'off'), default='auto', help='hf recipe: 8-bit weights when this GPU supports them and has enough memory (auto), else bfloat16')
    ap.add_argument('--route-tune', choices=('auto', 'off'), default='off', help='hf recipe: time the routes once per GPU and save the limits in --state-dir (off: defaults)')
    ap.add_argument('--precision-route', choices=('off', 'on'), default='off', help='hf recipe, opt-in: keep the bfloat16 weights too and run long inputs in bfloat16 where that is faster on this GPU')
    ap.add_argument('--memory-limit', type=float, default=0, help='hf: plan and allocate as on a card with this many GiB (0 = the whole card)')
    ap.add_argument('--exact-max', type=int, default=0, help='hf: exact path up to this many prompt tokens (0 = auto)')
    ap.add_argument('--graph-max', type=int, default=0, help='hf: CUDA graphs up to this many tail tokens (0 = auto)')
    ap.add_argument('--state-dir', default=str(Path.home() / '.cache' / 'systemone_runtime'))
    ap.add_argument('--graphs', choices=('background', 'start'), default='background', help='hf recipe: capture CUDA graphs after the server is up, while idle (background: answers at once, eagerly until each graph exists) or all before serving (start)')
    ap.add_argument('--prewarm-batches', choices=('off', 'on'), default='off', help='hf recipe: tune every concurrent batch shape at the first start (on) instead of at first use')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8010)
    ap.add_argument('--name', default=DEFAULT_NAME)
    return ap
HF_REPO = 'sky7350/Mica-v0.5-4B'
DEFAULT_NAME = 'mica-v0.5-4b'

def preflight(a):
    if sys.version_info < (3, 10):
        raise SystemExit('This server needs Python 3.10 or newer (this is Python %d.%d).' % sys.version_info[:2])
    if a.backend == 'gguf' and (not (a.model and a.tokenizer and a.runtime)):
        raise SystemExit('--backend gguf needs --model <file.gguf>, --tokenizer <folder> and --runtime <llama.cpp build>')
    if a.backend == 'hf' and str(a.device).startswith('cuda') and (not torch.cuda.is_available()):
        raise SystemExit('PyTorch sees no NVIDIA GPU (torch %s, CUDA build %s). Install the CUDA build of torch as in the README (on Windows: requirements-windows.txt), or use the GGUF backend on llama.cpp for AMD, Apple or CPU.' % (torch.__version__, torch.version.cuda or 'none'))

def fetch_weights(state_dir):
    import m59i8a56wgcx69l9bd9nz as w8_runtime
    target = Path(state_dir) / 'weights'
    folder = target / 'int8'
    if (folder / w8_runtime.FILE).is_file() and (folder / 'config.json').is_file():
        return str(folder)
    print('Downloading the INT8 weights (about 4.9 GB) from %s into %s. This happens once; later starts reuse them.' % (HF_REPO, target), flush=True)
    from huggingface_hub import snapshot_download
    snapshot_download(HF_REPO, allow_patterns=['int8/*'], local_dir=str(target))
    return str(folder)

def ready_message(a, report):
    url = 'http://%s:%d/v1/systemone' % (a.host, a.port)
    text = 'Ready: POST %s (model %s, %s).' % (url, a.name, (report.get('precision') or {}).get('precision', a.backend))
    graphs = report.get('graphs')
    if isinstance(graphs, dict) and graphs.get('mode') == 'background':
        text += ' CUDA graphs are being captured in the background; until then short requests run a little slower. GET /health shows how many are done.'
    return text

def main(argv=None):
    import m9qy9tkk89d48t0lxgr4e as ts
    from http.server import ThreadingHTTPServer
    a = parser().parse_args(argv)
    preflight(a)
    if a.backend == 'hf' and (not a.model):
        a.model = fetch_weights(a.state_dir)
    if not os.environ.get('TRITON_CACHE_DIR'):
        os.environ['TRITON_CACHE_DIR'] = str(Path(a.state_dir) / 'triton')
    judge, report = build(a)
    if a.backend == 'hf' and a.mode == 'recipe':
        judge = Batcher(judge)
    ts.JUDGE, ts.MODEL_NAME = (judge, a.name)
    handler = ts.Handler
    if a.mode == 'recipe':
        handler = ts.desktop_handler(types.SimpleNamespace(keep_alive=True, keep_alive_timeout=15.0, keep_warm_seconds=0))
    try:
        server = ThreadingHTTPServer((a.host, a.port), handler)
    except OSError as exc:
        raise SystemExit('Cannot listen on %s:%d (%s). Another server may be using the port; pass --port.' % (a.host, a.port, exc.strerror or exc))
    print(json.dumps({'serving': f'http://{a.host}:{a.port}/v1/systemone', 'model': a.name, 'start': report}), flush=True)
    print(ready_message(a, report), flush=True)
    server.serve_forever()

def llama_server_command(server_bin, gguf, port=18084, n_ctx=8192, slots=4):
    return [str(server_bin), '-m', str(gguf), '--port', str(port), '-c', str(n_ctx * slots), '-np', str(slots), '-ngl', '99', *DESKTOP_FLAGS]