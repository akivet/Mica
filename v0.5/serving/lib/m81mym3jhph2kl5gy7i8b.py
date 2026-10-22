from __future__ import annotations
import contextvars
import json
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager

class LRUCache:

    def __init__(self, max_entries=256, max_cost=None):
        if type(max_entries) is not int or max_entries < 0:
            raise ValueError('max_entries must be a non-negative integer')
        self.max_entries, self.max_cost = (max_entries, max_cost)
        self._data = OrderedDict()
        self._cost = 0
        self._lock = threading.Lock()
        self.hits = self.misses = self.evictions = 0

    def get(self, key, default=None):
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self.misses += 1
                return default
            self._data.move_to_end(key)
            self.hits += 1
            return entry[0]

    def put(self, key, value, cost=1):
        with self._lock:
            if self.max_entries == 0 or (self.max_cost is not None and cost > self.max_cost):
                return False
            old = self._data.pop(key, None)
            if old is not None:
                self._cost -= old[1]
            self._data[key] = (value, cost)
            self._cost += cost
            while len(self._data) > self.max_entries or (self.max_cost is not None and self._cost > self.max_cost):
                _key, (_value, dropped) = self._data.popitem(last=False)
                self._cost -= dropped
                self.evictions += 1
            return True

    def clear(self):
        with self._lock:
            self._data.clear()
            self._cost = 0

    def __len__(self):
        return len(self._data)

    def stats(self):
        with self._lock:
            return {'entries': len(self._data), 'cost': self._cost, 'hits': self.hits, 'misses': self.misses, 'evictions': self.evictions}

class TokenizationCache:

    def __init__(self, fn, max_entries=256, max_chars=4000000):
        self.fn = fn
        self.cache = LRUCache(max_entries, max_chars)

    def __call__(self, text, add_special=False):
        key = (text, bool(add_special))
        hit = self.cache.get(key)
        if hit is not None:
            return list(hit)
        tokens = self.fn(text, add_special)
        if self.cache.max_cost is None or len(text) * 4 <= self.cache.max_cost:
            self.cache.put(key, tuple(tokens), cost=len(text) + 8 * len(tokens))
        return list(tokens)

    def clear(self):
        self.cache.clear()

    def stats(self):
        return self.cache.stats()

def render_key(row, tokenizer_id=None):
    try:
        body = {k: v for k, v in row.items() if k != 'id'}
        return (tokenizer_id, json.dumps(body, sort_keys=True, ensure_ascii=False, default=repr))
    except (TypeError, ValueError):
        return None

def common_token_prefix(sequences):
    sequences = [s for s in sequences]
    if not sequences:
        return 0
    shortest = min((len(s) for s in sequences))
    for index in range(shortest):
        value = sequences[0][index]
        if any((s[index] != value for s in sequences)):
            return index
    return shortest

def template_prefix_length(token_lists, *, back_off=1, min_tokens=8):
    if len(token_lists) < 2:
        return 0
    n = common_token_prefix(token_lists) - back_off
    return n if n >= min_tokens else 0
_PROBE_STATES = ('alpha', '1234', '\n', ' ', '\t', '\r', '\u3000', '\u200b', '한국어', '{"a": 1}', '<b>', '- item', '"q"', '$5')

def probe_rows(states=_PROBE_STATES):
    rows = []
    for index, state in enumerate(states):
        if index % 3 == 0:
            rows.append({'id': f'p{index}', 'kind': 'noul', 'state': state, 'question': 'q?', 'candidates': [{'id': 'false', 'text': 'No'}, {'id': 'true', 'text': 'Yes'}]})
        elif index % 3 == 1:
            rows.append({'id': f'p{index}', 'kind': 'choice', 'state': state, 'question': 'which?', 'candidates': [{'id': 'a', 'text': 'A'}, {'id': 'b', 'text': 'B'}]})
        else:
            rows.append({'id': f'p{index}', 'kind': 'score', 'state': state, 'question': 'how much?', 'candidates': [{'id': '0', 'text': 'Low'}, {'id': '1', 'text': 'High'}]})
    return rows

class PrefixEntry:
    __slots__ = ('tokens', 'identity', 'seq', 'pinned', 'tick')

    def __init__(self, tokens, identity, seq, pinned, tick):
        self.tokens, self.identity, self.seq, self.pinned, self.tick = (list(tokens), identity, seq, pinned, tick)

    def __repr__(self):
        return f'PrefixEntry(seq={self.seq}, n={len(self.tokens)}, pinned={self.pinned})'

class PrefixSlotCache:

    def __init__(self, seq_ids):
        self.seq_ids = list(seq_ids)
        self.entries = []
        self._tick = 0
        self.hits = self.misses = self.builds = self.evictions = self.invalid = 0

    def _touch(self, entry):
        self._tick += 1
        entry.tick = self._tick

    def match(self, prompts, identity, valid=None):
        best = None
        for entry in self.entries:
            n = len(entry.tokens)
            if entry.identity != identity or n == 0 or (best is not None and n <= len(best.tokens)):
                continue
            if all((len(p) > n and p[:n] == entry.tokens for p in prompts)):
                if valid is not None and (not valid(entry)):
                    self.drop(entry)
                    self.invalid += 1
                    continue
                best = entry
        if best is None:
            self.misses += 1
        else:
            self.hits += 1
            self._touch(best)
        return best

    def reserve(self, tokens, identity, *, pinned=False):
        for entry in self.entries:
            if entry.identity == identity and entry.tokens == list(tokens):
                self.drop(entry)
                break
        used = {e.seq for e in self.entries}
        free = [s for s in self.seq_ids if s not in used]
        if free:
            seq = free[0]
        else:
            victims = [e for e in self.entries if not e.pinned] or [e for e in self.entries if e.identity != identity]
            if not victims:
                return None
            victim = min(victims, key=lambda e: e.tick)
            self.drop(victim)
            self.evictions += 1
            seq = victim.seq
        entry = PrefixEntry(tokens, identity, seq, pinned, 0)
        self._touch(entry)
        self.entries.append(entry)
        self.builds += 1
        return entry

    def drop(self, entry):
        if entry in self.entries:
            self.entries.remove(entry)

    def clear(self):
        self.entries.clear()

    def stats(self):
        return {'entries': [(e.seq, len(e.tokens), e.pinned) for e in self.entries], 'hits': self.hits, 'misses': self.misses, 'builds': self.builds, 'evictions': self.evictions, 'invalid': self.invalid}
STAGES = ('parse', 'render', 'tokenize', 'prefix_cache', 'prefill', 'decode', '_v67ub3xu71brsc1w0l52j', 'cpu_tool', 'reread', 'respond')
_MAX_EVENTS = 128
_CURRENT = contextvars.ContextVar('latency_tracer', default=None)

class _NullSpan:

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
_NULL = _NullSpan()

class LatencyTracer:

    def __init__(self, clock=time.perf_counter, t0=None):
        self.clock = clock
        self.t0 = clock() if t0 is None else t0
        self.stages = {}
        self.phases = {}
        self.events = []
        self.counters = {}
        self._stack = []

    @contextmanager
    def stage(self, name):
        frame = [name, self.clock(), 0.0]
        self._stack.append(frame)
        try:
            yield
        finally:
            end = self.clock()
            self._stack.pop()
            total = end - frame[1]
            own = max(0.0, total - frame[2])
            if self._stack:
                self._stack[-1][2] += total
            slot = self.stages.setdefault(name, [0.0, 0])
            slot[0] += own * 1000.0
            slot[1] += 1
            if len(self.events) < _MAX_EVENTS:
                self.events.append({'stage': name, 'at_ms': round((frame[1] - self.t0) * 1000.0, 3), 'ms': round(total * 1000.0, 3)})

    @contextmanager
    def phase(self, name):
        start = self.clock()
        try:
            yield
        finally:
            elapsed = self.clock() - start
            slot = self.phases.setdefault(name, [0.0, 0])
            slot[0] += elapsed * 1000.0
            slot[1] += 1
            if len(self.events) < _MAX_EVENTS:
                self.events.append({'phase': name, 'at_ms': round((start - self.t0) * 1000.0, 3), 'ms': round(elapsed * 1000.0, 3)})

    def add(self, name, ms):
        slot = self.stages.setdefault(name, [0.0, 0])
        slot[0] += float(ms)
        slot[1] += 1

    def count(self, name, value=1):
        self.counters[name] = self.counters.get(name, 0) + value

    def note(self, name, value):
        self.counters[name] = value

    def elapsed_ms(self):
        return (self.clock() - self.t0) * 1000.0

    def report(self, total_ms=None):
        total = self.elapsed_ms() if total_ms is None else float(total_ms)
        stages = {name: round(slot[0], 3) for name, slot in self.stages.items()}
        out = {'total_ms': round(total, 3), 'stages_ms': stages, 'stage_calls': {name: slot[1] for name, slot in self.stages.items()}, 'other_ms': round(max(0.0, total - sum(stages.values())), 3), 'timeline': list(self.events)}
        if self.phases:
            out['phases_ms'] = {name: round(slot[0], 3) for name, slot in self.phases.items()}
        if self.counters:
            out['counters'] = dict(self.counters)
        return out

def start(tracer=None):
    tracer = tracer or LatencyTracer()
    return (tracer, _CURRENT.set(tracer))

def finish(token):
    if token is None:
        return
    try:
        _CURRENT.reset(token)
    except (ValueError, RuntimeError):
        pass

def stage(name):
    tracer = _CURRENT.get()
    return _NULL if tracer is None else tracer.stage(name)