import argparse
import hashlib
import io
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
import torch
pass
from m7ml5d1vtymnzohdbris7 import layer_step, tail_mask, exp_rotary, gdn_step, SERVE
HEADER_TOKENS = 64
GRAPH_MAX = 1024
BUCKET = 64
_v6ff1myegbtwfltdvya89 = 1
HEADER_MAX = None

class Bench:

    def __init__(self, body, ro, tokenizer=None, info=None, header=()):
        self.body, self.ro, self.tokenizer, self.info = (body, ro, tokenizer, dict(info or {}))
        impl = str(body.config._attn_implementation)
        if impl == 'eager' or not (impl == 'sdpa' or impl.startswith('flash')):
            body.config._attn_implementation = 'sdpa'
        self.info.update(attn_implementation=body.config._attn_implementation, attn_implementation_loaded=impl)
        self.header = tuple(header)
        self.blocked = None
        self.kept = None
        self._runtimes = {}

    @property
    def device(self):
        return self.body.embed_tokens.weight.device

    def runtime(self, graph=False):
        if self.blocked:
            raise ValueError(self.blocked)
        if graph not in self._runtimes:
            self._runtimes[graph] = _v83m8rh6inetzovw1w89s(self, self.header, graph=graph)
        return self._runtimes[graph]

    def reset_runtimes(self):
        self._runtimes = {}

def system_header(tokenizer, size=HEADER_TOKENS):
    if tokenizer is None:
        return ()
    from m6lir2n4n221f2rp5z93i import prompt_for
    rows = [dict(id='h1', kind='choice', state='Alpha holds.', question='Which?', answer=['a'], candidates=[dict(id='a', text='A'), dict(id='b', text='B')]), dict(id='h2', kind='choice', state='{"zeta": 2}\nOther text.', question='What now?', answer=['y'], candidates=[dict(id='x', text='No'), dict(id='y', text='Yes'), dict(id='z', text='Maybe')])]
    ids = [tokenizer(prompt_for(r, tokenizer), add_special_tokens=False)['input_ids'] for r in rows]
    n = 0
    while n < min(len(ids[0]), len(ids[1]), size) and ids[0][n] == ids[1][n]:
        n += 1
    return tuple(ids[0][:n])

def rope_one(x, cos, sin, unsqueeze_dim=1):
    from transformers.models.qwen3_5.modeling_qwen3_5 import rotate_half
    cos, sin = (cos.unsqueeze(unsqueeze_dim), sin.unsqueeze(unsqueeze_dim))
    d = cos.shape[-1]
    rot, rest = (x[..., :d], x[..., d:])
    return torch.cat([rot * cos + rotate_half(rot) * sin, rest], dim=-1)
_FLASH = []

def flash_built():
    if not _FLASH:
        import m4042euhfp7ynv7osi045 as _vba9w2x4ebr0mfzzp69di
        _FLASH.append(_vba9w2x4ebr0mfzzp69di.FLASH_BUILT)
    return _FLASH[0]

def flash_gqa(q, k, v, scale, causal):
    if not (SERVE['gqa_flash'] and q.is_cuda and (q.dtype in (torch.bfloat16, torch.float16)) and (q.shape[-1] <= 256)):
        return None
    if not flash_built():
        import m4042euhfp7ynv7osi045 as _vba9w2x4ebr0mfzzp69di
        return _vba9w2x4ebr0mfzzp69di.aot_attention(q, k, v, scale) if causal or q.shape[2] == 1 else None
    try:
        return torch.ops.aten._scaled_dot_product_flash_attention(q, k, v, 0.0, causal, False, scale=scale)[0]
    except (RuntimeError, TypeError):
        return None

def prefix_attention_step(layer, h, pe, state, snapshot=False):
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
    a = layer.self_attn
    residual = h
    x = layer.input_layernorm(h)
    b, n, _ = x.shape
    q, gate = torch.chunk(a.q_proj(x).view(b, n, -1, a.head_dim * 2), 2, dim=-1)
    gate = gate.reshape(b, n, -1)
    q = a.q_norm(q.view(b, n, -1, a.head_dim)).transpose(1, 2)
    k = a.k_norm(a.k_proj(x).view(b, n, -1, a.head_dim)).transpose(1, 2)
    v = a.v_proj(x).view(b, n, -1, a.head_dim).transpose(1, 2)
    cos, sin = pe
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    if state is not None:
        k = torch.cat((state[0].expand(b, -1, -1, -1), k), 2)
        v = torch.cat((state[1].expand(b, -1, -1, -1), v), 2)
    kept = (k, v) if snapshot else None
    o = flash_gqa(q, k, v, a.scaling, n > 1)
    if o is not None:
        o = o.transpose(1, 2).reshape(b, n, -1).contiguous() * torch.sigmoid(gate)
        h = residual + a.o_proj(o)
        h = h + layer.mlp(layer.post_attention_layernorm(h))
        return (h, kept) if snapshot else h
    rep = q.shape[1] // k.shape[1]
    if rep > 1:
        k, v = (k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1))
    o = None
    if n == k.shape[2]:
        o = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=n > 1, scale=a.scaling)
    elif q.is_cuda and q.dtype in (torch.bfloat16, torch.float16) and (q.shape[-1] <= 256) and flash_built():
        try:
            o = torch.ops.aten._scaled_dot_product_flash_attention(q, k, v, 0.0, n > 1, False, scale=a.scaling)[0]
        except (RuntimeError, TypeError):
            o = None
    if o is None:
        mask = tail_mask(k.shape[2] - n, n, 'sdpa', q.dtype, q.device) if n > 1 else None
        o = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=a.scaling)
    o = o.transpose(1, 2).reshape(b, n, -1).contiguous() * torch.sigmoid(gate)
    h = residual + a.o_proj(o)
    h = h + layer.mlp(layer.post_attention_layernorm(h))
    return (h, kept) if snapshot else h

def step(layer, h, pe, pos, state):
    if state is not None and layer.layer_type == 'full_attention':
        return prefix_attention_step(layer, h, pe, state)
    return layer_step(layer, h, pe, pos, state, None, keep=False)[0]

class _v79scs9dr4oljw1o5ga3v:

    def __init__(self, ids, limit):
        self.ids, self.limit, self.graph, self.out = (ids, limit, None, None)

class _v83m8rh6inetzovw1w89s:

    def __init__(self, bench, header=(), graph=False, bucket=BUCKET, graph_max=GRAPH_MAX, header_max=HEADER_MAX, header_layers=None, last_only=True):
        self.bench, self.body = (bench, bench.body)
        self.layers = list(self.body.layers)
        self.last = len(self.layers) - 1
        self.header_layers = len(self.layers) if header_layers is None else int(header_layers)
        if not 0 < self.header_layers <= len(self.layers):
            raise ValueError('header_layers must be in 1..layers')
        self.last_only = bool(last_only)
        self.header_hidden = None
        self.dtype, self.device = (self.body.embed_tokens.weight.dtype, self.body.embed_tokens.weight.device)
        self.graph = bool(graph) and self.device.type == 'cuda'
        self.bucket, self.graph_max, self.header_max = (bucket, graph_max, header_max)
        self.impl = str(self.body.config._attn_implementation)
        self.cos = self.sin = self.positions = None
        self._retired = []
        self.header_ids = tuple(header)
        self.H = len(self.header_ids)
        self.header = None
        self.graphs, self.pool = ({}, None)
        self.defer, self.pending = (False, [])
        self._grow(max(4096, self.H + (header_max or 0) + 1))
        with torch.inference_mode():
            if self.H:
                self._build_header()

    def _grow(self, size):
        if self.cos is not None:
            self._retired.append((self.cos, self.sin, self.positions))
        p = torch.arange(size, device=self.device)
        cos, sin = exp_rotary(self.body, torch.empty(0, dtype=self.dtype, device=self.device), p)
        self.cos, self.sin, self.positions = (cos, sin, p[None])

    def rotary(self, start, stop):
        if stop > self.cos.shape[-2]:
            self._grow(max(stop, 2 * self.cos.shape[-2]))
        return (self.cos[:, start:stop], self.sin[:, start:stop])

    def _build_header(self):
        ids = torch.tensor([self.header_ids], device=self.device)
        h = self.body.embed_tokens(ids)
        pe = self.rotary(0, self.H)
        states = []
        for layer in self.layers[:self.header_layers]:
            h, state = layer_step(layer, h, pe, self.positions[:, :self.H], None, None, keep=True)
            states.append(state)
        self.header, self.header_hidden = (states, h)

    def header_digest(self):
        return [tuple((float(t.double().sum()) for t in s)) for s in self.header or []]

    def hit(self, ids, host=None):
        n = ids.shape[1]
        if not self.H or n <= self.H or (self.header_max is not None and n - self.H > self.header_max):
            return False
        if host is not None:
            return tuple(host[:self.H]) == self.header_ids
        return ids[0, :self.H].tolist() == list(self.header_ids)

    def plan(self, ids, host=None):
        full = self.header_layers == len(self.layers)
        start = self.H if full and self.hit(ids, host) else 0
        width = ids.shape[1] - start
        return (start, width, self.graph and width <= self.graph_max)

    @torch.inference_mode()
    def __call__(self, ids, host=None):
        start, width, graphed = self.plan(ids, host)
        if graphed:
            return self.replay(ids, start, width)
        return self.run(ids[:, start:], start, width)

    @torch.inference_mode()
    def padded(self, ids, host=None):
        start, width, _ = self.plan(ids, host)
        wb = -(-width // self.bucket) * self.bucket
        x = torch.zeros((1, wb), dtype=ids.dtype, device=ids.device)
        x[:, :width] = ids[:, start:]
        return self.run(x, start, wb, torch.tensor([start + width], device=ids.device))

    def lower(self, ids, stop, host=None):
        n = ids.shape[1]
        start = self.H if stop <= self.header_layers and self.hit(ids, host) else 0
        h = self.body.embed_tokens(ids[:, start:])
        pe = self.rotary(start, n)
        h = self.layers_range(h, pe, self.positions[:, start:n], start, 0, stop)
        return (h, start, pe)

    def layers_range(self, h, pe, pos, start, i0, i1):
        for i in range(i0, i1):
            h = step(self.layers[i], h, pe, pos, self.header[i] if start else None)
        return h

    def run(self, x, start, width, limit=None):
        h = self.body.embed_tokens(x)
        pe = self.rotary(start, start + width)
        h = self.layers_range(h, pe, self.positions[:, start:start + width], start, 0, self.last)
        return self.final(h, pe, self.header[self.last] if start else None, start, limit)

    def final(self, h, pe, state, start, limit):
        return self.bench.ro(self.body.norm(self.final_hidden(h, pe, state, start, limit)))

    def final_hidden(self, h, pe, state, start, limit):
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
        layer = self.layers[self.last]
        cos, sin = pe
        if not self.last_only:
            if limit is not None:
                raise ValueError('the padded bucket path needs last_only')
            m = None
            if state is not None and layer.layer_type == 'full_attention':
                m = tail_mask(state[0].shape[2], h.shape[1], self.impl, h.dtype, h.device)
            h, _ = layer_step(layer, h, pe, self.positions[:, start:start + h.shape[1]], state, m, keep=False)
            return h[:, -1]
        if limit is None:
            pick = lambda t: t[:, -1:]
        else:
            idx = (limit - 1 - start).reshape(-1)

            def pick(t):
                if t.shape[0] != idx.numel():
                    t = t.expand(idx.numel(), *t.shape[1:])
                return t.gather(1, idx.view(-1, 1, *[1] * (t.dim() - 2)).expand(-1, 1, *t.shape[2:]))
        residual = pick(h)
        x = layer.input_layernorm(h)
        if layer.layer_type == 'full_attention':
            a = layer.self_attn
            b, n, _ = x.shape
            k = a.k_norm(a.k_proj(x).view(b, n, -1, a.head_dim)).transpose(1, 2)
            v = a.v_proj(x).view(b, n, -1, a.head_dim).transpose(1, 2)
            k = rope_one(k, cos, sin)
            if state is not None:
                k, v = (torch.cat((state[0].expand(b, -1, -1, -1), k), 2), torch.cat((state[1].expand(b, -1, -1, -1), v), 2))
            q, gate = torch.chunk(a.q_proj(pick(x)).view(b, 1, -1, a.head_dim * 2), 2, dim=-1)
            gate = gate.reshape(b, 1, -1)
            q = a.q_norm(q.view(b, 1, -1, a.head_dim)).transpose(1, 2)
            q = rope_one(q, pick(cos), pick(sin))
            rep = q.shape[1] // k.shape[1]
            if rep > 1:
                k, v = (k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1))
            key_mask = None
            if limit is not None:
                keys = self.positions[0, :k.shape[2]]
                key_mask = torch.where(keys[None] < limit.reshape(-1, 1), 0.0, torch.finfo(q.dtype).min).to(q.dtype)[:, None, None]
            o = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=key_mask, scale=a.scaling)
            o = o.transpose(1, 2).reshape(b, 1, -1) * torch.sigmoid(gate)
            out = residual + a.o_proj(o)
        else:
            y, _ = gdn_step(layer.linear_attn, x, state, keep=False)
            out = residual + pick(y)
        out = out + layer.mlp(layer.post_attention_layernorm(out))
        return out[:, 0]

    def batch_states(self, states, b):
        if b == 1 or states is None:
            return states
        out = []
        for layer, s in zip(self.layers, states):
            if s is None or layer.layer_type == 'full_attention':
                out.append(s)
                continue
            conv, rec = s
            out.append((conv.expand(b, -1, -1), None if rec is None else rec.expand(b, *rec.shape[1:]).contiguous()))
        return out

    @torch.inference_mode()
    def run_batch(self, x, start, limit, states=None):
        b, width = x.shape
        if states is None:
            states = self.header if start else None
            if start and start != self.H:
                raise ValueError('a batch continuing after the header needs its prefix states')
        states = self.batch_states(states, b)
        h = self.body.embed_tokens(x)
        pe = self.rotary(start, start + width)
        pos = self.positions[:, start:start + width]
        for i in range(self.last):
            h = step(self.layers[i], h, pe, pos, states[i] if states else None)
        return self.final(h, pe, states[self.last] if states else None, start, limit)

    @torch.inference_mode()
    def prefix_states(self, ids, start):
        base = self.header if start else None
        h = self.body.embed_tokens(ids)
        n = ids.shape[1]
        pe = self.rotary(start, start + n)
        pos = self.positions[:, start:start + n]
        states = []
        for i, layer in enumerate(self.layers):
            prior = base[i] if base else None
            if layer.layer_type == 'full_attention':
                h, kv = prefix_attention_step(layer, h, pe, prior, snapshot=True)
            else:
                h, kv = layer_step(layer, h, pe, pos, prior, None, keep=True)
            states.append(kv)
        return states

    def capture(self, start, wb):
        ids = torch.zeros((1, wb), dtype=torch.long, device=self.device)
        limit = torch.full((1,), start + wb, dtype=torch.long, device=self.device)
        g = _v79scs9dr4oljw1o5ga3v(ids, limit)
        fn = lambda: self.run(g.ids, start, wb, g.limit)
        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(_v6ff1myegbtwfltdvya89):
                fn()
        torch.cuda.current_stream().wait_stream(side)
        g.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g.graph, pool=self.pool):
            g.out = fn()
        self.graphs[start, wb] = g
        return g

    def replay(self, ids, start, width):
        wb = -(-width // self.bucket) * self.bucket
        g = self.graphs.get((start, wb))
        if g is None and self.defer:
            if (start, wb) not in self.pending:
                self.pending.append((start, wb))
            return self.run(ids[:, start:], start, width)
        g = g or self.capture(start, wb)
        g.ids[:, :width].copy_(ids[:, start:])
        g.limit.fill_(start + width)
        g.graph.replay()
        return g.out

def graphed(fn, device):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    return (graph, out)