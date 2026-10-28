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
from fork import layer_step, tail_mask, exp_rotary, gdn_step, SERVE

HEADER_TOKENS = 64
GRAPH_MAX = 1024
BUCKET = 64
GRAPH_WARMUP = 1
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
            self._runtimes[graph] = PrefillEngine(self, self.header, graph=graph)
        return self._runtimes[graph]

    def reset_runtimes(self):
        self._runtimes = {}


def system_header(tokenizer, size=HEADER_TOKENS):
    if tokenizer is None:
        return ()
    from contract import prompt_for

    rows = [
        dict(
            id='h1',
            kind='choice',
            state='Alpha holds.',
            question='Which?',
            answer=['a'],
            candidates=[dict(id='a', text='A'), dict(id='b', text='B')],
        ),
        dict(
            id='h2',
            kind='choice',
            state='{"zeta": 2}\nOther text.',
            question='What now?',
            answer=['y'],
            candidates=[dict(id='x', text='No'), dict(id='y', text='Yes'), dict(id='z', text='Maybe')],
        ),
    ]
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
        import fast_runtime as fast_runtime

        _FLASH.append(fast_runtime.FLASH_BUILT)
    return _FLASH[0]


def flash_gqa(q, k, v, scale, causal):
    if not (SERVE['gqa_flash'] and q.is_cuda and (q.dtype in (torch.bfloat16, torch.float16)) and (q.shape[-1] <= 256)):
        return None
    if not flash_built():
        import fast_runtime as fast_runtime

        return fast_runtime.aot_attention(q, k, v, scale) if causal or q.shape[2] == 1 else None
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


class _Graph:
    def __init__(self, ids, limit):
        self.ids, self.limit, self.graph, self.out = (ids, limit, None, None)


class PrefillEngine:
    def __init__(
        self,
        bench,
        header=(),
        graph=False,
        bucket=BUCKET,
        graph_max=GRAPH_MAX,
        header_max=HEADER_MAX,
        header_layers=None,
        last_only=True,
    ):
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
        self.defer, self.pending, self.hits = (False, [], {})
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
        for layer in self.layers[: self.header_layers]:
            h, state = layer_step(layer, h, pe, self.positions[:, : self.H], None, None, keep=True)
            states.append(state)
        self.header, self.header_hidden = (states, h)

    def header_digest(self):
        return [tuple((float(t.double().sum()) for t in s)) for s in self.header or []]

    def hit(self, ids, host=None):
        n = ids.shape[1]
        if not self.H or n <= self.H or (self.header_max is not None and n - self.H > self.header_max):
            return False
        if host is not None:
            return tuple(host[: self.H]) == self.header_ids
        return ids[0, : self.H].tolist() == list(self.header_ids)

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
        h = self.layers_range(h, pe, self.positions[:, start : start + width], start, 0, self.last)
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
            h, _ = layer_step(layer, h, pe, self.positions[:, start : start + h.shape[1]], state, m, keep=False)
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
                k, v = (
                    torch.cat((state[0].expand(b, -1, -1, -1), k), 2),
                    torch.cat((state[1].expand(b, -1, -1, -1), v), 2),
                )
            q, gate = torch.chunk(a.q_proj(pick(x)).view(b, 1, -1, a.head_dim * 2), 2, dim=-1)
            gate = gate.reshape(b, 1, -1)
            q = a.q_norm(q.view(b, 1, -1, a.head_dim)).transpose(1, 2)
            q = rope_one(q, pick(cos), pick(sin))
            rep = q.shape[1] // k.shape[1]
            if rep > 1:
                k, v = (k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1))
            key_mask = None
            if limit is not None:
                keys = self.positions[0, : k.shape[2]]
                key_mask = torch.where(keys[None] < limit.reshape(-1, 1), 0.0, torch.finfo(q.dtype).min).to(q.dtype)[
                    :, None, None
                ]
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
        pos = self.positions[:, start : start + width]
        for i in range(self.last):
            h = step(self.layers[i], h, pe, pos, states[i] if states else None)
        return self.final(h, pe, states[self.last] if states else None, start, limit)

    @torch.inference_mode()
    def prefix_states(self, ids, start):
        base = self.header if start else None
        h = self.body.embed_tokens(ids)
        n = ids.shape[1]
        pe = self.rotary(start, start + n)
        pos = self.positions[:, start : start + n]
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
        g = _Graph(ids, limit)
        fn = lambda: self.run(g.ids, start, wb, g.limit)
        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(GRAPH_WARMUP):
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
            hits = getattr(self, 'hits', None)
            if hits is not None:
                hits[start, wb] = hits.get((start, wb), 0) + 1
            return self.run(ids[:, start:], start, width)
        g = g or self.capture(start, wb)
        g.ids[:, :width].copy_(ids[:, start:])
        g.limit.fill_(start + width)
        g.graph.replay()
        return g.out


FORK_SIZES = (2, 4, 8)
FORK_MAX_QUESTIONS = 8
FORK_TAILS = (16, 32, 64, 128, 256)


class ForkGraphs:
    def __init__(self, engine):
        self.engine, self.start = (engine, engine.H)
        self.kmax = engine.H + engine.graph_max
        self.prefix, self.tails = ({}, {})
        self.defer, self.pending, self.hits = (True, [], {})
        dev = engine.device
        self.n = torch.zeros((1,), dtype=torch.long, device=dev)
        self.kv, self.gdn = ({}, {})
        for i, (layer, s) in enumerate(zip(engine.layers, engine.header or ())):
            if layer.layer_type == 'full_attention':
                k = s[0]
                self.kv[i] = (
                    k.new_zeros(k.shape[0], k.shape[1], self.kmax, k.shape[3]),
                    s[1].new_zeros(k.shape[0], k.shape[1], self.kmax, s[1].shape[3]),
                )
            else:
                self.gdn[i] = (torch.zeros_like(s[0]), torch.zeros_like(s[1]))

    def keys(self, n, questions, tail):
        if not self.engine.H or n < 1 or n > self.engine.graph_max:
            return None
        b = next((x for x in FORK_SIZES if x >= min(questions, FORK_SIZES[-1])), None)
        t = next((x for x in FORK_TAILS if x >= tail), None)
        if b is None or t is None or questions > FORK_MAX_QUESTIONS:
            return None
        return (-(-n // BUCKET) * BUCKET, (b, t))

    def run_prefix(self, ids, sb):
        engine, start = (self.engine, self.start)
        h = engine.body.embed_tokens(ids)
        pe = engine.rotary(start, start + sb)
        pos = engine.positions[:, start : start + sb]
        valid = torch.arange(sb, device=ids.device)[None] < self.n
        for i, layer in enumerate(engine.layers):
            if layer.layer_type == 'full_attention':
                h, (k, v) = prefix_attention_step(layer, h, pe, engine.header[i], snapshot=True)
                self.kv[i][0][:, :, : start + sb].copy_(k)
                self.kv[i][1][:, :, : start + sb].copy_(v)
            else:
                h, (conv, rec) = layer_step(
                    layer, h, pe, pos, engine.header[i], None, keep=True, lengths=self.n, valid=valid
                )
                self.gdn[i][0].copy_(conv)
                self.gdn[i][1].copy_(rec)

    def run_tails(self, ids, lens):
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

        engine, (b, t) = (self.engine, ids.shape)
        at = self.start + self.n
        idx = at + torch.arange(t, device=ids.device)
        pe = (engine.cos[:, idx], engine.sin[:, idx])
        pos = idx[None]
        keys = torch.arange(self.kmax + t, device=ids.device)
        rows = torch.arange(t, device=ids.device)[:, None]
        ok = (keys[None] < at) | (keys[None] >= self.kmax) & (keys[None] - self.kmax <= rows)
        mask = torch.zeros(ok.shape, dtype=torch.float32, device=ids.device).masked_fill(
            ~ok, torch.finfo(torch.float32).min
        )
        mask_p, mask_t = (mask[:, : self.kmax], mask[:, self.kmax :])
        h = engine.body.embed_tokens(ids)
        for i, layer in enumerate(engine.layers):
            if layer.layer_type == 'full_attention':
                a = layer.self_attn
                residual = h
                x = layer.input_layernorm(h)
                q, gate = torch.chunk(a.q_proj(x).view(b, t, -1, a.head_dim * 2), 2, dim=-1)
                gate = gate.reshape(b, t, -1)
                q = a.q_norm(q.view(b, t, -1, a.head_dim)).transpose(1, 2)
                k = a.k_norm(a.k_proj(x).view(b, t, -1, a.head_dim)).transpose(1, 2)
                v = a.v_proj(x).view(b, t, -1, a.head_dim).transpose(1, 2)
                q, k = apply_rotary_pos_emb(q, k, *pe)
                kvh, d = (k.shape[1], q.shape[-1])
                rep = q.shape[1] // kvh
                qf = q.float().reshape(b, kvh, rep, t, d)
                kp, vp = (self.kv[i][0][0].float(), self.kv[i][1][0].float())
                sp = torch.matmul(qf.permute(1, 0, 2, 3, 4).reshape(kvh, b * rep * t, d), kp.transpose(-1, -2))
                sp = sp.reshape(kvh, b, rep, t, -1).permute(1, 0, 2, 3, 4) * a.scaling + mask_p
                st = torch.matmul(qf, k.float()[:, :, None].transpose(-1, -2)) * a.scaling + mask_t
                pr = torch.cat((sp, st), -1).softmax(-1)
                pp = pr[..., : self.kmax].permute(1, 0, 2, 3, 4).reshape(kvh, b * rep * t, -1)
                o = torch.matmul(pp, vp).reshape(kvh, b, rep, t, d).permute(1, 0, 2, 3, 4)
                o = o + torch.matmul(pr[..., self.kmax :], v.float()[:, :, None])
                o = o.reshape(b, -1, t, d).to(q.dtype)
                o = o.transpose(1, 2).reshape(b, t, -1) * torch.sigmoid(gate)
                h = residual + a.o_proj(o)
                h = h + layer.mlp(layer.post_attention_layernorm(h))
            else:
                conv, rec = self.gdn[i]
                state = (conv.expand(b, -1, -1), rec.expand(b, *rec.shape[1:]).contiguous())
                h = layer_step(layer, h, pe, pos, state, None, keep=False)[0]
        last = h.gather(1, (lens - 1).view(-1, 1, 1).expand(-1, 1, h.shape[-1]))[:, 0]
        return engine.bench.ro(engine.body.norm(last))

    def _capture(self, fn):
        engine = self.engine
        if engine.pool is None:
            engine.pool = torch.cuda.graph_pool_handle()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(GRAPH_WARMUP):
                fn()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=engine.pool):
            out = fn()
        return (graph, out)

    def capture(self, kind, key):
        dev = self.engine.device
        if kind == 'prefix':
            g = _Graph(torch.zeros((1, key), dtype=torch.long, device=dev), None)
            self.n.fill_(key)
            g.graph, _ = self._capture(lambda: self.run_prefix(g.ids, key))
            self.prefix[key] = g
        else:
            b, t = key
            g = _Graph(
                torch.zeros((b, t), dtype=torch.long, device=dev), torch.full((b,), t, dtype=torch.long, device=dev)
            )
            self.n.fill_(1)
            g.graph, g.out = self._capture(lambda: self.run_tails(g.ids, g.limit))
            self.tails[key] = g
        return g

    def run(self, state_ids, tails):
        keys = self.keys(len(state_ids), len(tails), max(map(len, tails))) if all(tails) else None
        if keys is None:
            return None
        missing = [
            (kind, key)
            for kind, key, have in (('prefix', keys[0], self.prefix), ('tails', keys[1], self.tails))
            if key not in have
        ]
        if missing:
            for m in missing:
                if m not in self.pending:
                    self.pending.append(m)
                self.hits[m] = self.hits.get(m, 0) + 1
            if self.defer:
                return None
            for kind, key in missing:
                self.capture(kind, key)
        dev = self.engine.device
        p, tg = (self.prefix[keys[0]], self.tails[keys[1]])
        n = len(state_ids)
        p.ids[0, :n].copy_(torch.tensor(state_ids, device=dev))
        self.n.fill_(n)
        p.graph.replay()
        width, out = (tg.ids.shape[0], [])
        for c in range(0, len(tails), width):
            chunk = list(tails[c : c + width])
            rows = chunk + [chunk[0]] * (width - len(chunk))
            x = torch.zeros(tg.ids.shape, dtype=torch.long)
            for r, ts in enumerate(rows):
                x[r, : len(ts)] = torch.tensor(ts)
            tg.ids.copy_(x.to(dev, non_blocking=True))
            tg.limit.copy_(torch.tensor([len(ts) for ts in rows]).to(dev, non_blocking=True))
            tg.graph.replay()
            out.append(tg.out[: len(chunk)].clone())
        return torch.cat(out)


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
