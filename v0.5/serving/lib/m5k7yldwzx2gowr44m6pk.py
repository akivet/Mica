import argparse
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
import torch
from torch.nn import functional as F
pass
import m8rhab841mmga1gnl3mp8 as fb
from m7ml5d1vtymnzohdbris7 import layer_step, gdn_step, SERVE, conv_channel_last
CUT = 16
W_MAX = 512
S_MAX = 32768

def stale_scan(layer, h, init=None):
    a = layer.linear_attn
    x, kv, key, value, g, beta = scan_inputs(layer, h, init)
    _, recurrent = a.chunk_gated_delta_rule(torch.zeros_like(key), key, value, g=g, beta=beta, initial_state=None if init is None else init[1], output_final_state=True, use_qk_l2norm_in_kernel=True)
    return (scan_tail(layer, x, kv, init), recurrent)

def proj_rows(linear, x, lo, hi=None):
    if hasattr(linear, 'qweight'):
        return linear(x)[..., lo:hi]
    w, bias = (linear.weight, linear.bias)
    return F.linear(x, w[lo:hi], None if bias is None else bias[lo:hi])

def repeat_keys(key, value):
    rep = value.shape[2] // key.shape[2]
    return key.repeat_interleave(rep, 2) if rep > 1 else key

def scan_inputs(layer, h, init=None):
    a = layer.linear_attn
    batch, length, _ = h.shape
    width = a.conv_kernel_size - 1
    x = layer.input_layernorm(h)
    _vae5z11bsktmjgjqwnorm = a.key_dim
    kv = proj_rows(a.in_proj_qkv, x, _vae5z11bsktmjgjqwnorm)
    raw = kv.transpose(1, 2)
    cw, cb = (a.conv1d.weight[_vae5z11bsktmjgjqwnorm:], None if a.conv1d.bias is None else a.conv1d.bias[_vae5z11bsktmjgjqwnorm:])
    conv = None
    if SERVE['conv_cl'] and a.causal_conv1d_fn is not None:
        conv = conv_channel_last(a.causal_conv1d_fn, raw, None if init is None else init[0][:, _vae5z11bsktmjgjqwnorm:], cw.squeeze(1), cb, a.activation)
    full = raw if init is None or conv is not None else torch.cat((init[0][:, _vae5z11bsktmjgjqwnorm:], raw), -1)
    if conv is not None:
        pass
    elif a.causal_conv1d_fn is not None:
        conv = a.causal_conv1d_fn(x=full, weight=cw.squeeze(1), bias=cb, activation=a.activation, seq_idx=None)
    else:
        conv = F.silu(F.conv1d(full, cw, cb, padding=width, groups=full.shape[1])[..., :full.shape[-1]])
    key, value = conv[..., -length:].transpose(1, 2).split((_vae5z11bsktmjgjqwnorm, a.value_dim), -1)
    key = key.reshape(batch, length, -1, a.head_k_dim)
    value = value.reshape(batch, length, -1, a.head_v_dim)
    if a.num_v_heads // a.num_k_heads > 1 and (not (SERVE.get('gdn_kernel') or {}).get('gva')):
        key = key.repeat_interleave(a.num_v_heads // a.num_k_heads, 2)
    beta = a.in_proj_b(x).sigmoid()
    g = -a.A_log.float().exp() * F.softplus(a.in_proj_a(x).float() + a.dt_bias)
    return (x, kv, key, value, g, beta)

def scan_tail(layer, x, kv, init=None):
    a = layer.linear_attn
    width = a.conv_kernel_size - 1
    _vae5z11bsktmjgjqwnorm = a.key_dim
    m = min(x.shape[1], width)
    q_last = proj_rows(a.in_proj_qkv, x[:, -m:], 0, _vae5z11bsktmjgjqwnorm)
    last = torch.cat((q_last, kv[:, -m:]), -1).transpose(1, 2)
    tail = last if init is None else torch.cat((init[0], last), -1)
    return F.pad(tail, (width - tail.shape[-1], 0)) if tail.shape[-1] < width else tail[..., -width:]

def _qkv(layer, h, cos, sin):
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
    a = layer.self_attn
    x = layer.input_layernorm(h)
    b, n, _ = x.shape
    q, gate = torch.chunk(a.q_proj(x).view(b, n, -1, a.head_dim * 2), 2, dim=-1)
    q = a.q_norm(q.view(b, n, -1, a.head_dim)).transpose(1, 2)
    k = a.k_norm(a.k_proj(x).view(b, n, -1, a.head_dim)).transpose(1, 2)
    v = a.v_proj(x).view(b, n, -1, a.head_dim).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    return (q, gate.reshape(b, n, -1), k, v)

def masked_attention_layer(layer, h, cos, sin, ctx, mask, rows=None):
    a = layer.self_attn
    q, gate, k, v = _qkv(layer, h, cos, sin)
    k, v = (torch.cat((ctx[0], k), 2), torch.cat((ctx[1], v), 2))
    residual = h
    if rows is not None:
        q, gate, residual = (q.index_select(2, rows), gate.index_select(1, rows), h.index_select(1, rows))
        mask = mask.index_select(2, rows)
    rep = q.shape[1] // k.shape[1]
    if rep > 1:
        k, v = (k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1))
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=a.scaling)
    o = o.transpose(1, 2).reshape(residual.shape[0], residual.shape[1], -1).contiguous() * torch.sigmoid(gate)
    out = residual + a.o_proj(o)
    return out + layer.mlp(layer.post_attention_layernorm(out))

def bucket_s(s):
    if s <= 1024:
        return max(128, -(-s // 128) * 128)
    step = 1 << s.bit_length() - 4
    return -(-s // step) * step

def bucket_w(w):
    return max(32, -(-w // 32) * 32)

class Fast(fb._v83m8rh6inetzovw1w89s):

    def __init__(self, bench, header=True, cut=CUT, w_max=W_MAX, s_max=S_MAX):
        self.cut = int(cut)
        self.header_mid = None
        super().__init__(bench, tuple(bench.header) if header else (), graph=False)
        L = len(self.layers)
        self.fa_up = [i for i in range(self.cut, L) if self.layers[i].layer_type == 'full_attention']
        self.gdn_up = [i for i in range(self.cut, L) if self.layers[i].layer_type == 'linear_attention']
        self.hdr_kv, self.hdr_gdn = ({}, {})
        self.w_max, self.s_max = (w_max, s_max)
        self.upper_graph = None
        if self.H:
            with torch.inference_mode():
                for i in self.fa_up:
                    self.hdr_kv[i] = self.kv(i, self.header_mid, 0)
                for i in self.gdn_up:
                    self.hdr_gdn[i] = stale_scan(self.layers[i], self.header_mid)

    def _build_header(self):
        ids = torch.tensor([self.header_ids], device=self.device)
        h = self.body.embed_tokens(ids)
        pe = self.rotary(0, self.H)
        states = []
        for i, layer in enumerate(self.layers[:self.header_layers]):
            if i == self.cut:
                self.header_mid = h
            h, state = layer_step(layer, h, pe, self.positions[:, :self.H], None, None, keep=True)
            states.append(state)
        self.header, self.header_hidden = (states, h)

    def kv(self, i, h, p0):
        layer = self.layers[i]
        a = layer.self_attn
        x = layer.input_layernorm(h)
        b, n, _ = x.shape
        k = a.k_norm(a.k_proj(x).view(b, n, -1, a.head_dim)).transpose(1, 2)
        v = a.v_proj(x).view(b, n, -1, a.head_dim).transpose(1, 2)
        cos, sin = self.rotary(p0, p0 + n)
        return (fb.rope_one(k, cos, sin), v)

    def start_of(self, ids, s, host=None):
        return self.H if self.H and s >= self.H and self.hit(ids, host) else 0

    def lower16(self, ids, s, host=None):
        start = self.start_of(ids, s, host)
        n = ids.shape[1]
        h = self.body.embed_tokens(ids[:, start:])
        pe = self.rotary(start, n)
        return (self.layers_range(h, pe, self.positions[:, start:n], start, 0, self.cut), start, pe)

    def context(self, h, start, s, mode):
        if s == 0:
            return ({}, {})
        rows = h[:, :s - start]
        ctx, gdn = ({}, {})
        for i in self.fa_up:
            if start:
                ctx[i] = self.hdr_kv[i] if rows.shape[1] == 0 else tuple((torch.cat((x, y), 2) for x, y in zip(self.hdr_kv[i], self.kv(i, rows, start))))
            else:
                ctx[i] = self.kv(i, rows, 0)
        if mode == '_v9qzrgapy1ah7g1jj5alh':
            for i in self.gdn_up:
                init = self.hdr_gdn[i] if start else None
                gdn[i] = init if rows.shape[1] == 0 else self.scan(self.layers[i], rows, init)
        elif mode != '_v654mofq6e6jk42xea5z6':
            raise ValueError('unknown mode')
        return (ctx, gdn)

    def scan(self, layer, rows, init=None):
        return stale_scan(layer, rows, init)

    def window(self, h, start, s, n, ctx, gdn):
        x = h[:, s - start:]
        pe, pos = (self.rotary(s, n), self.positions[:, s:n])
        for i in range(self.cut, self.last):
            layer = self.layers[i]
            state = ctx.get(i) if layer.layer_type == 'full_attention' else gdn.get(i)
            x = fb.step(layer, x, pe, pos, state)
        last = self.layers[self.last]
        return self.final(x, pe, ctx.get(self.last) if last.layer_type == 'full_attention' else gdn.get(self.last), s, None)

    def fast(self, ids, s, host=None, mode='_v654mofq6e6jk42xea5z6', graph=False, low=None):
        h, start, _ = low if low is not None else self.lower16(ids, s, host)
        n = ids.shape[1]
        ctx, gdn = self.context(h, start, s, mode)
        if graph:
            out = self.graph_window(h, start, s, n, ctx, gdn, mode)
            if out is not None:
                return out
        return self.window(h, start, s, n, ctx, gdn)

    def dense(self, low):
        h, start, pe = low
        h = self.layers_range(h, pe, self.positions[:, start:start + h.shape[1]], start, self.cut, self.last)
        return self.final(h, pe, self.header[self.last] if start else None, start, None)

    def cascade(self, ids, s, host=None, mode='_v654mofq6e6jk42xea5z6', tau=0.6, pick=None, graph=False):
        low = self.lower16(ids, s, host)
        z = self.fast(ids, s, host, mode, graph, low)
        if tau < 1 and confidence(z if pick is None else pick(z)) >= tau:
            return (z, False)
        return (self.dense(low), True)

    def graph_window(self, h, start, s, n, ctx, gdn, mode):
        if self.upper_graph is None:
            self.upper_graph = _v29sx5efmidklsdhdw8ye(self)
        return self.upper_graph(h, start, s, n, ctx, gdn, mode)

def confidence(z):
    return float(z.float().softmax(-1).max())

class _v29sx5efmidklsdhdw8ye:

    def __init__(self, rt):
        self.rt = rt
        dev, dt = (rt.device, rt.dtype)
        cfg = rt.body.config
        need = rt.s_max + rt.w_max + 1
        if need > rt.cos.shape[-2]:
            rt._grow(need)
        self.x = torch.zeros((1, rt.w_max, cfg.hidden_size), dtype=dt, device=dev)
        self.s_len = torch.zeros((1,), dtype=torch.long, device=dev)
        self.w_len = torch.ones((1,), dtype=torch.long, device=dev)
        self.kbuf, self.vbuf = ({}, {})
        for i in rt.fa_up:
            a = rt.layers[i].self_attn
            kvh = a.k_proj.out_features // a.head_dim
            self.kbuf[i] = torch.zeros((1, kvh, rt.s_max, a.head_dim), dtype=dt, device=dev)
            self.vbuf[i] = torch.zeros((1, kvh, rt.s_max, a.head_dim), dtype=dt, device=dev)
        self.gbuf = {}
        self.graphs, self.pool = ({}, None)
        self.cuda = rt.device.type == 'cuda'

    def fn(self, mode, sb, wb):
        rt = self.rt
        dev, dt = (rt.device, rt.dtype)
        x = self.x[:, :wb]
        pos = self.s_len + torch.arange(wb, device=dev)
        cos, sin = (rt.cos.index_select(1, pos), rt.sin.index_select(1, pos))
        neg = torch.finfo(dt).min
        ctx_mask = torch.where(torch.arange(sb, device=dev) < self.s_len, 0.0, neg).to(dt)
        win = torch.full((wb, wb), neg, dtype=dt, device=dev).triu(1)
        mask = torch.cat((ctx_mask.expand(wb, sb), win), 1)[None, None]
        for i in range(rt.cut, rt.last):
            layer = rt.layers[i]
            if layer.layer_type == 'full_attention':
                x = masked_attention_layer(layer, x, cos, sin, (self.kbuf[i][:, :, :sb], self.vbuf[i][:, :, :sb]), mask)
            else:
                state = self.gbuf.get(i) if mode != '_v654mofq6e6jk42xea5z6' else None
                x = layer_step(layer, x, (cos, sin), None, state, None, keep=False)[0]
        idx = self.w_len - 1
        layer = rt.layers[rt.last]
        if layer.layer_type == 'full_attention':
            out = masked_attention_layer(layer, x, cos, sin, (self.kbuf[rt.last][:, :, :sb], self.vbuf[rt.last][:, :, :sb]), mask, rows=idx)
        else:
            state = self.gbuf.get(rt.last) if mode != '_v654mofq6e6jk42xea5z6' else None
            y, _ = gdn_step(layer.linear_attn, layer.input_layernorm(x), state, keep=False)
            out = x.index_select(1, idx) + y.index_select(1, idx)
            out = out + layer.mlp(layer.post_attention_layernorm(out))
        return rt.bench.ro(rt.body.norm(out[:, 0]))

    def __call__(self, h, start, s, n, ctx, gdn, mode):
        rt = self.rt
        w = n - s
        sb, wb = (bucket_s(s), bucket_w(w))
        if s == 0 or wb > rt.w_max or sb > rt.s_max:
            return None
        for i in rt.fa_up:
            self.kbuf[i][:, :, :s].copy_(ctx[i][0])
            self.vbuf[i][:, :, :s].copy_(ctx[i][1])
        if mode != '_v654mofq6e6jk42xea5z6':
            for i in rt.gdn_up:
                if i not in self.gbuf:
                    self.gbuf[i] = tuple((t.clone() for t in gdn[i]))
                else:
                    for dst, src in zip(self.gbuf[i], gdn[i]):
                        dst.copy_(src)
        self.x[:, :w].copy_(h[:, s - start:])
        self.s_len.fill_(s)
        self.w_len.fill_(w)
        if not self.cuda:
            return self.fn(mode, sb, wb)
        key = (mode, sb, wb)
        g = self.graphs.get(key) or self.capture(key)
        g.graph.replay()
        return g.out

    def capture(self, key):
        g = fb._v79scs9dr4oljw1o5ga3v(None, None)
        fn = lambda: self.fn(*key)
        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(side)
        g.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g.graph, pool=self.pool):
            g.out = fn()
        self.graphs[key] = g
        return g