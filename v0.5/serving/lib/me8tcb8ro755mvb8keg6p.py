import argparse
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
import torch
from torch.nn import functional as F
pass
import m8rhab841mmga1gnl3mp8 as fb
import m5k7yldwzx2gowr44m6pk as _v6ur7q369jbjyz1zwxrpd
from m7ml5d1vtymnzohdbris7 import layer_step
EXT = 128
SCAN_PATHS = {}

def _torch_chunk():
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m
    return m.torch_chunk_gated_delta_rule

def torch_state(key, value, g, beta, initial_state=None, chunk_size=64):
    from transformers.models.qwen3_5.modeling_qwen3_5 import l2norm
    key = l2norm(key, dim=-1, eps=1e-06)
    key, value, beta, g = [x.transpose(1, 2).contiguous().to(torch.float32) for x in (key, value, beta, g)]
    batch, heads, length, kdim = key.shape
    vdim = value.shape[-1]
    pad = (chunk_size - length % chunk_size) % chunk_size
    key = F.pad(key, (0, 0, 0, pad))
    value = F.pad(value, (0, 0, 0, pad))
    beta = F.pad(beta, (0, pad))
    g = F.pad(g, (0, pad))
    total = length + pad
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    key, value, k_beta, v_beta = [x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (key, value, k_beta, v_beta)]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=key.device), diagonal=0)
    g = g.cumsum(dim=-1)
    decay_mask = (g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float().tril()
    attn = -(k_beta @ key.transpose(-1, -2) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    state = torch.zeros(batch, heads, kdim, vdim).to(value) if initial_state is None else initial_state.to(value)
    for i in range(0, total // chunk_size):
        k_i, v_i = (key[:, :, i], value[:, :, i])
        v_new = v_i - k_cumdecay[:, :, i] @ state
        state = state * g[:, :, i, -1, None, None].exp() + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
    return state
FLA_CHUNK = 'fla.ops.gated_delta_rule.chunk'

def fla_state(fn, key, value, g, beta, init):
    mod = sys.modules.get(FLA_CHUNK)
    orig = getattr(mod, 'chunk_fwd_o', None) if mod is not None else None
    calls = []
    if callable(orig):

        def skip(*args, **kw):
            calls.append(1)
            v = kw['v'] if 'v' in kw else args[2]
            return torch.empty_like(v)
        mod.chunk_fwd_o = skip
    try:
        _, state = fn(key, key, value, g=g, beta=beta, initial_state=init, output_final_state=True, use_qk_l2norm_in_kernel=True)
    finally:
        if callable(orig):
            mod.chunk_fwd_o = orig
    return (state, bool(calls))

def gdr_state(attn, key, value, g, beta, init):
    fn = attn.chunk_gated_delta_rule
    if fn is _torch_chunk():
        return (torch_state(key, value, g, beta, init), 'torch__vbeqq49pz5iwyndwqqult')
    state, skipped = fla_state(fn, key, value, g, beta, init)
    return (state, 'fla__vbeqq49pz5iwyndwqqult' if skipped else 'fla_full')

def state_scan(layer, h, init=None):
    x, kv, key, value, g, beta = _v6ur7q369jbjyz1zwxrpd.scan_inputs(layer, h, init)
    recurrent, path = gdr_state(layer.linear_attn, key, value, g, beta, None if init is None else init[1])
    SCAN_PATHS[path] = SCAN_PATHS.get(path, 0) + 1
    return (_v6ur7q369jbjyz1zwxrpd.scan_tail(layer, x, kv, init), recurrent)

class Fast2(_v6ur7q369jbjyz1zwxrpd.Fast):

    def __init__(self, bench, header=True, cut=_v6ur7q369jbjyz1zwxrpd.CUT, ext=0, _v2d09izj31gt2igvpejxb=True, **kw):
        self.ext, self._v2d09izj31gt2igvpejxb = (int(ext), bool(_v2d09izj31gt2igvpejxb))
        super().__init__(bench, header=header, cut=cut, **kw)

    def scan(self, layer, rows, init=None):
        return state_scan(layer, rows, init) if self._v2d09izj31gt2igvpejxb else _v6ur7q369jbjyz1zwxrpd.stale_scan(layer, rows, init)

    def context(self, h, start, s, mode):
        if mode in ('_v654mofq6e6jk42xea5z6', '_v9qzrgapy1ah7g1jj5alh'):
            return super().context(h, start, s, mode)
        if mode != '_v654mofq6e6jk42xea5z6_ct' and (not mode.startswith('_v9qzrgapy1ah7g1jj5alh_last')):
            raise ValueError('unknown mode ' + mode)
        ctx, _ = super().context(h, start, s, '_v654mofq6e6jk42xea5z6')
        if s == 0:
            return (ctx, {})
        rows = h[:, :s - start]
        gdn = {}
        for i in self.gdn_up:
            layer = self.layers[i]
            if mode == '_v654mofq6e6jk42xea5z6_ct':
                width = layer.linear_attn.conv_kernel_size - 1
                last = rows[:, -width:]
                init = self.hdr_gdn[i] if start else None
                if last.shape[1] == 0:
                    tail = init[0]
                else:
                    x = layer.input_layernorm(last)
                    kv = _v6ur7q369jbjyz1zwxrpd.proj_rows(layer.linear_attn.in_proj_qkv, x, layer.linear_attn.key_dim)
                    tail = _v6ur7q369jbjyz1zwxrpd.scan_tail(layer, x, kv, init)
                gdn[i] = (tail, zero_state(layer, tail))
            else:
                k = int(mode[len('_v9qzrgapy1ah7g1jj5alh_last'):])
                full = rows if rows.shape[1] >= k or not start else torch.cat((self.header_mid, rows), 1)
                gdn[i] = self.scan(layer, full[:, -k:], None)
        return (ctx, gdn)

    def fast(self, ids, s, host=None, mode='_v9qzrgapy1ah7g1jj5alh', graph=False, low=None):
        if self.ext:
            low = low if low is not None else self.lower16(ids, s, host)
            s = max(low[1], s - self.ext)
        return super().fast(ids, s, host, mode, graph, low)

    def extend(self, ctx, gdn, h, start, s0, s1, mode):
        if s1 <= s0:
            return (ctx, gdn)
        if not ctx:
            return self.context(h, start, s1, mode)
        rows = h[:, s0 - start:s1 - start]
        ctx2 = {i: tuple((torch.cat((x, y), 2) for x, y in zip(ctx[i], self.kv(i, rows, s0)))) for i in self.fa_up}
        gdn2 = {i: self.scan(self.layers[i], rows, gdn[i]) for i in self.gdn_up} if mode == '_v9qzrgapy1ah7g1jj5alh' else {}
        return (ctx2, gdn2)

def zero_state(layer, like):
    a = layer.linear_attn
    return torch.zeros((like.shape[0], a.num_v_heads, a.head_k_dim, a.head_v_dim), dtype=torch.float32, device=like.device)

def runtime(bench, cut=_v6ur7q369jbjyz1zwxrpd.CUT, ext=0, _v2d09izj31gt2igvpejxb=True):
    key = (int(cut), int(ext), bool(_v2d09izj31gt2igvpejxb))
    cache = bench.__dict__.setdefault('_rt_cache2', {})
    if key not in cache:
        if bench.blocked:
            raise ValueError(bench.blocked)
        cache[key] = Fast2(bench, cut=cut, ext=ext, _v2d09izj31gt2igvpejxb=_v2d09izj31gt2igvpejxb)
    return cache[key]