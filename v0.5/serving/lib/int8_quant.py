from __future__ import annotations
from typing import Iterable
import torch
from torch import nn

GDN = ('in_proj_qkv', 'in_proj_z', 'in_proj_a', 'in_proj_b', 'out_proj')
ATTN = ('q_proj', 'k_proj', 'v_proj', 'o_proj')
MLP = ('gate_proj', 'up_proj', 'down_proj')
GROUPS = {'gdn': GDN, 'attn': ATTN, 'mlp': MLP, 'ffn': MLP, 'all': GDN + ATTN + MLP}
DEFAULT_LAYERS = range(0, 16)
QMAX = {'int8': 127.0, 'fp8': 448.0}
FP8 = getattr(torch, 'float8_e4m3fn', None)
EPS = 1e-08


def parse_modules(spec):
    if spec is None:
        return GROUPS['all']
    items = spec.split(',') if isinstance(spec, str) else list(spec)
    out = []
    for item in (x.strip() for x in items):
        if not item:
            continue
        for name in GROUPS.get(item, (item,)):
            if name not in out:
                out.append(name)
    unknown = [n for n in out if n not in GROUPS['all']]
    if unknown:
        raise ValueError(f'unknown module names {unknown}; known {GROUPS["all"]}')
    return tuple(out)


def parse_layers(spec, n_layers=None):
    if spec is None:
        spec = DEFAULT_LAYERS
    if isinstance(spec, str):
        out = set()
        for part in spec.split(','):
            part = part.strip()
            if not part:
                continue
            if '-' in part:
                a, b = part.split('-')
                out.update(range(int(a), int(b) + 1))
            else:
                out.add(int(part))
        spec = out
    layers = sorted(set((int(i) for i in spec)))
    if n_layers is not None:
        layers = [i for i in layers if i < n_layers]
    return layers


def quant_weight(w, mode='int8'):
    wf = w.float()
    s = wf.abs().amax(1).clamp_min(EPS) / QMAX[mode]
    if mode == 'int8':
        q = torch.round(wf / s[:, None]).clamp(-127, 127).to(torch.int8)
    else:
        q = (wf / s[:, None]).clamp(-448.0, 448.0).to(FP8)
    return (q, s)


def quant_act(x, inv_smooth=None, mode='int8'):
    xf = x.float()
    if inv_smooth is not None:
        xf = xf * inv_smooth
    s = xf.abs().amax(-1).clamp_min(EPS) / QMAX[mode]
    if mode == 'int8':
        q = torch.round(xf / s[:, None]).clamp(-127, 127).to(torch.int8)
    else:
        q = (xf / s[:, None]).clamp(-448.0, 448.0).to(FP8)
    return (q, s)


def dequant(q, s):
    return q.float() * s.reshape(-1, 1)


def smooth_factors(act_amax, w, alpha=0.5, lo=0.0001, hi=10000.0):
    a = act_amax.float().to(w.device)
    wmax = w.float().abs().amax(0)
    s = a.clamp_min(EPS).pow(alpha) / wmax.clamp_min(EPS).pow(1.0 - alpha)
    s = torch.where((a > 0) & (wmax > 0), s, torch.ones_like(s))
    return s.clamp(lo, hi)


def int_gemm_exact(qx, qw):
    return (qx.double() @ qw.double().t()).round().to(torch.int32)


def torch_int_mm(qx, qw):
    m = qx.shape[0]
    if m <= 16:
        qx = torch.cat((qx, qx.new_zeros(32 - m, qx.shape[1])), 0)
    return torch._int_mm(qx, qw.t())[:m]


def target_linears(body, layers=None, modules=None):
    names = parse_modules(modules)
    wanted = parse_layers(layers, len(body.layers))
    out = []
    for i in wanted:
        layer = body.layers[i]
        for path, parent in layer.named_modules():
            for attr, child in parent.named_children():
                if attr in names and isinstance(child, nn.Linear):
                    q = f'layers.{i}.{(path + "." if path else "")}{attr}'
                    out.append((q, parent, attr, child))
    return out
