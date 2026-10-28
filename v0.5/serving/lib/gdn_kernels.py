from __future__ import annotations
import sys
from contextlib import contextmanager
import torch
import torch.nn.functional as F

FLA_CHUNK = 'fla.ops.gated_delta_rule.chunk'
PATHS = {}


def _count(name):
    PATHS[name] = PATHS.get(name, 0) + 1


def _long_state_module():
    import long_input_state as long_state

    return long_state


def _long_input_module():
    import long_input as long_input

    return long_input


def _unwrap(fn):
    return getattr(fn, '__wrapped__', fn)


def lean_available():
    try:
        import triton

        raise ImportError('not included in this release')
    except Exception:
        return False


def skipped_output(args, kw, u=None):
    v = kw['v'] if 'v' in kw else args[2] if len(args) > 2 else None
    if v is None:
        v = u
    if v is None:
        raise RuntimeError('chunk_fwd_o placeholder: neither v nor the h kernel value input is known')
    return torch.empty_like(v)


def fla_state_lean(fn, key, value, g, beta, init):
    mod = sys.modules.get(FLA_CHUNK)
    orig_o = getattr(mod, 'chunk_fwd_o', None) if mod is not None else None
    orig_h = getattr(mod, 'chunk_gated_delta_rule_fwd_h', None) if mod is not None else None
    used = dict(o=0, lean=0)
    if callable(orig_o) and callable(orig_h) and lean_available():
        raise ImportError('not included in this release')
    state, skipped = _long_state_module().fla_state(fn, key, value, g, beta, init)
    return (state, 'fla_state_only' if skipped else 'fla_full')


def delta_state(attn, key, value, g, beta, init, lean=False):
    long_state = _long_state_module()
    fn = _unwrap(attn.chunk_gated_delta_rule)
    if fn is long_state._torch_chunk():
        state, path = (long_state.torch_state(key, value, g, beta, init), 'torch_state_only')
    elif lean:
        state, path = fla_state_lean(fn, key, value, g, beta, init)
    else:
        state, skipped = long_state.fla_state(fn, key, value, g, beta, init)
        path = 'fla_state_only' if skipped else 'fla_full'
    _count(path)
    return (state, path)


def state_scan(layer, h, init=None, lean=False):
    long_input = _long_input_module()
    x, kv, key, value, g, beta = long_input.scan_inputs(layer, h, init)
    recurrent, _ = delta_state(layer.linear_attn, key, value, g, beta, None if init is None else init[1], lean)
    return (long_input.scan_tail(layer, x, kv, init), recurrent)


def batched_scan(layers, h, inits=None, lean=False):
    long_input = _long_input_module()
    inits = list(inits) if inits is not None else [None] * len(layers)
    if not layers:
        return []
    parts = [long_input.scan_inputs(layer, h, init) for layer, init in zip(layers, inits)]
    heads = [p[3].shape[2] for p in parts]
    key = torch.cat([p[2] for p in parts], 2)
    value = torch.cat([p[3] for p in parts], 2)
    g = torch.cat([p[4] for p in parts], 2)
    beta = torch.cat([p[5] for p in parts], 2)
    given = [init is not None for init in inits]
    if any(given) and (not all(given)):
        raise ValueError('batched_scan needs every start state or none')
    init = torch.cat([i[1] for i in inits], 1) if all(given) else None
    state, _ = delta_state(layers[0].linear_attn, key, value, g, beta, init, lean)
    states = state.split(heads, 1)
    return [(long_input.scan_tail(layer, p[0], p[1], i), s) for layer, p, i, s in zip(layers, parts, inits, states)]


def scan_valid(layer, rows, init, valid, lean=False):
    long_input = _long_input_module()
    a = layer.linear_attn
    x, kv, key, value, g, beta = long_input.scan_inputs(layer, rows, init)
    length = rows.shape[1]
    keep = (torch.arange(length, device=rows.device) < valid)[None, :, None]
    beta = beta * keep.to(beta.dtype)
    g = g * keep.to(g.dtype)
    recurrent, _ = delta_state(a, key, value, g, beta, None if init is None else init[1], lean)
    width = a.conv_kernel_size - 1
    kd = a.key_dim
    r = valid - width + torch.arange(width, device=rows.device)
    pick = r.clamp(0, length - 1)
    xr = x.index_select(1, pick)
    mixed = torch.cat((long_input.proj_rows(a.in_proj_qkv, xr, 0, kd), kv.index_select(1, pick)), -1).transpose(1, 2)
    prefix = init[0] if init is not None else mixed.new_zeros(mixed.shape[0], mixed.shape[1], width)
    from_prefix = prefix.index_select(-1, (r + width).clamp(0, width - 1))
    tail = torch.where((r >= 0)[None, None, :], mixed, from_prefix)
    return (tail, recurrent)


def short_dispatch(chunk_fn, recurrent_fn, max_len):

    def call(
        query,
        key,
        value,
        g=None,
        beta=None,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        **extra,
    ):
        if not extra and recurrent_fn is not None and (query.shape[1] <= max_len):
            _count('recurrent_short')
            return recurrent_fn(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=initial_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )
        _count('chunk')
        return chunk_fn(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            **extra,
        )

    call.__wrapped__ = chunk_fn
    return call


@contextmanager
def short_recurrent(layers, max_len=16):
    swapped = []
    try:
        if max_len and max_len > 0:
            for layer in layers:
                if getattr(layer, 'layer_type', None) != 'linear_attention':
                    continue
                a = layer.linear_attn
                had = 'chunk_gated_delta_rule' in vars(a)
                orig = a.chunk_gated_delta_rule
                a.chunk_gated_delta_rule = short_dispatch(orig, getattr(a, 'recurrent_gated_delta_rule', None), max_len)
                swapped.append((a, had, orig))
        yield
    finally:
        for a, had, orig in reversed(swapped):
            if had:
                a.chunk_gated_delta_rule = orig
            else:
                del a.chunk_gated_delta_rule
