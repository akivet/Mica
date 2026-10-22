import os
from contextlib import contextmanager
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class _vceeuht53lwl7cy30sonn:

    def __init__(self, k=None, v=None):
        self.k, self.v, self.new = (k, v, None)

    def update(self, k, v, layer_idx=None, cache_kwargs=None):
        self.new = (k, v)
        if self.k is None:
            return (k, v)
        return (torch.cat((self.k, k), 2), torch.cat((self.v, v), 2))

def gdn_step(attn, x, state=None, keep=True, lengths=None, valid=None):
    batch, length, _ = x.shape
    mixed, z, b, a = gdn_projections(attn, x)
    mixed = mixed.transpose(1, 2)
    z = z.reshape(batch, length, -1, attn.head_v_dim)
    width = attn.conv_kernel_size - 1
    if lengths is not None and keep and (state is None):
        state = (mixed.new_zeros(batch, mixed.shape[1], width), None)
    conv = None
    if SERVE['conv_cl'] and lengths is None and (attn.causal_conv1d_fn is not None):
        conv = conv_channel_last(attn.causal_conv1d_fn, mixed, None if state is None else state[0], attn.conv1d.weight.squeeze(1), attn.conv1d.bias, attn.activation)
    if conv is not None:
        full = mixed if state is None or length >= width else torch.cat((state[0], mixed), -1)
    else:
        full = mixed if state is None else torch.cat((state[0], mixed), -1)
    if not keep:
        tail = None
    elif lengths is not None:
        index = lengths[:, None] + torch.arange(width, device=full.device)
        tail = full.gather(-1, index[:, None, :].expand(batch, full.shape[1], width))
    else:
        tail = F.pad(full, (width - full.shape[-1], 0)) if full.shape[-1] < width else full[..., -width:]
    if conv is not None:
        pass
    elif attn.causal_conv1d_fn is not None:
        conv = attn.causal_conv1d_fn(x=full, weight=attn.conv1d.weight.squeeze(1), bias=attn.conv1d.bias, activation=attn.activation, seq_idx=None)
    else:
        conv = F.silu(attn.conv1d(full)[:, :, :full.shape[-1]])
    mixed = conv[..., -length:].transpose(1, 2)
    query, key, value = torch.split(mixed, [attn.key_dim, attn.key_dim, attn.value_dim], dim=-1)
    query = query.reshape(batch, length, -1, attn.head_k_dim)
    key = key.reshape(batch, length, -1, attn.head_k_dim)
    value = value.reshape(batch, length, -1, attn.head_v_dim)
    initial = None if state is None else state[1]
    kernel = gdn_kernel_options(attn) if FUSION['gdn'] else None
    if kernel is not None and kernel.get('gate'):
        g, beta, extra = (a, b, dict(kernel['gate'], A_log=attn.A_log, dt_bias=attn.dt_bias))
    else:
        beta = b.sigmoid()
        g = -attn.A_log.float().exp() * F.softplus(a.float() + attn.dt_bias)
        extra = {}
    if valid is not None and keep:
        if extra:
            beta = b.sigmoid()
            g = -attn.A_log.float().exp() * F.softplus(a.float() + attn.dt_bias)
            extra = {}
        beta = beta * valid[..., None].to(beta.dtype)
        g = g * valid[..., None].to(g.dtype)
    if attn.num_v_heads // attn.num_k_heads > 1 and (not (kernel is not None and kernel.get('gva'))):
        query = query.repeat_interleave(attn.num_v_heads // attn.num_k_heads, dim=2)
        key = key.repeat_interleave(attn.num_v_heads // attn.num_k_heads, dim=2)
    out, recurrent = attn.chunk_gated_delta_rule(query, key, value, g=g, beta=beta, initial_state=initial, output_final_state=keep, use_qk_l2norm_in_kernel=True, **extra)
    out = attn.norm(out.reshape(-1, attn.head_v_dim), z.reshape(-1, attn.head_v_dim)).reshape(batch, length, -1)
    return (attn.out_proj(out), (tail, recurrent) if keep else None)
SERVE = dict(conv_cl=False, gqa_flash=False)
CONV_CL = {'ok': 0, 'fallback': 0}

def conv_channel_last(fn, x, init, weight, bias, activation):
    if x.stride(1) != 1 or not (x.is_cuda or SERVE.get('conv_cl_cpu')):
        return None
    try:
        if init is not None:
            init = init.transpose(1, 2).contiguous().transpose(1, 2)
        out = fn(x=x, weight=weight, bias=bias, initial_states=init, activation=activation, seq_idx=None)
    except (TypeError, RuntimeError):
        CONV_CL['fallback'] += 1
        return None
    CONV_CL['ok'] += 1
    return out

def rms_norm_lean(weight1, eps, x):
    ss = torch.linalg.vector_norm(x, 2, -1, keepdim=True, dtype=torch.float32)
    rstd = torch.rsqrt(ss.square_().div_(x.shape[-1]).add_(eps))
    y = x * rstd
    if x.dtype == torch.float32:
        return y.mul_(weight1)
    return torch.mul(y, weight1, out=torch.empty_like(x))

def install_norms(body):
    import functools
    n = 0
    for m in body.modules():
        name = type(m).__name__
        if not name.endswith('RMSNorm') or 'Gated' in name or (not hasattr(m, 'eps')):
            continue
        w1 = (1.0 + m.weight.detach().float()).contiguous()
        m.forward = functools.partial(rms_norm_lean, w1, float(m.eps))
        n += 1
    return n
FUSION = dict(mlp=False, gdn=False, gdn_proj=False)
_GDN_OPTIONS = {}

def _gdn_probe_key(attn, dtype, device):
    return (attn.chunk_gated_delta_rule, str(device), dtype, attn.num_k_heads, attn.num_v_heads, attn.head_k_dim, attn.head_v_dim)

def gdn_kernel_options(attn):
    weight = _linear_weight(attn.in_proj_qkv)
    return _GDN_OPTIONS.get(_gdn_probe_key(attn, weight.dtype, weight.device))

def _linear_weight(module):
    weight = getattr(module, 'weight', None)
    return weight if weight is not None else module.base.weight

def _lora_type():
    try:
        raise ImportError('unavailable')
        return LoRALinear
    except ImportError:
        return None

def _plain_linear(module):
    return type(module) is torch.nn.Linear and module.bias is None and (not module.weight.requires_grad)

def gate_up_weight(mlp):
    lora = _lora_type()
    gate, up = (getattr(mlp, 'gate_proj', None), getattr(mlp, 'up_proj', None))
    if gate is None or up is None or (not hasattr(mlp, 'down_proj')) or (not hasattr(mlp, 'act_fn')):
        return None
    bases = []
    for module in (gate, up):
        if lora is not None and type(module) is lora:
            if getattr(module, '_ve4j7y0mumsnl4kmyp4ki_a', None) is not None:
                return None
            module = module.base
        if not _plain_linear(module):
            return None
        bases.append(module)
    g, u = (bases[0].weight, bases[1].weight)
    if g.dtype != u.dtype or g.device != u.device or g.shape[1] != u.shape[1]:
        return None
    fused = mlp.__dict__.get('_fused_gate_up')
    rows = g.shape[0]
    if fused is not None and fused.shape == (rows + u.shape[0], g.shape[1]) and (g.data_ptr() == fused.data_ptr()) and (u.data_ptr() == fused[rows:].data_ptr()) and g.is_contiguous() and u.is_contiguous():
        return fused
    with torch.no_grad():
        fused = torch.cat((g.data, u.data), 0)
        bases[0].weight.data = fused[:rows]
        bases[1].weight.data = fused[rows:]
    mlp.__dict__['_fused_gate_up'] = fused
    return fused

def _lora_delta(module, x, base_output):
    with torch.autocast(device_type=x.device.type, enabled=False):
        xf = x.to(module.a.dtype)
        delta = F.linear(F.linear(xf, module.a), module.b) * module.scale if getattr(module, 'active', True) else None
        if getattr(module, '_ve22609zwxy0resefwrru', None) is not None:
            frozen = F.linear(F.linear(xf, getattr(module, ''.join(map(chr, (97, 48))))), module.b0) * module.scale
            delta = frozen if delta is None else frozen + delta
    return base_output if delta is None else base_output + delta.to(dtype=base_output.dtype)
_GDN_PROJECTIONS = ('in_proj_qkv', 'in_proj_z', 'in_proj_b', 'in_proj_a')

def gdn_projection_weight(attn):
    lora = _lora_type()
    bases = []
    for name in _GDN_PROJECTIONS:
        module = getattr(attn, name)
        if lora is not None and type(module) is lora:
            if getattr(module, '_ve4j7y0mumsnl4kmyp4ki_a', None) is not None:
                return None
            module = module.base
        if not _plain_linear(module):
            return None
        bases.append(module)
    weights = [module.weight for module in bases]
    first = weights[0]
    if any((w.dtype != first.dtype or w.device != first.device or w.shape[1] != first.shape[1] for w in weights)):
        return None
    sizes = [w.shape[0] for w in weights]
    fused = attn.__dict__.get('_fused_gdn_proj')
    if fused is not None and fused.shape == (sum(sizes), first.shape[1]) and (fused.dtype == first.dtype) and (fused.device == first.device):
        pieces = fused.split(sizes, dim=0)
        if all((w.data_ptr() == p.data_ptr() and w.is_contiguous() for w, p in zip(weights, pieces))):
            return fused
    with torch.no_grad():
        fused = torch.cat([w.data for w in weights], dim=0)
        for module, piece in zip(bases, fused.split(sizes, dim=0)):
            module.weight.data = piece
    attn.__dict__['_fused_gdn_proj'] = fused
    return fused

def gdn_projections(attn, x):
    fused = gdn_projection_weight(attn) if FUSION['gdn_proj'] else None
    modules = [getattr(attn, name) for name in _GDN_PROJECTIONS]
    if fused is None:
        return tuple((module(x) for module in modules))
    sizes = [_linear_weight(module).shape[0] for module in modules]
    outputs = F.linear(x, fused).split(sizes, dim=-1)
    lora = _lora_type()
    return tuple((_lora_delta(module, x, output) if lora is not None and type(module) is lora else output for module, output in zip(modules, outputs)))

def lora_gate_up(modules, x, gate, up):
    with torch.autocast(device_type=x.device.type, enabled=False):
        xf = x.to(modules[0].a.dtype)
        active = [m for m in modules if getattr(m, 'active', True)]
        projected = iter(F.linear(xf, torch.cat([m.a for m in active])).split([m.a.shape[0] for m in active], -1)) if active else iter(())
        outputs = []
        for m, base in zip(modules, (gate, up)):
            delta = F.linear(next(projected), m.b) * m.scale if getattr(m, 'active', True) else None
            if getattr(m, '_ve22609zwxy0resefwrru', None) is not None:
                frozen = F.linear(F.linear(xf, getattr(m, ''.join(map(chr, (97, 48))))), m.b0) * m.scale
                delta = frozen if delta is None else frozen + delta
            outputs.append(base if delta is None else base + delta.to(base.dtype))
        return outputs

def mlp_forward(mlp, x):
    fused = gate_up_weight(mlp) if FUSION['mlp'] else None
    if fused is None:
        return mlp(x)
    rows = mlp.gate_proj.weight.shape[0] if hasattr(mlp.gate_proj, 'weight') else mlp.gate_proj.base.weight.shape[0]
    if x.device.type == 'cpu' and x.dtype == fused.dtype == torch.float32 and (fused.shape[0] == 2 * rows):
        flat = x.reshape(-1, x.shape[-1])
        grouped = torch.bmm(flat[None].expand(2, -1, -1), fused.view(2, rows, -1).transpose(1, 2))
        gate, up = (value.reshape(*x.shape[:-1], rows) for value in grouped.unbind(0))
    else:
        gate, up = F.linear(x, fused).split((rows, fused.shape[0] - rows), dim=-1)
    lora = _lora_type()
    modules = (mlp.gate_proj, mlp.up_proj)
    if lora is not None and all((type(m) is lora for m in modules)) and (modules[0].a.dtype == modules[1].a.dtype):
        gate, up = lora_gate_up(modules, x, gate, up)
    else:
        if lora is not None and type(mlp.gate_proj) is lora:
            gate = _lora_delta(mlp.gate_proj, x, gate)
        if lora is not None and type(mlp.up_proj) is lora:
            up = _lora_delta(mlp.up_proj, x, up)
    return mlp.down_proj(mlp.act_fn(gate) * up)

def tail_mask(prefix, length, implementation, dtype, device):
    if prefix == 0 or implementation.startswith('flash'):
        return None
    mask = torch.zeros(length, prefix + length, dtype=dtype, device=device)
    mask[:, prefix:] = torch.full((length, length), torch.finfo(dtype).min, dtype=dtype, device=device).triu(1)
    return mask[None, None]

def layer_step(layer, h, position_embeddings, position_ids, state=None, mask=None, keep=True, lengths=None, valid=None):
    residual = h
    x = layer.input_layernorm(h)
    if layer.layer_type == 'linear_attention':
        if lengths is None and valid is None:
            x, snapshot = gdn_step(layer.linear_attn, x, state, keep=keep)
        else:
            x, snapshot = gdn_step(layer.linear_attn, x, state, keep=keep, lengths=lengths, valid=valid)
    else:
        cache = _vceeuht53lwl7cy30sonn(*(state or (None, None)))
        x, _ = layer.self_attn(hidden_states=x, attention_mask=mask, position_ids=position_ids, past_key_values=cache, position_embeddings=position_embeddings)
        k, v = cache.new
        snapshot = None if not keep else (k, v) if state is None else (torch.cat((state[0], k), 2), torch.cat((state[1], v), 2))
    h = residual + x
    if FUSION['mlp']:
        return (h + mlp_forward(layer.mlp, layer.post_attention_layernorm(h)), snapshot)
    return (h + layer.mlp(layer.post_attention_layernorm(h)), snapshot)

def common_prefix(sequences):
    first = sequences[0]
    limit = min((len(s) for s in sequences)) - 1
    n = 0
    while n < limit and all((s[n] == first[n] for s in sequences[1:])):
        n += 1
    return n

def exp_rotary(body, h, p):
    return body.rotary_emb(h, p[None, None].expand(3, 1, -1))