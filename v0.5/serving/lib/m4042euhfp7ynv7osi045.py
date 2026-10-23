from __future__ import annotations
import os
import torch
import torch.nn.functional as F
import magmii2dz1iul56uojlmj as gk
GRAPH_BUCKETS = (128, 256, 512, 1024, 2048)
PATHS = {}

def _count(name):
    PATHS[name] = PATHS.get(name, 0) + 1

def _fb():
    import m8rhab841mmga1gnl3mp8 as fb
    return fb

def _v9kg568rycyvi6eqg5hqi():
    import m5k7yldwzx2gowr44m6pk as _v6ur7q369jbjyz1zwxrpd
    return _v6ur7q369jbjyz1zwxrpd

def payload(state):
    return 0 if state is None else sum((t.numel() * t.element_size() for t in state))

def _sdpa_has_gqa():
    import inspect
    try:
        return 'enable_gqa' in inspect.signature(F.scaled_dot_product_attention).parameters
    except (TypeError, ValueError):
        try:
            q = torch.zeros(1, 2, 1, 4)
            F.scaled_dot_product_attention(q, q[:, :1], q[:, :1], enable_gqa=True)
            return True
        except TypeError:
            return False
SDPA_GQA = _sdpa_has_gqa()

def _flash_built():
    if os.environ.get('SYSTEMONE_NO_FLASH') == '1':
        try:
            torch.backends.cuda.enable_flash_sdp(False)
        except (AttributeError, RuntimeError):
            pass
        return False
    try:
        return bool(torch.backends.cuda.is_flash_attention_available())
    except (AttributeError, RuntimeError):
        return True
FLASH_BUILT = _flash_built()

def aot_attention(q, k, v, scale):
    if FLASH_BUILT or not q.is_cuda:
        return None
    import m4q417rrk96tf7req4il7 as aot
    if not aot.ATTN['enabled']:
        return None
    return aot.attn_fwd(q, k, v, scale)
COARSE_AUTOTUNE = (('fla.modules.l2norm', ('l2norm_fwd_kernel',), ('NB',)), ('fla.modules.fused_norm_gate', ('layer_norm_gated_fwd_kernel',), ('NB',)), ('fla.ops.utils.cumsum', ('chunk_local_cumsum_scalar_kernel', 'chunk_local_cumsum_vector_kernel'), ('B',)))

def coarse_autotune_keys():
    import importlib
    try:
        from triton.runtime.autotuner import Autotuner
    except ImportError:
        return []
    changed = []
    for module, kernels, drop in COARSE_AUTOTUNE:
        try:
            mod = importlib.import_module(module)
        except ImportError:
            continue
        for name in kernels:
            tuner = getattr(mod, name, None)
            for _ in range(4):
                if tuner is None or isinstance(tuner, Autotuner):
                    break
                tuner = getattr(tuner, 'fn', None)
            if isinstance(tuner, Autotuner) and any((k in tuner.keys for k in drop)):
                tuner.keys = [k for k in tuner.keys if k not in drop]
                tuner.cache.clear()
                changed.append(name)
    return changed
GVA_VERIFIED_FLA = ('0.5.2',)

def gdn_serving_options(body):
    import importlib
    try:
        fla = importlib.import_module('fla')
    except ImportError:
        return None
    layers = [l.linear_attn for l in getattr(body, 'layers', ()) if getattr(l, 'layer_type', '') == 'linear_attention']
    if not layers or getattr(fla, '__version__', '') not in GVA_VERIFIED_FLA:
        return None
    a = layers[0]
    if a.num_v_heads % a.num_k_heads or a.num_v_heads == a.num_k_heads:
        return None
    return {'gva': True}

def _fused_mlp(self, x):
    import mdil3n8lo2da74g7xhdco as ffk
    gate, up = (self.gate_proj(x), self.up_proj(x))
    if gate.is_cuda and gate.is_contiguous() and up.is_contiguous():
        return self.down_proj(ffk.swi_glu(gate, up))
    return self.down_proj(F.silu(gate) * up)

def _v42fl7w8pieyiuhp0eilm(body):
    import types
    n = 0
    for layer in getattr(body, 'layers', ()):
        mlp = getattr(layer, 'mlp', None)
        if mlp is None or not all((hasattr(mlp, x) for x in ('gate_proj', 'up_proj', 'down_proj'))):
            continue
        if type(getattr(mlp, 'act_fn', None)).__name__ not in ('SiLU', 'SiLUActivation'):
            continue
        mlp.forward = types.MethodType(_fused_mlp, mlp)
        n += 1
    return n

def install_aot_attention():
    if FLASH_BUILT:
        return False
    from transformers.integrations.sdpa_attention import sdpa_attention_forward as base
    from transformers.modeling_utils import AttentionInterface
    if getattr(AttentionInterface._global_mapping.get('sdpa'), 'aot_wrapper', False):
        return True

    def sdpa_or_aot(module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None, **kwargs):
        n, keys = (query.shape[2], key.shape[2])
        causal = is_causal if is_causal is not None else getattr(module, 'is_causal', True)
        if attention_mask is None and (not dropout) and (scaling is not None) and (n == 1 or (n == keys and causal)):
            o = aot_attention(query, key, value, scaling)
            if o is not None:
                return (o.transpose(1, 2), None)
        return base(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs)
    sdpa_or_aot.aot_wrapper = True
    AttentionInterface.register('sdpa', sdpa_or_aot)
    return True

def bottom_right_mask(n, keys, device):
    return torch.arange(keys, device=device)[None, :] <= (keys - n + torch.arange(n, device=device))[:, None]

def attend_grouped(q, k, v, scale):
    b, hq, n, d = q.shape
    hk, keys = (k.shape[1], k.shape[2])
    rep = hq // hk
    qg = q.reshape(b, hk, rep * n, d)
    mask = None if n == 1 else bottom_right_mask(n, keys, q.device).repeat(rep, 1)
    o = F.scaled_dot_product_attention(qg, k, v, attn_mask=mask, scale=scale)
    return o.reshape(b, hq, n, v.shape[-1])

def attend_bottom_right(q, k, v, scale, prefer=None):
    n, keys = (q.shape[2], k.shape[2])
    same = q.shape[1] == k.shape[1]
    if prefer in (None, 'aot') and (not FLASH_BUILT):
        o = aot_attention(q, k, v, scale)
        if o is not None:
            return (o, 'aot')
        if prefer == 'aot':
            raise RuntimeError('attention cubin unavailable')
    if (prefer == 'flash' or (prefer is None and FLASH_BUILT)) and q.is_cuda and (q.dtype in (torch.bfloat16, torch.float16)) and (q.shape[-1] <= 256):
        try:
            o = torch.ops.aten._scaled_dot_product_flash_attention(q, k, v, 0.0, n > 1, False, scale=scale)[0]
            return (o, 'flash')
        except (RuntimeError, TypeError):
            if prefer == 'flash':
                raise
    if prefer in (None, 'sdpa_gqa') and SDPA_GQA and (not same):
        try:
            from torch.nn.attention.bias import causal_lower_right
            mask = causal_lower_right(n, keys) if n > 1 else None
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale, enable_gqa=True)
            return (o, 'sdpa_gqa')
        except (RuntimeError, TypeError, ImportError):
            if prefer == 'sdpa_gqa':
                raise
    return (attend_grouped(q, k, v, scale), 'grouped')

def prefix_attention_gqa(layer, h, pe, state):
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
    o, path = attend_bottom_right(q, k, v, a.scaling)
    _count('attn_' + path)
    o = o.transpose(1, 2).reshape(b, n, -1).contiguous() * torch.sigmoid(gate)
    h = residual + a.o_proj(o)
    return h + layer.mlp(layer.post_attention_layernorm(h))

def final_gqa(rt, h, pe, state):
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
    layer = rt.layers[rt.last]
    if layer.layer_type != 'full_attention':
        return rt.final(h, pe, state, 0, None)
    cos, sin = pe
    residual = h[:, -1:]
    x = layer.input_layernorm(h)
    a = layer.self_attn
    b, n, _ = x.shape
    k = a.k_norm(a.k_proj(x).view(b, n, -1, a.head_dim)).transpose(1, 2)
    v = a.v_proj(x).view(b, n, -1, a.head_dim).transpose(1, 2)
    _, k = apply_rotary_pos_emb(k, k, cos, sin)
    if state is not None:
        k, v = (torch.cat((state[0], k), 2), torch.cat((state[1], v), 2))
    q, gate = torch.chunk(a.q_proj(x[:, -1:]).view(b, 1, -1, a.head_dim * 2), 2, dim=-1)
    gate = gate.reshape(b, 1, -1)
    q = a.q_norm(q.view(b, 1, -1, a.head_dim)).transpose(1, 2)
    q, _ = apply_rotary_pos_emb(q, q, cos[:, -1:], sin[:, -1:])
    o, path = attend_bottom_right(q, k, v, a.scaling)
    _count('final_' + path)
    o = o.transpose(1, 2).reshape(b, 1, -1) * torch.sigmoid(gate)
    out = residual + a.o_proj(o)
    out = out + layer.mlp(layer.post_attention_layernorm(out))
    return rt.bench.ro(rt.body.norm(out[:, 0]))
SMALL_GRAPH_BUCKETS = (128, 256, 512)
RUNTIME_SHORT_MAX = 16

def runtime_options(stream=False, batch_heads=False, graph=False, short_max=RUNTIME_SHORT_MAX):
    opts = dict(stream=bool(stream), batch_heads=bool(batch_heads), lean=True, gqa=True, graph=bool(graph), short_max=int(short_max))
    if graph:
        opts['graph_buckets'] = SMALL_GRAPH_BUCKETS
    return opts

class FastRuntime:

    def __init__(self, rt, stream=False, batch_heads=False, lean=False, short_max=0, gqa=False, graph=False, graph_buckets=GRAPH_BUCKETS):
        self.rt = rt
        self.stream, self.batch_heads, self.lean = (bool(stream), bool(batch_heads), bool(lean))
        self.short_max, self.gqa, self.use_graph = (int(short_max or 0), bool(gqa), bool(graph))
        self.graph = F1cGraph(rt, graph_buckets, lean=lean) if graph else None
        self.stats = {}

    @property
    def active(self):
        return any((self.stream, self.batch_heads, self.lean, self.short_max > 0, self.gqa, self.use_graph))

    def __getattr__(self, name):
        if name == 'rt':
            raise AttributeError(name)
        return getattr(self.rt, name)

    @torch.inference_mode()
    def fast(self, ids, s, host=None, mode='_v9qzrgapy1ah7g1jj5alh', graph=False, low=None):
        rt = self.rt
        if mode not in ('_v654mofq6e6jk42xea5z6', '_v9qzrgapy1ah7g1jj5alh') or not self.active:
            return rt.fast(ids, s, host, mode, graph, low)
        if self.graph is not None:
            out = self.graph(ids, s, host, mode)
            if out is not None:
                return out
        h, start, _ = low if low is not None else rt.lower16(ids, s, host)
        if getattr(rt, 'ext', 0):
            s = max(start, s - rt.ext)
        return self.window(h, start, s, ids.shape[1], mode)

    def fa_context(self, h, start, s, i):
        rt = self.rt
        rows = h[:, :s - start]
        if not start:
            return rt.kv(i, rows, 0)
        if rows.shape[1] == 0:
            return rt.hdr_kv[i]
        return tuple((torch.cat((x, y), 2) for x, y in zip(rt.hdr_kv[i], rt.kv(i, rows, start))))

    def gdn_context(self, h, start, s, i):
        rt = self.rt
        rows = h[:, :s - start]
        init = rt.hdr_gdn[i] if start else None
        if rows.shape[1] == 0:
            return init
        if self.lean:
            return gk.state_scan(rt.layers[i], rows, init, lean=True)
        return rt.scan(rt.layers[i], rows, init)

    def gdn_all(self, h, start, s):
        rt = self.rt
        rows = h[:, :s - start]
        if rows.shape[1] == 0:
            return {i: rt.hdr_gdn[i] if start else None for i in rt.gdn_up}
        if self.batch_heads:
            inits = [rt.hdr_gdn[i] for i in rt.gdn_up] if start else None
            states = gk.batched_scan([rt.layers[i] for i in rt.gdn_up], rows, inits, lean=self.lean)
            return dict(zip(rt.gdn_up, states))
        return {i: self.gdn_context(h, start, s, i) for i in rt.gdn_up}

    def window(self, h, start, s, n, mode):
        rt = self.rt
        fb = _fb()
        has_ctx = s > 0
        stream = self.stream and has_ctx
        ctx = {} if stream or not has_ctx else {i: self.fa_context(h, start, s, i) for i in rt.fa_up}
        gdn = {}
        if has_ctx and mode == '_v9qzrgapy1ah7g1jj5alh' and (not stream or self.batch_heads):
            gdn = self.gdn_all(h, start, s)
        held = sum((payload(v) for v in (*ctx.values(), *gdn.values())))
        peak = held
        x = h[:, s - start:]
        pe, pos = (rt.rotary(s, n), rt.positions[:, s:n])
        layers = rt.layers[rt.cut:rt.last + 1]
        with gk.short_recurrent(layers, self.short_max):
            for i in range(rt.cut, rt.last + 1):
                layer = rt.layers[i]
                fa = layer.layer_type == 'full_attention'
                if fa:
                    state = (self.fa_context(h, start, s, i) if stream else ctx.pop(i, None)) if has_ctx else None
                elif mode == '_v9qzrgapy1ah7g1jj5alh' and has_ctx:
                    state = gdn.pop(i) if i in gdn else self.gdn_context(h, start, s, i)
                else:
                    state = None
                if stream:
                    peak = max(peak, sum((payload(v) for v in gdn.values())) + payload(state))
                if i == rt.last:
                    z = final_gqa(rt, x, pe, state) if self.gqa and fa else rt.final(x, pe, state, s, None)
                elif fa and state is not None and self.gqa:
                    x = prefix_attention_gqa(layer, x, pe, state)
                else:
                    x = fb.step(layer, x, pe, pos, state)
                del state
        self.stats = dict(context_payload_peak=peak, streamed=stream, start=start, s=s, n=n)
        return z

def bucket_pow2(w, low=32):
    b = low
    while b < w:
        b *= 2
    return b

class F1cGraph:

    def __init__(self, rt, buckets=GRAPH_BUCKETS, lean=False, max_graphs=64, max_window=512):
        self.rt, self.lean = (rt, bool(lean))
        self.max_window = int(max_window)
        self.buckets = tuple(sorted((int(b) for b in buckets)))
        self.max = self.buckets[-1]
        dev = rt.device
        need = self.max + bucket_pow2(self.max) + 1
        if need > rt.cos.shape[-2]:
            rt._grow(need)
        self.ids = torch.zeros((1, self.max), dtype=torch.long, device=dev)
        self.s_len = torch.zeros((1,), dtype=torch.long, device=dev)
        self.n_len = torch.ones((1,), dtype=torch.long, device=dev)
        self.graphs, self.pool, self.max_graphs = ({}, None, int(max_graphs))
        self.cuda = dev.type == 'cuda'
        self.defer, self.pending, self.hits = (False, [], {})

    def plan(self, ids, s, host=None):
        rt = self.rt
        n = ids.shape[1]
        start = rt.start_of(ids, s, host)
        s_eff = max(start, s - rt.ext) if getattr(rt, 'ext', 0) else s
        nb = next((b for b in self.buckets if b >= n), None)
        if nb is None or nb <= start or n <= s_eff:
            return None
        wb = min(bucket_pow2(n - s_eff), nb)
        if wb > self.max_window:
            return None
        return (start, s_eff, n, nb, wb)

    def fn(self, mode, start, nb, wb):
        rt = self.rt
        _v6ur7q369jbjyz1zwxrpd = _v9kg568rycyvi6eqg5hqi()
        from m7ml5d1vtymnzohdbris7 import gdn_step, layer_step
        dev, dt = (rt.device, rt.dtype)
        rows = nb - start
        h = rt.body.embed_tokens(self.ids[:, :rows])
        pe = rt.rotary(start, nb)
        h = rt.layers_range(h, pe, rt.positions[:, start:nb], start, 0, rt.cut)
        s_rel = self.s_len - start
        ctx = {}
        for i in rt.fa_up:
            kv = rt.kv(i, h, start)
            ctx[i] = tuple((torch.cat((x, y), 2) for x, y in zip(rt.hdr_kv[i], kv))) if start else kv
        gdn = {}
        if mode == '_v9qzrgapy1ah7g1jj5alh':
            for i in rt.gdn_up:
                gdn[i] = gk.scan_valid(rt.layers[i], h, rt.hdr_gdn[i] if start else None, s_rel, self.lean)
        elif mode != '_v654mofq6e6jk42xea5z6':
            raise ValueError('unknown mode')
        idx = (s_rel + torch.arange(wb, device=dev)).clamp(max=rows - 1)
        x = h.index_select(1, idx)
        del h
        wpos = self.s_len + torch.arange(wb, device=dev)
        cos, sin = (rt.cos.index_select(1, wpos), rt.sin.index_select(1, wpos))
        neg = torch.finfo(dt).min
        ctx_mask = torch.where(torch.arange(nb, device=dev) < self.s_len, 0.0, neg).to(dt)
        win = torch.full((wb, wb), neg, dtype=dt, device=dev).triu(1)
        mask = torch.cat((ctx_mask.expand(wb, nb), win), 1)[None, None]
        for i in range(rt.cut, rt.last):
            layer = rt.layers[i]
            if layer.layer_type == 'full_attention':
                x = _v6ur7q369jbjyz1zwxrpd.masked_attention_layer(layer, x, cos, sin, ctx.pop(i), mask)
            else:
                x = layer_step(layer, x, (cos, sin), None, gdn.pop(i, None), None, keep=False)[0]
        row = self.n_len - self.s_len - 1
        layer = rt.layers[rt.last]
        if layer.layer_type == 'full_attention':
            out = _v6ur7q369jbjyz1zwxrpd.masked_attention_layer(layer, x, cos, sin, ctx.pop(rt.last), mask, rows=row)
        else:
            y, _ = gdn_step(layer.linear_attn, layer.input_layernorm(x), gdn.pop(rt.last, None), keep=False)
            out = x.index_select(1, row) + y.index_select(1, row)
            out = out + layer.mlp(layer.post_attention_layernorm(out))
        return rt.bench.ro(rt.body.norm(out[:, 0]))

    @torch.inference_mode()
    def __call__(self, ids, s, host=None, mode='_v9qzrgapy1ah7g1jj5alh'):
        p = self.plan(ids, s, host)
        if p is None:
            _count('graph_eager_fallback')
            return None
        start, s_eff, n, nb, wb = p
        self.ids[:, :n - start].copy_(ids[:, start:n])
        self.s_len.fill_(s_eff)
        self.n_len.fill_(n)
        key = (mode, start, nb, wb)
        if not self.cuda:
            _count('graph_cpu_eager')
            return self.fn(*key)
        g = self.graphs.get(key)
        if g is None:
            if len(self.graphs) >= self.max_graphs:
                _count('graph_cap_fallback')
                return None
            if self.defer:
                if key not in self.pending:
                    self.pending.append(key)
                hits = getattr(self, 'hits', None)
                if hits is not None:
                    hits[key] = hits.get(key, 0) + 1
                _count('graph__vabpxxj6fxd83k3yr3p52')
                return None
            g = self.capture(key)
        g.graph.replay()
        _count('graph_replay')
        return g.out

    def capture(self, key, bounds=False):
        fb = _fb()
        g = fb._v79scs9dr4oljw1o5ga3v(None, None)
        fn = lambda: self.fn(*key)
        if bounds:
            _, start, nb, wb = key
            self.s_len.fill_(max(start, nb - min(wb, nb)))
            self.n_len.fill_(nb)
        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(fb._v6ff1myegbtwfltdvya89):
                fn()
        torch.cuda.current_stream().wait_stream(side)
        g.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g.graph, pool=self.pool):
            g.out = fn()
        self.graphs[key] = g
        return g