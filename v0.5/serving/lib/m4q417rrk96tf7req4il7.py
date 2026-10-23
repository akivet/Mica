from __future__ import annotations
import ctypes
import hashlib
import json
import os
import threading
from pathlib import Path
import torch
FORMAT = 'w8a8-aot/1'
ENV_DIR = '_vbttcdm5t019ge4njs3hy_W8A8_AOT_DIR'
DEFAULT_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / 'kernels'
DT = {torch.bfloat16: 'bf16', torch.float16: 'fp16', torch.float32: 'fp32'}
I32 = 2 ** 31
CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1
CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8
CU_FUNC_CACHE_PREFER_SHARED = 1
CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97
STATS = {'launch': 0, 'fallback': {}}

def _miss(reason):
    STATS['fallback'][reason] = STATS['fallback'].get(reason, 0) + 1
    return None

class _Driver:

    def __init__(self):
        names = ('nvcuda.dll',) if os.name == 'nt' else ('libcuda.so.1', 'libcuda.so')
        lib, err = (None, None)
        for n in names:
            try:
                lib = ctypes.CDLL(n)
                break
            except OSError as exc:
                err = exc
        if lib is None:
            raise OSError(f'CUDA driver library not found: {err}')
        P, I, U, VP = (getattr(ctypes, ''.join(map(chr, (80, 79, 73, 78, 84, 69, 82)))), ctypes.c_int, ctypes.c_uint, ctypes.c_void_p)
        proto = {'cuInit': (U,), 'cuGetErrorString': (I, P(ctypes.c_char_p)), 'cuCtxGetCurrent': (P(VP),), 'cuCtxSetCurrent': (VP,), 'cuDevicePrimaryCtxRetain': (P(VP), I), 'cuModuleLoadData': (P(VP), VP), 'cuModuleGetFunction': (P(VP), VP, ctypes.c_char_p), 'cuFuncGetAttribute': (P(I), I, VP), 'cuFuncSetAttribute': (VP, I, I), 'cuFuncSetCacheConfig': (VP, I), 'cuDeviceGetAttribute': (P(I), I, I), 'cuLaunchKernel': (VP, U, U, U, U, U, U, U, VP, P(VP), P(VP))}
        for name, args in proto.items():
            f = getattr(lib, name)
            f.argtypes, f.restype = (list(args), ctypes.c_int)
            setattr(self, name, f)
        self.check(self.cuInit(0), 'cuInit')

    def check(self, rc, what):
        if rc != 0:
            s = ctypes.c_char_p()
            self.cuGetErrorString(rc, ctypes.byref(s))
            raise RuntimeError(f"{what} failed: CUDA error {rc} {(s.value.decode() if s.value else '')}")
_DRV = None
_LOCK = threading.Lock()

def _driver():
    global _DRV
    if _DRV is None:
        _DRV = _Driver()
    return _DRV
_INDEX = {}
_FUNCS = {}
_CTX = {}

def kernel_dir():
    return Path(os.environ.get(ENV_DIR) or DEFAULT_DIR)
PACKED = 'k{}.pt'

def _load_index(cc):
    if cc in _INDEX:
        return _INDEX[cc]
    res = None
    j = kernel_dir() / f'w8a8_aot_sm{cc}.json'
    p = kernel_dir() / PACKED.format(cc)
    if j.is_file():
        idx = json.loads(j.read_text(encoding='utf-8'))
        blob = (kernel_dir() / idx.get('blob', f'w8a8_aot_sm{cc}.bin')).read_bytes()
        if idx.get('format') == FORMAT:
            idx['by_key'] = {e['key']: e for e in idx['entries']}
            res = (idx, blob)
    elif p.is_file():
        import zlib
        data = torch.load(str(p), map_location='cpu', weights_only=True)
        idx = json.loads(zlib.decompress(data['i'].numpy().tobytes()).decode('utf-8'))
        if idx.get('entries'):
            idx['by_key'] = {e['key']: e for e in idx['entries']}
            res = (idx, zlib.decompress(data['b'].numpy().tobytes()))
    _INDEX[cc] = res
    return res

def _pick_cc(major, minor):
    cc = major * 10 + minor
    if _load_index(cc):
        return cc
    for m in range(minor - 1, -1, -1):
        r = _load_index(major * 10 + m)
        if r and (not r[0].get('arch_specific')):
            return major * 10 + m
    return None

def _dev(device):
    dev = torch.device(device) if device is not None else torch.device('cuda')
    if dev.type == 'cuda' and dev.index is None:
        dev = torch.device('cuda', torch.cuda.current_device())
    return dev
_CC = {}

def _device_cc(dev):
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    if idx not in _CC:
        major, minor = torch.cuda.get_device_capability(idx)
        cc = _pick_cc(major, minor)
        if cc is None:
            return None
        _CC[idx] = cc
    return _CC[idx]

def available(device=None):
    if not torch.cuda.is_available():
        return (False, 'CUDA not available')
    dev = _dev(device)
    if dev.type != 'cuda':
        return (False, f'{dev} is not CUDA')
    try:
        _driver()
    except Exception as exc:
        return (False, f'driver: {exc!r}'[:200])
    cc = _device_cc(dev)
    if cc is None:
        return (False, f'no cubin container for capability {torch.cuda.get_device_capability(dev)} in {kernel_dir()}')
    return (True, f'sm_{cc} from {kernel_dir()}')

def _ensure_ctx(idx):
    drv = _driver()
    ctx = ctypes.c_void_p()
    drv.check(drv.cuCtxGetCurrent(ctypes.byref(ctx)), 'cuCtxGetCurrent')
    if not ctx.value:
        drv.check(drv.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), idx), 'cuDevicePrimaryCtxRetain')
        drv.check(drv.cuCtxSetCurrent(ctx), 'cuCtxSetCurrent')
    _CTX.setdefault(idx, ctx.value)

def _function(idx, cc, key):
    hit = _FUNCS.get((idx, cc, key))
    if hit is not None:
        return hit
    with _LOCK:
        hit = _FUNCS.get((idx, cc, key))
        if hit is not None:
            return hit
        index, blob = _load_index(cc)
        e = index['by_key'].get(key)
        if e is None:
            return None
        cubin = blob[e['offset']:e['offset'] + e['size']]
        if hashlib.sha256(cubin).hexdigest() != e['sha256']:
            raise RuntimeError(f'w8a8 AOT cubin {key} (sm_{cc}) fails its sha256 check')
        drv = _driver()
        _ensure_ctx(idx)
        mod, fn = (ctypes.c_void_p(), ctypes.c_void_p())
        buf = ctypes.create_string_buffer(cubin, len(cubin))
        drv.check(drv.cuModuleLoadData(ctypes.byref(mod), ctypes.cast(buf, ctypes.c_void_p)), f'cuModuleLoadData {key}')
        drv.check(drv.cuModuleGetFunction(ctypes.byref(fn), mod, e['name'].encode()), f'cuModuleGetFunction {key}')
        optin = ctypes.c_int()
        drv.check(drv.cuDeviceGetAttribute(ctypes.byref(optin), CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, idx), 'cuDeviceGetAttribute')
        if e['shared'] > optin.value:
            _FUNCS[idx, cc, key] = False
            return False
        if e['shared'] > 49152:
            static = ctypes.c_int()
            drv.check(drv.cuFuncSetCacheConfig(fn, CU_FUNC_CACHE_PREFER_SHARED), 'cuFuncSetCacheConfig')
            drv.check(drv.cuFuncGetAttribute(ctypes.byref(static), CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, fn), 'cuFuncGetAttribute')
            drv.check(drv.cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, optin.value - static.value), 'cuFuncSetAttribute')
        hit = (fn, e, mod)
        _FUNCS[idx, cc, key] = hit
        return hit

def preload(device=None):
    dev = _dev(device)
    cc = _device_cc(dev)
    if cc is None:
        return 0
    with torch.cuda.device(dev):
        torch.cuda.current_stream(dev)
        return sum((1 for k in _load_index(cc)[0]['by_key'] if _function(dev.index, cc, k)))
_TLS = threading.local()

def _prepared(dev, cc, key, fn, e):
    cache = _TLS.__dict__.setdefault('prepared', {})
    rec = cache.get((dev.index, cc, key))
    if rec is None:
        slots, names = ([], [])
        for p in e['params']:
            if p['type'].startswith('*'):
                slots.append(ctypes.c_uint64(0))
            elif p['type'] == 'i32':
                slots.append(ctypes.c_int32(0))
            else:
                return None
            names.append((p['name'], p['type'] == 'i32'))
        scratch = [h for h in e['hidden'] if h.get('size', 0) > 0]
        hidden = [ctypes.c_uint64(0) for _ in e['hidden']]
        params = (ctypes.c_void_p * (len(slots) + len(hidden)))(*[ctypes.addressof(v) for v in slots + hidden])
        rec = (slots, names, hidden, scratch, params, 32 * e['num_warps'], e['shared'], tuple(e['div16']), tuple(e['expect'].items()))
        cache[dev.index, cc, key] = rec
    return rec

def _stream(dev):
    try:
        return torch._C._cuda_getCurrentRawStream(dev.index)
    except (AttributeError, RuntimeError):
        return torch.cuda.current_stream(dev).cuda_stream

def _launch(dev, cc, key, args, grid):
    hit = _function(dev.index, cc, key)
    if not hit:
        return _miss('no_cubin' if hit is None else 'shared_mem')
    fn, e, _ = hit
    rec = _prepared(dev, cc, key, fn, e)
    if rec is None:
        return _miss('param_type')
    slots, names, hidden, scratch, params, block, shared, div16, expect = rec
    for name in div16:
        if args[name] % 16:
            return _miss(f'div16:{name}')
    for name, v in expect:
        if args[name] != v:
            return _miss(f'expect:{name}')
    for slot, (name, is_i32) in zip(slots, names):
        v = args[name]
        if is_i32 and (not -I32 <= v < I32):
            return _miss('i32_range')
        slot.value = v
    keep = []
    if scratch:
        for slot, h in zip(hidden, e['hidden']):
            ptr = 0
            if h.get('size', 0) > 0:
                t = torch.empty(grid * h['size'] + h.get('align', 1), dtype=torch.uint8, device=dev)
                keep.append(t)
                ptr = t.data_ptr()
                ptr += -ptr % max(1, h.get('align', 1))
            slot.value = ptr
    if grid <= 0:
        return True
    drv = _driver()
    if not getattr(_TLS, 'ctx_ok', False):
        ctx = ctypes.c_void_p()
        drv.cuCtxGetCurrent(ctypes.byref(ctx))
        if not ctx.value:
            _ensure_ctx(dev.index)
        _TLS.ctx_ok = True
    drv.check(drv.cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, shared, _stream(dev), params, None), f'cuLaunchKernel {key}')
    STATS['launch'] += 1
    return True

def _on(dev, call):
    if dev.index is None or torch.cuda.current_device() == dev.index:
        return call()
    with torch.cuda.device(dev):
        return call()

def _aligned(*ts):
    return all((t.data_ptr() % 16 == 0 for t in ts))

def act_quant(x, inv_smooth=None):
    if x.dim() != 2 or x.device.type != 'cuda' or x.dtype not in DT:
        return _miss('act_input')
    m, k = x.shape
    if x.stride(1) != 1 or x.stride(0) != k:
        return _miss('act_contiguous')
    cc = _device_cc(x.device)
    if cc is None:
        return _miss('no_container')
    if inv_smooth is not None and (inv_smooth.dtype != torch.float32 or not inv_smooth.is_contiguous() or inv_smooth.numel() != k or (inv_smooth.device != x.device)):
        return _miss('act_inv')
    if m * k >= I32:
        return _miss('i32_range')
    q = torch.empty((m, k), dtype=torch.int8, device=x.device)
    s = torch.empty((m,), dtype=torch.float32, device=x.device)
    inv = inv_smooth if inv_smooth is not None else s
    if not _aligned(x, inv, q, s):
        return _miss('align')
    key = f'act|{DT[x.dtype]}|K{k}|inv{int(inv_smooth is not None)}'
    args = dict(X=x.data_ptr(), INV=inv.data_ptr(), Q=q.data_ptr(), S=s.data_ptr(), K=k, s_xm=x.stride(0), s_qm=q.stride(0))
    ok = _on(x.device, lambda: _launch(x.device, cc, key, args, m))
    return (q, s) if ok else None
GEMM_CLASSES = ((128, 's'), (384, 'm'), (1024, 'n'))

def mm_i8(qx, sx, qw, sw, bias=None, out_dtype=torch.bfloat16):
    if qx.dim() != 2 or qw.dim() != 2 or qx.dtype != torch.int8 or (qw.dtype != torch.int8) or (out_dtype not in DT):
        return _miss('mm_input')
    m, k = qx.shape
    n = qw.shape[0]
    if qw.shape[1] != k or qx.stride(1) != 1 or qw.stride(1) != 1 or (qx.stride(0) != k) or (qw.stride(0) != k):
        return _miss('mm_contiguous')
    for t, size in ((sx, m), (sw, n)) + (((bias, n),) if bias is not None else ()):
        if t.dtype != torch.float32 or not t.is_contiguous() or t.numel() != size or (t.device != qx.device):
            return _miss('mm_scales')
    if qw.device != qx.device:
        return _miss('mm_device')
    cc = _device_cc(qx.device)
    if cc is None:
        return _miss('no_container')
    if m * max(k, n) >= I32:
        return _miss('i32_range')
    out = torch.empty((m, n), dtype=out_dtype, device=qx.device)
    b = bias if bias is not None else sw
    if not _aligned(qx, qw, sx, sw, b, out):
        return _miss('align')
    index = _load_index(cc)[0]['by_key']
    base = f'mm|N{n}|K{k}|bias{int(bias is not None)}|{DT[out_dtype]}|m16'
    key = base + '1' if m % 16 == 0 and base + '1' in index else base + '0'
    ckey = key + '|c' + next((c for hi, c in GEMM_CLASSES if m <= hi), 'l')
    if ckey in index:
        key = ckey
    e = index.get(key)
    if e is None:
        return _miss('no_cubin')
    bm, bn = (e['constexprs']['BM'], e['constexprs']['BN'])
    grid = -(-m // bm) * -(-n // bn)
    args = dict(A=qx.data_ptr(), B=qw.data_ptr(), SA=sx.data_ptr(), SB=sw.data_ptr(), BIAS=b.data_ptr(), C=out.data_ptr(), M=m, N=n, K=k, s_am=qx.stride(0), s_bn=qw.stride(0), s_cm=out.stride(0))
    ok = _on(qx.device, lambda: _launch(qx.device, cc, key, args, grid))
    return out if ok else None
ATTN = {'enabled': False}
ATTN_SMALL = 32
_SMS = {}

def _sm_count(dev):
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    if idx not in _SMS:
        _SMS[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
    return _SMS[idx]

def attn_key(hq, rep, d, scale, small):
    return 'attn|bf16|hd%d|hq%d|r%d|s%.8g|%s' % (d, hq, rep, scale, 'm16' if small else 'm64')

def attn_keys(device=None):
    cc = _device_cc(_dev(device))
    return [] if cc is None else sorted((k for k in _load_index(cc)[0]['by_key'] if k.startswith('attn|')))

def attn_fwd(q, k, v, scale):
    if q.dim() != 4 or k.dim() != 4 or v.shape != k.shape or (q.device.type != 'cuda') or (not q.dtype == k.dtype == v.dtype == torch.bfloat16):
        return _miss('attn_input')
    b, hq, n, d = q.shape
    hk, keys = (k.shape[1], k.shape[2])
    if k.shape[0] != b or k.shape[3] != d or hq % hk or (n < 1) or (keys < n) or (k.device != q.device):
        return _miss('attn_shape')
    if q.stride(3) != 1 or k.stride(3) != 1 or v.stride(3) != 1:
        return _miss('attn_stride')
    cc = _device_cc(q.device)
    if cc is None:
        return _miss('no_container')
    index = _load_index(cc)[0]['by_key']
    small = n <= ATTN_SMALL or -(-n // 64) * b * hq <= _sm_count(q.device) // 4
    key = attn_key(hq, hq // hk, d, scale, small)
    if key not in index:
        key = attn_key(hq, hq // hk, d, scale, False)
    e = index.get(key)
    if e is None:
        return _miss('no_cubin')
    o = torch.empty((b, n, hq, d), dtype=q.dtype, device=q.device).transpose(1, 2)
    if not _aligned(q, k, v, o):
        return _miss('align')
    args = dict(Q=q.data_ptr(), K=k.data_ptr(), V=v.data_ptr(), O=o.data_ptr(), n_q=n, n_k=keys, s_qb=q.stride(0), s_qh=q.stride(1), s_qm=q.stride(2), s_kb=k.stride(0), s_kh=k.stride(1), s_kn=k.stride(2), s_vb=v.stride(0), s_vh=v.stride(1), s_vn=v.stride(2), s_ob=o.stride(0), s_oh=o.stride(1), s_om=o.stride(2))
    grid = -(-n // e['constexprs']['BM']) * b * hq
    ok = _on(q.device, lambda: _launch(q.device, cc, key, args, grid))
    return o if ok else None