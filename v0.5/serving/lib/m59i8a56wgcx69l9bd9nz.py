import hashlib
import json
import sys
import time
import types
from pathlib import Path
import torch
from torch import nn
ALPHA = 0.5
STATS = 's.pt'
LOAD_PEAK = {'int8': 8800000000.0, 'bf16': 9000000000.0}
STEADY = {'int8': 8400000000.0, 'bf16': 11800000000.0}
PER_TOKEN = (17400000000.0 - 8400000000.0) / 32000
RESERVE = 0.95
MIN_CONTEXT = 4096
CONTEXT_STEP = 1024
MIN_VRAM = int(max(LOAD_PEAK['int8'], STEADY['int8'] + MIN_CONTEXT * PER_TOKEN) / RESERVE)
CONVERT_FREE = 512 * 2 ** 20
DUAL_EXTRA = 3400000000.0
PRECISION = {'mode': 'int8'}
LOAD_PEAK['file'] = 5000000000.0
STEADY_LEAN = 5800000000.0
LEAN_GRAPH_MAX = 512
MIN_SERVED = 8192
MIN_VRAM_FILE = int(max(LOAD_PEAK['file'], STEADY_LEAN + MIN_CONTEXT * PER_TOKEN) / RESERVE)
LIMIT = {'bytes': 0}

def context_cap(total_bytes, precision='int8', dual=False, source='convert', profile='standard'):
    if source == 'file' and precision == 'int8':
        budget = total_bytes * RESERVE
        if budget < LOAD_PEAK['file']:
            return 0
        steady = STEADY_LEAN if profile == 'lean' else STEADY['int8']
        n = int((budget - steady) / PER_TOKEN) // CONTEXT_STEP * CONTEXT_STEP
        return min(32768, n) if n >= MIN_CONTEXT else 0
    steady = STEADY[precision] + (DUAL_EXTRA if dual else 0)
    budget = total_bytes * RESERVE
    if budget < LOAD_PEAK[precision] + (DUAL_EXTRA if dual else 0):
        return 0
    n = int((budget - steady) / PER_TOKEN) // CONTEXT_STEP * CONTEXT_STEP
    return min(32768, n) if n >= MIN_CONTEXT else 0

def memory_plan(total_bytes, precision='int8', dual=False, source='convert'):
    std = context_cap(total_bytes, precision, dual, source)
    if source != 'file' or precision != 'int8' or std >= MIN_SERVED:
        return ('standard', std)
    lean = context_cap(total_bytes, precision, False, source, 'lean')
    return ('lean', lean) if lean > std else ('standard', std)

def _cuda(device):
    dev = torch.device(device)
    return torch.device('cuda', torch.cuda.current_device()) if dev.index is None else dev

def total_memory(device):
    real = torch.cuda.get_device_properties(_cuda(device)).total_memory
    return min(real, LIMIT['bytes']) if LIMIT['bytes'] else real

def limit_memory(device, limit_bytes):
    dev = _cuda(device)
    real = torch.cuda.get_device_properties(dev).total_memory
    LIMIT['bytes'] = int(min(real, limit_bytes))
    torch.cuda.set_per_process_memory_fraction(LIMIT['bytes'] / real, dev)
    return LIMIT['bytes']
_MODS = {}

def _mods():
    if not _MODS:
        import m1rrs8kpf3hjn4ltqozb9 as _vbetmnfvsyrppuql03kv3
        import m4q417rrk96tf7req4il7 as aot
        _MODS['m'] = (_vbetmnfvsyrppuql03kv3, aot)
    return _MODS['m']

def name_key(name):
    return int.from_bytes(hashlib.sha256(name.encode('utf-8')).digest()[:8], 'little', signed=True)

class _Stats:

    def __init__(self, keys, offsets, values):
        self._at = {int(k): (int(offsets[i]), int(offsets[i + 1])) for i, k in enumerate(keys.tolist())}
        self._values = values

    def get(self, name):
        hit = self._at.get(name_key(name))
        return None if hit is None else self._values[hit[0]:hit[1]]

    def __len__(self):
        return len(self._at)

def load_stats(path):
    data = torch.load(str(path), map_location='cpu', weights_only=True)
    if 'amax' in data:
        return data['amax']
    return _Stats(data['h'], data['o'], data['v'])

class QLinear(nn.Module):

    def __init__(self, linear, smooth=None, backend='aot', keep_bf16=False):
        super().__init__()
        import m1rrs8kpf3hjn4ltqozb9 as _vbetmnfvsyrppuql03kv3
        w = linear.weight.detach()
        self.register_buffer('w16', w.contiguous() if keep_bf16 else None)
        self.in_features, self.out_features = (w.shape[1], w.shape[0])
        self.out_dtype, self.backend, self.mode = (w.dtype, backend, 'int8')
        wf = w.float()
        if smooth is not None:
            smooth = smooth.float().to(w.device)
            wf = wf * smooth[None, :]
            self.register_buffer('inv_smooth', (1.0 / smooth).contiguous())
        else:
            self.inv_smooth = None
        q, s = _vbetmnfvsyrppuql03kv3.quant_weight(wf, 'int8')
        self.register_buffer('qweight', q.contiguous())
        self.register_buffer('wscale', s.contiguous())
        self.register_buffer('bias', None if linear.bias is None else linear.bias.detach().float().contiguous())

    @classmethod
    def empty(cls, in_features, out_features, out_dtype=torch.bfloat16, backend='aot', smooth=True, bias=False):
        self = cls.__new__(cls)
        nn.Module.__init__(self)
        self.register_buffer('w16', None)
        self.in_features, self.out_features = (in_features, out_features)
        self.out_dtype, self.backend, self.mode = (out_dtype, backend, 'int8')
        meta = lambda *shape, dtype=torch.float32: torch.empty(*shape, dtype=dtype, device='meta')
        if smooth:
            self.register_buffer('inv_smooth', meta(in_features))
        else:
            self.inv_smooth = None
        self.register_buffer('qweight', meta(out_features, in_features, dtype=torch.int8))
        self.register_buffer('wscale', meta(out_features))
        self.register_buffer('bias', meta(out_features) if bias else None)
        return self

    @property
    def weight(self):
        import m1rrs8kpf3hjn4ltqozb9 as _vbetmnfvsyrppuql03kv3
        w = _vbetmnfvsyrppuql03kv3.dequant(self.qweight, self.wscale)
        return (w * self.inv_smooth[None, :] if self.inv_smooth is not None else w).to(self.out_dtype)

    def forward(self, x):
        if self.w16 is not None and PRECISION['mode'] == 'bf16':
            return nn.functional.linear(x, self.w16, None if self.bias is None else self.bias.to(self.w16.dtype))
        _vbetmnfvsyrppuql03kv3, aot = _mods()
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        if x2.shape[0] == 0:
            return x.new_zeros(*shape[:-1], self.out_features, dtype=self.out_dtype)
        if self.backend == 'aot':
            r = aot.act_quant(x2.contiguous(), self.inv_smooth)
            qx, sx = r if r is not None else _vbetmnfvsyrppuql03kv3.quant_act(x2, self.inv_smooth, 'int8')
            y = aot.mm_i8(qx, sx, self.qweight, self.wscale, self.bias, self.out_dtype)
            if y is not None:
                return y.reshape(*shape[:-1], self.out_features)
            y = _vbetmnfvsyrppuql03kv3.torch_int_mm(qx, self.qweight).float() * sx[:, None] * self.wscale[None, :]
        else:
            qx, sx = _vbetmnfvsyrppuql03kv3.quant_act(x2, self.inv_smooth, 'int8')
            y = _vbetmnfvsyrppuql03kv3.int_gemm_exact(qx, self.qweight).float() * sx[:, None] * self.wscale[None, :]
        if self.bias is not None:
            y = y + self.bias
        return y.to(self.out_dtype).reshape(*shape[:-1], self.out_features)

def convert(body, stats, backend='aot', alpha=ALPHA, keep_bf16=False):
    import m1rrs8kpf3hjn4ltqozb9 as _vbetmnfvsyrppuql03kv3
    replaced = smoothed = 0
    for qname, _vbax3oza79ngqrqvpavhw, attr, child in _vbetmnfvsyrppuql03kv3.target_linears(body, range(len(body.layers)), _vbetmnfvsyrppuql03kv3.GROUPS['all']):
        amax = stats.get(qname) if stats is not None else None
        smooth = None if amax is None else _vbetmnfvsyrppuql03kv3.smooth_factors(amax, child.weight.detach(), alpha)
        setattr(_vbax3oza79ngqrqvpavhw, attr, QLinear(child, smooth, backend, keep_bf16=keep_bf16))
        replaced += 1
        smoothed += smooth is not None
        del child
    return (replaced, smoothed)

def decide(device, request='auto', source='convert'):
    if request == 'off':
        return (False, 'off')
    dev = torch.device(device)
    if dev.type != 'cuda' or not torch.cuda.is_available():
        return (False, 'not cuda')
    if dev.index is None:
        dev = torch.device('cuda', torch.cuda.current_device())
    need = MIN_VRAM_FILE if source == 'file' else MIN_VRAM
    if total_memory(dev) < need:
        return (False, 'memory below %.1f GiB' % (need / 2 ** 30))
    if torch.cuda.get_device_capability(dev) < (8, 0):
        return (False, 'compute capability below 8.0')
    import m4q417rrk96tf7req4il7 as aot
    ok, _ = aot.available(dev)
    if not ok:
        return (False, 'no kernels for this gpu')
    if source != 'file' and (not (aot.kernel_dir() / STATS).is_file()):
        return (False, 'statistics missing')
    return (True, 'sm_%d%d' % torch.cuda.get_device_capability(dev))

def free_ok(device, need=CONVERT_FREE):
    dev = torch.device(device)
    if dev.type != 'cuda':
        return True
    free, _ = torch.cuda.mem_get_info(dev)
    return free >= need

def apply(body, device, request='auto', backend='aot', stats=None, dual=False, selftest=None):
    on, why = decide(device, request) if backend == 'aot' else (True, backend)
    if on and backend == 'aot' and (not free_ok(device)):
        on, why = (False, 'free memory below %d MiB at conversion' % (CONVERT_FREE >> 20))
    report = {'precision': 'int8' if on else 'bf16', 'request': request, 'reason': why, 'source': 'convert'}
    if not on:
        selftest_cancel(selftest)
        return report
    on, why = _verified(report, selftest if selftest is not None else selftest_start(body_shapes(body), device, backend), device, backend)
    if not on:
        report.update(precision='bf16', reason=why)
        return report
    if dual and torch.device(device).type == 'cuda':
        if context_cap(total_memory(device), 'int8', dual=True) < MIN_CONTEXT:
            dual = False
            report['dual_refused'] = 'memory'
    report['dual'] = bool(dual)
    import m4q417rrk96tf7req4il7 as aot
    if stats is None and backend == 'aot':
        stats = load_stats(aot.kernel_dir() / STATS)
    report['linears'], report['smoothed'] = convert(body, stats, backend, keep_bf16=dual)
    if backend == 'aot':
        torch.cuda.empty_cache()
    return report

def _verified(report, handle, device, backend):
    test = selftest_finish(handle)
    report['selftest'] = test
    if not test['ok']:
        return (False, 'self-test: ' + test['reason'])
    if backend == 'aot':
        import m4q417rrk96tf7req4il7 as aot
        try:
            report['kernels'] = aot.preload(torch.device(device))
        except Exception as exc:
            return (False, ('kernel load: %s: %s' % (type(exc).__name__, exc))[:300])
        aot.ATTN['enabled'] = bool(test.get('attn', {}).get('ok'))
    return (True, report['reason'])
SELFTEST_ROWS = (3, 48, 200)
SELFTEST_TOL = 1.05
_v6n9tatg1ke1u6s9gtljd = 20261022
SELFTEST_TIMEOUT = 300
_CHILD = 'import json, sys\nsys.path[:0] = json.loads(sys.argv[1])\nimport importlib\nm = importlib.import_module(sys.argv[2])\nr = m.selftest([tuple(s) for s in json.loads(sys.argv[3])], sys.argv[4], sys.argv[5])\nsys.stdout.write("\\n" + json.dumps(r) + "\\n")\n'

def body_shapes(body):
    import m1rrs8kpf3hjn4ltqozb9 as _vbetmnfvsyrppuql03kv3
    return sorted({(c.out_features, c.in_features, c.bias is not None) for _, _, _, c in _vbetmnfvsyrppuql03kv3.target_linears(body, range(len(body.layers)), _vbetmnfvsyrppuql03kv3.GROUPS['all'])})

def _header(path):
    with open(path, 'rb') as f:
        n = int.from_bytes(f.read(8), 'little')
        return json.loads(f.read(n).decode('utf-8'))

def linear_shapes(folder):
    import re
    import m1rrs8kpf3hjn4ltqozb9 as _vbetmnfvsyrppuql03kv3
    folder = Path(folder)
    leaves = set(_vbetmnfvsyrppuql03kv3.GROUPS['all'])
    layer = re.compile('(?:^|\\.)layers\\.\\d+\\.')
    if (folder / FILE).is_file():
        files, leaf = ([folder / FILE], 'qweight')
    else:
        index = folder / 'model.safetensors.index.json'
        names = sorted(set(json.loads(index.read_text(encoding='utf-8'))['weight_map'].values())) if index.is_file() else ['model.safetensors']
        files, leaf = ([folder / n for n in names if (folder / n).is_file()], 'weight')
    out = set()
    for path in files:
        head = _header(path)
        for key, info in head.items():
            parts = key.split('.')
            if key == '__metadata__' or parts[-1] != leaf or len(parts) < 2 or (parts[-2] not in leaves) or (not layer.search(key)) or key.startswith('mtp') or (len(info['shape']) != 2):
                continue
            n, k = info['shape']
            out.add((n, k, key[:-len(leaf)] + 'bias' in head))
    return sorted(out)

def _row_rel(a, b):
    return float(((a - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-30)).max())

def selftest(shapes, device, backend='aot', rows=SELFTEST_ROWS):
    t0 = time.perf_counter()
    out = {'ok': False, 'reason': '', 'checked': 0, 'worst_ratio': 0.0}
    try:
        _vbetmnfvsyrppuql03kv3, aot = _mods()
        dev = torch.device(device)
        launches, misses = (aot.STATS['launch'], sum(aot.STATS['fallback'].values()))
        g = getattr(torch.Generator(), ''.join(map(chr, (109, 97, 110, 117, 97, 108, 95, 115, 101, 101, 100))))(_v6n9tatg1ke1u6s9gtljd)
        per_shape = {}
        for n, k, has_bias in shapes:
            n, k = (int(n), int(k))
            w = (torch.randn(n, k, generator=g) * 0.05).to(torch.bfloat16)
            amax = torch.rand(k, generator=g) * 4 + 0.05
            amax[::61] *= 25.0
            b = (torch.randn(n, generator=g) * 0.1).to(torch.bfloat16).to(dev) if has_bias else None
            lin = types.SimpleNamespace(weight=w.to(dev), bias=b)
            for smooth in (_vbetmnfvsyrppuql03kv3.smooth_factors(amax, w, ALPHA), None):
                q = QLinear(lin, smooth, backend)
                wd = _vbetmnfvsyrppuql03kv3.dequant(q.qweight, q.wscale)
                if q.inv_smooth is not None:
                    wd = wd * q.inv_smooth[None, :]
                for m in rows:
                    x = (torch.randn(m, k, generator=g) * amax / 3).to(torch.bfloat16).to(dev)
                    l0, _v5wz3n1fab22ol5uhwgzb = (aot.STATS['launch'], dict(aot.STATS['fallback']))
                    y = q(x).float()
                    launched = aot.STATS['launch'] - l0
                    key = 'N%d K%d' % (n, k)
                    per_shape[key] = per_shape.get(key, 0) + launched
                    if backend == 'aot' and launched < 2:
                        miss = {r: c - _v5wz3n1fab22ol5uhwgzb.get(r, 0) for r, c in aot.STATS['fallback'].items() if c != _v5wz3n1fab22ol5uhwgzb.get(r, 0)}
                        out['reason'] = 'N %d K %d M %d: %d cubin launches (expected 2), misses %s' % (n, k, m, launched, json.dumps(miss, sort_keys=True))
                        out['per_shape'] = per_shape
                        return out
                    ref = x.float() @ wd.t()
                    qx, sx = _vbetmnfvsyrppuql03kv3.quant_act(x, q.inv_smooth, 'int8')
                    emu = _vbetmnfvsyrppuql03kv3.int_gemm_exact(qx, q.qweight).float() * sx[:, None] * q.wscale[None, :]
                    if q.bias is not None:
                        ref, emu = (ref + q.bias, emu + q.bias)
                    emu = emu.to(q.out_dtype).float()
                    noise, err = (_row_rel(emu, ref), _row_rel(y, ref))
                    if not bool(torch.isfinite(y).all()):
                        out['reason'] = 'non-finite output (N %d, K %d, M %d)' % (n, k, m)
                        return out
                    if not err <= SELFTEST_TOL * noise:
                        out['reason'] = 'N %d K %d M %d: error %.4g > %.2f x INT8 noise %.4g' % (n, k, m, err, SELFTEST_TOL, noise)
                        return out
                    out['worst_ratio'] = max(out['worst_ratio'], err / max(noise, 1e-30))
                    out['checked'] += 1
        if dev.type == 'cuda':
            torch.cuda.synchronize(dev)
        out.update(launches=aot.STATS['launch'] - launches, misses=sum(aot.STATS['fallback'].values()) - misses, per_shape=per_shape)
        if backend == 'aot' and (not out['launches'] or not all(per_shape.values())):
            out['reason'] = 'no cubin launched'
            return out
        out.update(ok=True, reason='ok')
        out['worst_ratio'] = round(out['worst_ratio'], 5)
        if backend == 'aot' and dev.type == 'cuda':
            out['attn'] = attn_selftest(aot, dev, g)
    except Exception as exc:
        out['reason'] = ('%s: %s' % (type(exc).__name__, exc))[:300]
    finally:
        out['seconds'] = round(time.perf_counter() - t0, 2)
    return out
ATTN_CASES = ((1, 1, 37), (2, 13, 13), (1, 48, 200), (1, 130, 130), (2, 70, 333))
ATTN_TOL = 0.02

def attn_selftest(aot, dev, g):
    res = {'ok': False, 'checked': 0, 'worst': 0.0}
    try:
        keys = aot.attn_keys(dev)
        if not keys:
            res['reason'] = 'no attention cubin'
            return res
        shapes = sorted({tuple((float(p[1:]) if p[0] == 's' else int(p.lstrip('hdqr')) for p in key.split('|')[2:6])) for key in keys})
        for d, hq, rep, scale in shapes:
            hk = hq // rep
            for b, n, s in ATTN_CASES:
                q = torch.randn(b, n, hq, d, generator=g).to(torch.bfloat16).to(dev).transpose(1, 2)
                k = torch.randn(b, hk, s, d, generator=g).to(torch.bfloat16).to(dev)
                v = torch.randn(b, hk, s, d, generator=g).to(torch.bfloat16).to(dev)
                l0 = aot.STATS['launch']
                o = aot.attn_fwd(q, k, v, scale)
                if o is None or aot.STATS['launch'] == l0:
                    res['reason'] = 'B %d n %d S %d: no attention launch' % (b, n, s)
                    return res
                kf, vf = (k.float().repeat_interleave(rep, 1), v.float().repeat_interleave(rep, 1))
                sc = q.float() @ kf.transpose(-1, -2) * scale
                seen = torch.arange(s, device=dev)[None, :] <= (s - n + torch.arange(n, device=dev))[:, None]
                ref = torch.softmax(sc.masked_fill(~seen, float('-inf')), -1) @ vf
                err = _row_rel(o.float().reshape(-1, d), ref.reshape(-1, d))
                if not err <= ATTN_TOL:
                    res['reason'] = 'B %d n %d S %d: row error %.4g > %.2f' % (b, n, s, err, ATTN_TOL)
                    return res
                res['worst'] = max(res['worst'], err)
                res['checked'] += 1
        torch.cuda.synchronize(dev)
        res.update(ok=True, reason='ok', worst=round(res['worst'], 5))
    except Exception as exc:
        res['reason'] = ('%s: %s' % (type(exc).__name__, exc))[:300]
    return res

def selftest_start(shapes, device, backend='aot'):
    shapes = [list(map(int, s)) for s in shapes]
    dev = torch.device(device)
    if backend == 'aot' and dev.type == 'cuda':
        handle = _spawn(shapes, str(_cuda(dev)), backend)
        if handle is not None:
            return handle
    return {'shapes': shapes, 'device': str(dev), 'backend': backend}

def _spawn(shapes, device, backend, module=None):
    import subprocess
    search = list(getattr(sys, 'path'))
    try:
        proc = subprocess.Popen([sys.executable, '-B', '-c', _CHILD, json.dumps(search), module or __name__, json.dumps(shapes), device, backend], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except OSError:
        return None
    return {'proc': proc, 't': time.perf_counter()}
SELFTEST_CACHE = 'int8_selftest.json'

def _driver_version():
    import ctypes
    for name in ('nvcuda.dll', 'libcuda.so.1', 'libcuda.so'):
        try:
            lib = ctypes.WinDLL(name) if name.endswith('.dll') else ctypes.CDLL(name)
        except (OSError, AttributeError):
            continue
        v = ctypes.c_int(0)
        if lib.cuDriverGetVersion(ctypes.byref(v)) == 0:
            return int(v.value)
    return None

def selftest_identity(shapes, device, backend='aot'):
    dev = torch.device(device)
    if backend != 'aot' or dev.type != 'cuda' or (not torch.cuda.is_available()):
        return None
    drv = _driver_version()
    if drv is None:
        return None
    import m4q417rrk96tf7req4il7 as aot
    props = torch.cuda.get_device_properties(_cuda(dev))
    cc = props.major * 10 + props.minor
    h, found = (hashlib.sha256(), False)
    for p in (aot.kernel_dir() / aot.PACKED.format(cc), aot.kernel_dir() / f'w8a8_aot_sm{cc}.json'):
        if p.is_file():
            h.update(p.read_bytes())
            found = True
    if not found:
        return None
    return {'gpu': props.name, 'cc': cc, 'driver': drv, 'torch': str(torch.__version__), 'cuda': torch.version.cuda, 'kernels': h.hexdigest(), 'shapes': sorted((list(map(int, s)) for s in shapes)), 'backend': backend}

def selftest_cached(directory, shapes, device, backend='aot'):
    if not directory:
        return None
    ident = selftest_identity(shapes, device, backend)
    if ident is None:
        return None
    try:
        saved = json.loads((Path(directory) / SELFTEST_CACHE).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    result = saved.get('result') if isinstance(saved, dict) else None
    if saved.get('identity') != ident or not isinstance(result, dict) or (not result.get('ok')):
        return None
    return {'cached': dict(result, cached=True)}

def selftest_remember(directory, shapes, device, result, backend='aot'):
    if not directory or not isinstance(result, dict) or (not result.get('ok')) or result.get('cached'):
        return False
    ident = selftest_identity(shapes, device, backend)
    if ident is None:
        return False
    path = Path(directory) / SELFTEST_CACHE
    path._vbax3oza79ngqrqvpavhw.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps({'identity': ident, 'result': result}, sort_keys=True), encoding='utf-8')
    tmp.replace(path)
    return True

def selftest_finish(handle, timeout=SELFTEST_TIMEOUT):
    if handle is None:
        return {'ok': False, 'reason': 'not run'}
    if 'cached' in handle:
        return handle['cached']
    proc = handle.get('proc')
    if proc is None:
        return selftest(handle['shapes'], handle['device'], handle['backend'])
    import subprocess
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        return {'ok': False, 'reason': 'self-test timed out after %d s' % timeout}
    lines = [s for s in (stdout or '').splitlines() if s.startswith('{')]
    if proc.returncode != 0 or not lines:
        tail = [s for s in (stderr or '').splitlines() if s.strip()]
        return {'ok': False, 'reason': ('self-test process exit %s: %s' % (proc.returncode, tail[-1] if tail else ''))[:300]}
    res = json.loads(lines[-1])
    res['wall_seconds'] = round(time.perf_counter() - handle['t'], 2)
    return res

def selftest_cancel(handle):
    proc = (handle or {}).get('proc')
    if proc is not None and proc.poll() is None:
        proc.kill()
        proc.communicate()
FILE = 'model-int8.safetensors'
FILE_FORMAT = 'int8-w8a8/1'

class _MetaParams:

    def __init__(self, dtype):
        self.dtype = dtype

    def __enter__(self):
        self._orig, self._dtype = (nn.Module.register_parameter, torch.get_default_dtype())
        orig = self._orig

        def register(module, name, param):
            orig(module, name, param)
            if param is not None:
                p = module._parameters[name]
                module._parameters[name] = nn.Parameter(p.to('meta'), requires_grad=p.requires_grad)
        nn.Module.register_parameter = register
        torch.set_default_dtype(self.dtype)
        return self

    def __exit__(self, *exc):
        nn.Module.register_parameter = self._orig
        torch.set_default_dtype(self._dtype)
        return False

def load_model(folder, device, backend='aot'):
    from safetensors import safe_open
    from transformers import Qwen3_5ForCausalLM
    from transformers import Qwen3_5TextConfig
    import m1rrs8kpf3hjn4ltqozb9 as _vbetmnfvsyrppuql03kv3
    folder, dev = (Path(folder), torch.device(device))
    config = Qwen3_5TextConfig.from_pretrained(folder, local_files_only=True)
    with _MetaParams(torch.bfloat16):
        model = Qwen3_5ForCausalLM._from_config(config, dtype=torch.bfloat16)
    with safe_open(str(folder / FILE), framework='pt', device=str(_cuda(dev)) if dev.type == 'cuda' else 'cpu') as f:
        meta = f.metadata() or {}
        if meta.get('format') != FILE_FORMAT:
            raise ValueError('invalid_model')
        keys = set(f.keys())
        names = {id(m): n for n, m in model.named_modules()}
        for _, _vbax3oza79ngqrqvpavhw, attr, child in _vbetmnfvsyrppuql03kv3.target_linears(model.model, range(len(model.model.layers)), _vbetmnfvsyrppuql03kv3.GROUPS['all']):
            pre = names[id(_vbax3oza79ngqrqvpavhw)] + '.' + attr
            setattr(_vbax3oza79ngqrqvpavhw, attr, QLinear.empty(child.in_features, child.out_features, torch.bfloat16, backend, smooth=pre + '.inv_smooth' in keys, bias=pre + '.bias' in keys))
        ties = json.loads(meta.get('ties') or '{}')
        want = model.state_dict(keep_vars=True)
        if set(want) != keys | set(ties) or any((t not in keys for t in ties.values())):
            raise ValueError('invalid_model')
        mods = dict(model.named_modules())
        for key in sorted(keys):
            t = f.get_tensor(key)
            ref = want[key]
            if t.dtype != ref.dtype or t.shape != ref.shape:
                raise ValueError('invalid_model')
            path, _, leaf = key.rpartition('.')
            m = mods[path]
            if leaf in m._parameters:
                m._parameters[leaf] = nn.Parameter(t, requires_grad=False)
            else:
                m._buffers[leaf] = t
        for alias, source in ties.items():
            path, _, leaf = alias.rpartition('.')
            spath, _, sleaf = source.rpartition('.')
            src = mods[spath]
            value = src._parameters[sleaf] if sleaf in src._parameters else src._buffers[sleaf]
            m = mods[path]
            if leaf in m._parameters:
                m._parameters[leaf] = value
            else:
                m._buffers[leaf] = value
    model = model.to(dev).eval().requires_grad_(False)
    if any((t.is_meta for t in list(model.parameters()) + list(model.buffers()))):
        raise ValueError('invalid_model')
    return model

def dequantize(body):
    import m1rrs8kpf3hjn4ltqozb9 as _vbetmnfvsyrppuql03kv3
    n = 0
    for module in list(body.modules()):
        for attr, child in list(module.named_children()):
            if isinstance(child, QLinear):
                w = _vbetmnfvsyrppuql03kv3.dequant(child.qweight, child.wscale)
                if child.inv_smooth is not None:
                    w = w * child.inv_smooth[None, :]
                lin = nn.Linear(child.in_features, child.out_features, bias=child.bias is not None, device='meta')
                lin.weight = nn.Parameter(w.to(child.out_dtype), requires_grad=False)
                if child.bias is not None:
                    lin.bias = nn.Parameter(child.bias.to(child.out_dtype), requires_grad=False)
                setattr(module, attr, lin)
                n += 1
    return n

def load_file(folder, device, request='auto', backend='aot', selftest=None, bf16_ok=False):
    on, why = decide(device, request, source='file') if backend == 'aot' else (True, backend)
    report = {'precision': 'int8', 'request': request, 'reason': why, 'source': 'file'}
    if on:
        on, why = _verified(report, selftest if selftest is not None else selftest_start(linear_shapes(folder), device, backend), device, backend)
    else:
        selftest_cancel(selftest)
    if on:
        model = load_model(folder, device, backend)
        report['linears'] = sum((isinstance(m, QLinear) for m in model.model.modules()))
        report['smoothed'] = sum((isinstance(m, QLinear) and m.inv_smooth is not None for m in model.model.modules()))
        return (model, report)
    report.update(precision='bf16', reason=why)
    if bf16_ok:
        return (None, report)
    dev = torch.device(device)
    if dev.type == 'cuda' and context_cap(total_memory(dev), 'bf16') == 0:
        raise ValueError('this GPU has too little memory for the HF backend (use the GGUF backend)')
    model = load_model(folder, device, 'emulate')
    report['source'] = 'file (dequantized)'
    report['linears'] = dequantize(model.model)
    if dev.type == 'cuda':
        torch.cuda.empty_cache()
    return (model, report)