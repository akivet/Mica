import importlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys


def _primitive(value):
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if type(value) is list:
        return [_primitive(item) for item in value]
    if type(value) is dict and all((type(key) is str for key in value)):
        return {key: _primitive(item) for key, item in value.items()}
    raise ValueError('kernel state contains a nonserializable value')


def _config(config):
    return _primitive(dict(vars(config)))


def _key(key):
    if type(key) is not tuple or any((type(item) not in (bool, int, float, str, type(None)) for item in key)):
        raise ValueError('kernel autotune key is not a primitive tuple')
    return [_primitive(item) for item in key]


def _signature(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _runtime_identity(device):
    import torch

    device = torch.device(device)
    if device.type != 'cuda':
        raise ValueError('saved FLA kernel state requires the same CUDA device')
    gpu = torch.cuda.get_device_properties(device)
    return dict(
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        triton=importlib.metadata.version('triton'),
        fla=importlib.metadata.version('flash-linear-attention'),
        gpu=dict(name=gpu.name, capability=[gpu.major, gpu.minor], multiprocessors=gpu.multi_processor_count),
    )


def _source_file(module_name):
    module = sys.modules[module_name]
    path = Path(module.__file__)
    if path.suffix != '.py':
        raise ValueError('FLA kernel module source is unavailable: ' + module_name)
    if not path.is_file():
        raise FileNotFoundError('missing kernel source: ' + str(path))
    return path.name


def _tuners():
    module = sys.modules.get('triton.runtime.autotuner')
    autotuner = getattr(module, 'Autotuner', None)
    found, seen = ({}, set())
    if autotuner is None:
        return found
    for module_name, module in list(sys.modules.items()):
        if not module_name.startswith('fla.') or module is None:
            continue
        for value in list(vars(module).values()):
            chain = set()
            while value is not None and id(value) not in chain:
                chain.add(id(value))
                if isinstance(value, autotuner):
                    if id(value) in seen:
                        break
                    seen.add(id(value))
                    function = value.fn
                    unwrapped = set()
                    while hasattr(function, 'fn') and id(function) not in unwrapped:
                        unwrapped.add(id(function))
                        function = function.fn
                    owner, name = (function.__module__, function.__name__)
                    if not owner.startswith('fla.'):
                        raise ValueError('FLA autotuner has an unknown kernel owner')
                    identifier = owner + ':' + name
                    if identifier in found:
                        raise ValueError('multiple FLA autotuners share kernel identity: ' + identifier)
                    found[identifier] = value
                    break
                value = getattr(value, 'fn', None)
    return found


def _binding(identifier, tuner):
    return dict(
        source_file=_source_file(identifier.split(':', 1)[0]),
        tuner=type(tuner).__module__ + ':' + type(tuner).__qualname__,
        arg_names=list(tuner.arg_names),
        keys=list(tuner.keys),
    )


def _refuse_always():
    module = sys.modules.get('fla.ops.utils.cache')
    mode = getattr(module, 'FLA_CACHE_MODE', None)
    if getattr(mode, 'value', mode) == 'always':
        raise ValueError('FLA cache ALWAYS mode can override saved kernel choices')


def capture(device):
    import torch

    if torch.device(device).type != 'cuda':
        return None
    tuners = {
        name: tuner
        for name, tuner in _tuners().items()
        if any((key not in getattr(tuner, '_borrowed', ()) for key in tuner.cache))
    }
    if not tuners:
        return None
    _refuse_always()
    kernels = {}
    for identifier, tuner in sorted(tuners.items()):
        entries = [
            dict(key=_key(key), config=_config(config))
            for key, config in tuner.cache.items()
            if key not in getattr(tuner, '_borrowed', ())
        ]
        entries.sort(key=lambda entry: _signature(entry['key']))
        kernels[identifier] = dict(binding=_binding(identifier, tuner), entries=entries)
    return dict(version=1, runtime=_runtime_identity(device), kernels=kernels)


def restore(state, device):
    if state is None or state == {}:
        return
    if (
        type(state) is not dict
        or set(state) != {'version', 'runtime', 'kernels'}
        or type(state['version']) is not int
        or (state['version'] != 1)
    ):
        raise ValueError('invalid kernel state format')
    _primitive(state)
    _refuse_always()
    if state['runtime'] != _runtime_identity(device):
        raise ValueError('kernel runtime identity differs')
    if type(state['kernels']) is not dict:
        raise ValueError('invalid kernel state kernels')
    for identifier in state['kernels']:
        if not identifier.startswith('fla.') or identifier.count(':') != 1:
            raise ValueError('invalid FLA kernel identity')
        importlib.import_module(identifier.split(':', 1)[0])
    _refuse_always()
    tuners = _tuners()
    from triton import Config

    pending = []
    for identifier, saved in state['kernels'].items():
        tuner = tuners.get(identifier)
        if type(saved) is not dict:
            raise ValueError('invalid saved kernel binding')
        if tuner is None or saved.get('binding') != _binding(identifier, tuner):
            raise ValueError('kernel source or autotuner binding differs: ' + identifier)
        if set(saved) != {'binding', 'entries'} or type(saved['entries']) is not list:
            raise ValueError('invalid saved kernel entries')
        restored = {}
        for entry in saved['entries']:
            if type(entry) is not dict or set(entry) != {'key', 'config'} or type(entry['key']) is not list:
                raise ValueError('invalid saved kernel entry')
            key = tuple(entry['key'])
            _key(key)
            if type(entry['config']) is not dict:
                raise ValueError('invalid saved kernel config')
            try:
                config = Config(**entry['config'])
            except (TypeError, ValueError) as error:
                raise ValueError('kernel config cannot be restored exactly') from error
            if _config(config) != entry['config']:
                raise ValueError('kernel config cannot be restored exactly')
            if key in restored:
                raise ValueError('duplicate saved kernel autotune key')
            restored[key] = config
        pending.append((tuner, restored))
    for tuner, restored in pending:
        tuner.cache.update(restored)


def _rank(state, device):
    import torch

    gpu = torch.cuda.get_device_properties(torch.device(device))
    saved = ((state or {}).get('runtime') or {}).get('gpu') or {}
    if saved.get('name') == gpu.name and saved.get('multiprocessors') == gpu.multi_processor_count:
        return 0
    return 1 if saved.get('capability') == [gpu.major, gpu.minor] else 2


def _run_key(tuner, args, kwargs):
    all_args = {**dict(zip(tuner.arg_names, args)), **kwargs}
    named = {k: v for k, v in all_args.items() if k in tuner.arg_names}
    key = [named[k] for k in tuner.keys if k in named]
    key += [str(v.dtype) for v in named.values() if hasattr(v, 'dtype')]
    return tuple(key)


def pin(states, device):
    import collections
    import torch

    if torch.device(device).type != 'cuda' or not states:
        return {}
    from triton import Config

    table, default = ({}, {})
    for state in sorted((s for s in states if isinstance(s, dict)), key=lambda s: _rank(s, device)):
        for identifier, saved in (state.get('kernels') or {}).items():
            try:
                importlib.import_module(identifier.split(':', 1)[0])
                configs = [(tuple(e['key']), Config(**e['config'])) for e in saved['entries']]
            except Exception:
                continue
            keyed, rank = (table.setdefault(identifier, {}), _rank(state, device))
            for key, config in configs:
                keyed.setdefault(key, (config, rank))
            if identifier not in default and configs:
                count = collections.Counter((_signature(_config(c)) for _, c in configs))
                best = count.most_common(1)[0][0]
                default[identifier] = next((c for _, c in configs if _signature(_config(c)) == best))
    pinned = {}
    for identifier, tuner in _tuners().items():
        if identifier not in default or getattr(tuner, '_pinned', False):
            continue
        keyed, fallback, original = (table[identifier], default[identifier], tuner.run)

        def run(*args, _tuner=tuner, _keyed=keyed, _fallback=fallback, _original=original, **kwargs):
            if _tuner._pinned and len(_tuner.configs) > 1:
                key = _run_key(_tuner, args, kwargs)
                if key not in _tuner.cache:
                    config, rank = _keyed.get(key, (_fallback, None))
                    _tuner.cache[key] = config
                    if rank != 0:
                        _tuner._borrowed.add(key)
                    try:
                        return _original(*args, **kwargs)
                    except Exception:
                        _tuner.cache.pop(key, None)
                        _tuner._borrowed.discard(key)
                        _tuner._pinned = False
            return _original(*args, **kwargs)

        tuner.run, tuner._pinned, tuner._borrowed = (run, True, set())
        pinned[identifier.split(':', 1)[1]] = len(keyed)
    return pinned


def unpin_next():
    for identifier, tuner in sorted(_tuners().items()):
        if getattr(tuner, '_pinned', False) and tuner._borrowed and (not getattr(tuner, '_tried', False)):
            for key in tuner._borrowed:
                tuner.cache.pop(key, None)
            tuner._borrowed.clear()
            tuner._pinned, tuner._tried = (False, True)
            return identifier
    return None


def repin(identifier):
    tuner = _tuners().get(identifier)
    if tuner is not None and hasattr(tuner, '_borrowed'):
        tuner._pinned = True
