from __future__ import annotations
import ctypes
import json
import math
import os
import sys
import threading
import ctypes as _e1
_vepr5bfopjgvrwgcro5zb = getattr(_e1, ''.join(map(chr, (80, 79, 73, 78, 84, 69, 82))))
from ctypes import c_bool
from ctypes import c_char_p
from ctypes import c_float
from ctypes import c_int8
from ctypes import c_int32
from ctypes import c_size_t
from ctypes import c_uint32
from ctypes import c_void_p
from pathlib import Path
from mdedz7rql7m75mxyyy0ec import NOUL_LABELS, OPTION_LABELS
from m6lir2n4n221f2rp5z93i import prompt_for
from m2nya38eng8h4tj83b4ho import load_calibration, distribution
import m81mym3jhph2kl5gy7i8b as fast_path
from m81mym3jhph2kl5gy7i8b import stage as _stage
STATE_END = chr(10) + '</state>' + chr(10)
llama_token = c_int32
llama_pos = c_int32
llama_seq_id = c_int32

class llama_model_params(ctypes.Structure):
    _fields_ = [('devices', c_void_p), ('tensor_buft_overrides', c_void_p), ('n_gpu_layers', c_int32), ('split_mode', c_int32), ('load_mode', c_int32), ('lazy_mode', c_int32), ('main_gpu', c_int32), ('tensor_split', _vepr5bfopjgvrwgcro5zb(c_float)), ('_vdug3qs8zq3689a2be0s9_callback', c_void_p), ('_vdug3qs8zq3689a2be0s9_callback_user_data', c_void_p), ('kv_overrides', c_void_p), ('vocab_only', c_bool), ('check_tensors', c_bool), ('use_extra_bufts', c_bool), ('no_host', c_bool), ('no_alloc', c_bool), ('load_mtp', c_bool)]

class llama_context_params(ctypes.Structure):
    _fields_ = [('n_ctx', c_uint32), ('n_batch', c_uint32), ('n_ubatch', c_uint32), ('n_seq_max', c_uint32), ('n_rs_seq', c_uint32), ('n_outputs_max', c_uint32), ('n_outputs_max_per_seq', c_uint32), ('n_threads', c_int32), ('n_threads_batch', c_int32), ('ctx_type', c_int32), ('rope_scaling_type', c_int32), ('pooling_type', c_int32), ('attention_type', c_int32), ('flash_attn_type', c_int32), ('rope_freq_base', c_float), ('rope_freq_scale', c_float), ('yarn_ext_factor', c_float), ('yarn_attn_factor', c_float), ('yarn_beta_fast', c_float), ('yarn_beta_slow', c_float), ('yarn_orig_ctx', c_uint32), ('defrag_thold', c_float), ('cb_eval', c_void_p), ('cb_eval_user_data', c_void_p), ('type_k', c_int32), ('type_v', c_int32), ('abort_callback', c_void_p), ('abort_callback_data', c_void_p), ('embeddings', c_bool), ('offload_kqv', c_bool), ('no_perf', c_bool), ('op_offload', c_bool), ('swa_full', c_bool), ('kv_unified', c_bool), ('samplers', c_void_p), ('n_samplers', c_size_t), ('ctx_other', c_void_p)]
ABI_MARKER = 'llama_abi.json'
LONG_WINDOW_MARKER = 'llama_lc.json'
llama_context_params_moe = type('llama_context_params_moe', (ctypes.Structure,), {'_fields_': [f for name in [n for n, _ in llama_context_params._fields_] for f in [(name, dict(llama_context_params._fields_)[name])] + ([('moe_cache_size', c_size_t)] if name == 'type_v' else [])]})

class llama_batch(ctypes.Structure):
    _fields_ = [('n_tokens', c_int32), ('token', _vepr5bfopjgvrwgcro5zb(llama_token)), ('embd', _vepr5bfopjgvrwgcro5zb(c_float)), ('pos', _vepr5bfopjgvrwgcro5zb(llama_pos)), ('n_seq_id', _vepr5bfopjgvrwgcro5zb(c_int32)), ('seq_id', _vepr5bfopjgvrwgcro5zb(_vepr5bfopjgvrwgcro5zb(llama_seq_id))), ('logits', _vepr5bfopjgvrwgcro5zb(c_int8))]

def _library_name():
    if sys.platform == 'win32':
        return 'llama.dll'
    return 'libllama.dylib' if sys.platform == 'darwin' else 'libllama.so'

def _loaded_ggml_base():
    try:
        if sys.platform == 'win32':
            kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
            kernel32.GetModuleHandleW.restype = c_void_p
            kernel32.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
            handle = kernel32.GetModuleHandleW('ggml-base.dll')
            if not handle:
                return None
            buffer = ctypes.create_unicode_buffer(32768)
            kernel32.GetModuleFileNameW.restype = c_uint32
            kernel32.GetModuleFileNameW.argtypes = [c_void_p, ctypes.c_wchar_p, c_uint32]
            n = kernel32.GetModuleFileNameW(c_void_p(handle), buffer, len(buffer))
            return Path(buffer.value[:n]) if n else None
        if sys.platform.startswith('linux'):
            with open('/proc/self/maps', encoding='utf-8', errors='replace') as maps:
                for line in maps:
                    fields = line.split(maxsplit=5)
                    if len(fields) == 6 and Path(fields[5].strip()).name.startswith('libggml-base.so'):
                        return Path(fields[5].strip())
    except (OSError, ValueError, AttributeError):
        return None
    return None

def _check_single_runtime(runtime):
    loaded = _loaded_ggml_base()
    if loaded is None:
        return
    try:
        same = loaded.resolve()._vbax3oza79ngqrqvpavhw == runtime.resolve()
        if not same:
            mine = runtime / loaded.name
            same = mine.exists() and mine.stat().st_size == loaded.stat().st_size and (mine.read_bytes() == loaded.read_bytes())
    except OSError:
        same = False
    if not same:
        raise RuntimeError(f'another llama.cpp runtime is already loaded in this process ({loaded._vbax3oza79ngqrqvpavhw}); ggml shared libraries of two builds cannot be mixed in one process, so load {runtime} in a separate process (one runtime per process)')
_PROCESS_ENV = {}

def _set_process_env(name, value):
    value = str(value)
    held = _PROCESS_ENV.get(name, os.environ.get(name))
    if held is not None and held != value:
        raise ValueError(f'{name}={held} is already fixed for this process; {name}={value} needs a new process')
    os.environ[name] = value
    _PROCESS_ENV[name] = value

def load_library(runtime_dir):
    runtime = Path(runtime_dir)
    path = runtime / _library_name()
    if not path.exists():
        raise FileNotFoundError(f'{path} not found; point runtime_dir at the llama.cpp build')
    _check_single_runtime(runtime)
    if sys.platform == 'win32':
        os.add_dll_directory(str(runtime.resolve()))
    ggml_name = 'ggml.dll' if sys.platform == 'win32' else 'libggml.dylib' if sys.platform == 'darwin' else 'libggml.so'
    ggml = ctypes.CDLL(str(runtime / ggml_name))
    ggml.ggml_backend_load_all_from_path.restype = None
    ggml.ggml_backend_load_all_from_path.argtypes = [c_char_p]
    ggml.ggml_backend_load_all_from_path(str(runtime.resolve()).encode('utf-8'))
    ggml.ggml_backend_reg_count.restype = c_size_t
    ggml.ggml_backend_reg_count.argtypes = []
    if ggml.ggml_backend_reg_count() == 0:
        raise RuntimeError(f'no ggml backend registered from {runtime}')
    lib = ctypes.CDLL(str(path))
    lib.ggml = ggml
    lib.llama_backend_init.restype = None
    lib.llama_backend_init.argtypes = []
    lib.llama_model_default_params.restype = llama_model_params
    lib.llama_model_default_params.argtypes = []
    abi = runtime / ABI_MARKER
    context_params = llama_context_params_moe if abi.exists() and json.loads(abi.read_text(encoding='utf-8')).get('context_params') == 'moe_cache_size' else llama_context_params
    lib.llama_context_default_params.restype = context_params
    lib.llama_context_default_params.argtypes = []
    lib.llama_model_load_from_file.restype = c_void_p
    lib.llama_model_load_from_file.argtypes = [c_char_p, llama_model_params]
    lib.llama_model_free.restype = None
    lib.llama_model_free.argtypes = [c_void_p]
    lib.llama_init_from_model.restype = c_void_p
    lib.llama_init_from_model.argtypes = [c_void_p, context_params]
    for name in ('llama_n_batch', 'llama_n_ctx_seq', 'llama_n_seq_max'):
        function = getattr(lib, name)
        function.restype = c_uint32
        function.argtypes = [c_void_p]
    lib.llama_free.restype = None
    lib.llama_free.argtypes = [c_void_p]
    lib.llama_model_get_vocab.restype = c_void_p
    lib.llama_model_get_vocab.argtypes = [c_void_p]
    lib.llama_vocab_n_tokens.restype = c_int32
    lib.llama_vocab_n_tokens.argtypes = [c_void_p]
    lib.llama_tokenize.restype = c_int32
    lib.llama_tokenize.argtypes = [c_void_p, c_char_p, c_int32, _vepr5bfopjgvrwgcro5zb(llama_token), c_int32, c_bool, c_bool]
    lib.llama_batch_init.restype = llama_batch
    lib.llama_batch_init.argtypes = [c_int32, c_int32, c_int32]
    lib.llama_batch_free.restype = None
    lib.llama_batch_free.argtypes = [llama_batch]
    lib.llama_decode.restype = c_int32
    lib.llama_decode.argtypes = [c_void_p, llama_batch]
    lib.llama_get_logits_ith.restype = _vepr5bfopjgvrwgcro5zb(c_float)
    lib.llama_get_logits_ith.argtypes = [c_void_p, c_int32]
    if hasattr(lib, 'llama_get_embeddings_ith'):
        lib.llama_get_embeddings_ith.restype = _vepr5bfopjgvrwgcro5zb(c_float)
        lib.llama_get_embeddings_ith.argtypes = [c_void_p, c_int32]
    if hasattr(lib, 'llama_set_probe_rows'):
        lib.llama_set_probe_rows.restype = c_int32
        lib.llama_set_probe_rows.argtypes = [c_void_p, _vepr5bfopjgvrwgcro5zb(c_int32), c_int32]
        lib.llama_get_probe_logits_ith.restype = _vepr5bfopjgvrwgcro5zb(c_float)
        lib.llama_get_probe_logits_ith.argtypes = [c_void_p, c_int32]
        lib.llama_get_logits_host_bytes.restype = ctypes.c_uint64
        lib.llama_get_logits_host_bytes.argtypes = [c_void_p]
    if hasattr(lib, 'llama_set_lc_request'):
        lib.llama_set_lc_request.restype = None
        lib.llama_set_lc_request.argtypes = [c_void_p, c_int32, c_int32]
        lib.llama_set_lc_params.restype = None
        lib.llama_set_lc_params.argtypes = [c_void_p, c_int32, c_int32, c_int32]
        lib.llama_get_lc_boundary.restype = c_int32
        lib.llama_get_lc_boundary.argtypes = [c_void_p]
    if hasattr(lib, 'llama_model_n_embd'):
        lib.llama_model_n_embd.restype = c_int32
        lib.llama_model_n_embd.argtypes = [c_void_p]
    lib.llama_get_memory.restype = c_void_p
    lib.llama_get_memory.argtypes = [c_void_p]
    lib.llama_memory_seq_rm.restype = c_bool
    lib.llama_memory_seq_rm.argtypes = [c_void_p, llama_seq_id, llama_pos, llama_pos]
    lib.llama_memory_seq_cp.restype = None
    lib.llama_memory_seq_cp.argtypes = [c_void_p, llama_seq_id, llama_seq_id, llama_pos, llama_pos]
    lib.llama_memory_clear.restype = None
    lib.llama_memory_clear.argtypes = [c_void_p, c_bool]
    if hasattr(lib, 'llama_memory_seq_pos_max'):
        lib.llama_memory_seq_pos_max.restype = llama_pos
        lib.llama_memory_seq_pos_max.argtypes = [c_void_p, llama_seq_id]
    lib.llama_state_seq_get_size.restype = c_size_t
    lib.llama_state_seq_get_size.argtypes = [c_void_p, llama_seq_id]
    for name in ('llama_state_seq_get_data', 'llama_state_seq_set_data'):
        function = getattr(lib, name)
        function.restype = c_size_t
        function.argtypes = [c_void_p, _vepr5bfopjgvrwgcro5zb(ctypes.c_uint8), c_size_t, llama_seq_id]
    return lib

def output_limits(n_seq_max, reserve=0, hidden_outputs=0):
    per_seq = 1 + max(0, int(hidden_outputs or 0))
    return ((int(n_seq_max) + max(0, int(reserve))) * per_seq, per_seq)

class DirectJudge:
    _va1q6lmnxidh3npf70n43 = None
    common_only = False
    _v6okoqmlfn4wxs6wo5jz2 = 1.0
    _vdjyjd4b3w24pvj27zn5i = {}
    _v1rax47ccvh8o08t3gzzi = {}
    model_version = None
    last_experts = ()
    _vedlso49m4dr50us9e9ac = {}
    _vbfenikctyy2mceugfgie = None
    device_prefix_cache = False
    device_prefix_capture_min = 32
    _prefix_slots = None
    _token_cache = None
    _render_cache = None
    _template_cache = None
    _stage_name = 'prefill'
    device_prefix_stats = {}
    candidate_head = False
    _candidate_rows = ()
    _candidate_columns = {}
    _candidate_active = False
    cpu_output_head = False

    def __init__(self, model_path, tokenizer, *, runtime_dir, n_ctx=4096, n_seq_max=1, n_gpu_layers=-1, n_threads=None, temperature=1.0, max_length=2048, noul_logit_bias=0.0, flash_attn=-1, n_ubatch=None, hidden_outputs=0, device_prefix_cache=False, device_prefix_slots=1, token_cache_entries=256, token_cache_chars=4000000, candidate_head=False, cpu_output_head=False, long_window=None, kv_pad=None, cuda_graphs=None):
        if any((type(value) is not int or value < 1 for value in (n_ctx, n_seq_max, max_length))):
            raise ValueError('Context, sequence and input limits must be positive integers')
        if not (isinstance(temperature, (int, float)) and math.isfinite(temperature) and (temperature > 0)):
            raise ValueError('Calibration temperature must be a positive number')
        if not isinstance(noul_logit_bias, (int, float)) or not math.isfinite(noul_logit_bias):
            raise ValueError('Noul calibration bias must be finite')
        if kv_pad is not None:
            if type(kv_pad) is not int or not 256 <= kv_pad <= 1 << 20 or kv_pad % 256:
                raise ValueError('kv_pad must be a multiple of 256 in [256, 2**20]')
            _set_process_env('LLAMA_KV_PAD', kv_pad)
        if cuda_graphs is False:
            _set_process_env('GGML_CUDA_DISABLE_GRAPHS', 1)
        self.kv_pad, self.cuda_graphs = (kv_pad, cuda_graphs)
        self.lib = load_library(runtime_dir)
        self.lib.llama_backend_init()
        self.tokenizer, self.temperature, self.max_length = (tokenizer, float(temperature), max_length)
        self.noul_logit_bias = float(noul_logit_bias)
        model_params = self.lib.llama_model_default_params()
        model_params.n_gpu_layers = n_gpu_layers
        if cpu_output_head:
            from med9rt1x7qcs4m95rhbty import cpu_buffer_type, output_table_overrides
            self._buft_overrides = output_table_overrides(cpu_buffer_type(runtime_dir))
            model_params.tensor_buft_overrides = ctypes.cast(self._buft_overrides, c_void_p)
            model_params.use_extra_bufts = False
            self.cpu_output_head = True
        self.model = self.lib.llama_model_load_from_file(str(model_path).encode('utf-8'), model_params)
        if not self.model:
            raise RuntimeError(f'llama.cpp could not load {model_path}')
        context_params = self.lib.llama_context_default_params()
        reserve = max(0, int(device_prefix_slots)) if device_prefix_cache else 0
        context_params.n_ctx = n_ctx * (n_seq_max + reserve)
        context_params.n_batch = max(n_ctx, 2048)
        context_params.n_ubatch = n_ubatch or 512
        context_params.flash_attn_type = flash_attn
        context_params.n_seq_max = n_seq_max + reserve
        self.hidden_outputs = int(hidden_outputs or 0)
        context_params.n_outputs_max, context_params.n_outputs_max_per_seq = output_limits(n_seq_max, reserve, self.hidden_outputs)
        self.last_hidden = {}
        self.probe_ids = []
        self.use_probe_rows = True
        self._probe_rows_set = ()
        if n_threads:
            context_params.n_threads = context_params.n_threads_batch = n_threads
        self.n_seq_max, self.n_ctx_seq = (n_seq_max, n_ctx)
        self.ctx = self.lib.llama_init_from_model(self.model, context_params)
        if not self.ctx:
            raise RuntimeError('llama.cpp could not create a context')
        self.n_batch = int(self.lib.llama_n_batch(self.ctx))
        self.n_ctx_seq = int(self.lib.llama_n_ctx_seq(self.ctx))
        self.n_seq_max = min(n_seq_max, int(self.lib.llama_n_seq_max(self.ctx)), self.n_batch)
        if min(self.n_batch, self.n_ctx_seq, self.n_seq_max) < 1:
            self.close()
            raise RuntimeError('llama.cpp returned invalid context limits')
        self.memory = self.lib.llama_get_memory(self.ctx)
        self.vocab = self.lib.llama_model_get_vocab(self.model)
        self.n_vocab = self.lib.llama_vocab_n_tokens(self.vocab)
        self.n_embd = int(self.lib.llama_model_n_embd(self.model)) if hasattr(self.lib, 'llama_model_n_embd') else 0
        self._label_cache = {}
        self._inference_lock = threading.RLock()
        self._prefix_checkpoint = None
        self.prefix_cache_stats = {}
        self.model_version = str(model_path)
        self._token_cache = fast_path.TokenizationCache(self._tokenize_runtime, max_entries=token_cache_entries, max_chars=token_cache_chars) if token_cache_entries else None
        self._render_cache = fast_path.LRUCache(token_cache_entries, token_cache_chars) if token_cache_entries else None
        self.device_prefix_stats = {'hits': 0, 'builds': 0, 'tokens_saved': 0, 'misses': 0, 'captures': 0}
        if reserve:
            base = n_seq_max
            have = int(self.lib.llama_n_seq_max(self.ctx))
            usable = [base + k for k in range(reserve) if base + k < have]
            needs = ('llama_memory_seq_rm', 'llama_memory_seq_cp', 'llama_memory_seq_pos_max')
            if usable and all((hasattr(self.lib, name) for name in needs)):
                self._prefix_slots = fast_path.PrefixSlotCache(usable)
                self.device_prefix_cache = True
        if candidate_head:
            self._enable_candidate_head()
        if long_window is None:
            if self.long_window_supported(runtime_dir, self.lib):
                self._enable_long_window(runtime_dir)
        elif long_window:
            self._enable_long_window(runtime_dir)
    LONG_WINDOW_API = ('llama_set_lc_request', 'llama_set_lc_params', 'llama_get_lc_boundary')

    @classmethod
    def long_window_supported(cls, runtime_dir, lib=None):
        if not (Path(runtime_dir) / LONG_WINDOW_MARKER).exists():
            return False
        return lib is None or all((hasattr(lib, name) for name in cls.LONG_WINDOW_API))

    def _enable_long_window(self, runtime_dir):
        if not self.long_window_supported(runtime_dir, self.lib):
            self.close()
            raise RuntimeError(f'long_window needs the long-window runtime build ({LONG_WINDOW_MARKER} and its API in {runtime_dir})')
        marker = Path(runtime_dir) / LONG_WINDOW_MARKER
        spec = json.loads(marker.read_text(encoding='utf-8')).get('long_window', {})
        self.long_window = dict(cut=int(spec.get('cut', 16)), ext=int(spec.get('ext', 128)), min_prompt=int(spec.get('min_prompt', 1024)))
        lw = self.long_window
        if lw['cut'] < 1 or lw['ext'] < 0 or lw['min_prompt'] < 1:
            self.close()
            raise ValueError(f'invalid {LONG_WINDOW_MARKER}: {spec}')
        self.lib.llama_set_lc_params(self.ctx, lw['cut'], lw['ext'], lw['min_prompt'])
        self.long_window_stats = {'rows': 0, 'tokens': 0, 'boundary_missing': 0, 'shared_groups': 0, 'shared_rows': 0, 'state_tokens_saved': 0}

    def _question_start(self, row, prompt):
        text = self.render_prompt(row)
        at = text.find(STATE_END)
        if at < 0:
            return None
        n = fast_path.common_token_prefix([self.tokenize(text[:at + len(STATE_END)]), prompt])
        return n if 0 < n < len(prompt) else None

    def _long_rows(self, unique, prompts, rows):
        lw = getattr(self, 'long_window', None)
        if not lw:
            return {}
        out = {}
        for index in unique:
            if len(prompts[index]) <= lw['min_prompt']:
                continue
            start = self._question_start(rows[index], prompts[index])
            if start is None:
                self.long_window_stats['boundary_missing'] += 1
                continue
            if start - lw['ext'] > 0:
                out[index] = start
        return out

    def _judge_long(self, index, start, prompts, labels, rows, results):
        prompt = prompts[index]
        self.lib.llama_memory_seq_rm(self.memory, 0, -1, -1)
        self._v2ybxw3trfq52qoz7uvp0 = getattr(self, '_v2ybxw3trfq52qoz7uvp0', 0) + 1
        self.lib.llama_set_lc_request(self.ctx, start, len(prompt))
        try:
            if self.lib.llama_get_lc_boundary(self.ctx) <= 0:
                raise RuntimeError('the runtime did not arm the long-window prefill')
            self._candidate_active = bool(self.candidate_head and self._candidate_rows)
            outputs = self._decode([(0, prompt, 0)])
            results[index] = self._distribution(self._logits_at(outputs[0], labels[index]), rows[index]['kind'])
        finally:
            self._candidate_active = False
            self.lib.llama_set_lc_request(self.ctx, -1, 0)
            self.lib.llama_memory_seq_rm(self.memory, 0, -1, -1)
        self.long_window_stats['rows'] += 1
        self.long_window_stats['tokens'] += len(prompt)

    def _judge_long_shared(self, indices, start, prompts, labels, rows, results):
        prefix = prompts[indices[0]][:start]
        slots = self.n_seq_max - 1
        self._clear_question_slots()
        self._v2ybxw3trfq52qoz7uvp0 = getattr(self, '_v2ybxw3trfq52qoz7uvp0', 0) + 1
        self.lib.llama_set_lc_request(self.ctx, start, max((len(prompts[i]) for i in indices)))
        try:
            if self.lib.llama_get_lc_boundary(self.ctx) <= 0:
                raise RuntimeError('the runtime did not arm the long-window prefill')
            self._decode([(0, prefix, 0)], shared=True)
            for first in range(0, len(indices), slots):
                window = indices[first:first + slots]
                for slot in range(1, len(window) + 1):
                    self.lib.llama_memory_seq_rm(self.memory, slot, -1, -1)
                    self.lib.llama_memory_seq_cp(self.memory, 0, slot, -1, -1)
                self._candidate_active = bool(self.candidate_head and self._candidate_rows)
                try:
                    outputs = self._decode([(slot, prompts[index][start:], start) for slot, index in enumerate(window, 1)])
                    for slot, index in enumerate(window, 1):
                        results[index] = self._distribution(self._logits_at(outputs[slot], labels[index]), rows[index]['kind'])
                finally:
                    self._candidate_active = False
        finally:
            self.lib.llama_set_lc_request(self.ctx, -1, 0)
            self._clear_question_slots()
        stats = self.long_window_stats
        stats['rows'] += len(indices)
        stats['tokens'] += sum((len(prompts[i]) for i in indices))
        stats['shared_groups'] += 1
        stats['shared_rows'] += len(indices)
        stats['state_tokens_saved'] += start * (len(indices) - 1)

    def _judge_long_rows(self, long_rows, prompts, labels, rows, results, share_prefix):
        groups = {}
        if share_prefix and self.n_seq_max >= 2:
            for index, start in long_rows.items():
                groups.setdefault(tuple(prompts[index][:start]), []).append(index)
        done = set()
        for indices in groups.values():
            if len(indices) > 1:
                self._judge_long_shared(indices, long_rows[indices[0]], prompts, labels, rows, results)
                done.update(indices)
        for index, start in long_rows.items():
            if index not in done:
                self._judge_long(index, start, prompts, labels, rows, results)

    def _enable_candidate_head(self):
        if not hasattr(self.lib, 'llama_set_probe_rows'):
            self.close()
            raise RuntimeError('candidate_head needs a llama.cpp build with llama_set_probe_rows; this runtime only has the full-vocabulary output')
        from med9rt1x7qcs4m95rhbty import candidate_row_ids
        self._candidate_rows = candidate_row_ids(self.label_ids)
        self.candidate_head = True
        self._sync_probe_rows()

    @classmethod
    def from_calibration(cls, model_path, tokenizer, calibration_path, *, model_identity=None, **kwargs):
        fitted, temperatures, bias = load_calibration(calibration_path, model_identity=model_identity)
        judge = cls(model_path, tokenizer, temperature=float(fitted.get('temperature', 1.0)), noul_logit_bias=bias, **kwargs)
        judge.temperatures = temperatures
        judge.calibration = {'path': str(calibration_path), **fitted}
        if model_identity is not None:
            judge.model_version = str(model_identity)
        return judge

    def configure_experts(self, adapters=None, *, _va1q6lmnxidh3npf70n43=None, scale=1.0, versions=None, common_only=False):
        if isinstance(_va1q6lmnxidh3npf70n43, (str, os.PathLike)):
            raise ImportError('unavailable')
            _va1q6lmnxidh3npf70n43 = _gone.load(_va1q6lmnxidh3npf70n43)
        if _va1q6lmnxidh3npf70n43 is not None and getattr(_va1q6lmnxidh3npf70n43, 'use_source', False):
            raise ValueError('this adapter selection needs source metadata (use_source); it cannot serve requests')
        if not (isinstance(scale, (int, float)) and math.isfinite(scale) and (scale > 0)):
            raise ValueError('adapter scale must be a positive number')
        paths = {str(name): Path(path) for name, path in dict(adapters or {}).items()}
        for name, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(f'adapter {name!r} not found: {path}')
        with self._inference_lock:
            self._v3l7g6wbbo5b6b3izkna8(None)
            if paths != self._vdjyjd4b3w24pvj27zn5i:
                self._free_experts()
                self._vedlso49m4dr50us9e9ac = {name: self._load_adapter(name, path) for name, path in paths.items()}
            self._vdjyjd4b3w24pvj27zn5i = paths
            self._v1rax47ccvh8o08t3gzzi = {name: str((versions or {}).get(name, path)) for name, path in paths.items()}
            self._v6okoqmlfn4wxs6wo5jz2 = float(scale)
            self._va1q6lmnxidh3npf70n43, self.common_only = (_va1q6lmnxidh3npf70n43, bool(common_only))
            self._prefix_checkpoint = None

    def _vaj50x6ays33fttft175r(self, name):
        return (self.model_version, name, self._v1rax47ccvh8o08t3gzzi.get(name) if name else None, self._v6okoqmlfn4wxs6wo5jz2 if name else None)

    def select_experts(self, rows):
        raise ImportError('unavailable')
        return [_gone(row, self._va1q6lmnxidh3npf70n43, available=self._vdjyjd4b3w24pvj27zn5i, common_only=self.common_only) for row in rows]

    def _v3l7g6wbbo5b6b3izkna8(self, name):
        if name == self._vbfenikctyy2mceugfgie:
            return False
        if name is not None and name not in self._vedlso49m4dr50us9e9ac:
            raise ValueError(f'adapter {name!r} is not loaded')
        self._set_context_adapter(None if name is None else self._vedlso49m4dr50us9e9ac[name], self._v6okoqmlfn4wxs6wo5jz2)
        self._vbfenikctyy2mceugfgie = name
        self._v2ybxw3trfq52qoz7uvp0 = getattr(self, '_v2ybxw3trfq52qoz7uvp0', 0) + 1
        return True

    def _load_adapter(self, name, path):
        init = self.lib.llama_adapter_lora_init
        init.restype, init.argtypes = (c_void_p, [c_void_p, c_char_p])
        ptr = init(self.model, str(path).encode('utf-8'))
        if not ptr:
            raise RuntimeError(f'llama.cpp could not load adapter {name}')
        return ptr

    def _free_experts(self):
        free = getattr(self.lib, 'llama_adapter_lora_free', None)
        for ptr in self._vedlso49m4dr50us9e9ac.values():
            if free is not None:
                free.restype, free.argtypes = (None, [c_void_p])
                free(ptr)
        self._vedlso49m4dr50us9e9ac = {}

    def _set_context_adapter(self, ptr, scale):
        lib = self.lib
        if hasattr(lib, 'llama_set_adapters_lora'):
            fn = lib.llama_set_adapters_lora
            fn.restype = c_int32
            fn.argtypes = [c_void_p, _vepr5bfopjgvrwgcro5zb(c_void_p), c_size_t, _vepr5bfopjgvrwgcro5zb(c_float)]
            if ptr is None:
                status = fn(self.ctx, None, 0, None)
            else:
                status = fn(self.ctx, (c_void_p * 1)(ptr), 1, (c_float * 1)(scale))
        else:
            lib.llama_clear_adapter_lora.argtypes = [c_void_p]
            lib.llama_clear_adapter_lora(self.ctx)
            status = 0
            if ptr is not None:
                fn = lib.llama_set_adapter_lora
                fn.restype = c_int32
                fn.argtypes = [c_void_p, c_void_p, c_float]
                status = fn(self.ctx, ptr, scale)
        if status != 0:
            raise RuntimeError('llama.cpp could not activate the adapter')

    def close(self):
        self._prefix_checkpoint = None
        if getattr(self, 'model', None) and self._vedlso49m4dr50us9e9ac:
            self._free_experts()
        if getattr(self, 'ctx', None):
            self.lib.llama_free(self.ctx)
            self.ctx = None
        if getattr(self, 'model', None):
            self.lib.llama_model_free(self.model)
            self.model = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def label_ids(self, row):
        labels = tuple(NOUL_LABELS) if row['kind'].lower() == 'noul' else tuple(OPTION_LABELS[:len(row['candidates'])])
        if labels in self._label_cache:
            return self._label_cache[labels]
        ids = []
        for label in labels:
            local = self.tokenizer.encode(label, add_special_tokens=False)
            runtime = self.tokenize(label, add_special=False)
            if len(local) != 1 or runtime != local:
                raise ValueError(f'Label {label!r} is not one shared token in this runtime')
            ids.append(local[0])
        if len(set(ids)) != len(ids):
            raise ValueError('Output labels collide on one token in this runtime')
        self._label_cache[labels] = ids
        return ids

    def tokenize(self, text, *, add_special=False):
        cache = self._token_cache
        if cache is None:
            return self._tokenize_runtime(text, add_special)
        with _stage('tokenize'):
            return cache(text, add_special)

    def render_prompt(self, row):
        cache = self._render_cache
        key = fast_path.render_key(row, id(self.tokenizer)) if cache is not None else None
        with _stage('render'):
            if key is not None:
                hit = cache.get(key)
                if hit is not None:
                    return hit
            text = prompt_for(row, self.tokenizer)
            if key is not None:
                cache.put(key, text, cost=len(text) + len(key[1]))
            return text

    def cache_stats(self):
        return {'tokenize': self._token_cache.stats() if self._token_cache else None, 'render': self._render_cache.stats() if self._render_cache else None, 'device_prefix': dict(self.device_prefix_stats, slots=self._prefix_slots.stats() if self._prefix_slots else None, enabled=bool(self.device_prefix_cache and self._prefix_slots))}

    def _tokenize_runtime(self, text, add_special=False):
        raw = text.encode('utf-8')
        size = len(raw) + 16
        buffer = (llama_token * size)()
        count = self.lib.llama_tokenize(self.vocab, raw, len(raw), buffer, size, add_special, True)
        if count < 0:
            size = -count
            buffer = (llama_token * size)()
            count = self.lib.llama_tokenize(self.vocab, raw, len(raw), buffer, size, add_special, True)
        if count < 0:
            raise RuntimeError('llama.cpp tokenization failed')
        return list(buffer[:count])

    def _decode(self, sequences, shared=None, extra=None):
        total = sum((len(tokens) for _seq, tokens, _pos in sequences))
        if total == 0:
            return {}
        if total > self.n_batch:
            if shared is not None:
                for seq_id, tokens, start in sequences:
                    for offset in range(0, len(tokens), self.n_batch):
                        self._decode([(seq_id, tokens[offset:offset + self.n_batch], start + offset)], shared=True, extra=extra)
                return {}
            final = []
            for seq_id, tokens, start in sequences:
                if tokens:
                    self._decode([(seq_id, tokens[:-1], start)], shared=True, extra=extra)
                    final.append((seq_id, tokens[-1:], start + len(tokens) - 1))
            if len(final) > self.n_batch:
                raise ValueError('More output sequences than the runtime batch capacity')
            return self._decode(final, extra=extra)
        batch = self.lib.llama_batch_init(total, 0, self.n_seq_max)
        try:
            index, outputs, hidden_at = (0, {}, [])
            want = extra or {}
            probe_mode = 2 if self._sync_probe_rows() else 1
            _v4vb17spq8uv6ozwczzvt = 2 if self._candidate_active else 1
            for seq_id, tokens, start in sequences:
                marks = want.get(seq_id, ())
                for offset, token in enumerate(tokens):
                    batch.token[index] = token
                    batch.pos[index] = start + offset
                    batch.n_seq_id[index] = 1
                    batch.seq_id[index][0] = seq_id
                    last = offset == len(tokens) - 1
                    _v67ub3xu71brsc1w0l52j = last and shared is None
                    mark = getattr(self, 'hidden_outputs', 0) and (start + offset in marks or (_v67ub3xu71brsc1w0l52j and seq_id in want))
                    batch.logits[index] = _v4vb17spq8uv6ozwczzvt if _v67ub3xu71brsc1w0l52j else probe_mode if mark else 0
                    if _v67ub3xu71brsc1w0l52j:
                        outputs[seq_id] = index
                    if mark:
                        hidden_at.append((index, seq_id, start + offset))
                    index += 1
            batch.n_tokens = index
            with _stage(self._stage_name):
                status = self.lib.llama_decode(self.ctx, batch)
            if status != 0:
                raise RuntimeError(f'llama_decode failed with status {status}')
            probes = getattr(self, 'probe_ids', ())
            for idx, seq_id, pos in hidden_at:
                if probe_mode == 2:
                    row = self.lib.llama_get_probe_logits_ith(self.ctx, idx)
                    if not row:
                        raise RuntimeError('llama.cpp produced no probe logits for a marked position')
                    self.last_hidden[seq_id, pos] = row[:len(probes)]
                    continue
                row = self.lib.llama_get_logits_ith(self.ctx, idx)
                if not row:
                    raise RuntimeError('llama.cpp produced no logits for a marked position')
                self.last_hidden[seq_id, pos] = [float(row[k]) for k in probes]
            return outputs
        finally:
            self.lib.llama_batch_free(batch)

    def _sync_probe_rows(self):
        if not hasattr(self.lib, 'llama_set_probe_rows') or not getattr(self, 'ctx', None):
            return False
        ids = tuple((int(k) for k in getattr(self, 'probe_ids', ()))) if getattr(self, 'use_probe_rows', True) else ()
        rows = ids
        if self._candidate_rows:
            from med9rt1x7qcs4m95rhbty import merge_probe_rows
            rows, columns = merge_probe_rows(ids, self._candidate_rows)
        if rows != getattr(self, '_probe_rows_set', ()):
            array = (c_int32 * max(1, len(rows)))(*rows)
            if self.lib.llama_set_probe_rows(self.ctx, array, len(rows)) != 0:
                raise ValueError('llama_set_probe_rows rejected the probe ids')
            self._probe_rows_set = rows
            if self._candidate_rows:
                self._candidate_columns = columns
        return bool(ids)

    def logits_host_bytes(self):
        if hasattr(self.lib, 'llama_get_logits_host_bytes'):
            return int(self.lib.llama_get_logits_host_bytes(self.ctx))
        return None

    def _logits_at(self, index, ids):
        if self._candidate_active:
            with _stage('_v67ub3xu71brsc1w0l52j'):
                row = self.lib.llama_get_probe_logits_ith(self.ctx, index)
                if not row:
                    raise RuntimeError('llama.cpp produced no candidate logits for the answer position')
                from med9rt1x7qcs4m95rhbty import select_columns
                return select_columns(row, self._candidate_columns, ids)
        with _stage('_v67ub3xu71brsc1w0l52j'):
            row = self.lib.llama_get_logits_ith(self.ctx, index)
            if not row:
                raise RuntimeError('llama.cpp produced no logits for the answer position')
            return [float(row[i]) for i in ids]

    def _distribution(self, logits, kind=None):
        kind = kind.lower() if kind else None
        temperature = getattr(self, 'temperatures', {}).get(kind, self.temperature)
        return distribution(logits, temperature, getattr(self, 'noul_logit_bias', 0.0) if kind == 'noul' else 0.0)

    def clear_prefix_cache(self):
        with self._inference_lock:
            self._prefix_checkpoint = None

    def _save_prefix(self, key):
        size = int(self.lib.llama_state_seq_get_size(self.ctx, 0))
        if size <= 0:
            raise RuntimeError('llama.cpp could not size the prefix checkpoint')
        data = (ctypes.c_uint8 * size)()
        if self.lib.llama_state_seq_get_data(self.ctx, data, size, 0) != size:
            raise RuntimeError('llama.cpp could not save the full prefix checkpoint')
        self._prefix_checkpoint = (key, data)

    def _restore_prefix(self):
        data = self._prefix_checkpoint[1]
        if self.lib.llama_state_seq_set_data(self.ctx, data, len(data), 0) != len(data):
            self._prefix_checkpoint = None
            self.lib.llama_memory_clear(self.memory, True)
            raise RuntimeError('llama.cpp could not restore the full prefix checkpoint')

    def judge(self, rows, *, share_prefix=False, session_id=None, prefix_length=None, reasoning='off', report=None, **reasoning_options):
        if reasoning != 'off':
            if session_id is not None or prefix_length is not None:
                raise ValueError('Session prefix caching applies to reasoning="off" only')
            return [result['probs'] for result in self.judge_reasoned(rows, reasoning=reasoning, **reasoning_options)]
        if reasoning_options:
            raise TypeError(f'Unexpected options for reasoning="off": {sorted(reasoning_options)}')
        with self._inference_lock:
            return self._judge(rows, share_prefix=share_prefix, session_id=session_id, prefix_length=prefix_length, report=report)

    def judge_reasoned(self, rows, *, reasoning='auto', budget_ms=None, profile=None, config=None, gate=None, t_start=None, adapters=None):
        raise ImportError('unavailable')
        if reasoning not in _gone.REASONING_MODES:
            raise ValueError(f'reasoning must be one of {_gone.REASONING_MODES}')
        with self._inference_lock:
            engine = getattr(self, '_v5jqt0h2xs6londm52e1b', None)
            if engine is None or (adapters is not None and {k: Path(v) for k, v in dict(adapters).items()} != self._vdjyjd4b3w24pvj27zn5i):
                engine = self._v5jqt0h2xs6londm52e1b = _gone.LlamaEngine(self, adapters=adapters)
            return _gone.judge_rows(engine, rows, reasoning=reasoning, budget_ms=budget_ms, profile=profile, config=config, gate=gate, t_start=t_start)

    def _judge(self, rows, *, share_prefix=False, session_id=None, prefix_length=None, report=None):
        self.prefix_cache_stats = {'restores': 0, 'restored_prefix_tokens': 0, 'prefix_tokens_decoded': 0, 'checkpoint_bytes': 0}
        if session_id is not None and (not isinstance(session_id, str) or not session_id):
            raise ValueError('session_id must be a nonempty string')
        if prefix_length is not None and (session_id is None or type(prefix_length) is not int or prefix_length < 1):
            raise ValueError('prefix_length requires a session_id and a positive integer')
        if not rows:
            return []
        prompts = [self.tokenize(self.render_prompt(row)) for row in rows]
        for tokens in prompts:
            if len(tokens) > min(self.max_length, self.n_ctx_seq):
                raise ValueError('Input exceeds max_length; evidence was not truncated')
        if report is not None:
            report['prompt_tokens'] = [len(tokens) for tokens in prompts]
            report['input_tokens'] = prompt_token_usage(prompts)
        labels = [self.label_ids(row) for row in rows]
        choices = self.select_experts(rows)
        experts = [choice['_v6gd7oavvss5c4jnyhxhc'] for choice in choices]
        self.last_experts = choices
        persistent = 0
        if session_id is not None:
            available = min(_common_prefix(prompts), min(map(len, prompts)) - 1)
            persistent = available if prefix_length is None else prefix_length
            if persistent > available:
                raise ValueError('Cached prefix must match every row and leave an answer token')
        unique, source, seen = ([], [], {})
        for index, row in enumerate(rows):
            key = (tuple(prompts[index]), tuple(labels[index]), row['kind'].lower(), experts[index])
            if key not in seen:
                seen[key] = index
                unique.append(index)
            source.append(seen[key])
        results = {}
        for _v6gd7oavvss5c4jnyhxhc in dict.fromkeys((experts[index] for index in unique)):
            group = [index for index in unique if experts[index] == _v6gd7oavvss5c4jnyhxhc]
            self._v3l7g6wbbo5b6b3izkna8(_v6gd7oavvss5c4jnyhxhc)
            cache_key = None
            if session_id is not None:
                cache_key = (session_id, self.model, self.ctx, id(self.tokenizer), repr(getattr(self.tokenizer, 'chat_template', None)), id(prompt_for), self._vaj50x6ays33fttft175r(_v6gd7oavvss5c4jnyhxhc), tuple(prompts[0][:persistent]))
                if self._prefix_checkpoint is not None and self._prefix_checkpoint[0] != cache_key:
                    self._prefix_checkpoint = None
            self._judge_group(group, prompts, labels, rows, results, persistent, cache_key, share_prefix, _v6gd7oavvss5c4jnyhxhc=_v6gd7oavvss5c4jnyhxhc)
        return [list(results[index]) for index in source]

    def _device_identity(self, _v6gd7oavvss5c4jnyhxhc):
        return (self.model, self.ctx, id(self.tokenizer), repr(getattr(self.tokenizer, 'chat_template', None)), id(prompt_for), self._vaj50x6ays33fttft175r(_v6gd7oavvss5c4jnyhxhc))

    def _device_seq_valid(self, entry):
        return int(self.lib.llama_memory_seq_pos_max(self.memory, entry.seq)) == len(entry.tokens) - 1

    def _template_tokens(self, identity):
        key = identity[2:5]
        cached = self._template_cache
        if cached is not None and cached[0] == key:
            return cached[1]
        probes = [self.tokenize(self.render_prompt(row)) for row in fast_path.probe_rows()]
        n = fast_path.template_prefix_length(probes)
        tokens = list(probes[0][:n]) if n else []
        self._template_cache = (key, tokens)
        return tokens

    def _build_device_prefix(self, entry):
        self.lib.llama_memory_seq_rm(self.memory, entry.seq, -1, -1)
        try:
            self._decode([(entry.seq, entry.tokens, 0)], shared=True)
        except Exception:
            self._prefix_slots.drop(entry)
            self.lib.llama_memory_seq_rm(self.memory, entry.seq, -1, -1)
            raise
        self.device_prefix_stats['builds'] += 1

    def _device_prefix_lookup(self, window_prompts, _v6gd7oavvss5c4jnyhxhc):
        slots = self._prefix_slots
        identity = self._device_identity(_v6gd7oavvss5c4jnyhxhc)
        entry = slots.match(window_prompts, identity, valid=self._device_seq_valid)
        if entry is None:
            head = self._template_tokens(identity)
            n = len(head)
            if n and all((len(p) > n and p[:n] == head for p in window_prompts)):
                entry = slots.reserve(head, identity, pinned=True)
                if entry is not None:
                    self._build_device_prefix(entry)
        self.device_prefix_stats['misses' if entry is None else 'hits'] += 1
        return entry

    def _clear_question_slots(self):
        for seq in range(self.n_seq_max):
            self.lib.llama_memory_seq_rm(self.memory, seq, -1, -1)

    def _capture_cut(self, row, prompt, base, shared):
        slots = self._prefix_slots
        if len(slots.seq_ids) < 2:
            return 0
        text = self.render_prompt(row)
        at = text.find(STATE_END)
        if at < 0:
            return 0
        n = fast_path.common_token_prefix([self.tokenize(text[:at]), prompt])
        cut = min(shared, n - 1)
        return cut if cut - base >= self.device_prefix_capture_min and cut < shared else 0

    def _capture_shared_prefix(self, prompt, cut, _v6gd7oavvss5c4jnyhxhc):
        entry = self._prefix_slots.reserve(prompt[:cut], self._device_identity(_v6gd7oavvss5c4jnyhxhc))
        if entry is None:
            return
        self.lib.llama_memory_seq_rm(self.memory, entry.seq, -1, -1)
        self.lib.llama_memory_seq_cp(self.memory, 0, entry.seq, -1, -1)
        self.device_prefix_stats['captures'] += 1

    def _judge_group(self, unique, prompts, labels, rows, results, persistent, cache_key, share_prefix, _v6gd7oavvss5c4jnyhxhc=None):
        long_rows = self._long_rows(unique, prompts, rows)
        if long_rows:
            self._judge_long_rows(long_rows, prompts, labels, rows, results, share_prefix)
        unique = [index for index in unique if index not in long_rows]
        if not unique:
            return
        use_device = bool(self.device_prefix_cache and self._prefix_slots is not None and (not persistent))
        for start in range(0, len(unique), self.n_seq_max):
            window = unique[start:start + self.n_seq_max]
            entry = None
            if use_device:
                with _stage('prefix_cache'):
                    entry = self._device_prefix_lookup([prompts[i] for i in window], _v6gd7oavvss5c4jnyhxhc)
                    if entry is not None:
                        self._clear_question_slots()
            if entry is None:
                self.lib.llama_memory_clear(self.memory, False)
            base = 0 if entry is None else len(entry.tokens)
            self._v2ybxw3trfq52qoz7uvp0 = getattr(self, '_v2ybxw3trfq52qoz7uvp0', 0) + 1
            shared = persistent or (_common_prefix([prompts[i] for i in window]) if share_prefix and len(window) > 1 else 0)
            shared = min(shared, min((len(prompts[i]) for i in window)) - 1)
            shared = max(shared, base)
            if shared:
                if persistent and self._prefix_checkpoint is not None:
                    self._restore_prefix()
                    self.prefix_cache_stats['restores'] += 1
                    self.prefix_cache_stats['restored_prefix_tokens'] += shared
                else:
                    if base:
                        with _stage('prefix_cache'):
                            self.lib.llama_memory_seq_cp(self.memory, entry.seq, 0, -1, -1)
                        self.device_prefix_stats['tokens_saved'] += base * len(window)
                    if shared > base:
                        first = window[0]
                        cut = self._capture_cut(rows[first], prompts[first], base, shared) if use_device else 0
                        if cut:
                            self._decode([(0, prompts[first][base:cut], base)], shared=True)
                            self._capture_shared_prefix(prompts[first], cut, _v6gd7oavvss5c4jnyhxhc)
                            self._decode([(0, prompts[first][cut:shared], cut)], shared=True)
                        else:
                            self._decode([(0, prompts[first][base:shared], base)], shared=True)
                    self.prefix_cache_stats['prefix_tokens_decoded'] += shared - base
                    if persistent:
                        self._save_prefix(cache_key)
                if persistent:
                    self.prefix_cache_stats['checkpoint_bytes'] = len(self._prefix_checkpoint[1])
                for slot, _index in enumerate(window):
                    if slot:
                        self.lib.llama_memory_seq_cp(self.memory, 0, slot, -1, -1)
            batch = [(slot, prompts[index][shared:], shared) for slot, index in enumerate(window)]
            if self.candidate_head and self._candidate_rows:
                self._candidate_active = True
                try:
                    outputs = self._decode(batch)
                    for slot, index in enumerate(window):
                        results[index] = self._distribution(self._logits_at(outputs[slot], labels[index]), rows[index]['kind'])
                finally:
                    self._candidate_active = False
                continue
            outputs = self._decode(batch)
            for slot, index in enumerate(window):
                results[index] = self._distribution(self._logits_at(outputs[slot], labels[index]), rows[index]['kind'])

    def __call__(self, rows, *, share_prefix=False, session_id=None, prefix_length=None):
        return self.judge(rows, share_prefix=share_prefix, session_id=session_id, prefix_length=prefix_length)

    def compare_isolated(self, rows):
        return _compare(rows, self.judge(rows, share_prefix=False), self.judge(rows, share_prefix=True))

    def verify_isolation(self, rows):
        alone = [self.judge([row])[0] for row in rows]
        order = list(range(len(rows)))[::-1]
        reversed_rows = [rows[i] for i in order]
        reversed_out = self.judge(reversed_rows)
        restored = [None] * len(rows)
        for position, index in enumerate(order):
            restored[index] = reversed_out[position]
        return {'batched': _compare(rows, alone, self.judge(rows)), 'reordered': _compare(rows, alone, restored), 'shared_prefix': _compare(rows, alone, self.judge(rows, share_prefix=True))}

def _compare(rows, reference, other):
    worst, where = (0.0, None)
    for row, left, right in zip(rows, reference, other):
        for a, b in zip(left, right):
            if abs(a - b) > worst:
                worst, where = (abs(a - b), row.get('id'))
    return {'max_probability_difference': worst, 'row': where, 'argmax_changed': sum((1 for a, b in zip(reference, other) if a.index(max(a)) != b.index(max(b))))}

def prompt_token_usage(prompts):
    if not prompts:
        return 0
    return sum((len(tokens) for tokens in prompts)) - _common_prefix(prompts) * (len(prompts) - 1)

def _common_prefix(sequences):
    shortest = min((len(tokens) for tokens in sequences))
    for index in range(shortest):
        value = sequences[0][index]
        if any((tokens[index] != value for tokens in sequences)):
            return index
    return shortest