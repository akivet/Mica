"""The deployed judgment path: candidate logits read straight out of the prefill.

One prefill over the prompt (chunked at the runtime batch limit),
``llama_get_logits_ith`` at its last position, and the candidate rows indexed
out of that vector. No sampler chain is
constructed or run, no token is generated or accepted, no grammar or bias exists,
and nothing is requested twice: a candidate with a tiny logit costs exactly as
much as the top one, because the whole row vector is already in memory.

Independent questions are independent sequences. Each gets its own ``seq_id`` in
one batch, so llama.cpp keeps a separate state per question - attention KV and,
for the linear-attention blocks of this hybrid model, the convolution and
recurrent state - and one question can never attend to another's tokens.
More questions than ``n_seq_max`` are answered in windows of ``n_seq_max``
sequences.

A shared prefix is opt-in through ``share_prefix``. The prefix is decoded once
into a donor sequence and then copied with ``llama_memory_seq_cp``, which copies
the whole memory of that sequence (attention and recurrent alike) and keeps the
position offset, so each suffix continues from the right place. It stays an
optimization to verify rather than assume: ``verify_isolation`` answers every row
alone and then again batched, reordered and over a shared prefix. Batching alone
already moves probabilities slightly, because a different ubatch shape changes
the reduction order, so a shared-prefix difference only indicates shared state
when it is larger than the batched one.

Across calls, an explicit ``session_id`` retains one complete prefix checkpoint
in host memory. Restoring it includes attention KV and recurrent state; no
partial recurrent rollback is attempted. The final readout is always decoded.

This binds the llama.cpp shared library directly (``llama.dll`` / ``libllama.so``), b11010 or b11700 and later,
the same build the HTTP server uses.
"""
from __future__ import annotations

import ctypes
import math
import os
import sys
import threading
from ctypes import POINTER, c_bool, c_char_p, c_float, c_int8, c_int32, c_size_t, c_uint32, c_void_p
from pathlib import Path

from .codebook import NOUL_LABELS, OPTION_LABELS
from .native import prompt_for
from .calibration import load_calibration, distribution

llama_token = c_int32
llama_pos = c_int32
llama_seq_id = c_int32


class llama_model_params(ctypes.Structure):
    _fields_ = [
        ('devices', c_void_p),
        ('tensor_buft_overrides', c_void_p),
        ('n_gpu_layers', c_int32),
        ('split_mode', c_int32),
        ('load_mode', c_int32),
        ('lazy_mode', c_int32),
        ('main_gpu', c_int32),
        ('tensor_split', POINTER(c_float)),
        ('progress_callback', c_void_p),
        ('progress_callback_user_data', c_void_p),
        ('kv_overrides', c_void_p),
        ('vocab_only', c_bool),
        ('check_tensors', c_bool),
        ('use_extra_bufts', c_bool),
        ('no_host', c_bool),
        ('no_alloc', c_bool),
        ('load_mtp', c_bool),
    ]


class llama_context_params(ctypes.Structure):
    _fields_ = [
        ('n_ctx', c_uint32),
        ('n_batch', c_uint32),
        ('n_ubatch', c_uint32),
        ('n_seq_max', c_uint32),
        ('n_rs_seq', c_uint32),
        ('n_outputs_max', c_uint32),
        ('n_outputs_max_per_seq', c_uint32),
        ('n_threads', c_int32),
        ('n_threads_batch', c_int32),
        ('ctx_type', c_int32),
        ('rope_scaling_type', c_int32),
        ('pooling_type', c_int32),
        ('attention_type', c_int32),
        ('flash_attn_type', c_int32),
        ('rope_freq_base', c_float),
        ('rope_freq_scale', c_float),
        ('yarn_ext_factor', c_float),
        ('yarn_attn_factor', c_float),
        ('yarn_beta_fast', c_float),
        ('yarn_beta_slow', c_float),
        ('yarn_orig_ctx', c_uint32),
        ('defrag_thold', c_float),
        ('cb_eval', c_void_p),
        ('cb_eval_user_data', c_void_p),
        ('type_k', c_int32),
        ('type_v', c_int32),
        ('abort_callback', c_void_p),
        ('abort_callback_data', c_void_p),
        ('embeddings', c_bool),
        ('offload_kqv', c_bool),
        ('no_perf', c_bool),
        ('op_offload', c_bool),
        ('swa_full', c_bool),
        ('kv_unified', c_bool),
        ('samplers', c_void_p),
        ('n_samplers', c_size_t),
        ('ctx_other', c_void_p),
    ]


# llama.cpp from b11700 on (the build with the chunked CUDA / HIP kernel for this model's linear-attention layers)
# adds `size_t moe_cache_size` after type_v. load_library tells the two layouts apart from the library's defaults.
llama_context_params_new = type('llama_context_params_new', (ctypes.Structure,), {'_fields_': [
    f for name, kind in llama_context_params._fields_
    for f in ([(name, kind)] + ([('moe_cache_size', c_size_t)] if name == 'type_v' else []))]})


def _context_params_type(lib):
    """The llama_context_params layout of this library. The defaults are read into the larger layout (an older library
    writes fewer bytes, never more), then checked: offload_kqv and op_offload default to true in both versions, and in
    the wrong layout they land on padding or a null pointer."""
    lib.llama_context_default_params.restype = llama_context_params_new
    lib.llama_context_default_params.argtypes = []
    raw = bytes(lib.llama_context_default_params())
    old = llama_context_params.from_buffer_copy(raw[:ctypes.sizeof(llama_context_params)])
    new = llama_context_params_new.from_buffer_copy(raw)
    old_ok = old.offload_kqv and old.op_offload and not old.abort_callback_data
    new_ok = new.offload_kqv and new.op_offload and not new.abort_callback and not new.abort_callback_data
    if old_ok == new_ok:
        raise RuntimeError('unrecognised llama.cpp build: use b11010 or b11700 and later')
    return llama_context_params if old_ok else llama_context_params_new


class llama_batch(ctypes.Structure):
    _fields_ = [
        ('n_tokens', c_int32),
        ('token', POINTER(llama_token)),
        ('embd', POINTER(c_float)),
        ('pos', POINTER(llama_pos)),
        ('n_seq_id', POINTER(c_int32)),
        ('seq_id', POINTER(POINTER(llama_seq_id))),
        ('logits', POINTER(c_int8)),
    ]


def _library_name() -> str:
    if sys.platform == 'win32':
        return 'llama.dll'
    return 'libllama.dylib' if sys.platform == 'darwin' else 'libllama.so'


def _backend_names(runtime, ggml):
    """Names of the registered ggml backends ('Vulkan', 'CUDA', 'CPU', ...); [] when they cannot be read."""
    base_name = ('ggml-base.dll' if sys.platform == 'win32'
                 else 'libggml-base.dylib' if sys.platform == 'darwin' else 'libggml-base.so')
    try:
        base = ctypes.CDLL(str(runtime / base_name)) if (runtime / base_name).exists() else ggml
        ggml.ggml_backend_reg_get.restype = c_void_p           # registry: ggml; names: ggml-base
        ggml.ggml_backend_reg_get.argtypes = [c_size_t]
        base.ggml_backend_reg_name.restype = c_char_p
        base.ggml_backend_reg_name.argtypes = [c_void_p]
        return [base.ggml_backend_reg_name(ggml.ggml_backend_reg_get(i)).decode('utf-8', 'replace')
                for i in range(ggml.ggml_backend_reg_count())]
    except (OSError, AttributeError):
        return []


def load_library(runtime_dir):
    """Load the llama.cpp shared library, with its own directory on the search path."""
    runtime = Path(runtime_dir)
    path = runtime / _library_name()
    if not path.exists():
        raise FileNotFoundError(f'{path} not found; point runtime_dir at the llama.cpp build')
    if sys.platform == 'win32':
        # ggml*.dll sit beside it and are resolved relative to the process.
        os.add_dll_directory(str(runtime.resolve()))
    # llama.cpp builds ship their backends as separate shared libraries, which
    # are registered by ggml's loader. Without this, model loading fails with
    # "no backends are loaded" even though the files sit right there.
    ggml_name = ('ggml.dll' if sys.platform == 'win32'
                 else 'libggml.dylib' if sys.platform == 'darwin' else 'libggml.so')
    ggml = ctypes.CDLL(str(runtime / ggml_name))
    # The loader searches the executable's directory, which is python's, not the
    # build's. Point it at the build so the Vulkan backend is actually found.
    ggml.ggml_backend_load_all_from_path.restype = None
    ggml.ggml_backend_load_all_from_path.argtypes = [c_char_p]
    ggml.ggml_backend_load_all_from_path(str(runtime.resolve()).encode('utf-8'))
    ggml.ggml_backend_reg_count.restype = c_size_t
    ggml.ggml_backend_reg_count.argtypes = []
    if ggml.ggml_backend_reg_count() == 0:
        raise RuntimeError(f'no ggml backend registered from {runtime}')
    lib = ctypes.CDLL(str(path))
    lib.ggml = ggml
    lib.backends = _backend_names(runtime, ggml)
    # A GPU backend file that did not register (missing driver, or a DLL from another build) leaves only the CPU,
    # which still works but is many times slower; say so instead of running slowly in silence.
    gpu_files = sorted(p.name for p in runtime.glob('*ggml-*') if any(k in p.name.lower() for k in
                                                                       ('vulkan', 'cuda', 'hip', 'metal', 'sycl')))
    if gpu_files and lib.backends and set(lib.backends) <= {'CPU', 'RPC', 'BLAS'}:
        print(f'warning: {", ".join(gpu_files)} did not load; running on the CPU only', file=sys.stderr)
    lib.llama_backend_init.restype = None
    lib.llama_backend_init.argtypes = []
    lib.llama_model_default_params.restype = llama_model_params
    lib.llama_model_default_params.argtypes = []
    context_params = _context_params_type(lib)
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
    lib.llama_tokenize.argtypes = [c_void_p, c_char_p, c_int32, POINTER(llama_token), c_int32, c_bool, c_bool]
    lib.llama_batch_init.restype = llama_batch
    lib.llama_batch_init.argtypes = [c_int32, c_int32, c_int32]
    lib.llama_batch_free.restype = None
    lib.llama_batch_free.argtypes = [llama_batch]
    lib.llama_decode.restype = c_int32
    lib.llama_decode.argtypes = [c_void_p, llama_batch]
    lib.llama_get_logits_ith.restype = POINTER(c_float)
    lib.llama_get_logits_ith.argtypes = [c_void_p, c_int32]
    lib.llama_get_memory.restype = c_void_p
    lib.llama_get_memory.argtypes = [c_void_p]
    lib.llama_memory_seq_rm.restype = c_bool
    lib.llama_memory_seq_rm.argtypes = [c_void_p, llama_seq_id, llama_pos, llama_pos]
    lib.llama_memory_seq_cp.restype = None
    lib.llama_memory_seq_cp.argtypes = [c_void_p, llama_seq_id, llama_seq_id, llama_pos, llama_pos]
    lib.llama_memory_clear.restype = None
    lib.llama_memory_clear.argtypes = [c_void_p, c_bool]
    lib.llama_memory_seq_pos_max.restype = llama_pos
    lib.llama_memory_seq_pos_max.argtypes = [c_void_p, llama_seq_id]
    lib.llama_state_seq_get_size.restype = c_size_t
    lib.llama_state_seq_get_size.argtypes = [c_void_p, llama_seq_id]
    for name in ('llama_state_seq_get_data', 'llama_state_seq_set_data'):
        function = getattr(lib, name)
        function.restype = c_size_t
        function.argtypes = [c_void_p, POINTER(ctypes.c_uint8), c_size_t, llama_seq_id]
    return lib


class DirectJudge:
    """Typed judgment from one prefill, with no sampling and no generation."""

    def __init__(self, model_path, tokenizer, *, runtime_dir, n_ctx=4096, n_seq_max=1,
                 n_gpu_layers=-1, n_threads=None, temperature=1.0, max_length=2048, noul_logit_bias=0.0,
                 flash_attn=-1, n_ubatch=None, header_cache=True, state_cache=True):
        if any(type(value) is not int or value < 1 for value in (n_ctx, n_seq_max, max_length)):
            raise ValueError('Context, sequence and input limits must be positive integers')
        if not (isinstance(temperature, (int, float)) and math.isfinite(temperature) and temperature > 0):
            raise ValueError('Calibration temperature must be a positive number')
        if not isinstance(noul_logit_bias,(int,float)) or not math.isfinite(noul_logit_bias):
            raise ValueError('Noul calibration bias must be finite')
        self.lib = load_library(runtime_dir)
        self.lib.llama_backend_init()
        self.tokenizer, self.temperature, self.max_length = tokenizer, float(temperature), max_length
        self.noul_logit_bias=float(noul_logit_bias)
        model_params = self.lib.llama_model_default_params()
        model_params.n_gpu_layers = n_gpu_layers
        self.model = self.lib.llama_model_load_from_file(str(model_path).encode('utf-8'), model_params)
        if not self.model:
            raise RuntimeError(f'llama.cpp could not load {model_path}')
        context_params = self.lib.llama_context_default_params()
        # state_cache: one more sequence (on the GPU) keeps the most recent long state, so its memory grows by one
        # sequence of --n-ctx tokens; llama.cpp splits n_ctx evenly over the sequences.
        slots = n_seq_max + (1 if state_cache else 0)
        context_params.n_ctx = n_ctx * slots
        context_params.n_batch = max(n_ctx, 2048)
        # A ubatch as large as n_ctx spills VRAM on 8 GB cards; 512 is several times faster there.
        context_params.n_ubatch = n_ubatch or 512
        # -1 auto, 0 off, 1 on. Auto on a Vulkan-only build means off: measured on an RX 6600 XT, flash attention made
        # a 4k-token prompt 2x slower (8.3 s vs 4.2 s) and 1k 1.2x slower. CUDA, ROCm and Metal keep llama.cpp's auto.
        if flash_attn == -1 and 'Vulkan' in self.lib.backends and not {'CUDA', 'ROCm', 'HIP', 'Metal'} & set(self.lib.backends):
            flash_attn = 0
        self.flash_attn = flash_attn
        context_params.flash_attn_type = flash_attn
        context_params.n_seq_max = slots
        # Only the final prompt position needs output projection. It still
        # computes all vocabulary logits; label_ids selects from that vector.
        context_params.n_outputs_max = slots
        context_params.n_outputs_max_per_seq = 1
        if n_threads:
            context_params.n_threads = context_params.n_threads_batch = n_threads
        self.n_seq_max, self.n_ctx_seq = n_seq_max, n_ctx
        self.ctx = self.lib.llama_init_from_model(self.model, context_params)
        if not self.ctx:
            raise RuntimeError('llama.cpp could not create a context')
        self.n_batch = int(self.lib.llama_n_batch(self.ctx))
        self.n_ctx_seq = int(self.lib.llama_n_ctx_seq(self.ctx))
        runtime_seqs = int(self.lib.llama_n_seq_max(self.ctx))
        self.n_seq_max = min(n_seq_max, runtime_seqs - (slots - n_seq_max), self.n_batch)
        if min(self.n_batch, self.n_ctx_seq, self.n_seq_max) < 1:
            self.close()
            raise RuntimeError('llama.cpp returned invalid context limits')
        # header_cache: the decoded prompt header (system prompt and template up to the state, the same tokens in
        # every request) is kept as a sequence checkpoint in host memory and restored into sequence 0 instead of
        # being decoded again; with a short state the header is about a third of the prompt.
        self.header_cache = bool(header_cache)
        self._header, self._header_state = None, None
        # state cache: after a long state is decoded, sequence 0 at the end of the state is copied into a reserved
        # sequence on the GPU (a device copy, no host transfer). A later request whose prompt starts with those
        # tokens - the same state with another question, or a transcript that grew - copies it back and decodes only
        # the rest.
        self._slot = runtime_seqs - 1 if state_cache else None
        self._slot_tokens = None
        self.memory = self.lib.llama_get_memory(self.ctx)
        self.vocab = self.lib.llama_model_get_vocab(self.model)
        self.n_vocab = self.lib.llama_vocab_n_tokens(self.vocab)
        self._label_cache = {}
        self._inference_lock = threading.RLock()
        self._prefix_checkpoint = None
        self.prefix_cache_stats = {}

    @classmethod
    def from_calibration(cls, model_path, tokenizer, calibration_path, *, model_identity=None, **kwargs):
        fitted, temperatures, bias = load_calibration(calibration_path, model_identity=model_identity)
        judge = cls(model_path, tokenizer, temperature=float(fitted.get('temperature', 1.)), noul_logit_bias=bias, **kwargs)
        judge.temperatures = temperatures
        judge.calibration = {'path': str(calibration_path), **fitted}
        return judge

    def close(self):
        self._prefix_checkpoint = None
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
        """Codebook token ids for this row, checked once against this runtime."""
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

    def _decode(self, sequences, shared=None):
        """Decode (seq_id, tokens, start_pos) triples, asking for the last row only."""
        total = sum(len(tokens) for _seq, tokens, _pos in sequences)
        if total == 0:
            return {}
        if total > self.n_batch:
            if shared is not None:
                for seq_id, tokens, start in sequences:
                    for offset in range(0, len(tokens), self.n_batch):
                        self._decode([(seq_id, tokens[offset:offset + self.n_batch], start + offset)],
                                     shared=True)
                return {}
            # Logits belong to the most recent decode only. Keep every sequence's
            # readout token for the final batch so no output is overwritten by a
            # later prefill chunk. All preceding chunks request no logits.
            final = []
            for seq_id, tokens, start in sequences:
                if tokens:
                    self._decode([(seq_id, tokens[:-1], start)], shared=True)
                    final.append((seq_id, tokens[-1:], start + len(tokens) - 1))
            if len(final) > self.n_batch:
                raise ValueError('More readout sequences than the runtime batch capacity')
            return self._decode(final)
        batch = self.lib.llama_batch_init(total, 0, self.n_seq_max)
        try:
            index, outputs = 0, {}
            for seq_id, tokens, start in sequences:
                for offset, token in enumerate(tokens):
                    batch.token[index] = token
                    batch.pos[index] = start + offset
                    batch.n_seq_id[index] = 1
                    batch.seq_id[index][0] = seq_id
                    last = offset == len(tokens) - 1
                    batch.logits[index] = 1 if (last and shared is None) else 0
                    if last and shared is None:
                        outputs[seq_id] = index
                    index += 1
            batch.n_tokens = index
            status = self.lib.llama_decode(self.ctx, batch)
            if status != 0:
                raise RuntimeError(f'llama_decode failed with status {status}')
            return outputs
        finally:
            self.lib.llama_batch_free(batch)

    def _header_tokens(self, row):
        """Token ids every prompt starts with: two renderings of a row that differ only in the state agree up to
        the state; one token less in case the first state token merges with the text before it."""
        if self._header is None:
            a = self.tokenize(prompt_for({**row, 'state': 'alpha'}, self.tokenizer))
            b = self.tokenize(prompt_for({**row, 'state': 'omega'}, self.tokenizer))
            n = _common_prefix([a, b]) - 1
            self._header = a[:n] if n > 0 else []
        return self._header

    def _checkpoint(self):
        """Sequence 0 as bytes (attention KV and recurrent state), or None when this build cannot save it."""
        size = int(self.lib.llama_state_seq_get_size(self.ctx, 0))
        if size <= 0:
            return None
        data = (ctypes.c_uint8 * size)()
        return data if self.lib.llama_state_seq_get_data(self.ctx, data, size, 0) == size else None

    def _clear(self):
        """Empty the question sequences; the state-cache sequence keeps its state."""
        if self._slot is None:
            self.lib.llama_memory_clear(self.memory, False)
        else:
            for seq in range(self.n_seq_max):
                self.lib.llama_memory_seq_rm(self.memory, seq, -1, -1)

    def _restore_base(self, window_prompts, row):
        """Empty the question sequences and put the longest cached prefix of these prompts into sequence 0: the
        state cache, or the prompt header (header cache, decoded and checkpointed on first use). Returns its token
        count; 0 when nothing applies."""
        self._clear()
        limit = min(len(p) for p in window_prompts) - 1        # leave a readout token
        cached = self._slot_tokens
        if cached is not None:
            n = len(cached)
            if int(self.lib.llama_memory_seq_pos_max(self.memory, self._slot)) != n - 1:
                self._slot_tokens = None                         # the reserved sequence was cleared or changed
            elif n <= limit and all(p[:n] == cached for p in window_prompts):
                self.lib.llama_memory_seq_cp(self.memory, self._slot, 0, -1, -1)
                self.prefix_cache_stats['state_tokens_reused'] = n
                return n
        if not self.header_cache:
            return 0
        head = self._header_tokens(row)
        n = len(head)
        if not n or n > limit or not all(p[:n] == head for p in window_prompts):
            return 0
        if self._header_state is None:
            self._decode([(0, head, 0)], shared=True)
            data = self._checkpoint()
            if data is None:                       # this build cannot checkpoint a sequence: decode headers as before
                self.header_cache = False
                self._clear()
                return 0
            self._header_state = data
            self.prefix_cache_stats['header_builds'] = self.prefix_cache_stats.get('header_builds', 0) + 1
        elif self.lib.llama_state_seq_set_data(self.ctx, self._header_state, len(self._header_state), 0) != \
                len(self._header_state):
            self.header_cache, self._header_state = False, None
            self._clear()
            return 0
        self.prefix_cache_stats['header_tokens_reused'] = n
        return n

    def _state_cut(self, row, prompt, base):
        """Token count of this prompt up to the end of its state (two tokens short, in case the state's last token
        merges differently once the state grows), or 0 when the state is too short to be worth a checkpoint."""
        if self._slot is None or len(prompt) - base < STATE_CACHE_MIN:
            return 0
        text = prompt_for(row, self.tokenizer)
        at = text.find(STATE_END)
        if at < 0:
            return 0
        head = self.tokenize(text[:at])
        cut = len(head) - 2
        if cut - base < STATE_CACHE_MIN or cut >= len(prompt) - 1 or prompt[:cut] != head[:cut]:
            return 0
        return cut

    def _remember_state(self, tokens):
        """Copy sequence 0 (holding exactly `tokens`) into the state-cache sequence."""
        self.lib.llama_memory_seq_rm(self.memory, self._slot, -1, -1)
        self.lib.llama_memory_seq_cp(self.memory, 0, self._slot, -1, -1)
        self._slot_tokens = list(tokens)
        self.prefix_cache_stats['state_saved_tokens'] = len(tokens)

    def _logits_at(self, index, ids):
        row = self.lib.llama_get_logits_ith(self.ctx, index)
        if not row:
            raise RuntimeError('llama.cpp produced no logits for the readout position')
        return [float(row[i]) for i in ids]

    def _distribution(self, logits, kind=None):
        kind = kind.lower() if kind else None
        temperature = getattr(self, 'temperatures', {}).get(kind, self.temperature)
        return distribution(logits, temperature, getattr(self, 'noul_logit_bias', 0.) if kind == 'noul' else 0.)

    def clear_prefix_cache(self):
        """Release the single in-memory checkpoint at session teardown."""
        with self._inference_lock:
            self._prefix_checkpoint = None

    def _save_prefix(self, key):
        # The non-ext API uses flags=0: hybrid KV AND recurrent R/S tensors.
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

    def judge(self, rows, *, share_prefix=False, session_id=None, prefix_length=None):
        """Return typed distributions, optionally reusing an explicit session prefix.

        A session retains one complete KV/recurrent checkpoint in host memory.
        `prefix_length` counts rendered runtime tokens, must match every row, and
        must leave at least one readout token. Omit it to use the common prefix
        (all but the final token for one row). Changed prefixes are recomputed,
        never trimmed out of a later recurrent state. Session IDs must be unique
        per client session; call clear_prefix_cache at teardown. No disk cache.
        """
        with self._inference_lock:
            return self._judge(rows, share_prefix=share_prefix, session_id=session_id,
                               prefix_length=prefix_length)

    def _judge(self, rows, *, share_prefix=False, session_id=None, prefix_length=None):
        self.prefix_cache_stats = {'restores': 0, 'restored_prefix_tokens': 0,
                                   'prefix_tokens_decoded': 0, 'checkpoint_bytes': 0}
        if session_id is not None and (not isinstance(session_id, str) or not session_id):
            raise ValueError('session_id must be a nonempty string')
        if prefix_length is not None and (session_id is None or type(prefix_length) is not int or prefix_length < 1):
            raise ValueError('prefix_length requires a session_id and a positive integer')
        if not rows:
            return []
        prompts = [self.tokenize(prompt_for(row, self.tokenizer)) for row in rows]
        self.last_prompts = prompts   # the server's usage count reads these instead of tokenizing again
        for tokens in prompts:
            if len(tokens) > min(self.max_length, self.n_ctx_seq):
                raise ValueError('Input exceeds max_length; evidence was not truncated')
        labels = [self.label_ids(row) for row in rows]
        persistent = 0
        cache_key = None
        if session_id is not None:
            available = min(_common_prefix(prompts), min(map(len, prompts)) - 1)
            persistent = available if prefix_length is None else prefix_length
            if persistent > available:
                raise ValueError('Cached prefix must match every row and leave a readout token')
            # Bound to this loaded model/context, tokenizer/template and exact
            # runtime token sequence. No checkpoint crosses judge instances.
            cache_key = (session_id, self.model, self.ctx, id(self.tokenizer),
                         repr(getattr(self.tokenizer, 'chat_template', None)), id(prompt_for),
                         tuple(prompts[0][:persistent]))
            if self._prefix_checkpoint is not None and self._prefix_checkpoint[0] != cache_key:
                self._prefix_checkpoint = None
        # Identical questions with different request IDs have identical inputs.
        # Deduplicate readouts inside this call, independently of prefix caching.
        unique, source, seen = [], [], {}
        for index, row in enumerate(rows):
            key = (tuple(prompts[index]), tuple(labels[index]), row['kind'].lower())
            if key not in seen:
                seen[key] = index
                unique.append(index)
            source.append(seen[key])
        results = {}
        for start in range(0, len(unique), self.n_seq_max):
            window = unique[start:start + self.n_seq_max]
            # data=False resets cell metadata only; zeroing every KV/recurrent buffer (data=True) cost ~600 ms per
            # call and is unnecessary: new sequences re-initialise their recurrent state on first use (identical
            # probabilities to data=True). _header_base clears, then restores the cached header into sequence 0.
            if persistent:
                self.lib.llama_memory_clear(self.memory, False)
                self._slot_tokens = None
                base = 0
            else:
                base = self._restore_base([prompts[i] for i in window], rows[window[0]])
            shared = persistent or (_common_prefix([prompts[i] for i in window])
                                    if share_prefix and len(window) > 1 else 0)
            # Even a complete-prefix match needs a final token to produce logits.
            shared = min(shared, min(len(prompts[i]) for i in window) - 1)
            shared = max(shared, base)   # a restored prefix is a shared prefix too (it leaves a readout token)
            first = window[0]
            cut = 0 if persistent else self._state_cut(rows[first], prompts[first], base)
            if cut and not all(prompts[i][:cut] == prompts[first][:cut] for i in window):
                cut = 0
            shared = max(shared, cut)
            if shared:
                # Decode the shared state once into sequence 0, then copy the whole
                # memory of that sequence - attention and recurrent alike - into
                # each suffix, which continues from position `shared`.
                if persistent and self._prefix_checkpoint is not None:
                    self._restore_prefix()
                    self.prefix_cache_stats['restores'] += 1
                    self.prefix_cache_stats['restored_prefix_tokens'] += shared
                else:
                    if cut:
                        self._decode([(0, prompts[first][base:cut], base)], shared=True)
                        self._remember_state(prompts[first][:cut])
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
                        # A ranged copy is rejected for these buffers; the donor
                        # sequence holds exactly the prefix, so copy all of it.
                        self.lib.llama_memory_seq_cp(self.memory, 0, slot, -1, -1)
            batch = [(slot, prompts[index][shared:], shared) for slot, index in enumerate(window)]
            outputs = self._decode(batch)
            for slot, index in enumerate(window):
                results[index] = self._distribution(self._logits_at(outputs[slot], labels[index]), rows[index]['kind'])
        return [list(results[index]) for index in source]

    def __call__(self, rows, *, share_prefix=False, session_id=None, prefix_length=None):
        return self.judge(rows, share_prefix=share_prefix, session_id=session_id,
                          prefix_length=prefix_length)

    def compare_isolated(self, rows):
        """Largest probability difference between shared-prefix and isolated runs."""
        return _compare(rows, self.judge(rows, share_prefix=False), self.judge(rows, share_prefix=True))

    def verify_isolation(self, rows):
        """Four ways a question could be contaminated by its neighbours.

        A question answered alone is the reference. Answering it beside others,
        answering it in a different position, and answering it over a shared
        prefix must all give the same distribution. ``batched`` isolates the
        floating-point cost of batching itself, so a shared-prefix difference is
        only attributable to the shared state when it exceeds that.
        """
        alone = [self.judge([row])[0] for row in rows]
        order = list(range(len(rows)))[::-1]
        reversed_rows = [rows[i] for i in order]
        reversed_out = self.judge(reversed_rows)
        restored = [None] * len(rows)
        for position, index in enumerate(order):
            restored[index] = reversed_out[position]
        return {
            'batched': _compare(rows, alone, self.judge(rows)),
            'reordered': _compare(rows, alone, restored),
            'shared_prefix': _compare(rows, alone, self.judge(rows, share_prefix=True)),
        }


def _compare(rows, reference, other):
    """Worst per-candidate probability gap and any argmax that moved."""
    worst, where = 0.0, None
    for row, left, right in zip(rows, reference, other):
        for a, b in zip(left, right):
            if abs(a - b) > worst:
                worst, where = abs(a - b), row.get('id')
    return {'max_probability_difference': worst, 'row': where,
            'argmax_changed': sum(1 for a, b in zip(reference, other)
                                  if a.index(max(a)) != b.index(max(b)))}


STATE_END = chr(10) + '</state>' + chr(10)   # rendered prompt: end of the state block (native.prompt_for)
STATE_CACHE_MIN = 256                         # state tokens beyond the restored prefix worth a checkpoint


def _common_prefix(sequences):
    shortest = min(len(tokens) for tokens in sequences)
    for index in range(shortest):
        value = sequences[0][index]
        if any(tokens[index] != value for tokens in sequences):
            return index
    return shortest
