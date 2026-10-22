from __future__ import annotations
import argparse
import ctypes
import json
import math
import sys
from ctypes import c_char_p
from ctypes import c_void_p
from dataclasses import dataclass
from pathlib import Path
from mdedz7rql7m75mxyyy0ec import NOUL_LABELS, OPTION_LABELS

def candidate_label_rows():
    return ({'kind': 'noul', 'candidates': [None, None]}, {'kind': 'choice', 'candidates': [None] * len(OPTION_LABELS)})

def candidate_row_ids(label_ids):
    out = []
    for row in candidate_label_rows():
        for token in label_ids(row):
            if token not in out:
                out.append(int(token))
    return tuple(out)

def merge_probe_rows(probe_ids, candidate_rows):
    rows = [int(k) for k in probe_ids]
    columns = {}
    for position, token in enumerate(rows):
        columns.setdefault(token, position)
    for token in candidate_rows:
        if token not in columns:
            columns[token] = len(rows)
            rows.append(int(token))
    return (tuple(rows), columns)

def select_columns(vector, columns, ids):
    try:
        return [float(vector[columns[int(token)]]) for token in ids]
    except KeyError as missing:
        raise ValueError(f'label id {missing} is not among the candidate head rows') from None
OUTPUT_TABLE_PATTERN = '^(token_embd|output)\\.weight$'

class llama_model_tensor_buft_override(ctypes.Structure):
    _fields_ = [('pattern', c_char_p), ('buft', c_void_p)]

def _ggml_base_name():
    if sys.platform == 'win32':
        return 'ggml-base.dll'
    return 'libggml-base.dylib' if sys.platform == 'darwin' else 'libggml-base.so'

def cpu_buffer_type(runtime_dir):
    base = ctypes.CDLL(str(Path(runtime_dir) / _ggml_base_name()))
    base.ggml_backend_cpu_buffer_type.restype = c_void_p
    base.ggml_backend_cpu_buffer_type.argtypes = []
    buft = base.ggml_backend_cpu_buffer_type()
    if not buft:
        raise RuntimeError('ggml_backend_cpu_buffer_type returned NULL')
    return buft

def output_table_overrides(buft, pattern=OUTPUT_TABLE_PATTERN):
    array = (llama_model_tensor_buft_override * 2)()
    array[0].pattern = pattern.encode('ascii')
    array[0].buft = buft
    array[1].pattern = None
    array[1].buft = None
    return array