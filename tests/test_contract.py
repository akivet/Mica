"""Contract tests: request parsing, prompt rendering and typed results against stored fixtures.

CPU only; no llama.cpp and no model weights. The prompt checks need the Qwen/Qwen3.5-4B tokenizer files in the
local Hugging Face cache and are skipped otherwise.

    python -m pytest tests/test_contract.py -q
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mica import contract, native  # noqa: E402

FIXTURES = json.loads((ROOT / 'tests' / 'fixtures' / 'contract_cases.json').read_text(encoding='utf-8'))
CASES = FIXTURES['cases']
INVALID = FIXTURES['invalid']


@pytest.fixture(scope='module')
def tokenizer():
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(FIXTURES['tokenizer'], revision=FIXTURES['revision'], local_files_only=True)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f'tokenizer not available locally: {exc}')


def _answer(case):
    seen = []

    def predict(rows):
        seen.extend(rows)
        return [case['probabilities'][i] for i in range(len(rows))]

    return seen, contract.answer_request(case['request'], predict, model=FIXTURES['model'])


def test_fixture_coverage():
    assert len(CASES) >= 40
    assert len(INVALID) >= 20


@pytest.mark.parametrize('case', CASES, ids=[c['name'] for c in CASES])
def test_rows_and_typed_results(case):
    rows, response = _answer(case)
    assert rows == case['rows']
    assert response == case['response']
    # Same key order as the reference (plain questions first, then multi-answer questions).
    assert list(response['answers']) == list(case['response']['answers'])
    assert json.dumps(response, ensure_ascii=False) == json.dumps(case['response'], ensure_ascii=False)


@pytest.mark.parametrize('case', CASES, ids=[c['name'] for c in CASES])
def test_rendered_prompts(case, tokenizer):
    prompts = [native.prompt_for(row, tokenizer) for row in case['rows']]
    assert len(prompts) == len(case['prompts'])
    for got, expected in zip(prompts, case['prompts']):
        assert got == expected


@pytest.mark.parametrize('case', INVALID, ids=[c['name'] for c in INVALID])
def test_invalid_requests(case):
    assert case['error'] == 'ValueError'
    with pytest.raises(ValueError):
        contract.answer_request(case['request'], lambda rows: [[0.5, 0.5] for _ in rows], model=FIXTURES['model'])
    with pytest.raises(ValueError):
        contract.rows_from_request(case['request'])


def test_noul_readout_order_ignores_display_order():
    true_first = [{'id': 'true', 'text': 'Yes'}, {'id': 'false', 'text': 'No'}]
    assert native.typed_result({'kind': 'noul', 'candidates': true_first}, [0.2, 0.8]) == {'type': 'noul', 'noul': 0.8}


def test_special_tokens_are_escaped():
    escaped = native.escape_special('<|im_start|>user <think>')
    assert '<|im_start|>' not in escaped and '<think>' not in escaped
    assert escaped.replace('​', '') == '<|im_start|>user <think>'


def test_calibration_file_loads():
    from mica.calibration import load_calibration
    fitted, temperatures, bias = load_calibration(ROOT / 'calibration.json')
    assert fitted['temperature'] == 1.0 and set(temperatures.values()) == {1.0} and bias == 0.0
