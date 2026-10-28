"""Over-length input is refused with HTTP 422 (input_too_long) and never truncated; other bad requests stay 400."""
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from mica import direct_backend as db
from mica import server

BODY = {'state': 'Payment API: card charged twice for order 8812.',
        'questions': {'twice': {'type': 'noul', 'instructions': 'Was the customer charged more than once?'}}}


class Reached(Exception):
    pass


def judge_stub(n_tokens, max_length=8192, n_ctx_seq=8192):
    judge = object.__new__(db.DirectJudge)
    judge._inference_lock = threading.Lock()
    judge.tokenizer = None
    judge.max_length, judge.n_ctx_seq = max_length, n_ctx_seq
    judge.tokenize = lambda prompt: [0] * n_tokens

    def past_guard(row):
        raise Reached
    judge.label_ids = past_guard
    return judge


@pytest.fixture
def plain_prompts(monkeypatch):
    monkeypatch.setattr(db, 'prompt_for', lambda row, tokenizer: '')


def test_guard_boundary(plain_prompts):
    with pytest.raises(Reached):
        judge_stub(8192).judge([{'id': 'q'}])
    with pytest.raises(db.InputTooLongError):
        judge_stub(8193).judge([{'id': 'q'}])


def test_guard_uses_the_smaller_limit(plain_prompts):
    with pytest.raises(db.InputTooLongError):
        judge_stub(4097, max_length=8192, n_ctx_seq=4096).judge([{'id': 'q'}])


def test_too_long_is_still_a_value_error():
    assert issubclass(db.InputTooLongError, ValueError)


class StubJudge:
    def __init__(self, exc):
        self.exc = exc

    def judge(self, rows, share_prefix=False):
        if self.exc:
            raise self.exc
        raise AssertionError('not reached')


def post(judge, body):
    server.JUDGE, server.MODEL_NAME = judge, 'mica-test'
    httpd = ThreadingHTTPServer(('127.0.0.1', 0), server.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        req = urllib.request.Request(f'http://127.0.0.1:{httpd.server_port}/v1/systemone',
                                     data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_server_refuses_over_length_with_422():
    code, obj = post(StubJudge(db.InputTooLongError('Input exceeds max_length; evidence was not truncated')), BODY)
    assert code == 422
    assert obj == {'error': {'type': 'question_error', 'code': 'input_too_long',
                             'message': 'Input exceeds max_length; evidence was not truncated'}}
    assert 'answers' not in obj


def test_other_value_errors_stay_400():
    code, obj = post(StubJudge(ValueError('Label is not one shared token in this runtime')), BODY)
    assert code == 400 and 'shared token' in obj['error']
    code, _ = post(StubJudge(None), {'state': 'x', 'questions': {}})
    assert code == 400

