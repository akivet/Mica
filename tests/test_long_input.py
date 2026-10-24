"""Long-input mode bookkeeping in DirectJudge, without llama.cpp: which rows are prefilled in that mode, and when."""
from mica import direct_backend as db


class Calls:
    def __init__(self):
        self.calls = []


def judge_stub(long_input=True):
    judge = object.__new__(db.DirectJudge)
    judge.long_input = long_input
    log = Calls()
    judge._decode = lambda seqs, shared=None: log.calls.extend(('exact', s, t[0], p, len(t)) for s, t, p in seqs)
    judge._decode_long = lambda seq, tokens, start: log.calls.append(('long', seq, tokens[0], start, len(tokens)))
    return judge, log


def test_prefill_splits_at_long_start():
    judge, log = judge_stub()
    tokens = list(range(100))
    judge._prefill(0, tokens, 10, 60, 40)
    assert log.calls == [('long', 0, 10, 10, 30), ('exact', 0, 40, 40, 20)]


def test_prefill_all_exact_or_all_long():
    judge, log = judge_stub()
    tokens = list(range(100))
    judge._prefill(0, tokens, 50, 70, 40)       # starts past long_start
    judge._prefill(0, tokens, 0, 30, 40)        # ends before it
    judge._prefill(0, tokens, 20, 20, 40)       # nothing to decode
    judge._prefill(0, tokens, 0, 30, 0)         # mode off
    assert log.calls == [('exact', 0, 50, 50, 20), ('long', 0, 0, 0, 30), ('exact', 0, 0, 0, 30)]


def test_long_start_needs_a_long_prompt_and_the_mode():
    judge, _ = judge_stub()
    assert judge._long_start({}, [0] * db.LONG_INPUT_MIN) == 0
    judge.long_input = False
    assert judge._long_start({}, [0] * (db.LONG_INPUT_MIN + 500)) == 0
