"""Module 4 acceptance test (§4 of SPEC.md). Run from repo root: python -m pytest."""

import time

from core.datasets import make_dataset, make_tier0
from core.laws import KEPLER
from core.sandbox import run_batch

# N_PARAMS=1 (not 0): with 0 params, evaluator's curve_fit fails BEFORE ever
# calling evaluate_law, so the loop never runs and we'd test "fit failed" instead
# of the sandbox timeout path. With >=1 param, curve_fit invokes evaluate_law ->
# the loop actually spins -> the worker hangs -> sandbox's 0.5s timeout must kill
# it (well before evaluator's own 2s SIGALRM net would fire in-process).
INF_LOOP = "N_PARAMS=1\ndef evaluate_law(inputs, params):\n    while True:\n        pass"
SYNTAX_ERR = "def evaluate_law(inputs, params)\n    return params[0]\nN_PARAMS=1"  # missing colon
TRUE_KEPLER = "N_PARAMS = 2\ndef evaluate_law(inputs, params):\n    x = inputs[0]\n    return params[0] * x ** params[1]"


def test_infinite_loop_times_out_no_hang_no_raise():
    ds = make_tier0(0)
    t0 = time.time()
    results = run_batch([INF_LOOP], ds)
    elapsed = time.time() - t0
    assert len(results) == 1
    assert results[0].valid is False
    assert results[0].note == "timeout"
    assert elapsed < 5.0          # killed, not hung


def test_syntax_error_short_circuits():
    ds = make_tier0(0)
    results = run_batch([SYNTAX_ERR], ds)
    assert len(results) == 1
    assert results[0].valid is False
    assert results[0].note == "syntax"


def test_one_bad_candidate_does_not_poison_batch():
    ds = make_dataset(KEPLER, "anon", 0)
    codes = [TRUE_KEPLER, INF_LOOP, SYNTAX_ERR]
    results = run_batch(codes, ds)
    assert len(results) == len(codes)
    # the good one is scored correctly despite its toxic neighbours
    assert results[0].valid is True
    assert results[0].test_error < 1e-6
    assert results[1].note == "timeout"
    assert results[2].note == "syntax"


def test_results_match_input_order_and_length():
    ds = make_tier0(0)
    codes = [TRUE_KEPLER, SYNTAX_ERR, TRUE_KEPLER, INF_LOOP]
    results = run_batch(codes, ds)
    assert len(results) == len(codes)
    assert results[1].note == "syntax"
    assert results[3].note == "timeout"
    assert results[0].valid is True and results[2].valid is True


def test_mixed_batch_of_25_completes_quickly():
    ds = make_tier0(0)
    polys = [
        f"N_PARAMS=2\ndef evaluate_law(inputs, params):\n    x=inputs[0]\n    return params[0]*x**{k}+params[1]"
        for k in range(1, 24)  # 23 distinct valid forms
    ]
    codes = polys + [INF_LOOP, SYNTAX_ERR]   # 25 total, 1 loop + 1 syntax
    t0 = time.time()
    results = run_batch(codes, ds)
    elapsed = time.time() - t0
    assert len(results) == 25
    assert results[-2].note == "timeout"
    assert results[-1].note == "syntax"
    assert sum(r.valid for r in results) >= 20   # the polynomials score fine
    # concurrent, not serial: 23 valid + 1 loop must not take 25*0.5s serially
    assert elapsed < 10.0
