"""Module 5 PURE-LOGIC unit tests (§4 of SPEC.md) — NO network.

The live Tier 0 convergence test is marked @pytest.mark.live and is SKIPPED by
the normal sweep (see tests/conftest.py). Run it with: python -m pytest --run-live
"""

import numpy as np
import pytest

from core.contracts import ScoreResult
from core.search import (
    ProgramDB,
    TraceRecorder,
    temperature_schedule,
)

# --- synthetic candidate builders (structurally distinct unless noted) --------
LINEAR = "N_PARAMS=1\ndef evaluate_law(inputs, params):\n    x=inputs[0]\n    return params[0]*x"
AFFINE = "N_PARAMS=2\ndef evaluate_law(inputs, params):\n    x=inputs[0]\n    return params[0]*x + params[1]"
RATIO = "N_PARAMS=1\ndef evaluate_law(inputs, params):\n    x=inputs[0]\n    return params[0]/x"
SQUARE = "N_PARAMS=1\ndef evaluate_law(inputs, params):\n    x=inputs[0]\n    return params[0]*x*x"
CONST = "N_PARAMS=1\ndef evaluate_law(inputs, params):\n    return params[0]"

# same STRUCTURE as power form, differing only in the constant exponent
POW_A = "N_PARAMS=2\ndef evaluate_law(inputs, params):\n    x=inputs[0]\n    return params[0]*x**1.5"
POW_B = "N_PARAMS=2\ndef evaluate_law(inputs, params):\n    x=inputs[0]\n    return params[0]*x**2.7"


def _res(test_error: float, score: float) -> ScoreResult:
    return ScoreResult(valid=True, train_error=test_error, test_error=test_error,
                       length=5, fitted_params=(1.0,), score=score, note="ok")


# --- temperature schedule -----------------------------------------------------
def test_temperature_schedule_endpoints_and_strictly_decreasing():
    assert temperature_schedule(0.0) == pytest.approx(0.8)
    assert temperature_schedule(1.0) == pytest.approx(0.4)
    fracs = np.linspace(0.0, 1.0, 25)
    temps = [temperature_schedule(f) for f in fracs]
    assert all(b < a for a, b in zip(temps, temps[1:]))   # strictly decreasing
    # clamps outside [0,1]
    assert temperature_schedule(-1.0) == pytest.approx(0.8)
    assert temperature_schedule(2.0) == pytest.approx(0.4)


# --- weighted exemplar sampler ------------------------------------------------
def test_sampler_favors_high_score_but_can_return_tail():
    db = ProgramDB()
    # five structurally-distinct programs, descending score (LINEAR best, CONST worst)
    db.insert(LINEAR, _res(0.1, score=10.0))
    db.insert(AFFINE, _res(0.2, score=8.0))
    db.insert(RATIO, _res(0.3, score=6.0))
    db.insert(SQUARE, _res(0.4, score=4.0))
    db.insert(CONST, _res(0.5, score=2.0))
    assert len(db) == 5

    rng = np.random.default_rng(0)
    counts: dict[str, int] = {}
    for _ in range(3000):
        (pick,) = db.sample_exemplars(1, rng)
        counts[pick] = counts.get(pick, 0) + 1

    # high-score item dominates...
    assert max(counts, key=counts.get) == LINEAR
    assert counts[LINEAR] > counts[CONST]
    # ...but the tail (lowest score) is still reachable -> not deterministic top-K
    assert counts.get(CONST, 0) > 0


def test_sampler_empty_pool_returns_empty():
    rng = np.random.default_rng(0)
    assert ProgramDB().sample_exemplars(4, rng) == []


# --- program DB dedup ---------------------------------------------------------
def test_dedup_collapses_constant_only_variants():
    db = ProgramDB()
    db.insert(POW_A, _res(0.3, score=5.0))
    db.insert(POW_B, _res(0.2, score=6.0))   # same structure, only constant differs
    assert len(db) == 1                       # collapsed to one structure
    # the better (lower test_error) representative is kept
    assert db.best()[0] == POW_B


def test_dedup_keeps_structurally_different():
    db = ProgramDB()
    db.insert(LINEAR, _res(0.3, score=5.0))
    db.insert(AFFINE, _res(0.3, score=5.0))   # genuinely different structure
    assert len(db) == 2


def test_invalid_results_are_not_inserted():
    db = ProgramDB()
    bad = ScoreResult(valid=False, train_error=float("inf"), test_error=float("inf"),
                      length=0, fitted_params=(), score=float("-inf"), note="timeout")
    assert db.insert(LINEAR, bad) is False
    assert len(db) == 0


# --- best-so-far tracking -----------------------------------------------------
def test_best_returns_lowest_test_error():
    db = ProgramDB()
    db.insert(LINEAR, _res(0.5, score=2.0))
    db.insert(AFFINE, _res(0.05, score=9.0))   # lowest test_error
    db.insert(RATIO, _res(0.2, score=6.0))
    assert db.best()[0] == AFFINE
    assert db.best_test_error() == pytest.approx(0.05)


# --- trace bookkeeping monotonicity ------------------------------------------
def test_trace_bookkeeping_is_monotonic():
    rec = TraceRecorder()
    # feed rounds; round_best_error is intentionally noisy (one goes UP)
    updates = [(25, 0.5, 10, 100), (25, 0.3, 12, 120),
               (25, 0.4, 9, 90), (25, 0.05, 11, 110)]
    for n, err, p, c in updates:
        rec.update(n, err, p, c)

    xs = [x for x, _ in rec.error_trace]
    errs = [e for _, e in rec.error_trace]
    tok_xs = [x for x, _ in rec.token_trace]
    toks = [t for _, t in rec.token_trace]

    assert xs == [25, 50, 75, 100]                 # x strictly cumulative
    assert all(b <= a for a, b in zip(errs, errs[1:]))   # best_error non-increasing
    assert errs[-1] == pytest.approx(0.05)
    assert xs == tok_xs                             # traces share the x-axis
    assert all(b >= a for a, b in zip(toks, toks[1:]))   # cumulative tokens non-decreasing
    assert rec.total_completion_tokens == 420
    assert rec.total_prompt_tokens == 42


# --- live convergence (SKIPPED in normal sweep) -------------------------------
@pytest.mark.live
def test_run_search_tier0_converges():
    from core.datasets import make_tier0
    from core.search import run_search

    ds = make_tier0(seed=0)
    log = run_search(ds, budget=300, seed=0, batch_size=25)

    assert log.law_name == "tier0"
    assert log.error_trace and log.token_trace
    assert log.best_test_error < 1e-6        # converged within budget
    assert log.best_code
