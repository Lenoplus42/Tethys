"""Module 6 pure-logic tests (NO network)."""

import pytest

from core.ablation_experiment import (
    compute_prior_price,
    first_crossing,
    load_runlog,
    plot_ablation,
    save_runlog,
)
from core.contracts import RunLog


def _mk(condition, seed, error_trace, token_trace):
    return RunLog(
        law_name="newton", condition=condition, budget=300, seed=seed,
        best_test_error=error_trace[-1][1], error_trace=error_trace,
        best_code="N_PARAMS=1\ndef evaluate_law(inputs, params):\n    return params[0]",
        total_prompt_tokens=0,
        total_completion_tokens=(token_trace[-1][1] if token_trace else 0),
        token_trace=token_trace,
    )


# anon crosses E_TARGET (1e-4) at evaluated=50, then KEEPS going to 90 (post-Occam)
ANON_ERR = [(10, 1.0), (50, 1e-4), (90, 1e-12)]
ANON_TOK = [(10, 100), (50, 500), (90, 900)]
# priors already below at its first recorded point (evaluated=8)
PRI_ERR = [(8, 1e-4), (20, 1e-9)]
PRI_TOK = [(8, 80), (20, 200)]


def test_prior_price_uses_first_crossing_not_converged():
    logs = {"anon": [_mk("anon", 0, ANON_ERR, ANON_TOK)],
            "priors": [_mk("priors", 0, PRI_ERR, PRI_TOK)]}
    res = compute_prior_price(logs, e_target=1e-4)
    e = res["per_seed"][0]
    # FIRST crossing of anon is 50, NOT the converged count 90
    assert e["B_anon"] == pytest.approx(50.0)
    assert e["B_priors"] == pytest.approx(8.0)
    assert e["price"] == pytest.approx(50.0 / 8.0)
    assert res["prior_price"]["mean"] == pytest.approx(6.25)


def test_token_price_path():
    logs = {"anon": [_mk("anon", 0, ANON_ERR, ANON_TOK)],
            "priors": [_mk("priors", 0, PRI_ERR, PRI_TOK)]}
    res = compute_prior_price(logs, e_target=1e-4)
    e = res["per_seed"][0]
    assert e["B_anon_tokens"] == pytest.approx(500.0)     # tokens at first crossing
    assert e["B_priors_tokens"] == pytest.approx(80.0)
    assert e["price_tokens"] == pytest.approx(6.25)
    assert res["prior_price_tokens"]["mean"] == pytest.approx(6.25)


def test_crossing_interpolates_straddle():
    err = [(10, 1e-2), (20, 1e-6)]    # straddles 1e-4 between the two points
    tok = [(10, 100), (20, 300)]
    bp, bt = first_crossing(err, tok, 1e-4)
    frac = (1e-2 - 1e-4) / (1e-2 - 1e-6)
    assert bp == pytest.approx(10 + frac * 10)
    assert bt == pytest.approx(100 + frac * 200)


def test_no_crossing_returns_none():
    bp, bt = first_crossing([(10, 1.0), (20, 1e-2)], [(10, 100), (20, 200)], 1e-4)
    assert bp is None and bt is None
    logs = {"anon": [_mk("anon", 0, [(10, 1.0), (20, 1e-2)], [(10, 100), (20, 200)])],
            "priors": [_mk("priors", 0, PRI_ERR, PRI_TOK)]}
    res = compute_prior_price(logs, e_target=1e-4)
    assert res["per_seed"][0]["price"] is None     # anon never reached target
    assert res["prior_price"]["n"] == 0


def test_multi_seed_aggregation():
    logs = {
        "anon": [_mk("anon", 0, ANON_ERR, ANON_TOK), _mk("anon", 1, ANON_ERR, ANON_TOK)],
        "priors": [_mk("priors", 0, PRI_ERR, PRI_TOK), _mk("priors", 1, PRI_ERR, PRI_TOK)],
    }
    res = compute_prior_price(logs, e_target=1e-4)
    assert res["prior_price"]["n"] == 2
    assert res["prior_price"]["mean"] == pytest.approx(6.25)
    assert res["prior_price"]["std"] == pytest.approx(0.0)


def test_plot_ablation_writes_png(tmp_path):
    logs = {
        "anon": [_mk("anon", 0, ANON_ERR, ANON_TOK), _mk("anon", 1, ANON_ERR, ANON_TOK)],
        "priors": [_mk("priors", 0, PRI_ERR, PRI_TOK), _mk("priors", 1, PRI_ERR, PRI_TOK)],
    }
    price = compute_prior_price(logs, e_target=1e-4)
    out = tmp_path / "ablation.png"
    plot_ablation(logs, out, e_target=1e-4, law_name="newton", price=price)
    assert out.exists() and out.stat().st_size > 0


def test_runlog_serialization_roundtrip(tmp_path):
    log = _mk("anon", 0, ANON_ERR, ANON_TOK)
    path = tmp_path / "rl.json"
    save_runlog(log, path)
    back = load_runlog(path)
    assert (back.law_name, back.condition, back.seed) == ("newton", "anon", 0)
    assert [tuple(p) for p in back.error_trace] == [tuple(p) for p in log.error_trace]
    assert back.best_code == log.best_code
    # loaded (list-based) traces still feed the price computation
    bp, _ = first_crossing(back.error_trace, back.token_trace, 1e-4)
    assert bp == pytest.approx(50.0)
