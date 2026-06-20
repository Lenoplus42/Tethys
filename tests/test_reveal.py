"""Reveal unit tests (pure-logic, NO network)."""

from core.contracts import RunLog
from core.laws import NEWTON, ROCKET
from core.reveal import format_equation, reveal, simplify_discovered

# A bloated-but-equivalent best_code (the kind anon produces): full quadratic basis
# + a guarded ratio term. With all coeffs ~0 except the ratio term ~1, it reduces
# to a*b/c**2 — exactly the Newton true form (modulo the +1e-8 divide-guard).
BLOATED_NEWTON = (
    "N_PARAMS = 7\n"
    "def evaluate_law(inputs, params):\n"
    "    x = inputs[0]\n"
    "    y = inputs[1]\n"
    "    z = inputs[2]\n"
    "    return (params[0]*x**2 + params[1]*y**2 + params[2]*z**2 + params[3]*x*y\n"
    "            + params[4]*y*z + params[5]*x*z + params[6]*(x*y/(z**2 + 1e-8)))\n"
)
# fitted constants: six negligible, the ratio coefficient ~1
BLOATED_PARAMS = (4.68e-10, -7.03e-11, -1.47e-10, -1.21e-9, 6.59e-10, -8.46e-11, 1.0000000086)


def test_reveal_simplifies_bloated_superset_to_true_law():
    substituted, simplified, true_expr, matches = simplify_discovered(
        BLOATED_NEWTON, BLOATED_PARAMS, NEWTON)
    # [3] folds to the true form after dropping the +1e-8 divide-guard
    # (matches uses mathematical equality, not sympy's structural ==)
    assert matches is True
    # [2] keeps the divide-guard visible (honesty): substituted still differs from true
    assert substituted != true_expr


def test_clean_form_also_matches():
    # a clean ~4-param power form must of course match too
    clean = (
        "N_PARAMS = 4\n"
        "def evaluate_law(inputs, params):\n"
        "    a = inputs[0]\n"
        "    b = inputs[1]\n"
        "    c = inputs[2]\n"
        "    return params[0] * a**params[1] * b**params[2] / c**params[3]\n"
    )
    params = (1.0, 1.0, 1.0, 2.0)   # c0=1, exponents 1,1,2 -> a*b/c**2
    _, simplified, true_expr, matches = simplify_discovered(clean, params, NEWTON)
    assert matches is True


def test_reveal_handles_transcendental_log_forms():
    # the symbolic reveal must run on log forms (math.log / bare log) without
    # crashing on math.log(symbol), and fold to the Tsiolkovsky truth.
    mathlog = "N_PARAMS=1\ndef evaluate_law(inputs, params):\n    v, r = inputs\n    return params[0]*v*math.log(r)\n"
    _, simplified, _, matches = simplify_discovered(mathlog, (1.0,), ROCKET)
    assert matches is True
    # the readable one-line formatter works for math.log AND bare log (no import)
    bare = "N_PARAMS=1\ndef evaluate_law(inputs, params):\n    v, r = inputs\n    return params[0]*v*log(r)\n"
    eq = format_equation(bare, (1.0,), ROCKET, simplify=True, with_matches=True)
    assert "matches_true: True" in eq and "log(mass_ratio)" in eq


def test_reveal_from_runlog_offline_uses_stored_params():
    # reveal(runlog) must work from stored fitted_params alone — no dataset/LLM
    log = RunLog(
        law_name="newton", condition="priors", budget=300, seed=0,
        best_test_error=0.0, error_trace=[(8, 0.0)],
        best_code=BLOATED_NEWTON,
        fitted_params=BLOATED_PARAMS, true_law_str=NEWTON.true_law_str,
    )
    assert reveal(log) is True     # prints the 3 forms; folds to a*b/c**2, matches=True
