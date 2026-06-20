"""Module 2 acceptance test (§4 of SPEC.md)."""

import numpy as np

from contracts import Dataset
from datasets import make_dataset
from evaluator import (
    EPS,
    combine_score,
    description_length,
    evaluate_fit,
    fit_params,
    score_program,
)
from laws import KEPLER

TRUE_KEPLER = "N_PARAMS = 2\ndef evaluate_law(inputs, params):\n    x = inputs[0]\n    return params[0] * x ** params[1]\n"


def test_true_kepler_recovers_form():
    ds = make_dataset(KEPLER, "anon", 0)
    res = score_program(TRUE_KEPLER, ds)
    assert res.valid
    assert res.test_error < EPS            # generalizes to machine precision
    assert res.test_error < 1e-12
    assert abs(res.fitted_params[1] - 1.5) < 1e-6   # recovered the exponent
    assert abs(res.fitted_params[0] - 1.0) < 1e-6   # and the constant


def _build_overfit_program(train_rows) -> str:
    """A ~100-line lookup table: memorizes every TRAIN point exactly, returns a
    fitted constant otherwise -> near-zero TRAIN error but useless on TEST."""
    lines = [
        "N_PARAMS = 1",
        "def evaluate_law(inputs, params):",
        "    x = inputs[0]",
    ]
    for x, y in [(r[0], r[1]) for r in train_rows]:
        lines.append(f"    if abs(x - {x!r}) < 1e-9:")
        lines.append(f"        return {y!r}")
    lines.append("    return params[0]")
    return "\n".join(lines) + "\n"


def test_overfit_loses_to_true_kepler():
    ds = make_dataset(KEPLER, "anon", 0)
    overfit_code = _build_overfit_program(ds.train)

    true_res = score_program(TRUE_KEPLER, ds)
    over_res = score_program(overfit_code, ds)

    # the lookup table fits TRAIN essentially perfectly...
    assert over_res.train_error < 1e-9
    # ...yet generalizes terribly and is enormous -> strictly worse score.
    assert over_res.test_error > true_res.test_error
    assert over_res.score < true_res.score
    assert description_length(overfit_code) > description_length(TRUE_KEPLER)


def test_infinite_loop_does_not_hang_or_raise():
    ds = make_dataset(KEPLER, "anon", 0)
    looping = (
        "N_PARAMS = 1\n"
        "def evaluate_law(inputs, params):\n"
        "    while True:\n"
        "        pass\n"
        "    return params[0]\n"
    )
    res = score_program(looping, ds)        # must return, not hang
    assert res.valid is False


def test_raising_program_is_invalid():
    ds = make_dataset(KEPLER, "anon", 0)
    bad = (
        "N_PARAMS = 1\n"
        "def evaluate_law(inputs, params):\n"
        "    raise RuntimeError('boom')\n"
    )
    res = score_program(bad, ds)
    assert res.valid is False
    assert res.test_error == float("inf")


def _synthetic_dataset(scale: float) -> Dataset:
    """Outputs = 2*scale*x over x in [1,10]; lets us probe NMSE scale-invariance."""
    rng = np.random.default_rng(0)
    x = rng.uniform(1.0, 10.0, size=60)
    y = 2.0 * scale * x
    rows = [(float(xi), float(yi)) for xi, yi in zip(x, y)]
    return Dataset(
        law_name=f"scale_{scale:g}",
        condition="anon",
        n_inputs=1,
        input_names=["Sensor_X"],
        output_name="Sensor_Y",
        domain_hint="",
        train=rows[:40],
        test=rows[40:],
        true_law_fn=lambda inputs, s=scale: 2.0 * s * inputs[0],
        true_law_str="y = 2*s*x",
    )


def test_nmse_is_scale_invariant():
    # Same 1% relative error on a small-output law and a Newton-scale law must
    # yield test_error on the SAME order (raw MSE would differ by ~1e12).
    prog = "N_PARAMS = 1\ndef evaluate_law(inputs, params):\n    return params[0] * inputs[0]\n"
    ns = {}
    exec(prog, ns)

    small = _synthetic_dataset(1.0)         # outputs O(10)
    large = _synthetic_dataset(1e6)         # outputs O(1e7), Newton-and-beyond scale

    # impose an identical +1% relative bias on each via the slope param
    _, err_small = evaluate_fit(ns, (2.0 * 1.0 * 1.01,), small)
    _, err_large = evaluate_fit(ns, (2.0 * 1e6 * 1.01,), large)

    assert err_small > 0 and err_large > 0
    assert abs(err_small - err_large) / err_small < 1e-6   # identical, scale-free


def test_combine_score_lexicographic():
    # any sub-EPS program beats any supra-EPS program...
    assert combine_score(1e-9, 999) > combine_score(1e-3, 1)
    # ...and within sub-EPS, shorter code wins...
    assert combine_score(1e-9, 5) > combine_score(1e-9, 50)
    # ...and within supra-EPS, smaller error wins.
    assert combine_score(1e-3, 5) > combine_score(1e-1, 5)
