"""Module 1 acceptance test (§4 of SPEC.md).

- make_dataset(kepler,"priors",0) and (kepler,"anon",0) have bitwise-identical
  train/test numbers and different names.
- true_law_fn reproduces clean test outputs to machine precision.
"""

from core.datasets import make_dataset, make_tier0
from core.laws import KEPLER, NEWTON


def test_identical_numbers_different_names():
    p = make_dataset(KEPLER, "priors", 0)
    a = make_dataset(KEPLER, "anon", 0)

    # bitwise-identical numeric data across conditions (the SOLE difference is framing)
    assert p.train == a.train
    assert p.test == a.test

    # but the names / hint differ — the prior is present in one, stripped in the other
    assert p.input_names != a.input_names
    assert p.output_name != a.output_name
    assert p.input_names == ["orbital_radius"]
    assert a.input_names == ["Sensor_X"]
    assert p.domain_hint == "These are orbital measurements of bodies around a star."
    assert a.domain_hint == ""


def test_true_law_reproduces_test_outputs():
    ds = make_dataset(KEPLER, "anon", 0)
    for row in ds.test:
        inputs, out = row[:-1], row[-1]
        assert ds.true_law_fn(inputs) == out  # machine precision: exact for noiseless law


def test_disjoint_train_test():
    ds = make_dataset(KEPLER, "anon", 0)
    assert set(ds.train).isdisjoint(set(ds.test))


def test_newton_three_inputs_and_range():
    ds = make_dataset(NEWTON, "priors", 0)
    assert ds.n_inputs == 3
    for row in ds.train + ds.test:
        inputs, out = row[:-1], row[-1]
        assert len(inputs) == 3
        assert all(1.0 <= v <= 10.0 for v in inputs)
        assert ds.true_law_fn(inputs) == out


def test_tier0_smoke():
    ds = make_tier0(0)
    assert ds.law_name == "tier0"
    for row in ds.test:
        assert ds.true_law_fn(row[:-1]) == row[-1]
