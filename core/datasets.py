"""Datasets (Module 1, §4 of SPEC.md) — render a law under one condition.

CRITICAL FOR A CLEAN ABLATION: for a given (law, seed) the underlying numeric
data is generated ONCE, deterministically. The two conditions differ ONLY in
input_names / output_name / domain_hint. The numeric train/test rows are
bitwise-identical across conditions, so the gap is attributable to prior
knowledge alone.
"""

import numpy as np

from core.contracts import Condition, Dataset, LawSpec
from core.laws import TIER0

# Inputs are sampled uniformly in [1, 10] (Newton requirement in §3; also keeps
# all laws' outputs O(1)-O(100) for curve_fit stability and avoids x<=0).
INPUT_LO = 1.0
INPUT_HI = 10.0
N_TRAIN = 80
N_TEST = 40


def _generate_rows(law: LawSpec, seed: int):
    """Generate (train, test) rows ONCE from (law, seed). Condition-independent.

    Disjoint train/test (first N_TRAIN rows vs the rest of one shared sample).
    Gaussian noise (proportional to signal) is added on BOTH train and test only
    when law.noise_std > 0 (robustness law).
    """
    rng = np.random.default_rng(seed)
    n_total = N_TRAIN + N_TEST
    X = rng.uniform(INPUT_LO, INPUT_HI, size=(n_total, law.n_inputs))

    clean = np.array([law.true_law_fn(tuple(x)) for x in X])

    if law.noise_std > 0.0:
        out = clean + rng.normal(0.0, 1.0, size=n_total) * law.noise_std * np.abs(clean)
    else:
        out = clean

    rows = [tuple(float(v) for v in x) + (float(o),) for x, o in zip(X, out)]
    return rows[:N_TRAIN], rows[N_TRAIN:]


def make_dataset(law: LawSpec, condition: Condition, seed: int) -> Dataset:
    """Render `law` under `condition`. Numeric data depends only on (law, seed)."""
    train, test = _generate_rows(law, seed)

    if condition == "priors":
        input_names = list(law.semantic_inputs)
        output_name = law.semantic_output
        domain_hint = law.domain_hint
    elif condition == "anon":
        input_names = list(law.anon_inputs)
        output_name = law.anon_output
        domain_hint = ""  # per Contract 1b: "" for anon — the prior is stripped
    else:
        raise ValueError(f"unknown condition: {condition!r}")

    return Dataset(
        law_name=law.name,
        condition=condition,
        n_inputs=law.n_inputs,
        input_names=input_names,
        output_name=output_name,
        domain_hint=domain_hint,
        train=train,
        test=test,
        true_law_fn=law.true_law_fn,
        true_law_str=law.true_law_str,
        noise_std=law.noise_std,
        meta={"seed": seed},
    )


def make_tier0(seed: int) -> Dataset:
    """Pipeline-validator dataset (§3 Tier 0). Single condition is fine here."""
    return make_dataset(TIER0, "anon", seed)
