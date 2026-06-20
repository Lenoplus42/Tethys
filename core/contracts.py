"""The seam (§1 of SPEC.md) — LOCKED contracts between logic core and plumbing.

Boundary between the logic core (laws, datasets, scoring, proposer, ablation
analysis) and the plumbing (async transport, sandbox, program DB, dashboard).
"""

from dataclasses import dataclass, field
from typing import Callable, Literal

Condition = Literal["priors", "anon"]   # the experiment's independent variable


# ---- Contract 1a: the law (condition-independent ground truth) ----
@dataclass
class LawSpec:
    name: str                       # "kepler"
    n_inputs: int
    true_law_fn: Callable           # ground truth f(inputs)->output. NEVER shown to the LLM.
    true_law_str: str               # human-readable, for reveal + distance-to-truth
    target_form_hint: str           # the form curve_fit must recover, e.g. "c0 * x**c1"
    n_params: int                   # free constants the true form needs
    # semantic (WITH-PRIORS) framing — engages the LLM's pretrained knowledge:
    semantic_inputs: list[str]      # ["orbital_radius"]
    semantic_output: str            # "orbital_period"
    domain_hint: str                # short physics context shown ONLY in "priors" condition
    # anonymized (PRIOR-STRIPPED) framing:
    anon_inputs: list[str]          # ["Sensor_X"]
    anon_output: str                # "Sensor_Y"
    noise_std: float = 0.0          # >0 only for robustness law


# ---- Contract 1b: a concrete dataset = a law rendered under one condition ----
@dataclass
class Dataset:
    law_name: str
    condition: Condition
    n_inputs: int
    input_names: list[str]          # semantic OR anon, per condition
    output_name: str
    domain_hint: str                # the hint string for "priors"; "" for "anon"
    train: list[tuple]              # rows = (in_0,...,in_{n-1}, out). SHOWN to proposer.
    test:  list[tuple]              # HELD OUT — scoring only, never in prompt.
    true_law_fn: Callable           # for scoring distance-to-truth; NEVER in prompt
    true_law_str: str
    noise_std: float = 0.0
    meta: dict = field(default_factory=dict)


# CRITICAL FOR A CLEAN ABLATION: for a given law+seed, the underlying NUMERIC DATA is IDENTICAL
# across conditions. Only input_names / output_name / domain_hint differ. The semantic framing is
# the SOLE independent variable, so the gap is attributable to prior knowledge alone.


# ---- Contract 3: scoring ----
@dataclass
class ScoreResult:
    valid: bool
    train_error: float
    test_error: float               # THE signal: generalization on held-out test
    length: int                     # AST node count (Occam)
    fitted_params: tuple
    score: float                    # higher = better
    note: str


# ---- Contract 4: one logged run at a fixed compute budget ----
@dataclass
class RunLog:
    law_name: str
    condition: Condition
    budget: int                     # total programs evaluated (the compute x-axis)
    seed: int
    best_test_error: float          # at end of budget
    error_trace: list[tuple]        # [(programs_evaluated, best_test_error_so_far), ...]
    best_code: str
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    token_trace: list[tuple] = field(default_factory=list)  # (programs_evaluated, cumulative_completion_tokens)
    # self-describing record so a saved RunLog can redraw curves + reveal the law
    # WITHOUT re-running the engine or calling the LLM:
    fitted_params: tuple = ()       # constants fitted to best_code (for the symbolic reveal)
    true_law_str: str = ""          # ground-truth law string (e.g. "Y = G*A*B/C**2")
