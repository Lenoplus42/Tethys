"""Evaluator (Module 2, §4 of SPEC.md) — the Occam scoring core.

Pure functions, no concurrency, no engine logic. numpy + scipy only.

A candidate program is a STRING that defines, at minimum, the FIXED symbols of
SPEC §1 Contract 2:

    N_PARAMS = <int>
    def evaluate_law(inputs, params):   # inputs: tuple[float,...] -> float

We exec the string in a fresh namespace and pull those two symbols out.

Scoring decisions (§4 Module 2):
  * Constants are fit with scipy.optimize.curve_fit on TRAIN only.
  * Errors are NORMALIZED MSE (NMSE = mean((pred-true)**2) / var(true)). This
    makes the headline signal comparable across laws of wildly different output
    scale (Kepler ~O(10) vs Newton ~O(100)), which is load-bearing for the
    cross-law PRIOR_PRICE comparison (§2.2). Raw MSE would NOT be comparable.
  * Selection is LEXICOGRAPHIC, not weighted: among programs that already
    generalize (test_error < EPS) we rank purely by shorter code (Occam);
    otherwise we rank by smaller test_error. The headline signal is always
    test_error.
"""

import ast
import math

import numpy as np
from scipy.optimize import curve_fit

from core.contracts import Dataset, ScoreResult

# CONDITION-BLIND execution namespace (scoring is condition-INDEPENDENT — this is
# IDENTICAL for both priors and anon; never branch it on condition). Presetting
# transcendentals lets a candidate run whether it writes `math.log(y)`, `np.log(y)`,
# or a bare `log(y)`, WITH or WITHOUT importing — eliminating the "forgot to import"
# failure mode (not promptable; models forget). `ln` is aliased to natural log for
# the aerospace habit. This changes nothing about WHICH form wins — only whether a
# math-using candidate is runnable at all.
_EXEC_BUILTINS = {
    "math": math, "np": np, "numpy": np,
    "log": math.log, "ln": math.log, "log10": math.log10,
    "exp": math.exp, "sqrt": math.sqrt,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "pi": math.pi, "e": math.e,
}

# Lexicographic generalization threshold. A program with test NMSE below EPS is
# treated as "has found the law"; further competition is purely on simplicity.
# (Distinct from §2.2's E_TARGET=1e-4, which is the gap-metric target used by the
# ablation module, not here.)
EPS = 1e-6

# Variance floor so NMSE is finite for (near-)constant-output splits.
_VAR_EPS = 1e-12

# Score encoding constants (see combine_score docstring).
_SUBEPS_OFFSET = 1e6


def _exec_program(code: str) -> dict:
    """exec a candidate string in a namespace preseeded with math building blocks
    (see _EXEC_BUILTINS). A fresh COPY per call (exec mutates the namespace). The
    namespace is condition-INDEPENDENT — scoring stays condition-blind."""
    ns: dict = dict(_EXEC_BUILTINS)
    exec(code, ns)
    if "evaluate_law" not in ns or "N_PARAMS" not in ns:
        raise ValueError("program missing evaluate_law/N_PARAMS")
    return ns


def _split_arrays(rows: list[tuple]):
    """rows of (in_0,...,in_{n-1}, out) -> (X shape (n_rows, n_inputs), y)."""
    arr = np.asarray(rows, dtype=float)
    return arr[:, :-1], arr[:, -1]


def _predict(evaluate_law, params, X: np.ndarray) -> np.ndarray:
    out = np.empty(X.shape[0], dtype=float)
    for i, row in enumerate(X):
        out[i] = evaluate_law(tuple(row), params)
    return out


def _nmse(pred: np.ndarray, true: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=float)
    if not np.all(np.isfinite(pred)):
        return float("inf")
    var = float(np.var(true))
    mse = float(np.mean((pred - true) ** 2))
    return mse / max(var, _VAR_EPS)


def fit_params(program_ns: dict, dataset: Dataset):
    """Fit N_PARAMS constants on dataset.train ONLY via curve_fit.

    Returns a tuple of fitted params, or None on ANY failure. NEVER raises.
    """
    try:
        evaluate_law = program_ns["evaluate_law"]
        n_params = int(program_ns["N_PARAMS"])
        X, y = _split_arrays(dataset.train)

        # curve_fit calls model(xdata, *params); we route each row through
        # evaluate_law so the program's FIXED signature is honored exactly.
        def model(_xdata, *params):
            return _predict(evaluate_law, params, X)

        p0 = np.ones(n_params, dtype=float)
        popt, _ = curve_fit(model, X, y, p0=p0, maxfev=10000)
        popt = tuple(float(p) for p in np.atleast_1d(popt))

        if len(popt) != n_params or not all(np.isfinite(p) for p in popt):
            return None
        return popt
    except Exception:
        return None


def evaluate_fit(program_ns: dict, params: tuple, dataset: Dataset) -> tuple[float, float]:
    """Return (train_error, test_error) as NMSE on each split."""
    evaluate_law = program_ns["evaluate_law"]

    Xtr, ytr = _split_arrays(dataset.train)
    Xte, yte = _split_arrays(dataset.test)

    train_error = _nmse(_predict(evaluate_law, params, Xtr), ytr)
    test_error = _nmse(_predict(evaluate_law, params, Xte), yte)
    return train_error, test_error


def description_length(code: str) -> int:
    """AST node count of the evaluate_law function BODY (Occam, not gameable).

    Counts every node within the function's body statements — not characters,
    so whitespace/comment padding cannot shrink it. The signature itself is
    excluded so all candidates are measured on equal footing.
    """
    tree = ast.parse(code)
    fn = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.FunctionDef) and n.name == "evaluate_law"),
        None,
    )
    if fn is None:
        raise ValueError("no evaluate_law definition found")
    return sum(1 for stmt in fn.body for _ in ast.walk(stmt))


def combine_score(test_error: float, length: int) -> float:
    """Single float encoding the LEXICOGRAPHIC order (§4 Module 2).

    Two tiers:
      * sub-EPS  (test_error < EPS): the law is found; rank purely by simplicity.
            score = _SUBEPS_OFFSET - length        (shorter code -> higher score)
        Since length is small (<< _SUBEPS_OFFSET=1e6), every sub-EPS score is
        strictly positive.
      * supra-EPS (test_error >= EPS): the law is not yet found; rank by error.
            score = -test_error                    (smaller error -> higher score)
        Every supra-EPS score is <= -EPS < 0.

    Because every sub-EPS score is > 0 and every supra-EPS score is < 0, ANY
    generalizing program outranks ANY non-generalizing one, and ties are broken
    by code length — exactly the lexicographic (test_error, then length) order.
    """
    if test_error < EPS:
        return _SUBEPS_OFFSET - float(length)
    return -float(test_error)


def score_program(code: str, dataset: Dataset) -> ScoreResult:
    """The seam fn the sandbox wraps. Catches everything; never raises. Runaway/timeout
    protection is the sandbox's job — do not call score_program on untrusted code without sandbox."""
    try:
        ns = _exec_program(code)
        length = description_length(code)
        params = fit_params(ns, dataset)
        if params is None:
            return ScoreResult(
                valid=False,
                train_error=float("inf"),
                test_error=float("inf"),
                length=length,
                fitted_params=(),
                score=float("-inf"),
                note="fit failed",
            )
        train_error, test_error = evaluate_fit(ns, params, dataset)
        score = combine_score(test_error, length)
        return ScoreResult(
            valid=True,
            train_error=train_error,
            test_error=test_error,
            length=length,
            fitted_params=params,
            score=score,
            note="ok",
        )
    except Exception as e:  # noqa: BLE001 — by contract, failures become valid=False
        return ScoreResult(
            valid=False,
            train_error=float("inf"),
            test_error=float("inf"),
            length=0,
            fitted_params=(),
            score=float("-inf"),
            note=f"error: {type(e).__name__}",
        )
