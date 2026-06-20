"""core/reveal.py — human-readable display of a discovered law.

Honesty-first (the Occam story depends on it): we print THREE forms, never hiding
the raw form behind the pretty one:

  [1] RAW best_code — exactly as the engine selected it, params[i] placeholders intact.
  [2] CONSTANT-SUBSTITUTED — fitted constants filled in, near-zero coefficients and
      near-integer exponents cleaned (so a bloated skeleton shows which terms vanished).
  [3] SYMPY-SIMPLIFIED — additionally drops negligible ADDITIVE literal constants
      (e.g. a "+1e-8" divide-guard, harmless where the data has |denominator| >> 1e-8),
      folded to a clean symbolic law and compared side-by-side with LawSpec.true_law_str.

No network, no engine coupling beyond reusing fit_params for re-fitting. sympy only.

CLI:  python -m core.reveal <path-to-runlog.json>
      (a JSON dict with at least law_name, condition, seed, best_code)
"""

import json
import sys

import sympy as sp

from core.contracts import RunLog
from core.datasets import make_dataset
from core.evaluator import fit_params
from core.laws import KEPLER, NEWTON, ROCKET, TIER0

_LAW_REGISTRY = {law.name: law for law in (TIER0, KEPLER, NEWTON, ROCKET)}

ZERO_TOL = 1e-5       # |fitted coefficient| below this -> treated as 0
ADDITIVE_TOL = 1e-6   # additive literal constant below this (divide-guard) -> dropped
_INPUT_NAMES = ["a", "b", "c", "d", "e"]


def _symbols_tuple(n: int) -> tuple:
    syms = sp.symbols(" ".join(_INPUT_NAMES[: max(n, 1)]), positive=True)
    return syms if isinstance(syms, tuple) else (syms,)


def _clean_params(params) -> tuple:
    """Zero near-zero constants, snap near-integers (handles both coefficients and
    fitted exponents); round the rest for readable display."""
    out = []
    for p in params:
        p = float(p)
        if abs(p) < ZERO_TOL:
            out.append(0)
        elif abs(p - round(p)) < ZERO_TOL:
            out.append(int(round(p)))
        else:
            out.append(round(p, 6))
    return tuple(out)


def _drop_small_additives(expr):
    """Recursively drop additive numeric literals with |value| < ADDITIVE_TOL
    (e.g. the +1e-8 inside (c**2 + 1e-8)). Multiplicative factors are left alone."""
    if expr.is_Add:
        kept = [_drop_small_additives(t) for t in expr.args
                if not (t.is_number and abs(float(t)) < ADDITIVE_TOL)]
        return sp.Add(*kept) if kept else sp.Integer(0)
    if expr.args:
        return expr.func(*[_drop_small_additives(a) for a in expr.args])
    return expr


def simplify_discovered(best_code: str, fitted_params, law):
    """Pure (no print). Returns (substituted_expr, simplified_expr, true_expr, matches)."""
    syms = _symbols_tuple(law.n_inputs)
    ns: dict = {}
    exec(best_code, ns)
    evaluate_law = ns["evaluate_law"]

    cleaned = _clean_params(fitted_params)
    substituted = sp.sympify(evaluate_law(syms, cleaned))        # [2] params filled + cleaned
    simplified = sp.simplify(_drop_small_additives(substituted))  # [3] guards dropped + folded
    true_expr = sp.simplify(sp.sympify(law.true_law_fn(syms)))    # ground truth, symbolically
    matches = sp.simplify(simplified - true_expr) == 0
    return substituted, simplified, true_expr, matches


def reveal_from_parts(best_code: str, fitted_params, law) -> bool:
    """Print all three forms + side-by-side true law. Returns the matches flag."""
    print("=" * 64)
    print(f"DISCOVERED LAW REVEAL  (law={law.name})")
    print("=" * 64)
    print("\n[1] RAW best_code (as the engine selected it — placeholders intact):")
    print(best_code)

    try:
        substituted, simplified, true_expr, matches = simplify_discovered(best_code, fitted_params, law)
    except Exception as e:  # never let a display tool crash a run
        print(f"\n(could not symbolically reveal: {type(e).__name__}: {e})")
        return False

    syms = ", ".join(_INPUT_NAMES[: max(law.n_inputs, 1)])
    print(f"\n[2] CONSTANT-SUBSTITUTED (fitted params filled, near-0/near-int cleaned; inputs = {syms}):")
    print(f"    {substituted}")
    print("\n[3] SYMPY-SIMPLIFIED (negligible additive divide-guards dropped):")
    print(f"    {simplified}")
    print(f"\n    true law (LawSpec.true_law_str): {law.true_law_str}")
    print(f"    true law (symbolic)            : {true_expr}")
    print(f"    matches true law               : {matches}")
    print("=" * 64)
    return matches


def reveal(runlog: RunLog) -> bool:
    """Reveal the law in a RunLog. Prefers the stored fitted_params (so a saved
    RunLog reveals OFFLINE — no dataset rebuild, no LLM); falls back to re-fitting
    on the deterministic dataset only if fitted_params is absent (older logs)."""
    law = _LAW_REGISTRY.get(runlog.law_name)
    if law is None:
        print(f"(unknown law {runlog.law_name!r}; cannot reveal)")
        return False
    if not runlog.best_code:
        print("(no best_code in runlog; nothing to reveal)")
        return False

    params = tuple(runlog.fitted_params) if runlog.fitted_params else None
    if params is None:                      # fallback for pre-schema logs (no LLM, just curve_fit)
        ds = make_dataset(law, runlog.condition, runlog.seed)
        ns: dict = {}
        try:
            exec(runlog.best_code, ns)
        except Exception as e:
            print(f"(best_code exec failed: {e})")
            return False
        params = fit_params(ns, ds)
        if params is None:
            print("(re-fit failed; cannot substitute constants)")
            return False
    return reveal_from_parts(runlog.best_code, params, law)


def _load_runlog(path: str) -> RunLog:
    with open(path) as f:
        d = json.load(f)
    return RunLog(
        law_name=d["law_name"],
        condition=d["condition"],
        budget=d.get("budget", 0),
        seed=d.get("seed", 0),
        best_test_error=d.get("best_test_error", float("inf")),
        error_trace=d.get("error_trace", []),
        best_code=d.get("best_code", ""),
        total_prompt_tokens=d.get("total_prompt_tokens", 0),
        total_completion_tokens=d.get("total_completion_tokens", 0),
        token_trace=d.get("token_trace", []),
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m core.reveal <path-to-runlog.json>")
        raise SystemExit(2)
    reveal(_load_runlog(sys.argv[1]))
