"""Proposer (Module 3, §4 of SPEC.md) — the steering module.

This is the CRUX of the experiment: build_prompt is the literal mechanism that
gives or removes the LLM's prior. PURE module — no network, no async, no engine
logic. Only stdlib (re, ast) + the §1 contracts.

AUDITABILITY (load-bearing, §4): the ONLY textual difference between a "priors"
prompt and an "anon" prompt for the same law+seed must be the variable NAMES and
the domain HINT. The body template is byte-identical across conditions; the
"scientific-discovery vs fit-a-function" framing EMERGES from the names + hint
alone (priors: orbital_period vs orbital_radius + physics context; anon: bare
Sensor_* + no context). This mirrors datasets.py's "identical numbers, only names
differ" and is what makes the measured gap attributable to the prior alone.
"""

import ast
import re

from core.contracts import Dataset

# ---- 1. System prompt (condition-INDEPENDENT; states the fixed §1 contract) ----
SYSTEM_PROMPT = """You are an equation-discovery engine. You propose candidate \
FUNCTIONAL FORMS for a program named `evaluate_law`.

Building blocks available to you: arithmetic (+, -, *, /), powers (x**c), and
transcendental functions (log, exp, sqrt). If you use a transcendental,
`import math` (or numpy) inside the code — e.g. math.log, math.exp, math.sqrt.
All of these are available; choose whatever form fits the data, with no preference
among them.

Rules — follow every one exactly:

1. Propose only the FORM (the structure) of evaluate_law. Leave ALL numeric
   constants as `params[i]` placeholders. Do NOT compute, guess, fit, or hardcode
   any constant value — an external optimizer fits the constants. A literal
   number anywhere a coefficient belongs is a mistake.

2. Honor the FIXED contract. Every candidate MUST define both, with this exact
   signature:
       N_PARAMS = <int>
       def evaluate_law(inputs, params):
           ...
   `inputs` is a tuple of floats — index it (x = inputs[0], y = inputs[1], ...).
   Use params[0], params[1], ... for every constant.

3. Declare N_PARAMS as EXACTLY the number of params[i] slots you use, i.e.
   (highest params index referenced) + 1. A mismatch makes the candidate invalid,
   so keep N_PARAMS and your usage consistent.

4. Return EACH candidate in its OWN ```python fenced code block. Nothing else is
   required between blocks.
"""


# ---- 2. Prompt builder (the ONLY place the prior enters) ----
def build_prompt(dataset: Dataset, exemplars: list[str], data_preview: str) -> str:
    """Build the user prompt for one proposal round.

    Branches on dataset.condition ONLY through the names + hint that get
    interpolated; the surrounding prose is identical across conditions (see the
    module docstring on auditability). `exemplars` empty => early round, push
    DIVERSITY; non-empty => late round, REFINE the leaders.
    """
    in_names = dataset.input_names
    out_name = dataset.output_name
    cols = ", ".join(in_names) + f", {out_name}"

    parts: list[str] = []

    # Task framing — neutral body; the names carry the framing.
    parts.append(
        f"You are given a dataset of rows. Each row lists the inputs "
        f"({', '.join(in_names)}) and the resulting output ({out_name}). "
        f"Propose candidate functional forms for {out_name} as a function of "
        f"the inputs."
    )

    # The hint is the ONLY extra line in "priors"; absent (empty) in "anon".
    if dataset.domain_hint:
        parts.append(f"Context: {dataset.domain_hint}")

    # Data preview (a few train rows). Numbers are identical across conditions;
    # only the column names differ.
    parts.append(f"Data preview (columns: {cols}):\n{data_preview}")

    # Restate the fixed contract so it appears in the user prompt too.
    parts.append(
        "Fixed contract — every candidate must define:\n"
        "  N_PARAMS = <int>\n"
        "  def evaluate_law(inputs, params):\n"
        f"      # inputs is a tuple of {dataset.n_inputs} float(s); index it\n"
        "      ...\n"
        "Leave every numeric constant as params[i]; do not hardcode numbers. "
        "N_PARAMS must equal (highest params index used) + 1."
    )

    # Diversity (early) vs refinement (late) — condition-independent steering.
    if exemplars:
        seeds = "\n".join(f"```python\n{code.strip()}\n```" for code in exemplars)
        parts.append(
            "Current highest-scoring candidate forms (seeds):\n" + seeds + "\n\n"
            "REFINE the leaders: propose small structural variations and "
            "improvements on these. Keep what works; change one thing at a time."
        )
    else:
        parts.append(
            "This is an early round with no seeds yet. Propose DIVERSE, "
            "structurally DIFFERENT candidate forms — explore distinct structures "
            "(powers, products, ratios, sums, and transcendental forms like logs "
            "and exponentials). Do not converge prematurely."
        )

    parts.append(
        "Return each candidate in its own ```python fenced code block."
    )

    return "\n\n".join(parts)


# ---- 3. Parser (drops malformed blocks silently; never raises) ----
_FENCE_RE = re.compile(r"```[ \t]*[\w+\-.]*[ \t]*\r?\n(.*?)```", re.DOTALL)


def _defines_contract(code: str) -> bool:
    """True iff `code` parses AND defines both evaluate_law and N_PARAMS."""
    try:
        tree = ast.parse(code)
    except Exception:
        return False

    has_fn = any(
        isinstance(n, ast.FunctionDef) and n.name == "evaluate_law"
        for n in ast.walk(tree)
    )
    has_np = any(
        (isinstance(n, ast.Assign)
         and any(isinstance(t, ast.Name) and t.id == "N_PARAMS" for t in n.targets))
        or (isinstance(n, ast.AnnAssign)
            and isinstance(n.target, ast.Name) and n.target.id == "N_PARAMS")
        for n in ast.walk(tree)
    )
    return has_fn and has_np


def parse_programs(llm_response: str) -> list[str]:
    """Extract clean candidate code from fenced blocks. Never raises.

    Keeps a block ONLY if it parses and defines both evaluate_law and N_PARAMS;
    silently drops prose / syntax-error / contract-less blocks.
    """
    if not llm_response:
        return []
    out: list[str] = []
    for block in _FENCE_RE.findall(llm_response):
        code = block.strip()
        if code and _defines_contract(code):
            out.append(code)
    return out
