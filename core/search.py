"""Search (Module 5, §4 of SPEC.md) — the evolutionary loop.

Stitches proposer + sandbox + evaluator + a program DB into one self-improving
search. Reuses those modules; reimplements no scoring/execution/prompts.

ARCHITECTURE — pure logic is split from the live loop so the policy surface is
unit-testable with NO network:
  * Pure, network-free: ProgramDB (insert/dedup/best), the weighted exemplar
    sampler, temperature_schedule, TraceRecorder bookkeeping.
  * Live: run_search wires those to proposer + sandbox + the Prime LLM.

Importing this module performs NO network / env access — `openai` and `dotenv`
are imported lazily inside the live helpers only.
"""

import ast
import time

import numpy as np

from core.contracts import Dataset, RunLog, ScoreResult
from core.evaluator import EPS
from core.proposer import SYSTEM_PROMPT, build_prompt, parse_programs
from core.sandbox import run_batch

MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
BATCH_SIZE = 25
EXEMPLAR_K = 4              # 3-5: sample this many seeds per round
TEMP_HI, TEMP_LO = 0.8, 0.4
BACKOFF_S = 2.0            # brief 429 back-off
MAX_NO_PROGRESS = 8        # consecutive dead rounds (LLM down / 0 candidates) -> stop


# ---------------------------------------------------------------------------
# Pure logic (no network)
# ---------------------------------------------------------------------------
def structure_key(code: str) -> str:
    """Normalized structural key: ast.dump of the evaluate_law body with numeric
    Constants flattened, so forms differing ONLY in constants are the SAME
    structure (mirrors scripts/smoke_proposer.py)."""
    tree = ast.parse(code)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "evaluate_law")

    class _Norm(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, (int, float, complex)):
                return ast.copy_location(ast.Constant(value=0), node)
            return node

    return "\n".join(ast.dump(_Norm().visit(s)) for s in fn.body)


def temperature_schedule(budget_fraction: float) -> float:
    """High early -> low late, linearly in budget fraction consumed. Strictly
    decreasing on [0,1]; endpoints TEMP_HI (0.8) and TEMP_LO (0.4). This is the
    v1 diversity source that replaces islands."""
    f = min(max(budget_fraction, 0.0), 1.0)
    return TEMP_HI - (TEMP_HI - TEMP_LO) * f


class ProgramDB:
    """Single shared pool of (code, ScoreResult), deduped by normalized structure,
    keeping the best (lowest test_error) representative per structure.

    ISLANDS SEAM (out of scope for v1): to add islands, hold a list of these pools
    and route insert()/sample_exemplars() per island id, with periodic reset of a
    stagnant island. v1 is deliberately ONE pool.
    """

    def __init__(self):
        self._by_key: dict[str, tuple[str, ScoreResult]] = {}

    def insert(self, code: str, result: ScoreResult) -> bool:
        """Insert a VALID result. Collapses structural duplicates, keeping the
        lower test_error. Returns True if it became (or replaced) the rep."""
        if not result.valid:
            return False
        try:
            key = structure_key(code)
        except Exception:
            key = "raw::" + code  # unparsable structure -> treat raw code as key
        existing = self._by_key.get(key)
        if existing is None or result.test_error < existing[1].test_error:
            self._by_key[key] = (code, result)
            return True
        return False

    def __len__(self) -> int:
        return len(self._by_key)

    def best(self) -> tuple[str, ScoreResult] | None:
        if not self._by_key:
            return None
        return min(self._by_key.values(), key=lambda cr: cr[1].test_error)

    def best_test_error(self) -> float:
        b = self.best()
        return b[1].test_error if b else float("inf")

    def sample_exemplars(self, k: int, rng: np.random.Generator) -> list[str]:
        """Weighted sampling that FAVORS high score but RETAINS tail exploration
        (not top-K). Rank-weighted: best score gets the largest weight, worst gets
        the smallest but strictly-positive weight, so any item can be drawn."""
        entries = list(self._by_key.values())
        if not entries:
            return []
        n = len(entries)
        k = min(k, n)
        # rank by score descending; weight = (n - rank) so all weights >= 1 > 0.
        order = sorted(range(n), key=lambda i: entries[i][1].score, reverse=True)
        rank = {idx: r for r, idx in enumerate(order)}
        weights = np.array([n - rank[i] for i in range(n)], dtype=float)
        weights /= weights.sum()
        chosen = rng.choice(n, size=k, replace=False, p=weights)
        return [entries[i][0] for i in chosen]


class TraceRecorder:
    """Cumulative bookkeeping for the ablation x-axis. Guarantees error_trace x is
    non-decreasing (cumulative count) and best_error is non-increasing (running
    min), and accumulates token usage."""

    def __init__(self):
        self.error_trace: list[tuple] = []
        self.token_trace: list[tuple] = []
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_evaluated = 0
        self.best_test_error = float("inf")

    def update(self, n_evaluated: int, round_best_error: float,
               prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        self.total_evaluated += int(n_evaluated)
        self.best_test_error = min(self.best_test_error, round_best_error)
        self.total_prompt_tokens += int(prompt_tokens)
        self.total_completion_tokens += int(completion_tokens)
        self.error_trace.append((self.total_evaluated, self.best_test_error))
        self.token_trace.append((self.total_evaluated, self.total_completion_tokens))


def data_preview(dataset: Dataset, n: int = 5) -> str:
    """First few train rows, formatted for the prompt (same shape as the smoke)."""
    return "\n".join(", ".join(f"{v:.4f}" for v in row) for row in dataset.train[:n])


# ---------------------------------------------------------------------------
# Live loop (network) — lazy imports so the module stays import-clean
# ---------------------------------------------------------------------------
def _make_client():
    import os
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    kwargs = dict(api_key=os.environ["PRIME_API_KEY"],
                  base_url="https://api.pinference.ai/api/v1")
    if os.environ.get("PRIME_TEAM_ID"):
        kwargs["default_headers"] = {"X-Prime-Team-ID": os.environ["PRIME_TEAM_ID"]}
    return OpenAI(**kwargs)


def _call_llm(client, prompt: str, temperature: float):
    """ONE LLM round, robust: retry once, brief back-off on 429, never raise.
    Returns (text, prompt_tokens, completion_tokens) or None on persistent failure."""
    from openai import RateLimitError

    for attempt in range(2):  # one try + one retry
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=4096,
            )
            text = resp.choices[0].message.content or ""
            u = resp.usage
            return text, getattr(u, "prompt_tokens", 0), getattr(u, "completion_tokens", 0)
        except RateLimitError:
            time.sleep(BACKOFF_S)
        except Exception:
            time.sleep(0.5)
    return None


def run_search(dataset: Dataset, budget: int, seed: int, batch_size: int = BATCH_SIZE) -> RunLog:
    """Evolutionary search to a GLOBAL budget (total programs evaluated, the
    ablation x-axis). Stops early if best_test_error < EPS. Returns a fully
    populated RunLog. Never crashes on a failing LLM call — logs and continues."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    db = ProgramDB()
    rec = TraceRecorder()
    preview = data_preview(dataset)
    client = _make_client()
    no_progress = 0

    while rec.total_evaluated < budget and rec.best_test_error >= EPS:
        frac = rec.total_evaluated / budget if budget else 1.0
        temperature = temperature_schedule(frac)
        exemplars = db.sample_exemplars(EXEMPLAR_K, rng)
        prompt = build_prompt(dataset, exemplars, preview)

        out = _call_llm(client, prompt, temperature)
        if out is None:
            no_progress += 1
            if no_progress >= MAX_NO_PROGRESS:
                break
            continue
        text, ptok, ctok = out

        candidates = parse_programs(text)
        if not candidates:
            no_progress += 1
            # still bank the token cost of the dead round
            rec.total_prompt_tokens += ptok
            rec.total_completion_tokens += ctok
            if no_progress >= MAX_NO_PROGRESS:
                break
            continue
        no_progress = 0

        results = run_batch(candidates, dataset, max_workers=8)  # never raises, len==input
        for code, r in zip(candidates, results):
            db.insert(code, r)

        rec.update(len(results), db.best_test_error(), ptok, ctok)

    best = db.best()
    return RunLog(
        law_name=dataset.law_name,
        condition=dataset.condition,
        budget=budget,
        seed=seed,
        best_test_error=rec.best_test_error,
        error_trace=rec.error_trace,
        best_code=best[0] if best else "",
        total_prompt_tokens=rec.total_prompt_tokens,
        total_completion_tokens=rec.total_completion_tokens,
        token_trace=rec.token_trace,
    )
