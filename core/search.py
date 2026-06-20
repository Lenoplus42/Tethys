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
import os
import time
from pathlib import Path

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
OCCAM_PATIENCE_ROUNDS = 8  # after first sub-EPS crossing, keep searching up to this many more
                           # rounds so lexicographic selection can find a SHORTER exact form...
OCCAM_STALL_ROUNDS = 3     # ...or stop early once the shortest sub-EPS form hasn't shrunk in this many rounds
REQUEST_TIMEOUT_S = 60.0   # HARD per-request timeout: a silent stall (server never
                           # responds) becomes a failed round, never a 0%-CPU freeze.
PROGRESS_LOG = Path("runlogs/progress.log")  # local cost/health meter — Prime's dashboard is dark


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
        """Lexicographically-best program — the one we REVEAL as best_code. Mirrors
        evaluator.combine_score: once below EPS the SHORTEST form wins (Occam),
        not merely the lowest test_error; supra-EPS programs are ranked by
        test_error and always lose to any sub-EPS one. Higher ScoreResult.score is
        better (score encodes the tiers + length), with lower test_error as the
        intra-tier tiebreak."""
        if not self._by_key:
            return None
        return max(self._by_key.values(), key=lambda cr: (cr[1].score, -cr[1].test_error))

    def min_test_error(self) -> float:
        """Lowest test_error in the pool — the raw DISCOVERY error for the
        error_trace / EPS-crossing check. DECOUPLED from best(): after crossing
        EPS, best() optimizes for shortness while this keeps tracking the curve."""
        if not self._by_key:
            return float("inf")
        return min(cr[1].test_error for cr in self._by_key.values())

    def best_test_error(self) -> float:
        """Alias for min_test_error() — the discovery-curve value (NOT best().test_error)."""
        return self.min_test_error()

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
def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _log_progress(line: str) -> None:
    """Append one flushed line to runlogs/progress.log — our ONLY observability
    (Prime's usage dashboard shows no data). Logging must NEVER crash the run."""
    try:
        PROGRESS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(PROGRESS_LOG, "a") as f:
            f.write(line + "\n")
            f.flush()
    except Exception:
        pass


def _make_client():
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    kwargs = dict(api_key=os.environ["PRIME_API_KEY"],
                  base_url="https://api.pinference.ai/api/v1")
    if os.environ.get("PRIME_TEAM_ID"):
        kwargs["default_headers"] = {"X-Prime-Team-ID": os.environ["PRIME_TEAM_ID"]}
    # HARD per-request timeout; max_retries=0 because _call_llm does our SINGLE
    # retry — stacking the SDK's retries on ours could turn one stall into 6x60s.
    return OpenAI(**kwargs).with_options(timeout=REQUEST_TIMEOUT_S, max_retries=0)


def _call_llm(client, prompt: str, temperature: float):
    """ONE LLM round. Robust: one retry, brief back-off on 429, NEVER raises and
    NEVER blocks forever (client carries a hard per-request timeout). A silent
    stall surfaces as APITimeoutError and is treated as a failed round.

    Returns (payload, reason): payload is (text, prompt_tokens, completion_tokens)
    on success with reason='ok'; otherwise None with reason in
    {'timeout','conn','ratelimit','other'}."""
    from openai import APIConnectionError, APITimeoutError, RateLimitError

    reason = "other"
    for _attempt in range(2):  # one try + one retry; SDK retry is OFF (max_retries=0)
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
            payload = (text, getattr(u, "prompt_tokens", 0), getattr(u, "completion_tokens", 0))
            return payload, "ok"
        except RateLimitError:
            reason = "ratelimit"
            time.sleep(BACKOFF_S)
        except APITimeoutError:        # subclass of APIConnectionError -> catch FIRST
            reason = "timeout"
            time.sleep(0.5)
        except APIConnectionError:
            reason = "conn"
            time.sleep(0.5)
        except Exception:
            reason = "other"
            time.sleep(0.5)
    return None, reason


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
    round_idx = 0
    # Occam-patience convergence state (do NOT stop at first sub-EPS hit):
    crossed = False            # have we crossed EPS yet?
    rounds_since_cross = 0     # rounds elapsed since the first crossing
    occam_stall = 0           # rounds since the shortest sub-EPS form last shrank
    best_sub_len = None        # AST length of the current shortest sub-EPS form

    _log_progress(f"{_now()} START law={dataset.law_name} condition={dataset.condition} "
                  f"budget={budget} seed={seed} batch_size={batch_size}")

    # Budget caps everything; early-stop is handled by the Occam-patience block below.
    while rec.total_evaluated < budget:
        round_idx += 1
        frac = rec.total_evaluated / budget if budget else 1.0
        temperature = temperature_schedule(frac)
        exemplars = db.sample_exemplars(EXEMPLAR_K, rng)
        prompt = build_prompt(dataset, exemplars, preview)

        payload, reason = _call_llm(client, prompt, temperature)
        if payload is None:
            # stall/timeout/conn/etc. -> a failed round, never a freeze. Make it VISIBLE.
            _log_progress(f"{_now()} round={round_idx} LLM_CALL_FAILED reason={reason}")
            no_progress += 1
            if no_progress >= MAX_NO_PROGRESS:
                break
            continue
        text, ptok, ctok = payload

        candidates = parse_programs(text)
        if not candidates:
            no_progress += 1
            # still bank the token cost of the dead round
            rec.total_prompt_tokens += ptok
            rec.total_completion_tokens += ctok
            _log_progress(f"{_now()} round={round_idx} parsed=0 (no valid candidates) "
                          f"ctok={rec.total_completion_tokens} ptok={rec.total_prompt_tokens}")
            if no_progress >= MAX_NO_PROGRESS:
                break
            continue
        no_progress = 0

        results = run_batch(candidates, dataset, max_workers=8)  # never raises, len==input
        for code, r in zip(candidates, results):
            db.insert(code, r)

        rec.update(len(results), db.best_test_error(), ptok, ctok)

        best = db.best()
        snippet = " ".join(best[0].split())[:80] if best else "-"
        _log_progress(f"{_now()} round={round_idx} evald={rec.total_evaluated} "
                      f"best_err={rec.best_test_error:.3e} pool={len(db)} temp={temperature:.3f} "
                      f"ctok={rec.total_completion_tokens} ptok={rec.total_prompt_tokens} "
                      f"best_code='{snippet}'")

        # --- Occam-patience convergence ---------------------------------------
        # Don't freeze on the first (often bloated) sub-EPS program. Once below
        # EPS, keep searching so lexicographic db.best() can find a SHORTER exact
        # form; stop after a patience window OR once shortness stops improving.
        if db.min_test_error() < EPS:
            shortest_len = best[1].length if best else None
            if not crossed:
                crossed = True
                rounds_since_cross = 0
                occam_stall = 0
                best_sub_len = shortest_len
                _log_progress(f"{_now()} round={round_idx} EPS_CROSSED best_len={best_sub_len} "
                              f"-> Occam refinement (patience={OCCAM_PATIENCE_ROUNDS}, stall={OCCAM_STALL_ROUNDS})")
            else:
                rounds_since_cross += 1
                if best_sub_len is None or shortest_len < best_sub_len:
                    best_sub_len = shortest_len
                    occam_stall = 0
                else:
                    occam_stall += 1
                if rounds_since_cross >= OCCAM_PATIENCE_ROUNDS or occam_stall >= OCCAM_STALL_ROUNDS:
                    reason = "patience" if rounds_since_cross >= OCCAM_PATIENCE_ROUNDS else "stall"
                    _log_progress(f"{_now()} round={round_idx} OCCAM_STOP reason={reason} "
                                  f"shortest_len={best_sub_len}")
                    break

    best = db.best()
    _log_progress(f"{_now()} DONE law={dataset.law_name} condition={dataset.condition} "
                  f"rounds={round_idx} evald={rec.total_evaluated} "
                  f"best_err={rec.best_test_error:.3e} converged={rec.best_test_error < EPS} "
                  f"ctok={rec.total_completion_tokens} ptok={rec.total_prompt_tokens}")
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
        fitted_params=tuple(best[1].fitted_params) if best else (),
        true_law_str=dataset.true_law_str,
    )
