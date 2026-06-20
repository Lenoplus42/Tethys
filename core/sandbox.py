"""Sandbox (Module 4, §4 of SPEC.md) — safe concurrent execution layer (plumbing).

WRAPS core.evaluator.score_program; it does NOT reimplement scoring. Its only
jobs are isolation, a hard timeout, a syntax pre-filter, warning hygiene, and
never letting one bad candidate take down a batch.

PROCESS isolation, NOT threads: candidate code may spin in `while True: pass`,
which the GIL makes unkillable in a thread. We run each candidate in a
ProcessPoolExecutor worker so a runaway can be force-killed from outside.

Note on pickling: Dataset.true_law_fn is a lambda (laws.py) and is NOT picklable,
but ProcessPoolExecutor pickles every task argument. score_program only ever
reads dataset.train/test, never true_law_fn, so we ship a lambda-stripped copy
(dataclasses.replace(..., true_law_fn=None)) into the workers.
"""

import ast
import dataclasses
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout

import numpy as np

from core.contracts import ScoreResult
from core.evaluator import score_program

TIMEOUT_S = 0.5          # hard per-candidate wall-clock budget (~500ms, §4)
DEFAULT_WORKERS = 8      # Prime rate-limit unprobed -> stay conservative


def _invalid(note: str) -> ScoreResult:
    """A uniform invalid result so a failure never escapes as an exception."""
    return ScoreResult(
        valid=False,
        train_error=float("inf"),
        test_error=float("inf"),
        length=0,
        fitted_params=(),
        score=float("-inf"),
        note=note,
    )


def _worker(code: str, dataset) -> ScoreResult:
    """Runs in a child process. Suppress numerical warnings (illegal math from
    candidates: neg base ** frac, overflow, /0) so a batch doesn't flood stderr.
    The NaN/inf still flows through evaluator's existing invalidation logic
    untouched — only the console spam is silenced, never the verdict."""
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        return score_program(code, dataset)


def _strip_dataset(dataset):
    """Lambda-free, picklable copy for shipping into worker processes."""
    try:
        return dataclasses.replace(dataset, true_law_fn=None)
    except Exception:
        return dataset


def _force_shutdown(executor: ProcessPoolExecutor) -> None:
    """Terminate worker processes (kill any still spinning) and drop the pool.
    Grab the process handles BEFORE shutdown clears them. fresh-pool-per-batch,
    so terminating idle workers too is harmless and guarantees no leak."""
    procs = list(getattr(executor, "_processes", {}).values())
    for p in procs:
        try:
            p.terminate()          # SIGTERM — kills a `while True` worker
        except Exception:
            pass
    try:
        executor.shutdown(wait=False, cancel_futures=True)
    except Exception:
        pass
    for p in procs:
        try:
            p.join(0.2)
            if p.is_alive():
                p.kill()           # SIGKILL backstop
        except Exception:
            pass


def run_batch(codes: list[str], dataset, max_workers: int = DEFAULT_WORKERS) -> list[ScoreResult]:
    """Score every candidate concurrently across a process pool.

    Returns one ScoreResult per input code, IN ORDER, same length as `codes`.
    NEVER raises: syntax errors, timeouts, and worker crashes all map to a
    valid=False ScoreResult with a descriptive note.
    """
    results: list[ScoreResult] = [None] * len(codes)  # type: ignore[list-item]

    # Cheap last-gate pre-filter: skip spawning a process for unparsable code.
    to_run: list[tuple[int, str]] = []
    for i, code in enumerate(codes):
        try:
            ast.parse(code)
        except SyntaxError:
            results[i] = _invalid("syntax")
        except Exception:
            results[i] = _invalid("syntax")
        else:
            to_run.append((i, code))

    if not to_run:
        return results

    safe_ds = _strip_dataset(dataset)
    executor = ProcessPoolExecutor(max_workers=max(1, min(max_workers, len(to_run))))
    try:
        future_to_idx = {
            executor.submit(_worker, code, safe_ds): i for i, code in to_run
        }
        for future, i in future_to_idx.items():
            try:
                results[i] = future.result(timeout=TIMEOUT_S)
            except FuturesTimeout:
                results[i] = _invalid("timeout")
            except Exception as e:  # worker crash, broken pool, unpicklable, ...
                results[i] = _invalid(f"worker error: {type(e).__name__}")
    finally:
        # Always force-kill: any candidate may have left a spinning worker.
        _force_shutdown(executor)

    return results
