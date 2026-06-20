"""Ablation experiment (Module 6, §4 of SPEC.md) — THE product.

Runs the two-condition ablation, computes PRIOR_PRICE (the compute price of the
prior), and renders the hero plot. Reuses run_search; reimplements no engine.

THE measured object is the GAP between the two error-vs-compute curves: with the
prior available ("priors") vs stripped ("anon"). PRIOR_PRICE = B_anon / B_priors,
where B_cond = programs (or completion tokens) evaluated to FIRST reach E_TARGET.

CLI (the unified entry — replaces the /tmp temp runners):
  python -m core.ablation_experiment --law newton --budget 300 --seeds 0,1,2 --out runlogs/
  python -m core.ablation_experiment --law newton --no-run --out runlogs/   # re-plot from saved logs
"""

import dataclasses
import json
import time
import warnings
from pathlib import Path

import numpy as np

from core.contracts import RunLog
from core.datasets import make_dataset
from core.laws import KEPLER, NEWTON, ROCKET
from core.search import BATCH_SIZE, run_search

E_TARGET = 1e-4          # §2.2 headline target error
_ERR_FLOOR = 1e-18       # clamp exact-zero errors for log-scale plotting
_LAW_BY_NAME = {"kepler": KEPLER, "newton": NEWTON, "rocket": ROCKET}
# Function class per law (for plot labels) — these are DISTINCT classes, not a
# strict complexity ladder; the thesis is the control + the range across classes.
_FUNCTION_CLASS = {
    "kepler": "power law · control",
    "rocket": "transcendental · log",
    "newton": "interaction · 3-input",
}


# ---------------------------------------------------------------------------
# Live runners (network)
# ---------------------------------------------------------------------------
def run_condition(law, condition, budget, seed) -> RunLog:
    """Thin wrapper: render the dataset for one condition and search it."""
    ds = make_dataset(law, condition, seed)
    return run_search(ds, budget=budget, seed=seed, batch_size=BATCH_SIZE)


def sweep_ablation(law, budget, seeds=(0, 1, 2)) -> dict:
    """Run BOTH conditions x each seed. Returns {condition: [RunLog, ...]}."""
    out = {"priors": [], "anon": []}
    for condition in ("priors", "anon"):
        for seed in seeds:
            out[condition].append(run_condition(law, condition, budget, seed))
    return out


# ---------------------------------------------------------------------------
# Pure logic: the gap metric
# ---------------------------------------------------------------------------
def first_crossing(error_trace, token_trace, e_target):
    """FIRST point at which best_test_error reaches <= e_target, interpolated.

    Returns (B_programs, B_tokens): programs-evaluated and cumulative-completion-
    tokens at the crossing. Uses the FIRST crossing (NOT the converged / post-Occam
    count). error_trace and token_trace share the same x (evaluated) per round, so
    the same interpolation fraction is applied to both. Returns (None, None) if the
    trace never reaches e_target.
    """
    if not error_trace:
        return None, None
    errs = [(float(p[0]), float(p[1])) for p in error_trace]
    toks = [(float(p[0]), float(p[1])) for p in token_trace] if token_trace else None

    # already at/below target on the very first recorded point
    if errs[0][1] <= e_target:
        return errs[0][0], (toks[0][1] if toks else None)

    for i in range(1, len(errs)):
        y0, y1 = errs[i - 1][1], errs[i][1]
        if y1 <= e_target:                       # straddle: y0 > e_target >= y1
            frac = 1.0 if y0 == y1 else (y0 - e_target) / (y0 - y1)
            x0, x1 = errs[i - 1][0], errs[i][0]
            b_prog = x0 + frac * (x1 - x0)
            b_tok = None
            if toks:
                t0, t1 = toks[i - 1][1], toks[i][1]
                b_tok = t0 + frac * (t1 - t0)
            return b_prog, b_tok
    return None, None                            # never reached e_target


def _agg(values):
    if not values:
        return {"mean": None, "geomean": None, "std": None, "min": None, "max": None, "n": 0}
    arr = np.asarray(values, dtype=float)
    # geometric mean is the principled aggregate for a RATIO like PRIOR_PRICE: it
    # isn't dragged up by a single heavy-tailed seed the way the arithmetic mean is.
    geomean = float(np.exp(np.mean(np.log(arr)))) if np.all(arr > 0) else None
    return {"mean": float(arr.mean()), "geomean": geomean, "std": float(arr.std(ddof=0)),
            "min": float(arr.min()), "max": float(arr.max()), "n": int(arr.size)}


def compute_prior_price(logs, e_target=E_TARGET) -> dict:
    """PRIOR_PRICE = B_anon / B_priors at e_target, by program count AND by
    completion tokens, averaged across matched seeds with per-seed spread.

    Program and token prices should agree closely (Qwen is non-reasoning, so
    tokens ~ programs*batch) — agreement is a robustness claim for the pitch.
    """
    priors = {log.seed: log for log in logs.get("priors", [])}
    anon = {log.seed: log for log in logs.get("anon", [])}
    seeds = sorted(set(priors) & set(anon))

    per_seed, prog_prices, tok_prices = {}, [], []
    for s in seeds:
        bp_prog, bp_tok = first_crossing(priors[s].error_trace, priors[s].token_trace, e_target)
        ba_prog, ba_tok = first_crossing(anon[s].error_trace, anon[s].token_trace, e_target)
        entry = {
            "B_priors": bp_prog, "B_anon": ba_prog,
            "B_priors_tokens": bp_tok, "B_anon_tokens": ba_tok,
            "price": None, "price_tokens": None,
        }
        if bp_prog and ba_prog and bp_prog > 0:
            entry["price"] = ba_prog / bp_prog
            prog_prices.append(entry["price"])
        if bp_tok and ba_tok and bp_tok > 0:
            entry["price_tokens"] = ba_tok / bp_tok
            tok_prices.append(entry["price_tokens"])
        per_seed[s] = entry

    return {
        "e_target": e_target,
        "per_seed": per_seed,
        "prior_price": _agg(prog_prices),          # program-count price
        "prior_price_tokens": _agg(tok_prices),    # completion-token price
    }


# ---------------------------------------------------------------------------
# Plotting (the hero artifact)
# ---------------------------------------------------------------------------
def _step_interp(xs, ys, grid, floor):
    """Carry-forward (best-so-far) step interpolation of one seed onto `grid`.
    NaN before the seed's first evaluation (no left extrapolation)."""
    out = np.full(len(grid), np.nan)
    for j, gx in enumerate(grid):
        idx = int(np.searchsorted(xs, gx, side="right")) - 1
        if idx >= 0:
            out[j] = max(ys[idx], floor)
    return out


def _draw_ablation_on_ax(ax, logs, e_target, law_name=None, price=None, show_ylabel=True):
    """Draw ONE law's ablation onto a given Axes (shared by single + comparison plots):
    per-seed faint lines + mean, std band over seeds, shaded gap, E_TARGET line, title."""
    colors = {"priors": "tab:blue", "anon": "tab:red"}

    curves, gxmin, gxmax = {}, np.inf, -np.inf
    for cond in ("anon", "priors"):
        cs = []
        for log in logs.get(cond, []):
            if not log.error_trace:
                continue
            xs = np.array([float(p[0]) for p in log.error_trace])
            ys = np.array([max(float(p[1]), _ERR_FLOOR) for p in log.error_trace])
            cs.append((xs, ys))
            gxmin, gxmax = min(gxmin, xs.min()), max(gxmax, xs.max())
        curves[cond] = cs

    if not np.isfinite(gxmin):
        gxmin, gxmax = 1.0, 2.0
    grid = np.logspace(np.log10(max(gxmin, 1.0)), np.log10(max(gxmax, 2.0)), 60)

    means = {}
    for cond in ("anon", "priors"):
        cs = curves.get(cond, [])
        if not cs:
            continue
        for xs, ys in cs:                          # faint per-seed lines
            ax.plot(xs, ys, color=colors[cond], alpha=0.22, lw=1)
        stacked = np.array([_step_interp(xs, ys, grid, _ERR_FLOOR) for xs, ys in cs])
        logy = np.log10(stacked)
        with warnings.catch_warnings():   # all-NaN columns (grid below first eval) are expected
            warnings.simplefilter("ignore", RuntimeWarning)
            m, sd = np.nanmean(logy, axis=0), np.nanstd(logy, axis=0)
        mean_y = 10 ** m
        ax.plot(grid, mean_y, color=colors[cond], lw=2.6, zorder=5,
                label=f"{cond} (n={len(cs)})")
        if len(cs) > 1:                            # std band over seeds (the error bars)
            ax.fill_between(grid, 10 ** (m - sd), 10 ** (m + sd),
                            color=colors[cond], alpha=0.15)
        means[cond] = mean_y

    if "priors" in means and "anon" in means:      # shaded gap = the prior's price
        ax.fill_between(grid, means["priors"], means["anon"],
                        where=(means["anon"] >= means["priors"]),
                        color="gray", alpha=0.18, label="gap = prior's price")

    ax.axhline(e_target, ls="--", color="black", lw=1, alpha=0.7)
    ax.text(grid[0], e_target * 1.4, f"E_TARGET={e_target:g}", fontsize=8, va="bottom")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("total programs evaluated  (inference compute →)")
    if show_ylabel:
        ax.set_ylabel("test error (NMSE)  ↓ better")

    ppd = (price or {}).get("prior_price", {})
    pptd = (price or {}).get("prior_price_tokens", {})
    pp = ppd.get("geomean") or ppd.get("mean")        # geometric mean: honest for a ratio
    ppt = pptd.get("geomean") or pptd.get("mean")
    title = (law_name or "").strip()
    if pp:
        ann = f"PRIOR_PRICE ≈ {pp:.1f}× (programs)"
        if ppt:
            ann += f" · {ppt:.1f}× (tokens)"
        title = f"{title}\n{ann}" if title else ann
    if title:
        ax.set_title(title)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, which="both", ls=":", alpha=0.3)


def plot_ablation(logs, save_path, e_target=E_TARGET, law_name=None, price=None):
    """Save a single-law log-log PNG (per-seed lines + mean, std band, shaded gap)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    _draw_ablation_on_ax(ax, logs, e_target, f"The price of a prior — {law_name or ''}".strip(" —"), price)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    return str(save_path)


def plot_comparison(law_logs, save_path, e_target=E_TARGET, prices=None, order=None):
    """THE everything-plot: one panel per law, shared log-y, ordered by increasing
    gap so Kepler (control, ~1×) sits left of Newton (~10×) — the escalation story.
    `law_logs` = {law_name: sweep_dict}; `prices` = {law_name: compute_prior_price(...)}."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prices = prices or {}
    # order by GEOMEAN price (the headline aggregate; consistent with the labels)
    names = order or sorted(
        law_logs,
        key=lambda n: (prices.get(n, {}).get("prior_price", {}) or {}).get("geomean") or 0.0,
    )
    fig, axes = plt.subplots(1, len(names), figsize=(7.5 * len(names), 6), sharey=True)
    if len(names) == 1:
        axes = [axes]
    for i, (ax, name) in enumerate(zip(axes, names)):
        label = f"{name}  ({_FUNCTION_CLASS.get(name, '')})" if _FUNCTION_CLASS.get(name) else name
        _draw_ablation_on_ax(ax, law_logs[name], e_target, law_name=label,
                             price=prices.get(name), show_ylabel=(i == 0))

    fig.suptitle("The price of a prior — across function classes (geomean over 3 seeds, "
                 "first E* crossing)", fontsize=14, y=1.02)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)


# ---------------------------------------------------------------------------
# Persistence  — the JSON SCHEMA below is the CONTRACT a downstream dashboard
# reads to redraw EVERYTHING (curves, gap, reveal) WITHOUT re-running the engine
# or calling the LLM. This decouples "running the experiment" from "displaying it".
#
# Layout (each sweep is self-contained and never overwrites another):
#   runlogs/<law>_b<budget>_n<seeds>_<YYYYMMDD-HHMMSS>/
#       <law>_<condition>_seed<seed>.json   # one COMPLETE RunLog each (see below)
#       summary.json                        # PRIOR_PRICE (program+token), per-seed B's, convergence
#       ablation_<law>.png                  # the rendered curve
#   runlogs/<law>_latest -> <that sweep dir>  # symlink to the most recent good sweep
#
# RunLog JSON (reload-sufficient, via dataclasses.asdict on the amended RunLog):
#   law_name, condition, seed, budget, best_test_error,
#   error_trace [[evaluated, best_err], ...], token_trace [[evaluated, cum_completion_tokens], ...],
#   total_prompt_tokens, total_completion_tokens,
#   best_code, fitted_params [..], true_law_str
#   -> best_code + fitted_params + true_law_str are enough for core.reveal OFFLINE.
#
# summary.json:
#   {law, true_law_str, budget, seeds, e_target, timestamp,
#    convergence: {anon: "k/n", priors: "k/n"},
#    prior_price: {mean,std,min,max,n}, prior_price_tokens: {...},
#    per_seed: {seed: {B_priors,B_anon,price, B_priors_tokens,B_anon_tokens,price_tokens}}}
# ---------------------------------------------------------------------------
def save_runlog(log, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(dataclasses.asdict(log), f, indent=2)


def load_runlog(path) -> RunLog:
    with open(path) as f:
        return RunLog(**json.load(f))


def sweep_dirname(law_name, budget, seeds):
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{law_name}_b{budget}_n{'-'.join(str(s) for s in seeds)}_{stamp}"


def save_sweep(logs, sweep_dir):
    """Write every RunLog (one COMPLETE JSON per condition×seed) into sweep_dir."""
    sweep_dir = Path(sweep_dir)
    sweep_dir.mkdir(parents=True, exist_ok=True)
    for condition, runs in logs.items():
        for log in runs:
            save_runlog(log, sweep_dir / f"{log.law_name}_{condition}_seed{log.seed}.json")
    return sweep_dir


def build_summary(law, budget, seeds, e_target, price, logs):
    def _rate(cond):
        runs = logs.get(cond, [])
        crossed = sum(1 for lg in runs if first_crossing(lg.error_trace, lg.token_trace, e_target)[0] is not None)
        return f"{crossed}/{len(runs)}"
    return {
        "law": law.name,
        "true_law_str": law.true_law_str,
        "budget": budget,
        "seeds": list(seeds),
        "e_target": e_target,
        "timestamp": time.strftime("%Y%m%d-%H%M%S"),
        "convergence": {"anon": _rate("anon"), "priors": _rate("priors")},
        "prior_price": price["prior_price"],
        "prior_price_tokens": price["prior_price_tokens"],
        "per_seed": price["per_seed"],
    }


def save_summary(summary, sweep_dir):
    with open(Path(sweep_dir) / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def load_sweep(sweep_dir):
    """Load a sweep DIRECTORY -> (logs {condition:[RunLog]}, summary dict or None).
    Groups by each RunLog's own `condition` field (filenames are just convenience)."""
    sweep_dir = Path(sweep_dir)
    logs = {"priors": [], "anon": []}
    for p in sorted(sweep_dir.glob("*.json")):
        if p.name == "summary.json":
            continue
        log = load_runlog(p)
        logs.setdefault(log.condition, []).append(log)
    summary = None
    sp = sweep_dir / "summary.json"
    if sp.exists():
        with open(sp) as f:
            summary = json.load(f)
    return logs, summary


def update_latest(out_dir, law_name, sweep_dir):
    """Point runlogs/<law>_latest at the most recent good sweep (relative symlink)."""
    link = Path(out_dir) / f"{law_name}_latest"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(Path(sweep_dir).name)
    return link


def resolve_sweep_dir(out_dir, law_name, explicit=None):
    """A specific --dir, else runlogs/<law>_latest."""
    if explicit:
        return Path(explicit)
    return Path(out_dir) / f"{law_name}_latest"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _print_summary(law, price, png_path):
    print("\n================ ABLATION SUMMARY ================")
    print(f"law            : {law.name}    true: {law.true_law_str}")
    print(f"E_TARGET       : {price['e_target']:g}")
    pp, ppt = price["prior_price"], price["prior_price_tokens"]
    if pp["mean"]:
        print(f"PRIOR_PRICE    : geomean {pp.get('geomean'):.2f}x  (programs)  "
              f"[n={pp['n']}, range {pp['min']:.2f}-{pp['max']:.2f}, arith.mean {pp['mean']:.2f}, std {pp['std']:.2f}]")
    else:
        print("PRIOR_PRICE    : (insufficient crossings — anon and/or priors never reached E_TARGET)")
    if ppt["mean"]:
        print(f"PRIOR_PRICE_tok: geomean {ppt.get('geomean'):.2f}x  (completion tokens)  "
              f"[n={ppt['n']}, range {ppt['min']:.2f}-{ppt['max']:.2f}, arith.mean {ppt['mean']:.2f}, std {ppt['std']:.2f}]")
    print("\nper-seed:")
    for s, e in sorted(price["per_seed"].items()):
        print(f"  seed {s}: B_priors={e['B_priors']} B_anon={e['B_anon']} "
              f"price={e['price']!s:.6} | tok B_priors={e['B_priors_tokens']} "
              f"B_anon={e['B_anon_tokens']} price_tok={e['price_tokens']!s:.6}")
    print(f"\nplot           : {png_path}")
    print("==================================================")


def main(argv=None):
    import argparse

    p = argparse.ArgumentParser(description="Run the price-of-a-prior ablation.")
    p.add_argument("--law", choices=sorted(_LAW_BY_NAME),
                   help="run/plot a single law")
    p.add_argument("--compare", default=None,
                   help="comma-separated laws -> the combined everything-plot from SAVED logs "
                        "(implies --no-run), e.g. --compare kepler,newton")
    p.add_argument("--budget", type=int, default=300)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--out", default="runlogs")
    p.add_argument("--e-target", type=float, default=E_TARGET)
    p.add_argument("--no-run", action="store_true",
                   help="reload the saved sweep (--dir or <law>_latest) and re-plot/re-price")
    p.add_argument("--dir", default=None, help="explicit sweep dir to reload (else <law>_latest)")
    args = p.parse_args(argv)

    seeds = tuple(int(s) for s in args.seeds.split(","))
    out_dir = Path(args.out)

    # combined everything-plot from each law's latest saved sweep (the escalation story)
    if args.compare:
        names = [n.strip() for n in args.compare.split(",")]
        law_logs, prices = {}, {}
        for n in names:
            logs, _ = load_sweep(resolve_sweep_dir(out_dir, n))
            law_logs[n] = logs
            prices[n] = compute_prior_price(logs, e_target=args.e_target)
        png = plot_comparison(law_logs, out_dir / "ablation_comparison.png",
                              e_target=args.e_target, prices=prices)
        for n in names:
            _print_summary(_LAW_BY_NAME[n], prices[n], f"(in {png})")
        print(f"\ncombined plot  : {png}")
        return

    if not args.law:
        p.error("either --law or --compare is required")
    law = _LAW_BY_NAME[args.law]

    if args.no_run:
        sweep_dir = resolve_sweep_dir(out_dir, law.name, args.dir)
        logs, _ = load_sweep(sweep_dir)
        price = compute_prior_price(logs, e_target=args.e_target)
        png = plot_ablation(logs, Path(sweep_dir) / f"ablation_{law.name}.png",
                            e_target=args.e_target, law_name=law.name, price=price)
        _print_summary(law, price, png)
        return

    logs = sweep_ablation(law, args.budget, seeds)
    price = compute_prior_price(logs, e_target=args.e_target)
    sweep_dir = out_dir / sweep_dirname(law.name, args.budget, seeds)
    save_sweep(logs, sweep_dir)
    save_summary(build_summary(law, args.budget, seeds, args.e_target, price, logs), sweep_dir)
    png = plot_ablation(logs, sweep_dir / f"ablation_{law.name}.png",
                        e_target=args.e_target, law_name=law.name, price=price)
    update_latest(out_dir, law.name, sweep_dir)
    _print_summary(law, price, png)
    print(f"sweep dir      : {sweep_dir}  (runlogs/{law.name}_latest -> it)")


if __name__ == "__main__":
    main()
