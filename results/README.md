# Results bundle — published artifacts

A self-contained snapshot of the final experiment so anyone can view and **replay it
offline — no API key, no tokens, no engine run**. (The working scratch dir `runlogs/`
stays gitignored; this is the curated copy.)

## The headline
| law | PRIOR_PRICE (geomean, programs) | tokens | range | n | anon convergence |
|---|---|---|---|---|---|
| **kepler** (control) | **1.05×** | 0.94× | 1.0–1.15× | 3 | 3/3 |
| **newton** | **20.1×** | 27.2× | 1.7–100.4× | 3 | 3/3 @ budget 800 |

Full write-up + caveats: [`../RESULTS.md`](../RESULTS.md).

## What's here
- `ablation_comparison.png` — the hero plot (Kepler ≈1× vs Newton ≈20×, side by side).
- `newton_latest/` → `newton_b800_n0-1-2_*/` — complete sweep: one JSON per condition×seed
  (with `fitted_params`, `true_law_str`), `summary.json`, `ablation_newton.png`.
- `kepler_latest/` → `kepler_b300_n0-1-2_*/` — same, for the control.

## View / replay (offline, from this folder)
```bash
# recompute prices + redraw the combined plot from saved JSON (no LLM):
python -m core.ablation_experiment --compare kepler,newton --no-run --out results/

# reveal a discovered law (Newton priors -> a*b/c**2, matches=True):
python -m core.reveal results/newton_latest/newton_priors_seed0.json
```
Every RunLog JSON is reload-sufficient: best_code + fitted_params + true_law_str are enough
to redraw the curves and reveal the law without re-running the engine.
