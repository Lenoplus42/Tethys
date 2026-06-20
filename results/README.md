# Results bundle — published artifacts

A self-contained snapshot so anyone can view and **replay it offline — no API key, no tokens,
no engine run**. (Working scratch lives in gitignored `runlogs/`; this is the curated copy.)

## Headline
Across three **distinct function classes**, the compute price of the prior (B_anon / B_priors at
E_TARGET, geomean over 3 seeds, first-crossing) ranges from ~1× (useless) to ~12–20× (real):

| law | function class | PRIOR_PRICE (geomean, programs) | tokens | range | n |
|---|---|---|---|---|---|
| **kepler** | power law · **control** | **1.05×** | 0.94× | 1.0–1.15× | 3 |
| **rocket** | transcendental · log | **11.9×** | 14.9× | 8.7–14.5× | 3 |
| **newton** | interaction · 3-input | **20.1×** | 27.2× | 1.7–100× | 3 |

Not a monotonic "complexity ladder" — the thesis is the **control + the range across classes**.
Rocket's ~12× is the tight, reproducible headline; Newton's ~20× is heavy-tailed (quote with range).
Full write-up + honest caveats (incl. rocket-anon seed2 finding a power-law *approximation* of log):
[`../RESULTS.md`](../RESULTS.md).

## What's here
- `ablation_comparison.png` — the 3-panel hero plot (function class + geomean + n=3 error bands).
- `kepler_latest/`, `rocket_latest/`, `newton_latest/` → each a complete sweep: one JSON per
  condition×seed (with `fitted_params`, `true_law_str`), `summary.json`, `ablation_<law>.png`.

## View / replay (offline, from this folder)
```bash
# recompute prices + redraw the 3-panel plot from saved JSON (no LLM):
python -m core.ablation_experiment --compare kepler,newton,rocket --no-run --out results/

# reveal a discovered law:
python -m core.reveal results/rocket_latest/rocket_priors_seed0.json   # -> delta_v = exhaust_velocity*log(mass_ratio)
python -m core.reveal results/newton_latest/newton_priors_seed0.json   # -> a*b/c**2
```
Every RunLog JSON is reload-sufficient: `best_code` + `fitted_params` + `true_law_str` are enough
to redraw the curves and reveal the law without re-running the engine.
