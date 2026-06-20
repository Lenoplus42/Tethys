# Results — The Price of a Prior

> Headline: **stripping an AI's prior costs real inference compute, and the cost grows
> with problem hardness.** On Kepler the prior is worthless (~1×); on Newton it is worth
> ~**20× compute** (geometric mean; per-seed 1.7–100×). The measured gap between the two
> error-vs-compute curves *is* the price of the prior.

All numbers from the engine in this repo (LLM-SR-style search; Qwen3-235B via Prime),
`E_TARGET = 1e-4`, 3 seeds per condition. Data + plots regenerate offline from saved
JSON — see "Reproducing" below.

## PRIOR_PRICE (B_anon / B_priors at E_TARGET)

| law | geomean (programs) | geomean (tokens) | range (programs) | n | anon convergence |
|---|---|---|---|---|---|
| **kepler** (control) | **1.05×** | 0.94× | 1.00–1.15× | 3 | 3/3 @ budget 300 |
| **newton** | **20.1×** | 27.2× | 1.67–100.4× | 3 | 3/3 @ budget 800 |

Geometric mean is the headline (it is the principled aggregate for a ratio and is not
dragged up by one heavy-tailed seed; arithmetic mean on Newton is 50× with std 40×).

### Per-seed (Newton, budget 800)
| seed | B_priors | B_anon | price (programs) | price (tokens) |
|---|---|---|---|---|
| 0 | 6 | 10 | 1.67× | 1.74× |
| 1 | 5 | 244 | 48.8× | 78.3× |
| 2 | 5 | 502 | 100.4× | 148.1× |

### Per-seed (Kepler, budget 300)
| seed | B_priors | B_anon | price |
|---|---|---|---|
| 0 | 8 | 8 | 1.00× |
| 1 | 6 | 6.9 | 1.15× |
| 2 | 6 | 6 | 1.00× |

## The discovered laws (offline reveal, `python -m core.reveal <runlog.json>`)
- **Newton, priors** → `params[0]*m1**params[1]*m2**params[2]/r**2` → simplified
  **`a*b/c**2`**, `matches true law: True`. A clean rediscovery of universal gravitation —
  the hero reveal.
- **Newton, anon** → a bloated superset (full quadratic basis + a guarded ratio term) whose
  extra coefficients fit to ~0; the symbolic reveal folds it to the same `a*b/c**2`.
- **Kepler, priors** → `c0 * orbital_radius**1.5` (fitted `(1.0, 1.5)`).

## Key findings (honest)
1. **The control works.** Kepler's prior is useless (`c0*x**c1` is trivially discoverable
   blind), so the gap is ~1×. This proves the method does not manufacture gaps — it
   distinguishes a *useful* prior from a *useless* one.
2. **The gap is robustly large on Newton** — every converged seed shows 1.7–100×, never ≈1.
   The thesis ("the harder the science, the more the prior is worth") holds.
3. **The gap magnitude is heavy-tailed / high-variance**, intrinsic to a stochastic search
   over a hard form (and the LLM's non-deterministic sampling): the *same* seed can give a
   wildly different gap run-to-run. Report the geometric mean + range, not a single point.
4. **Budget sensitivity of Newton-anon convergence**: 1/3 seeds @ budget 300 → 2/3 @ 600 →
   3/3 @ 800. Priors always converges in round 1 (≤8 programs).
5. **Program price ≠ token price**: token price runs higher (Newton 27× vs 20×) because the
   no-prior search burns disproportionately more LLM rounds. (Do NOT claim they agree.)

## Cost
Newton budget-800 sweep (6 runs): ~306k tokens (~$0.06). Kepler 300 sweep: ~few k tokens.
The whole experiment cost well under $0.50 of inference.

## Artifacts & reproducing
Data is gitignored (`runlogs/`), but each sweep is self-contained and replays offline — no
LLM, no engine:
- `runlogs/<law>_latest/` → newest good sweep: one complete RunLog JSON per condition×seed
  (incl. `fitted_params`, `true_law_str`), `summary.json`, and `ablation_<law>.png`.
  - newton: `runlogs/newton_b800_n0-1-2_*/` · kepler: `runlogs/kepler_b300_n0-1-2_*/`
  - (earlier Newton sweeps `newton_b600_*`, `newton_b300_*` retained for comparison.)
- Combined hero plot: `runlogs/ablation_comparison.png`.
- Regenerate plots/prices from saved JSON (no tokens spent):
  - `python -m core.ablation_experiment --compare kepler,newton --no-run --out runlogs/`
  - `python -m core.reveal runlogs/newton_latest/newton_priors_seed0.json`
- Re-run live (spends tokens): `python -m core.ablation_experiment --law newton --budget 800 --seeds 0,1,2 --out runlogs/`
