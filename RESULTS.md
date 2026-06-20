# Results — The Price of a Prior

> **Headline.** Stripping an AI's prior knowledge costs real inference compute. Across three
> *distinct function classes* — a **power law** (control), a **transcendental/log** law, and a
> **3-input interaction** law — the prior's price ranges from **~1× (useless)** to **~12–20×
> (worth real compute)**. The measured gap between the two error-vs-compute curves (prior on
> vs. off, identical data) *is* the price of the prior.

This is **not** a claim of a monotonic "complexity ladder" (rocket-vs-newton complexity is
debatable). The thesis is the **control + the range across function classes**: the method
returns ~1× when the prior is useless and a large, real number when it isn't.

All numbers: the engine in this repo (LLM-SR-style search; Qwen3-235B via Prime Intellect),
`E_TARGET = 1e-4`, **3 seeds per condition**, **geometric mean** of the per-seed PRIOR_PRICE,
B measured at the **first crossing of E_TARGET** (not the post-Occam refinement count). Data +
plots regenerate offline from saved JSON — see "Reproducing".

## PRIOR_PRICE = B_anon / B_priors  (geomean over 3 seeds)

| law | function class | geomean (programs) | geomean (tokens) | range (programs) | std | anon conv. |
|---|---|---|---|---|---|---|
| **kepler** | power law · **control** | **1.05×** | 0.94× | 1.00–1.15× | 0.07 | 3/3 @ b300 |
| **rocket** | transcendental · log | **11.9×** | 14.9× | 8.7–14.5× | 2.5 | 3/3 @ b800 |
| **newton** | interaction · 3-input | **20.1×** | 27.2× | 1.7–100× | 40 | 3/3 @ b800 |

Geomean is the headline (the principled aggregate for a ratio; not dragged up by one
heavy-tailed seed — Newton's arithmetic mean is 50× with std 40×).

**Headline recommendation.** **Rocket's ~12× is the most defensible pitch number** — tight,
reproducible (range 8.7–14.5×, std 2.5), with a clean iconic reveal. **Newton's ~20×** is the
"it can be even larger" point, but it is **heavy-tailed** (1.7–100×) — quote it with its range,
never as a clean point. **Kepler ~1×** is the control that proves the method doesn't fabricate gaps.

### Per-seed (first E_TARGET crossing)
**rocket** (budget 800): seed0 `B_p=8 B_a=116 → 14.5×`; seed1 `9 / 120 → 13.3×`; seed2 `6 / 52 → 8.7×`.
**newton** (budget 800): seed0 `6 / 10 → 1.7×`; seed1 `5 / 244 → 48.8×`; seed2 `5 / 502 → 100.4×`.
**kepler** (budget 300): seed0 `8 / 8 → 1.0×`; seed1 `6 / 6.9 → 1.15×`; seed2 `6 / 6 → 1.0×`.

## The discovered laws (offline reveal, `python -m core.reveal <runlog.json>`)
- **Rocket, priors (all 3 seeds)** → **`delta_v = exhaust_velocity * log(mass_ratio)`**,
  `matches_true: True` — a clean rediscovery of the **Tsiolkovsky rocket equation**, in 4–6
  rounds. The cleanest hero reveal of the three.
- **Rocket, anon (seed0, seed1)** → also the clean `log` form.
  **Rocket, anon (seed2)** → a **power-law approximation of the log**:
  `v_e·(mass_ratio^0.00108 − 1) − …` (≈ `v_e·ln(mass_ratio)`, since `xᵋ−1 ≈ ε·ln x`).
  Numerically converged (test err ≈ 6e-8) but **`matches_true: False`** — recorded honestly:
  without the prior, the search found a *different but numerically-equivalent* form. Interesting,
  not a bug; the per-DONE `final_eqn` flagged it at a glance.
- **Newton, priors** → `c0·m1·m2 / r²` → `a*b/c**2`, `matches_true: True`.
  **Newton, anon** → a bloated 7-parameter superset whose junk coefficients fit to ≈0; the
  symbolic reveal folds it back to the same `a*b/c**2`.
- **Kepler, priors** → `c0 · orbital_radius**1.5` (fitted `(1.0, 1.5)`).

## Key findings (honest)
1. **The control works.** Kepler's prior is useless (`c0·x^c1` is trivially discoverable blind),
   so the gap is ~1×. The method distinguishes a *useful* prior from a *useless* one.
2. **Two real laws, two large prices.** Rocket ~12× (tight) and Newton ~20× (heavy-tailed) —
   the prior is worth real compute on both, across different function classes.
3. **The gap magnitude is law- and run-specific.** Newton is heavy-tailed/high-variance
   (same seed varies run-to-run under the LLM's non-deterministic sampling); rocket is tight.
   Always report geomean + range, never a single point.
4. **A transcendental capability was required, and is condition-blind.** The engine presets a
   math namespace (`log/exp/sqrt/…`) so log-form candidates run with or without an `import` —
   identical for both conditions (the prior enters only via prompt names+hint). Without this,
   rocket-priors stalled (almost every log candidate failed on a missing import).
5. **Program price ≠ token price**: token price runs higher (search without the prior burns
   disproportionately more LLM rounds). Reported separately; do not claim they agree.

## Cost
Rocket b800 (6 runs) ≈ 110k tokens (~$0.02) · Newton b800 ≈ 306k (~$0.06) · Kepler b300 ≈ few k.
The whole experiment cost well under **$0.50** of inference.

## Artifacts & reproducing
Data lives in gitignored `runlogs/`; the curated, pullable copy is in **`results/`**. Each sweep
is self-contained and replays offline — **no LLM, no engine run**:
- `<law>_latest/` → newest sweep: one complete RunLog JSON per condition×seed (incl.
  `fitted_params`, `true_law_str`), `summary.json`, and `ablation_<law>.png`.
- Combined 3-panel hero plot: `results/ablation_comparison.png`.
- Regenerate plots/prices from saved JSON (no tokens):
  `python -m core.ablation_experiment --compare kepler,newton,rocket --no-run --out results/`
- Reveal a discovered law: `python -m core.reveal results/rocket_latest/rocket_priors_seed0.json`
- Re-run live (spends tokens): `python -m core.ablation_experiment --law rocket --budget 800 --seeds 0,1,2 --out runlogs/`
