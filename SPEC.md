# The Price of a Prior — Build Spec

> **The measured object is a gap between two curves.** We run the *same* proven equation-discovery
> engine (FunSearch / LLM-SR — we cite them) on the *same* hidden law under two conditions:
> once with the LLM's prior knowledge **available** (semantic variable names + light domain
> context), once with it **stripped** (anonymized `Sensor_*`, no context). We plot
> discovery-error vs **inference compute** for both. **The gap between them is the answer to a
> question nobody in this literature has asked: how much inference compute does it cost to
> rediscover a law when the model can't recite it?** That gap is *the price of prior knowledge,
> paid in compute* — and when compute is free, the price goes to zero. That is the world Etched
> is building. Built for the *Inference-Time Compute Hackathon 2026*, Track 3 — "Build the
> machine + Build the future" (*imagine near-infinite compute, near-0 latency, build as if that
> world is today*).
>
> **We claim NO new method.** The engine is the LLM-SR family. Equation discovery is our clean,
> exactly-verifiable **testbed**, not our achievement. Our deliverable is the **two-curve
> ablation, the gap metric, and the live show**. Read §7 (Positioning) before writing any slide.

This file is the single source of truth. Implement modules **one at a time**, in order. Each has
an interface contract and an acceptance test — a module is "done" only when its test passes.

---

## 0. Core paradigm + the three things that made us pivot here

The engine is a high-concurrency **FunSearch/LLM-SR evolutionary loop**: an LLM proposes many
diverse candidate **programs** in parallel; each is executed against data (exact, free
verification — no LLM judge); failures are pruned; survivors are scored by **generalization
error first, then code simplicity (Occam)**; the best are written back to a program database and
become few-shot seeds for the next round. More inference compute → wider search → lower error.

Three hard truths from prior-art review that *define* this spec (do not relitigate them):

1. **The method is not novel.** LLM proposes skeleton → external optimizer fits constants →
   evolutionary Occam selection IS LLM-SR (ICLR 2025) + In-Context SR (Merler 2024). We build on
   it and say so. Our contribution is the **ablation and the compute framing**, not the loop.
2. **Constant mutation (`3.8927`) is nearly a no-op.** `curve_fit` fits the constant; the LLM
   never sees or recites it, so changing `1.0 → 3.8927` does not change the LLM's problem. The
   ONLY real prior-stripper is **semantic anonymization** (`Sensor_X` + no domain hint). In the
   pitch, lean the no-recitation claim on **anonymization**, not the constant. (Keep the mutated
   constant only as an optional stage flourish in the standalone reveal — §3 note.)
3. **Rediscovering Kepler/Newton from data is 4 years old** (Lemos & Cranmer 2022, from *real*
   trajectories — strictly harder than our synthetic sensors). So **the rediscovery is NOT the
   achievement.** It is the controlled instrument that makes the gap measurable. Demote it
   accordingly everywhere.

**The independent variable of the whole project is: does the LLM have its prior or not.**
Everything serves measuring what that variable costs in compute.

---

## 1. The seam (contracts — LOCK FIRST)

Boundary between the **logic core** (laws, datasets, scoring, proposer, ablation analysis) and
the **plumbing** (async transport, sandbox, program DB, dashboard). Hold these and the two
workstreams never block each other.

```python
from dataclasses import dataclass, field
from typing import Callable, Literal

Condition = Literal["priors", "anon"]   # the experiment's independent variable

# ---- Contract 1a: the law (condition-independent ground truth) ----
@dataclass
class LawSpec:
    name: str                       # "kepler"
    n_inputs: int
    true_law_fn: Callable           # ground truth f(inputs)->output. NEVER shown to the LLM.
    true_law_str: str               # human-readable, for reveal + distance-to-truth
    target_form_hint: str           # the form curve_fit must recover, e.g. "c0 * x**c1"
    n_params: int                   # free constants the true form needs
    # semantic (WITH-PRIORS) framing — engages the LLM's pretrained knowledge:
    semantic_inputs: list[str]      # ["orbital_radius"]
    semantic_output: str            # "orbital_period"
    domain_hint: str                # short physics context shown ONLY in "priors" condition
    # anonymized (PRIOR-STRIPPED) framing:
    anon_inputs: list[str]          # ["Sensor_X"]
    anon_output: str                # "Sensor_Y"
    noise_std: float = 0.0          # >0 only for robustness law

# ---- Contract 1b: a concrete dataset = a law rendered under one condition ----
@dataclass
class Dataset:
    law_name: str
    condition: Condition
    n_inputs: int
    input_names: list[str]          # semantic OR anon, per condition
    output_name: str
    domain_hint: str                # the hint string for "priors"; "" for "anon"
    train: list[tuple]              # rows = (in_0,...,in_{n-1}, out). SHOWN to proposer.
    test:  list[tuple]              # HELD OUT — scoring only, never in prompt.
    true_law_fn: Callable           # for scoring distance-to-truth; NEVER in prompt
    true_law_str: str
    noise_std: float = 0.0
    meta: dict = field(default_factory=dict)

# CRITICAL FOR A CLEAN ABLATION: for a given law+seed, the underlying NUMERIC DATA is IDENTICAL
# across conditions. Only input_names / output_name / domain_hint differ. The semantic framing is
# the SOLE independent variable, so the gap is attributable to prior knowledge alone.

# ---- Contract 2: the candidate program shape (fixed symbols) ----
#     N_PARAMS = <int>
#     def evaluate_law(inputs, params):     # inputs: tuple[float,...] length n_inputs
#         x = inputs[0]                      # (Newton: a,b,c = inputs)
#         return params[0] * x ** params[1]
# `evaluate_law`, the signature, and `N_PARAMS` are FIXED. Only thing the sandbox executes.

# ---- Contract 3: scoring ----
@dataclass
class ScoreResult:
    valid: bool
    train_error: float
    test_error: float               # THE signal: generalization on held-out test
    length: int                     # AST node count (Occam)
    fitted_params: tuple
    score: float                    # higher = better
    note: str

# ---- Contract 4: one logged run at a fixed compute budget ----
@dataclass
class RunLog:
    law_name: str
    condition: Condition
    budget: int                     # total programs evaluated (the compute x-axis)
    seed: int
    best_test_error: float          # at end of budget
    error_trace: list[tuple]        # [(programs_evaluated, best_test_error_so_far), ...]
    best_code: str
    # token accounting (added during impl; honest inference-compute proxy alongside `budget`):
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    token_trace: list[tuple] = field(default_factory=list)  # (programs_evaluated, cumulative_completion_tokens)
```

---

## 2. THE EXPERIMENT (the product — read twice)

This section, not the engine, is the project. Everything else is instrumentation for it.

### 2.1 The ablation
For each law, run the identical engine under both `"priors"` and `"anon"`, at a sweep of compute
budgets, ≥3 seeds each. Overlay the two error-vs-compute curves on one axis with shaded gap.

- **"priors" curve** drops early: a physics-aware LLM told `orbital_period` vs `orbital_radius`
  often proposes the `c0 * x**1.5` form within the first few programs.
- **"anon" curve** drops later: given `Sensor_X`, it must *search* the form space to find the
  same structure.
- **The gap = compute price of the missing prior.**

### 2.2 The gap metric (the quotable number — this is the headline result)
Define a fixed target test-error `E_TARGET` (e.g. `1e-4`). For each condition, find
`B_cond` = programs evaluated to first reach `E_TARGET` (interpolate on `error_trace`).

> `test_error` is **normalized MSE** (`mean((pred-true)**2) / var(true)`, see Module 2), so a
> single `E_TARGET` is comparable across laws of different output scale (Kepler ~O(10) vs
> Newton ~O(100)). This scale-invariance is what makes the **cross-law** `PRIOR_PRICE` comparison
> in §2.3 legitimate — raw MSE would not be comparable.

```
PRIOR_PRICE = B_anon / B_priors        # "the prior was worth N× inference compute"
```
Report `PRIOR_PRICE` per law. Secondary metric: area-between-curves on log-log axes.

### 2.3 The escalation story (why ≥2 laws)
Expectation: **the gap GROWS with law complexity.** Kepler (1 input, 2 params) → modest prior
price; Newton (3 inputs, interaction structure) → larger prior price. That trend is the Etched
money line: *"the harder the science, the more the prior is worth, the more compute you need to
buy it back — and that compute is exactly what near-infinite inference makes free."* If the trend
holds across Kepler→Newton, you have a *general* claim, not an anecdote.

### 2.4 Hero artifacts
1. **The ablation plot** — two curves, shaded gap, error bars over seeds, log-log, per law. The
   single most important deliverable. Generate from the **overnight run** so it exists no matter
   what the live demo does.
2. **The live show** — the dashboard (§3 Module 7): thousands of candidate universes flickering,
   mass die-offs in red, the best program collapsing toward an ever-shorter law. Run it live on
   the **anon** condition (the harder, more dramatic search).
3. **The PRIOR_PRICE numbers** — "12× on Kepler, 40× on Newton" on a slide.

---

## 3. The laws (concrete `LawSpec`s)

Priority order is strict. **The anon curve on Kepler is the guaranteed deliverable; everything
else is an upgrade gated behind the Hour-6 health check.**

### Tier 0 — pipeline validator (must converge by Hour 6; not shown)
`Y = 2*X**2 + 5`, 1 input, noiseless. Only job: prove DB ↔ Sampler ↔ Evaluator is wired. Single
condition is fine here. If this doesn't converge by Hour 6, STOP and simplify.

### Law 1 — KEPLER (primary gap demo, must-land)
- `true_law_fn`: `Y = c0 * X**1.5`, `n_params=2`, `target_form_hint="c0 * x**c1"`, 1 input.
- **priors framing:** `semantic_inputs=["orbital_radius"]`, `semantic_output="orbital_period"`,
  `domain_hint="These are orbital measurements of bodies around a star."`
- **anon framing:** `anon_inputs=["Sensor_X"]`, `anon_output="Sensor_Y"`, `domain_hint=""`.
- This law must converge in *both* conditions so the gap is clean. Kepler is chosen precisely
  because it is famous enough that the prior fires hard (big, visible gap) yet simple enough to
  converge blind.

### Law 2 — NEWTON (gap-generalizes / gap-grows, should-land)
- `true_law_fn`: `Y = G * A * B / C**2`, `n_params=4`,
  `target_form_hint="c0 * a**c1 * b**c2 / c**c3"`, 3 inputs.
- **priors framing:** `semantic_inputs=["mass_1","mass_2","distance"]`,
  `semantic_output="gravitational_force"`,
  `domain_hint="These describe the gravitational attraction between two masses."`
- **anon framing:** `anon_inputs=["Sensor_A","Sensor_B","Sensor_C"]`, `anon_output="Sensor_Y"`.
- Data care: sample `A,B,C` in `[1,10]`, keep outputs O(1)–O(100) for `curve_fit` stability.
- This is the **escalation** law: expect a *larger* prior price than Kepler. That trend is the
  point. The same `(inputs, params)` seam handles it with no refactor.

### Law 3 — ROBUSTNESS under noise (nice-to-land, only if ahead)
- Single-variable so noise is the only added challenge: `Y = c0 * X**2.3`, `noise_std ≈ 8%` of
  signal (Gaussian, on train+test). `n_params=2`.
- Demonstrates Occam selection survives messy data: a high-degree polynomial drives *train* error
  low by fitting noise, but the simple true form wins on *test* error AND is shorter →
  lexicographic selection picks truth. (Can be run as a single curve; ablation optional here.)

### Optional stage flourish — the alien-constant reveal
For a standalone "no-recitation" beat, render Kepler-anon with a **mutated constant** (`3.8927`)
and watch it find the law blind. NOTE per §0.2: this is *cosmetic* — it does not change the
LLM's problem and is NOT the ablation. Keep the ablation data identical across conditions.

---

## 4. Module breakdown (implement in this order)

### Module 1 — `laws.py` + `datasets.py` (ship FIRST; unblocks plumbing)
`laws.py` holds the `LawSpec`s above. `datasets.py`:
```python
def make_dataset(law: LawSpec, condition: Condition, seed: int) -> Dataset
def make_tier0(seed: int) -> Dataset
```
Requirements: for a given `(law, seed)`, generate the numeric data ONCE, then render two
`Dataset`s that differ ONLY in names + `domain_hint` (clean ablation). Disjoint train/test. Fixed
seed → reproducible. Robustness law adds Gaussian noise controlled by `noise_std`.

**Acceptance:** `make_dataset(kepler,"priors",0)` and `make_dataset(kepler,"anon",0)` have
**bitwise-identical** `train`/`test` numbers and **different** names; `true_law_fn` reproduces
clean test outputs to machine precision.

### Module 2 — `evaluator.py` (the Occam core)
```python
def fit_params(program_ns, dataset) -> tuple                 # curve_fit over N_PARAMS on TRAIN
def evaluate_fit(program_ns, params, dataset) -> tuple       # (train_err, test_err)
def description_length(code: str) -> int                     # AST node count of evaluate_law body
def combine_score(test_error, length) -> float              # LEXICOGRAPHIC, see below
def score_program(code: str, dataset) -> ScoreResult        # seam fn the sandbox wraps
```
Decisions: fit constants with `scipy.optimize.curve_fit` on TRAIN only (try/except → return
`None`/`valid=False` on ANY failure — raise, non-finite params, arity mismatch — never propagate).
**Errors are normalized MSE (NMSE):** `test_error = mean((pred-true)**2) / var(true)` per split
(variance floored by a small eps for constant-output splits). NMSE makes `E_TARGET` comparable
across laws (§2.2) — do NOT use raw MSE. **Selection is lexicographic, not weighted:** among
programs with `test_error < EPS` (impl: `EPS = 1e-6`, the "law found" threshold — distinct from
§2.2's `E_TARGET = 1e-4`), rank purely by `-length`; above `EPS`, rank by `-test_error`.
`combine_score` encodes this single order so every sub-EPS program outranks every supra-EPS one.
`description_length` = AST node count of the `evaluate_law` body (NOT chars — whitespace can't be
gamed). `score_program` also wraps a lightweight single-thread SIGALRM timeout so a pathological
candidate (`while True`) can't hang a pure call; the REAL hard timeout stays in the sandbox
(Module 4). Headline signal is always `test_error`.

**Acceptance:** hand-written true Kepler → `test_error ≈ machine-eps`, `fitted_params ≈ (c0,1.5)`;
a 100-line if/else overfit → worse `score` despite low `train_error`; an infinite-loop/raising
program → `valid=False`, no hang, no raise; NMSE equal-order across a constant-scale and a
Newton-scale law for an equally-good fit.

### Module 3 — `proposer.py` (steering — owns prompt + parser; plumbing owns the async call)
```python
SYSTEM_PROMPT: str
def build_prompt(dataset, exemplars, data_preview) -> str   # MUST honor dataset.condition
def parse_programs(llm_response: str) -> list[str]
```
**Condition handling is the crux of the experiment:** in `"priors"`, include
`dataset.domain_hint` and the semantic names. In `"anon"`, include NEITHER — only `Sensor_*` and
raw numbers. The prompt builder is the literal mechanism that gives/removes the prior. State the
fixed contract (`N_PARAMS`, `evaluate_law`). Early rounds push diversity; late rounds refine the
leader. Parser drops malformed blocks without raising (keep a block only if it `ast.parse`s AND
defines both `evaluate_law` and `N_PARAMS`).

**Auditability (load-bearing, impl decision):** the prompt body template is **byte-identical**
across conditions; the ONLY textual difference is the interpolated names + the one `Context:`
hint line (present in `"priors"`, absent in `"anon"`). The "scientific-discovery vs
fit-a-function" framing is **not written as two prose variants** — it EMERGES from the names +
hint alone (mirrors datasets.py's "identical numbers, only names differ"). Do NOT add any semantic
flavor/units/context to the anon prompt; a reviewer must be able to diff the two prompts and see
ONLY names/hint differ. (Enforced by a test that reconstructs the priors prompt from the anon one
by swapping names + re-adding the hint line.)

**Acceptance:** same law, two conditions → the `"priors"` prompt contains the domain hint +
semantic names and no `Sensor`; the `"anon"` prompt contains `Sensor_*` and neither the hint nor
any semantic name; both state the fixed contract. ≥20 valid distinct functions per batch on
Tier 0.

### Module 4 — `sandbox.py` (plumbing — infra)
`ProcessPoolExecutor` across cores; hard ~500ms timeout per candidate; kill runaway loops;
`ast.parse`/`ruff` pre-filter; calls `evaluator.score_program`. **Acceptance:** infinite-loop
program is killed and scored `valid=False` without hanging; hundreds of candidates/sec.

### Module 5 — `search.py` (convergence policy — co-owned)
Producer–consumer evolutionary loop + program DB with islands; exemplar sampling favoring high
score with retained exploration; island reset on stagnation. Logic core owns policy knobs; infra
owns DB + async transport. **Acceptance:** Tier 0 converges end-to-end; Kepler converges in both
conditions within budget.

### Module 6 — `ablation_experiment.py` (THE product — logic core / DS)
```python
def run_condition(law, condition, budget, seed) -> RunLog
def sweep_ablation(law, budgets, seeds=3) -> dict           # {condition: [RunLog,...]}
def compute_prior_price(logs, e_target=1e-4) -> float       # B_anon / B_priors
def plot_ablation(logs) -> None                             # 2 curves, shaded gap, error bars, log-log
```
x-axis = **total programs evaluated** (honest compute proxy, not wall-clock). ≥3 seeds → error
bars. Kick the full Kepler+Newton ablation off as the **overnight background run by Hour 6** so
the hero plot and PRIOR_PRICE numbers exist regardless of the live demo.

**Acceptance:** saved `ablation_kepler.png` with two converging curves, a visible gap, error
bars; `compute_prior_price` returns a finite ratio > 1.

### Module 7 — `dashboard.py` (plumbing — infra)
Live terminal: thousands of universes flickering, failures dying red, the best `evaluate_law`
collapsing to an ever-shorter law. Runs live on the **anon** condition. Ablation plot rendered
separately by Module 6.

---

## 5. Division of labor
| Owner | Modules |
|---|---|
| **DS / ML** (logic core) | `laws.py`, `datasets.py`, `evaluator.py`, `proposer.py` (prompt+parser, **condition logic**), policy knobs in `search.py`, **`ablation_experiment.py`** |
| **SWE / Infra** (plumbing) | `sandbox.py`, async transport + DB in `search.py`, `dashboard.py`, vLLM/Groq/DeepSeek serving |

Logic core ships pure, independently-testable functions; infra wraps concurrency. The §1 seam +
`Condition` are the only coordination needed.

---

## 6. Timeline (20h; Hour-6 is a hard gate) + scoping

- **0–2** — `laws.py` + `datasets.py` (Tier 0 + Kepler both conditions). Ship to infra now.
- **1–3** — `evaluator.py` pure functions; hand `score_program` to infra.
- **2–4** — `proposer.py` prompt + parser **with condition logic**; test on Tier 0.
- **4–6** — integrate. **HOUR-6 GATE: does Tier 0 converge AND does Kepler converge in the `anon`
  condition?** No → simplify until yes; do not proceed.
- **6** — launch overnight ablation run (Kepler both conditions; add Newton if loop is healthy).
- **6–10** — `search.py` tuning; get Kepler `priors` converging early, `anon` converging at all;
  confirm a visible gap on a quick low-budget sweep.
- **10–14** — `ablation_experiment.py`: the two-curve plot + `PRIOR_PRICE`. Push Newton.
- **14–16** — finalize hero plot from overnight data; lock PRIOR_PRICE numbers (Kepler, Newton).
- **16–18** — **FEATURE FREEZE.** Record backup video of the live anon demo. Draft slides (§7).
- **18–20** — rehearse + buffer.

**Scoping ladder (defend in this order):**
1. Kepler `anon` curve converges → you have a working engine + live demo. (Floor.)
2. Kepler **two-curve gap** + PRIOR_PRICE → you have the product. (Target.)
3. Newton gap > Kepler gap → you have the *general* claim + Etched escalation line. (Win.)
4. Robustness curve → bonus credibility slide. (Only if ahead.)
Never let "all of it perfect" sink the Hour-18 freeze.

---

## 7. Positioning & pitch (READ before any slide — this is how we win)

### What this IS
An answer to a **storytelling** prompt (*build as if infinite compute is today*), delivered as a
**measurement nobody in the literature has made**: the compute price of an AI's prior knowledge,
shown as a gap between two scaling curves. We are not competing on discovery; we are competing on
making a thesis **felt and quantified**.

### The telescope framing (core stance)
FunSearch / LLM-SR are *cartographers* — they proved machines **can** discover; that's settled. We
built a **telescope**: we point their proven machinery at a different object — *the future of
discovery itself* — and make it visible and measurable. They optimized for a better equation or a
benchmark score. We produce what they never tried to: **the exchange rate between compute and
truth, and the price of a prior in compute.**

### Lineage slide — title it "What we stand on" (put it EARLY)
Name ancestors before any judge can; it disarms the only real attack and signals literacy.
- **FunSearch** (Nature 2023) — LLM + evolution discovers new math.
- **LLM-SR** (ICLR 2025) — equation discovery via program skeletons + evolution. **Our engine.**
- **In-Context SR** (Merler 2024) — LLM finds form, optimizer fits constants. **Our `curve_fit`.**
- **LLM-SRBench / LSR-Transform** (2025) — anti-recitation benchmark. We acknowledge it covers the
  anonymization idea; we **don't** sell anti-cheat as our novelty (see below).
- **Rediscovering orbital mechanics** (Lemos & Cranmer 2022) — Newton from *real* data, 4 yrs ago.
  So **our rediscovery is the instrument, not the result.**

*They* asked "**can** machines discover?" *We* ask "**what does the prior cost**, in compute?"

### The bulletproof one-liner
> "Equation discovery is solved — FunSearch and LLM-SR did it, and we build directly on them. We
> use it as a clean, exactly-verifiable testbed to measure something nobody has: the exchange
> rate between inference compute and discovery, and the **compute price of stripping away an AI's
> prior knowledge**. On Kepler that prior is worth ~Nx compute; on Newton, more. When compute is
> free, that price goes to zero — and that's the world Etched is building."

### Claim discipline (the room knows the field — do NOT overclaim)
- ✅ "We measured the compute price of a prior, on a verifiable testbed, under a no-recitation
  guarantee enforced by anonymization." / "The asymptote is the *seed* of automated science."
- ❌ "We invented this." / "Nobody has done equation discovery." / "We measured the cost of curing
  cancer." / Leaning the no-recitation claim on the mutated constant (it's a near-no-op).

### Rehearsed judge Q&A (both teammates memorize identical lines)
- **"Isn't this just LLM-SR?"** → "The engine is, and we cite it. LLM-SR proved priors *help*. It
  never measured **what removing them costs in compute** — that gap is our result, and it's a
  pure inference-compute story, which is this hackathon's whole theme."
- **"What's actually original?"** → "The two-curve ablation and the gap metric. We turn the LLM's
  prior into an *independent variable* and price it in compute. Nobody foregrounds that."
- **"Your anti-cheat — isn't that just LLM-SRBench?"** → "The *recitation-prevention* idea, yes —
  we don't claim it. We repurpose anonymization not as a defense but as the **knob of the
  experiment**: priors on vs off, same data. That's a different use they didn't make."
- **"Isn't rediscovering Newton old (Cranmer 2022)?"** → "Yes — that's why rediscovery is our
  *instrument*, not our claim. We need a law with known ground truth so the gap is exactly
  measurable. The result is the gap, not the law."
- **"Why does this matter to Etched?"** → "The gap is denominated in inference compute, and it
  grows with problem hardness. Your silicon is what pays that price down to zero. We show the
  exact exchange rate; you make it free."

### What kills the pitch (beyond §6 build risks)
- Selling the engine or the rediscovery as novel → instant credibility loss.
- Selling anti-cheat as our contribution → LLM-SRBench already owns it; reframe it as the knob.
- Leaning no-recitation on the constant mutation → a sharp judge pokes it and you deflate; lean
  on anonymization.
- The cosmic claim ("cost of science/cancer") → keep the asymptote rhetorical, the gap measured.
