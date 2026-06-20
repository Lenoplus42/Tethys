"""The laws (§3 of SPEC.md) — concrete LawSpec ground truths.

Each LawSpec is condition-INDEPENDENT. The semantic vs anonymized framings live
side by side on the same spec; datasets.py renders one or the other per condition
while keeping the numeric data identical. true_law_fn takes a tuple of inputs
(matching Contract 2's evaluate_law(inputs, params) convention) and returns a
scalar. It is NEVER shown to the LLM.
"""

from contracts import LawSpec

# Newton constant, chosen as 1.0 to keep outputs O(1)-O(100) for curve_fit stability
# (we are not modelling SI units — only numeric structure).
G = 1.0


# ---- Tier 0 — pipeline validator (§3): Y = 2*X**2 + 5, 1 input, noiseless ----
TIER0 = LawSpec(
    name="tier0",
    n_inputs=1,
    true_law_fn=lambda inputs: 2.0 * inputs[0] ** 2 + 5.0,
    true_law_str="Y = 2*X**2 + 5",
    target_form_hint="c0 * x**c1 + c2",
    n_params=3,
    semantic_inputs=["x"],
    semantic_output="y",
    domain_hint="",
    anon_inputs=["Sensor_X"],
    anon_output="Sensor_Y",
)


# ---- Law 1 — KEPLER (§3): Y = c0 * X**1.5, n_params=2 ----
KEPLER = LawSpec(
    name="kepler",
    n_inputs=1,
    true_law_fn=lambda inputs: 1.0 * inputs[0] ** 1.5,
    true_law_str="Y = 1.0 * X**1.5",
    target_form_hint="c0 * x**c1",
    n_params=2,
    semantic_inputs=["orbital_radius"],
    semantic_output="orbital_period",
    domain_hint="These are orbital measurements of bodies around a star.",
    anon_inputs=["Sensor_X"],
    anon_output="Sensor_Y",
)


# ---- Law 2 — NEWTON (§3): Y = G * A * B / C**2, n_params=4 ----
NEWTON = LawSpec(
    name="newton",
    n_inputs=3,
    true_law_fn=lambda inputs: G * inputs[0] * inputs[1] / inputs[2] ** 2,
    true_law_str="Y = G * A * B / C**2  (G=1.0)",
    target_form_hint="c0 * a**c1 * b**c2 / c**c3",
    n_params=4,
    semantic_inputs=["mass_1", "mass_2", "distance"],
    semantic_output="gravitational_force",
    domain_hint="These describe the gravitational attraction between two masses.",
    anon_inputs=["Sensor_A", "Sensor_B", "Sensor_C"],
    anon_output="Sensor_Y",
)
