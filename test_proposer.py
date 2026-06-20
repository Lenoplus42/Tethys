"""Module 3 acceptance test (§4 of SPEC.md) — NO network."""

from datasets import make_dataset
from laws import KEPLER
from proposer import SYSTEM_PROMPT, build_prompt, parse_programs

PREVIEW = "1.000, 1.000\n4.000, 8.000\n9.000, 27.000"

HINT = "These are orbital measurements of bodies around a star."

VALID_A = "N_PARAMS = 2\ndef evaluate_law(inputs, params):\n    x = inputs[0]\n    return params[0] * x ** params[1]"
VALID_B = "N_PARAMS = 1\ndef evaluate_law(inputs, params):\n    x = inputs[0]\n    return params[0] * x"


def test_priors_prompt_has_hint_and_semantic_names_no_sensor():
    ds = make_dataset(KEPLER, "priors", 0)
    p = build_prompt(ds, [], PREVIEW)
    assert HINT in p
    assert "orbital_radius" in p
    assert "orbital_period" in p
    assert "Sensor" not in p


def test_anon_prompt_has_sensor_no_hint_no_semantic():
    ds = make_dataset(KEPLER, "anon", 0)
    p = build_prompt(ds, [], PREVIEW)
    assert "Sensor_X" in p
    assert "Sensor_Y" in p
    assert HINT not in p
    assert "orbital_radius" not in p
    assert "orbital_period" not in p


def test_both_prompts_state_fixed_contract():
    for cond in ("priors", "anon"):
        ds = make_dataset(KEPLER, cond, 0)
        p = build_prompt(ds, [], PREVIEW)
        assert "N_PARAMS" in p
        assert "evaluate_law" in p


def test_auditability_only_names_and_hint_differ():
    # Stronger than the spec's minimum: replacing anon names + adding the hint
    # back must reproduce the priors prompt byte-for-byte (the body is identical).
    pri = build_prompt(make_dataset(KEPLER, "priors", 0), [], PREVIEW)
    ano = build_prompt(make_dataset(KEPLER, "anon", 0), [], PREVIEW)
    reconstructed = (
        ano.replace("Sensor_X", "orbital_radius").replace("Sensor_Y", "orbital_period")
    )
    # the only remaining difference is the single Context hint line
    assert reconstructed + f"\n\nContext: {HINT}" != pri  # hint isn't appended at end
    # remove the hint line from priors -> must equal the reconstructed anon body
    pri_no_hint = pri.replace(f"\n\nContext: {HINT}", "")
    assert pri_no_hint == reconstructed


def test_parse_keeps_only_valid_blocks():
    response = (
        "Here are my ideas:\n\n"
        f"```python\n{VALID_A}\n```\n\n"
        "```python\ndef evaluate_law(inputs, params)\n    return params[0]\n```\n\n"  # syntax error
        f"```python\n{VALID_B}\n```\n\n"
        "```\nThis block is just prose, no code here.\n```\n"  # prose
    )
    progs = parse_programs(response)
    assert len(progs) == 2
    assert any("** params[1]" in p for p in progs)
    assert all("evaluate_law" in p and "N_PARAMS" in p for p in progs)


def test_parse_empty_and_no_code_return_empty():
    assert parse_programs("") == []
    assert parse_programs("just some prose, nothing fenced at all") == []
    # a fenced block that parses but lacks the contract is dropped too
    assert parse_programs("```python\nx = 1 + 2\n```") == []


def test_exemplars_change_prompt_diversity_vs_refine():
    ds = make_dataset(KEPLER, "anon", 0)
    early = build_prompt(ds, [], PREVIEW)
    late = build_prompt(ds, [VALID_A], PREVIEW)
    assert early != late
    assert "DIVERSE" in early
    assert "REFINE" in late
    assert VALID_A in late  # the seed is shown


def test_system_prompt_forbids_hardcoded_constants():
    assert "params[i]" in SYSTEM_PROMPT
    assert "N_PARAMS" in SYSTEM_PROMPT
    assert "evaluate_law" in SYSTEM_PROMPT
