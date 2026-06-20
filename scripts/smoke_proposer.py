"""STANDALONE manual smoke (NOT pytest, NOT imported by the engine, NOT in the suite).

The ONLY thing in the repo that hits the network. Purpose:
  (1) verify "≥20 valid distinct functions per batch on Tier 0",
  (2) probe the Prime Intellect rate limit before any overnight run.
Reuses proposer/evaluator/datasets — reimplements nothing. Run: python smoke_proposer.py
"""

from dotenv import load_dotenv
load_dotenv()                      # reads .env -> os.environ
from openai import OpenAI
from openai import RateLimitError, AuthenticationError
import os, ast

from core.proposer import SYSTEM_PROMPT, build_prompt, parse_programs
from core.evaluator import score_program
from core.datasets import make_tier0

MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"

_kwargs = dict(api_key=os.environ["PRIME_API_KEY"], base_url="https://api.pinference.ai/api/v1")
if os.environ.get("PRIME_TEAM_ID"):
    _kwargs["default_headers"] = {"X-Prime-Team-ID": os.environ["PRIME_TEAM_ID"]}
client = OpenAI(**_kwargs)


def structure_key(code: str) -> str:
    """Normalized structural key: ast.dump of evaluate_law body with numeric
    Constants flattened to a placeholder, so forms differing ONLY in constants
    count as the SAME structure."""
    tree = ast.parse(code)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "evaluate_law")

    class _Norm(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, (int, float, complex)):
                return ast.copy_location(ast.Constant(value=0), node)
            return node

    return "\n".join(ast.dump(_Norm().visit(s)) for s in fn.body)


def main():
    ds = make_tier0(seed=0)
    preview = "\n".join(", ".join(f"{v:.4f}" for v in row) for row in ds.train[:5])
    prompt = build_prompt(ds, exemplars=[], data_preview=preview) + (
        "\n\nPropose approximately 30 structurally distinct candidate forms, each "
        "in its own ```python block."
    )

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=4096,
        )
    except RateLimitError as e:
        print("RATE LIMITED (429) — this sets the search.py semaphore ceiling.")
        print("error:", e)
        hdrs = getattr(getattr(e, "response", None), "headers", {}) or {}
        print("retry-after:", hdrs.get("retry-after"))
        print("limit headers:", {k: v for k, v in hdrs.items() if "limit" in k.lower() or "remaining" in k.lower()})
        return
    except AuthenticationError as e:
        print("AUTH FAILED (401): API key likely missing the Inference permission — check the Prime dashboard.")
        print("error:", e)
        return

    text = resp.choices[0].message.content
    finish = resp.choices[0].finish_reason

    candidates = parse_programs(text)
    scored = [(c, score_program(c, ds)) for c in candidates]
    valid = [(c, r) for c, r in scored if r.valid]

    keys = set()
    for c, _ in valid:
        try:
            keys.add(structure_key(c))
        except Exception:
            pass
    distinct_valid = len(keys)

    u = resp.usage
    print("=== Tier 0 proposer smoke — funnel ===")
    print(f"requested        : ~30 (target stated in prompt)")
    print(f"finish_reason    : {finish}        # 'length' => OUTPUT TRUNCATED, distinct undercount is an artifact")
    print(f"parsed           : {len(candidates)}")
    print(f"valid            : {len(valid)}")
    print(f"distinct_valid   : {distinct_valid}")
    print(f"tokens           : prompt={u.prompt_tokens} completion={u.completion_tokens}")
    print(f"RESULT           : {'PASS' if distinct_valid >= 20 else f'FAIL ({distinct_valid})'}")
    print("--- interpretation ---")
    print("- low parsed       -> format/parse issue OR finish_reason=length truncation")
    print("- valid high, distinct low -> model repeating few forms; diversity too weak")
    print("- valid low        -> malformed programs; proposer contract not landing")
    print("note: distinct_valid = runnable + structurally unique, NOT correct. Most forms are wrong; that's expected.")
    print("usage:", u)


if __name__ == "__main__":
    main()
