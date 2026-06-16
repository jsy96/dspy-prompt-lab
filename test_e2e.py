"""End-to-end smoke test — exercises both endpoints against the REAL DeepSeek API.

Requires DEEPSEEK_API_KEY in the environment. Costs a handful of DeepSeek calls
(bootstrap teacher + judge + 2 compare calls).

Run (this machine uses uv, no system python):
    DEEPSEEK_API_KEY=sk-xxx uv run --no-project \\
        --with 'dspy==3.2.1' --with 'flask>=3.0,<4.0' \\
        --env-file .env   # or export DEEPSEEK_API_KEY first
        python test_e2e.py

On Windows git bash, set the key first:
    export DEEPSEEK_API_KEY=sk-xxx
    PYTHONPATH=api uv run --no-project --with 'dspy==3.2.1' --with 'flask>=3.0,<4.0' python test_e2e.py
"""
import os
import sys

# Make `api/` importable (dspy_lab, app) without a running server.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "api"))

from app import app  # noqa: E402


def main() -> int:
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("ERROR: set DEEPSEEK_API_KEY first (e.g. export DEEPSEEK_API_KEY=sk-...)")
        return 1

    client = app.test_client()

    print("=== /api/optimize ===")
    r = client.post(
        "/api/optimize",
        json={
            "examples": [
                {"input": "Translate: good morning", "output": "早上好"},
                {"input": "Translate: thank you", "output": "谢谢"},
                {"input": "Translate: sorry", "output": "对不起"},
            ],
            "instruction": "Translate English phrases to Simplified Chinese.",
            "max_bootstrapped_demos": 3,
        },
    )
    d = r.get_json()
    assert d and d.get("ok"), f"optimize failed: {d}"
    print("ok=True  stats=", d["stats"])
    print("instruction:", d["instruction"])
    for i, x in enumerate(d.get("demos", []), 1):
        print(f"  demo#{i}  in={x['input']!r}  out={x['output']!r}")
    assert d["stats"]["demos_total"] >= 1, "expected at least one demo"

    print("\n=== /api/compare ===")
    r2 = client.post(
        "/api/compare",
        json={
            "plain_prompt": "把下面的英文翻译成中文。",
            "instruction": d["instruction"],
            "demos": d["demos"],
            "test_input": "Translate: good night",
        },
    )
    e = r2.get_json()
    assert e and e.get("ok"), f"compare failed: {e}"
    print("ok=True  elapsed_ms=", e["elapsed_ms"])
    print("plain     :", (e.get("plain") or {}).get("output"))
    print("optimized :", (e.get("optimized") or {}).get("output"))

    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
