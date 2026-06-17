"""dspy-prompt-lab backend — a Flask single-app.

Two endpoints, both backed by the real Python ``dspy`` package:

  POST /api/optimize
      Run dspy.BootstrapFewShot over user-supplied (input, output) examples and
      return the optimized prompt (instruction + few-shot demos).

  POST /api/compare
      Feed a hand-written "plain prompt" and the "optimized prompt" to the same
      LLM on one test input and return both outputs + latencies.

  GET  /api/health
      Cheap liveness probe that does not call the LLM.

The LLM is chosen by which API key is set: GLM_API_KEY -> Zhipu GLM-5.1 via the
anthropic-compatible endpoint, otherwise DEEPSEEK_API_KEY -> DeepSeek. The key is
read from the environment on every call (never hardcoded). Deployed as a Vercel
Function; locally the same app runs via `run_local.py` / `start.bat`.
"""
from __future__ import annotations

import os
import sys

# Vercel's @vercel/python runtime imports this entry module by absolute path but
# does not put its own directory on sys.path, so the sibling `dspy_lab` helper
# raises ModuleNotFoundError in production (locally it only works because we run
# with PYTHONPATH=api). Insert this file's directory first so `from dspy_lab
# import ...` resolves identically in both environments.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time

import dspy
import litellm
from dspy.adapters import ChatAdapter
from dspy.predict import ChainOfThought, Predict
from dspy.teleprompt import BootstrapFewShot
from flask import Flask, jsonify, request, send_from_directory

from dspy_lab import (
    DEFAULT_INSTRUCTION,
    LLM_BASE,
    LLM_MODEL,
    LLM_PREFIX,
    build_signature,
    build_cot_signature,
    configure,
    make_lm,
)

app = Flask(__name__)
# Keep CJK / unicode readable in JSON responses instead of \uXXXX escapes.
app.json.ensure_ascii = False

MAX_EXAMPLES = 8
MAX_BOOTSTRAPPED_DEMOS = 3


def _err(msg: str, code: int = 400):
    return jsonify({"ok": False, "error": msg}), code


def _llm_completion(key: str, messages, *, max_tokens: int, temperature: float = 0.0):
    """One direct LLM call via litellm on the active provider (GLM or DeepSeek)."""
    return litellm.completion(
        model=f"{LLM_PREFIX}/{LLM_MODEL}",
        api_base=LLM_BASE,
        api_key=key,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        num_retries=1,  # cap tail latency under 429 storms (mirrored in make_lm)
    )


@app.post("/api/optimize")
def optimize():
    """BootstrapFewShot -> export instruction + demos as the optimized prompt."""
    body = request.get_json(force=True, silent=True) or {}

    examples = body.get("examples") or []
    if not isinstance(examples, list) or not (1 <= len(examples) <= MAX_EXAMPLES):
        return _err(f"`examples` must be a list of 1..{MAX_EXAMPLES} objects.")
    cleaned = []
    for e in examples:
        if not isinstance(e, dict):
            return _err("each example must be an object with `input` and `output`.")
        inp = (e.get("input") or "").strip()
        out = (e.get("output") or "").strip()
        if not inp or not out:
            return _err("each example needs non-empty `input` and `output`.")
        cleaned.append({"input": inp, "output": out})

    try:
        max_d = int(body.get("max_bootstrapped_demos", MAX_BOOTSTRAPPED_DEMOS))
    except (TypeError, ValueError):
        max_d = MAX_BOOTSTRAPPED_DEMOS
    max_d = max(1, min(MAX_BOOTSTRAPPED_DEMOS, max_d))

    instruction = (body.get("instruction") or "").strip()

    try:
        configure()
        lm = make_lm(max_tokens=512, temperature=0.0)
        dspy.configure(lm=lm)
        from dspy_lab import _require_key  # local import keeps top-level light
        key = _require_key()

        signature = build_signature(instruction)
        student = ChainOfThought(signature)

        trainset = [
            dspy.Example(input=e["input"], output=e["output"]).with_inputs("input")
            for e in cleaned
        ]

        # LLM-as-judge metric: a tiny DeepSeek call deciding whether the teacher's
        # output is semantically equivalent to the user's expected output.
        def metric(example, pred, trace=None):
            expected = (getattr(example, "output", "") or "").strip()
            got = (getattr(pred, "output", "") or "").strip()
            if not got:
                return False
            prompt = (
                "Judge whether the Candidate is semantically equivalent to the "
                "Reference for the task. Reply with ONLY one word: yes or no.\n\n"
                f"Reference: {expected}\nCandidate: {got}"
            )
            try:
                resp = _llm_completion(
                    key,
                    [{"role": "user", "content": prompt}],
                    max_tokens=4,
                    temperature=0.0,
                )
                verdict = (resp.choices[0].message.content or "").strip().lower()
                return verdict.startswith("y")
            except Exception:
                # A failed judge must not crash bootstrap — just drop this demo.
                return False

        optimizer = BootstrapFewShot(
            metric=metric,
            max_bootstrapped_demos=max_d,
            max_labeled_demos=4,
            max_rounds=1,
            max_errors=3,
        )
        compiled = optimizer.compile(student, trainset=trainset)

        # --- export the optimized prompt ------------------------------------
        adapter = ChatAdapter()
        predictors = compiled.named_predictors()
        pred = predictors[0][1] if predictors else None

        demos_out = []
        template_text = ""
        out_instruction = instruction or DEFAULT_INSTRUCTION
        bootstrapped = 0
        if pred is not None:
            out_instruction = pred.signature.instructions or out_instruction
            system_msg = adapter.format_system_message(pred.signature)
            demos_msgs = adapter.format_demos(pred.signature, pred.demos)
            for ex in pred.demos:
                if getattr(ex, "augmented", False):
                    bootstrapped += 1
                demos_out.append(
                    {
                        "input": getattr(ex, "input", "") or "",
                        "reasoning": getattr(ex, "reasoning", "") or "",
                        "output": getattr(ex, "output", "") or "",
                    }
                )
            parts = [f"[SYSTEM]\n{system_msg}"]
            for m in demos_msgs:
                parts.append(f"[{str(m.get('role')).upper()}]\n{m.get('content', '')}")
            template_text = "\n\n".join(parts)

        return jsonify(
            {
                "ok": True,
                "instruction": out_instruction,
                "demos": demos_out,
                "template_text": template_text,
                "stats": {
                    "examples": len(cleaned),
                    "bootstrapped": bootstrapped,
                    "demos_total": len(demos_out),
                },
                "note": "" if demos_out else (
                    "No demos passed the judge metric; returning the base "
                    "instruction with your examples as plain few-shot."
                ),
            }
        )
    except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
        return _err(f"optimization failed: {exc}", 500)


@app.post("/api/compare")
def compare():
    """Run the plain prompt and the optimized prompt on one test input."""
    body = request.get_json(force=True, silent=True) or {}

    plain_prompt = (body.get("plain_prompt") or "").strip()
    test_input = (body.get("test_input") or "").strip()
    demos = body.get("demos") or []
    instruction = (body.get("instruction") or "").strip()

    if not test_input:
        return _err("`test_input` is required.")
    has_optimized = bool(demos) or bool(instruction)
    if not plain_prompt and not has_optimized:
        return _err("Provide a `plain_prompt` and/or optimized (`instruction`+`demos`).")

    try:
        from dspy_lab import _require_key
        key = _require_key()
    except RuntimeError as exc:
        return _err(str(exc), 500)

    t_start = time.time()
    result = {"ok": True, "plain": None, "optimized": None}

    # --- plain side: user's hand-written prompt + the test input ------------
    if plain_prompt:
        try:
            t0 = time.time()
            resp = _llm_completion(
                key,
                [{"role": "user", "content": f"{plain_prompt}\n\nInput:\n{test_input}"}],
                max_tokens=512,
            )
            result["plain"] = {
                "output": (resp.choices[0].message.content or "").strip(),
                "latency_ms": int((time.time() - t0) * 1000),
            }
        except Exception as exc:  # noqa: BLE001
            result["plain"] = {"output": "", "latency_ms": 0, "error": str(exc)}

    # --- optimized side: real dspy.Predict with the compiled demos ----------
    if has_optimized:
        try:
            configure()
            dspy.configure(lm=make_lm(max_tokens=512, temperature=0.0))
            sig = build_cot_signature(instruction)
            pred = Predict(sig)
            pred.demos = [
                dspy.Example(
                    input=d.get("input", ""),
                    reasoning=d.get("reasoning", "") or "",
                    output=d.get("output", ""),
                )
                for d in demos
                if isinstance(d, dict)
            ]
            t0 = time.time()
            res = pred(input=test_input)
            result["optimized"] = {
                "output": (getattr(res, "output", "") or "").strip(),
                "latency_ms": int((time.time() - t0) * 1000),
            }
        except Exception as exc:  # noqa: BLE001
            result["optimized"] = {"output": "", "latency_ms": 0, "error": str(exc)}

    result["elapsed_ms"] = int((time.time() - t_start) * 1000)
    return jsonify(result)


@app.get("/api/health")
def health():
    """Cheap liveness probe that does not call DeepSeek."""
    return jsonify({"ok": True, "model": LLM_MODEL})


# Project root (this file lives in api/, so root is one level up). Used only by
# the local `/` route that serves the single-page frontend; on Vercel the static
# index.html is served directly and this route is never reached.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@app.get("/")
def index():
    """Serve the single-page frontend so the local server is a one-stop URL."""
    return send_from_directory(_ROOT, "index.html")


# Expose `app` for Vercel's Python runtime (it looks for an `app` variable at
# supported entrypoints such as api/app.py).
app = app
