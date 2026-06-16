"""Shared DSPy + DeepSeek helpers for dspy-prompt-lab.

All DeepSeek calls (bootstrap teacher, LLM-as-judge metric, compare) reuse a
single ``DEEPSEEK_API_KEY`` environment variable, read at call time — never
hardcoded. Configured for Vercel's serverless filesystem (read-only): DSPy's
disk cache is disabled so nothing writes to ``~/.dspy_cache``.
"""
from __future__ import annotations

import os

import dspy
from dspy.signatures import make_signature

# DeepSeek OpenAI-compatible endpoint. Both "https://api.deepseek.com" and
# ".../v1" work; the bare host is used so litellm's openai provider appends
# "/chat/completions" to a path DeepSeek accepts.
DEEPSEEK_BASE = "https://api.deepseek.com"

# DeepSeek-V4-Flash (non-thinking mode), available since 2026-04-24. The legacy
# "deepseek-chat" alias still routes to this same model but is fully retired on
# 2026/07/24 15:59 UTC (after that it 404s), so we use the explicit new name.
# Base URL and API key are unchanged; only the model string moves.
DEEPSEEK_MODEL = "deepseek-v4-flash"

DEFAULT_INSTRUCTION = (
    "Given the input, produce the best possible output that fulfills the task."
)

# ChainOfThought auto-prepends a `reasoning` output field, so a Signature
# declared as "input -> output" becomes "input -> reasoning, output" in the
# compiled program. compare() rebuilds with the same expanded form.
SIGNATURE_STRING = "input -> output"
SIGNATURE_STRING_COT = "input -> reasoning, output"

_configured = False


def configure() -> None:
    """Idempotently disable DSPy disk cache (serverless has a read-only FS).

    Safe to call on every request: a module-level guard makes it a no-op after
    the first invocation within a warm function instance.
    """
    global _configured
    if _configured:
        return
    # Memory-only cache keeps cross-call speed within one warm instance without
    # touching disk. See dspy/clients/__init__.py:19-55.
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=True)
    _configured = True


def _require_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY environment variable is not set. "
            "Add it in the Vercel project settings (or export it locally)."
        )
    return key


def make_lm(max_tokens: int = 512, temperature: float = 0.0) -> dspy.LM:
    """Return a DSPy LM pointing at DeepSeek's OpenAI-compatible endpoint."""
    configure()
    return dspy.LM(
        f"openai/{DEEPSEEK_MODEL}",
        api_base=DEEPSEEK_BASE,
        api_key=_require_key(),
        cache=False,  # serverless double-insurance: never read/write disk cache
        max_tokens=max_tokens,
        temperature=temperature,
        num_retries=1,  # cap tail latency under DeepSeek 429 storms (mirrored in _deepseek_completion)
    )


def build_signature(instruction: str | None):
    """Build the user-task Signature. Empty instruction falls back to a default."""
    instr = (instruction or "").strip() or DEFAULT_INSTRUCTION
    return make_signature(SIGNATURE_STRING, instructions=instr)


def build_cot_signature(instruction: str | None):
    """Expanded form used when reconstructing a compiled program in compare()."""
    instr = (instruction or "").strip() or DEFAULT_INSTRUCTION
    return make_signature(SIGNATURE_STRING_COT, instructions=instr)
