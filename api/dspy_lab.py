"""Shared DSPy + LLM helpers for dspy-prompt-lab.

The LLM provider is chosen by which API key is present in the environment, so
the same code runs both locally (GLM-5.1) and on Vercel (DeepSeek):

  GLM_API_KEY present  -> Zhipu GLM-5.1 via the anthropic-compatible endpoint
                         (https://open.bigmodel.cn/api/anthropic). This is the
                         GLM Coding plan that also powers Claude Code here; the
                         openai-compatible paas/v4 endpoint has no balance
                         (429 / code 1113 余额不足), so the anthropic one is used.
  otherwise            -> DeepSeek via its openai-compatible endpoint
                         (https://api.deepseek.com), the Vercel-deployed default.

The bootstrap teacher, the LLM-as-judge metric, and the compare side all reuse
the same provider + key, read at call time — never hardcoded. DSPy's disk cache
is disabled (serverless read-only FS); memory cache stays on for warm-instance
speed.
"""
from __future__ import annotations

import os

import dspy
from dspy.signatures import make_signature


def _select_provider() -> tuple[str, str, str, str]:
    """Pick (base_url, model, litellm_prefix, key_env) from the environment.

    GLM wins when GLM_API_KEY is set; otherwise DeepSeek keeps working for the
    Vercel deployment that only has DEEPSEEK_API_KEY configured.
    """
    if os.environ.get("GLM_API_KEY"):
        return (
            "https://open.bigmodel.cn/api/anthropic",  # anthropic-compat endpoint
            "glm-5.1",
            "anthropic",  # litellm provider prefix -> anthropic message format
            "GLM_API_KEY",
        )
    return (
        "https://api.deepseek.com",
        "deepseek-v4-flash",
        "openai",
        "DEEPSEEK_API_KEY",
    )


LLM_BASE, LLM_MODEL, LLM_PREFIX, KEY_ENV = _select_provider()

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
    key = os.environ.get(KEY_ENV)
    if not key:
        raise RuntimeError(
            f"{KEY_ENV} environment variable is not set. For local GLM-5.1 set "
            f"GLM_API_KEY=<your Zhipu token>; for DeepSeek set DEEPSEEK_API_KEY. "
            f"(On Vercel, add it in the project environment settings.)"
        )
    return key


def make_lm(max_tokens: int = 512, temperature: float = 0.0) -> dspy.LM:
    """Return a DSPy LM pointing at the active provider's endpoint."""
    configure()
    return dspy.LM(
        f"{LLM_PREFIX}/{LLM_MODEL}",
        api_base=LLM_BASE,
        api_key=_require_key(),
        cache=False,  # double-insurance: never read/write disk cache
        max_tokens=max_tokens,
        temperature=temperature,
        num_retries=1,  # cap tail latency under 429 storms
    )


def build_signature(instruction: str | None):
    """Build the user-task Signature. Empty instruction falls back to a default."""
    instr = (instruction or "").strip() or DEFAULT_INSTRUCTION
    return make_signature(SIGNATURE_STRING, instructions=instr)


def build_cot_signature(instruction: str | None):
    """Expanded form used when reconstructing a compiled program in compare()."""
    instr = (instruction or "").strip() or DEFAULT_INSTRUCTION
    return make_signature(SIGNATURE_STRING_COT, instructions=instr)
