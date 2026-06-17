"""Local launcher for dspy-prompt-lab.

Serves the single-page frontend (index.html) + every /api/* endpoint on one
Flask dev server, so opening the printed URL gives the whole app. The LLM
provider is chosen by which key is set: GLM_API_KEY -> GLM-5.1, otherwise
DEEPSEEK_API_KEY -> DeepSeek.

Run via start.bat, or directly:
    set PYTHONPATH=api
    uv run --no-project --with "dspy==3.2.1" --with "flask>=3.0,<4.0" python run_local.py
"""
from __future__ import annotations

import os

from app import app


def main() -> None:
    port = int(os.environ.get("PORT", "5000"))
    host = os.environ.get("HOST", "127.0.0.1")
    # The provider block in dspy_lab runs on import, so report which one won.
    from dspy_lab import KEY_ENV, LLM_MODEL

    print(f"[dspy-prompt-lab] model={LLM_MODEL} (key from ${KEY_ENV})")
    print(f"[dspy-prompt-lab] serving on http://{host}:{port}  (Ctrl+C to stop)")
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
