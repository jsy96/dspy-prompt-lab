@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM  dspy-prompt-lab local launcher
REM  Serves the frontend (index.html) + /api/* on one Flask server.
REM  LLM = GLM-5.1 via the Zhipu anthropic endpoint (needs GLM_API_KEY).
REM  Falls back to DeepSeek if only DEEPSEEK_API_KEY is set.
REM ============================================================

set "PORT=5000"

REM --- locate uv (no system python on this box; uv manages everything) ---
set "UV="
where uv >nul 2>nul && set "UV=uv"
if not defined UV if exist "C:\ProgramData\chocolatey\bin\uv.exe" set "UV=C:\ProgramData\chocolatey\bin\uv.exe"
if not defined UV (
    echo [ERROR] uv not found on PATH. Install it: https://docs.astral.sh/uv/
    exit /b 1
)

REM --- optional .env loader (KEY=VALUE lines, '#' comments) ---
REM     Existing environment variables always win (dotenv semantics).
if exist ".env" (
    for /f "usebackq eol=# tokens=1,* delims==" %%a in (".env") do (
        if not "%%a"=="" if not defined %%a set "%%a=%%b"
    )
)

REM --- require at least one provider key ---
set "HAS_KEY=0"
if defined GLM_API_KEY set "HAS_KEY=1"
if defined DEEPSEEK_API_KEY set "HAS_KEY=1"
if "!HAS_KEY!"=="0" (
    echo [ERROR] No LLM key found in the environment or in .env.
    echo [ERROR] For GLM-5.1:  set GLM_API_KEY=your-zhipu-token
    echo [ERROR] or create a .env file next to this script with:  GLM_API_KEY=your-zhipu-token
    exit /b 1
)

if defined GLM_API_KEY (
    echo [INFO] LLM provider: GLM-5.1 ^(Zhipu anthropic endpoint^)
) else (
    echo [INFO] LLM provider: DeepSeek ^(GLM_API_KEY not set, falling back to DEEPSEEK_API_KEY^)
)
echo [INFO] Serving on http://127.0.0.1:!PORT!  ^(Ctrl+C to stop^)

REM dspy_lab / app live under api/, so put that on the module search path.
set "PYTHONPATH=api"
"!UV!" run --no-project --with "dspy==3.2.1" --with "flask>=3.0,<4.0" python run_local.py

endlocal
