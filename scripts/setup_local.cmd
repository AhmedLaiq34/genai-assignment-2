@echo off
setlocal
set "PROJECT_ROOT=%~dp0.."
for %%I in ("%PROJECT_ROOT%") do set "PROJECT_ROOT=%%~fI"
cd /d "%PROJECT_ROOT%"
set "HF_HOME=%PROJECT_ROOT%\.cache\huggingface"
set "HF_HUB_CACHE=%HF_HOME%\hub"
set "UV_CACHE_DIR=%PROJECT_ROOT%\.cache\uv"
set "PIP_CACHE_DIR=%PROJECT_ROOT%\.cache\pip"
set "PROJECT_RUNTIME_PROFILE=local"

if not exist ".venv\Scripts\python.exe" (
  uv venv --python 3.11 ".venv"
  if errorlevel 1 exit /b 1
)

if exist "requirements-local.lock" (
  uv pip install --python ".venv\Scripts\python.exe" -r "requirements-local.lock" --extra-index-url https://download.pytorch.org/whl/cu128
  if errorlevel 1 exit /b 1
) else (
  uv pip install --python ".venv\Scripts\python.exe" torch --index-url https://download.pytorch.org/whl/cu128
  if errorlevel 1 exit /b 1
  uv pip install --python ".venv\Scripts\python.exe" -r "requirements-base.txt"
  if errorlevel 1 exit /b 1
)
uv pip freeze --python ".venv\Scripts\python.exe" > "requirements-local.lock"
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" "scripts\doctor.py" --smoke-test --output "results\environments\local.json"
if errorlevel 1 exit /b 1
echo Local environment setup completed.
