@echo off
set "PROJECT_ROOT=%~dp0.."
for %%I in ("%PROJECT_ROOT%") do set "PROJECT_ROOT=%%~fI"
set "HF_HOME=%PROJECT_ROOT%\.cache\huggingface"
set "HF_HUB_CACHE=%HF_HOME%\hub"
set "UV_CACHE_DIR=%PROJECT_ROOT%\.cache\uv"
set "PIP_CACHE_DIR=%PROJECT_ROOT%\.cache\pip"
set "PROJECT_RUNTIME_PROFILE=local"
call "%PROJECT_ROOT%\.venv\Scripts\activate.bat"
