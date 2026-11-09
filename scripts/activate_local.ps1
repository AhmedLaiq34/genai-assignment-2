$projectRoot = Split-Path -Parent $PSScriptRoot
$env:HF_HOME = Join-Path $projectRoot ".cache\huggingface"
$env:HF_HUB_CACHE = Join-Path $env:HF_HOME "hub"
$env:UV_CACHE_DIR = Join-Path $projectRoot ".cache\uv"
$env:PIP_CACHE_DIR = Join-Path $projectRoot ".cache\pip"
$env:PROJECT_RUNTIME_PROFILE = "local"
. (Join-Path $projectRoot ".venv\Scripts\Activate.ps1")
