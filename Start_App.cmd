@echo off
setlocal
set "PROJECT_ROOT=%~dp0"
if not exist "%PROJECT_ROOT%app\main.py" (
  echo The local screening application has not been implemented yet.
  echo It is scheduled for Phase 10.
  exit /b 2
)
if not exist "%PROJECT_ROOT%.venv\Scripts\python.exe" (
  call "%PROJECT_ROOT%Setup_Local.cmd"
  if errorlevel 1 exit /b 1
)
cd /d "%PROJECT_ROOT%"
set "HF_HOME=%PROJECT_ROOT%.cache\huggingface"
set "HF_HUB_CACHE=%HF_HOME%\hub"
set "PROJECT_RUNTIME_PROFILE=local"
".venv\Scripts\python.exe" "app\main.py"
exit /b %errorlevel%
