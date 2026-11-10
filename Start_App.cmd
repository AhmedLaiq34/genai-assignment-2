@echo off
setlocal
set "PROJECT_ROOT=%~dp0"
if not exist "%PROJECT_ROOT%app\main.py" (
  echo The local screening application entry point is missing.
  pause
  exit /b 2
)
if not exist "%PROJECT_ROOT%.venv\Scripts\python.exe" (
  call "%PROJECT_ROOT%Setup_Local.cmd"
  if errorlevel 1 (
    echo Local environment setup failed.
    pause
    exit /b 1
  )
)
cd /d "%PROJECT_ROOT%"
".venv\Scripts\python.exe" "scripts\launch_app.py"
if errorlevel 1 pause
exit /b %errorlevel%
