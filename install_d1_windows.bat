@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Please run install_windows.bat first so the Spatial Desktop virtual environment exists.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -u install_d1_model.py 600m
pause
