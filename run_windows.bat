@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  set "PYTHON_OK="
  for %%V in (3.12 3.11 3.10) do (
    py -%%V -c "import sys; print(sys.version)" >nul 2>&1
    if not errorlevel 1 set "PYTHON_OK=1"
  )
  if not defined PYTHON_OK (
    echo A supported Python installation is missing. Starting the installer...
    call install_windows.bat
    if errorlevel 1 exit /b 1
  ) else (
    py -3 -m venv .venv
    if errorlevel 1 (
      echo Could not create the virtual environment. Starting the installer...
      call install_windows.bat
      if errorlevel 1 exit /b 1
    )
  )
)

if not exist ".venv\Scripts\python.exe" (
  echo ERROR: Python setup did not create the virtual environment.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" main.py
exit /b %errorlevel%

:failed
echo ERROR: Setup or application launch failed.
pause
exit /b 1
