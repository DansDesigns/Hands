@echo on
setlocal EnableExtensions
cd /d "%~dp0"

echo ============================================================
echo Spatial Desktop Prototype - Windows installer
echo Working directory: %CD%
echo ============================================================

echo [1/9] Checking the Python Launcher...
where py >nul 2>&1
if errorlevel 1 (
  call :install_python
  if errorlevel 1 goto :failed
)

set "PYTHON_CMD="
echo [2/9] Searching for Python 3.10, 3.11, or 3.12...
for %%V in (3.12 3.11 3.10) do (
    py -%%V -c "import sys; print(sys.version)" 2>nul
    if not errorlevel 1 if not defined PYTHON_CMD set "PYTHON_CMD=py -%%V"
)
if not defined PYTHON_CMD (
  call :install_python
  if errorlevel 1 goto :failed
  set "PYTHON_CMD="
  for %%V in (3.12 3.11 3.10) do (
      py -%%V -c "import sys; print(sys.version)" 2>nul
      if not errorlevel 1 if not defined PYTHON_CMD set "PYTHON_CMD=py -%%V"
  )
)
if not defined PYTHON_CMD goto :no_supported_python

echo Selected interpreter: %PYTHON_CMD%
echo.
echo [3/9] Creating the virtual environment (this may take a moment)...
if exist ".venv\Scripts\python.exe" (
    echo Existing virtual environment found; reusing it.
) else (
    %PYTHON_CMD% -m venv .venv
    if errorlevel 1 goto :error
)

set "VPY=%CD%\.venv\Scripts\python.exe"
echo.
echo [4/9] Installing Spatial Desktop and local document-search dependencies...
"%VPY%" -c "import pip; print('Using pip', pip.__version__)"
if errorlevel 1 (
  echo pip is missing; bootstrapping it with ensurepip...
  "%VPY%" -m ensurepip
  if errorlevel 1 goto :error
)
"%VPY%" -u -m pip install --verbose -r requirements.txt
if errorlevel 1 goto :error

echo.
echo [4b] Installing the optional 3D hand renderer (moderngl / OpenGL)...
"%VPY%" -u -m pip install -r requirements-3d.txt
if errorlevel 1 (
  echo WARNING: 3D hand rendering could not be installed. The Skeleton and Flat hand styles still work.
  echo Retry later with: .venv\Scripts\python.exe -m pip install -r requirements-3d.txt
)

echo.
echo [5/8] Installing optional offline StreamTTS runtime...
"%VPY%" -u -m pip install --verbose -r requirements-voice.txt
if errorlevel 1 (
  echo WARNING: StreamTTS dependencies could not be installed. Hand tracking and file search will still work.
  echo You can retry later with: .venv\Scripts\python.exe -m pip install -r requirements-voice.txt
) else (
  echo.
  echo [6/9] Downloading StreamTTS voice files for offline speech...
  echo This download is only needed once and is skipped if files already exist.
  "%VPY%" -u install_voice_models.py
  if errorlevel 1 (
    echo WARNING: StreamTTS model download failed. The app will still install.
    echo Retry later with: .venv\Scripts\python.exe install_voice_models.py
  )
)

echo.
echo [7/9] Downloading optional local face welcome models...
echo YuNet/SFace run locally. Face recognition remains disabled until enrolled in Settings.
"%VPY%" -u install_face_models.py
if errorlevel 1 (
  echo WARNING: Face-recognition model download failed. The app will still work without Face Welcome.
  echo Retry later with: .venv\Scripts\python.exe install_face_models.py
)

echo.
echo [8/9] Downloading the default quantized Liquid AI d1-omni-600M model...
echo The Q8 model plus projector is about 670 MB. The installer also downloads llama.cpp for local inference.
echo The d1-omni-600M Q8 model is the only model option enabled for now.
"%VPY%" -u install_d1_model.py 600m
if errorlevel 1 (
  echo WARNING: d1 setup did not complete. Spatial Desktop remains usable without d1.
  echo Retry later with: .venv\Scripts\python.exe install_d1_model.py 600m
)


echo.
echo [9/9] Assigning the local Freehand launcher icon and creating the Start Menu shortcut...
set "ICON_PNG=%CD%\Freehand_logo_small.png"
set "ICON_ICO=%CD%\Freehand_logo_small.ico"
rem Windows shortcuts need an ICO resource; convert the bundled PNG locally, without network access.
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; try { if (Test-Path $env:ICON_PNG) { Add-Type -AssemblyName System.Drawing; $src=[System.Drawing.Image]::FromFile($env:ICON_PNG); try { $bmp=[System.Drawing.Bitmap]::new(256,256,[System.Drawing.Imaging.PixelFormat]::Format32bppArgb); try { $g=[System.Drawing.Graphics]::FromImage($bmp); try { $g.Clear([System.Drawing.Color]::Transparent); $scale=[Math]::Min(256.0/$src.Width,256.0/$src.Height); $w=[int]($src.Width*$scale); $h=[int]($src.Height*$scale); $g.DrawImage($src,[int]((256-$w)/2),[int]((256-$h)/2),$w,$h) } finally { $g.Dispose() }; $hicon=$bmp.GetHicon(); $icon=[System.Drawing.Icon]::FromHandle($hicon); $fs=[System.IO.File]::Create($env:ICON_ICO); try { $icon.Save($fs) } finally { $fs.Dispose() } } finally { $bmp.Dispose() } } finally { $src.Dispose() } } } catch { Write-Warning ('Could not prepare Freehand launcher icon: ' + $_.Exception.Message) }"
if not exist "%ICON_ICO%" echo WARNING: Freehand_logo_small.png is missing or could not be converted; the shortcut will use the Windows default icon.
set "STARTMENU=%APPDATA%\Microsoft\Windows\Start Menu\Programs"
if not exist "%STARTMENU%" mkdir "%STARTMENU%"
set "SHORTCUT=%STARTMENU%\Spatial Desktop Prototype.lnk"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$s=(New-Object -ComObject WScript.Shell).CreateShortcut('%SHORTCUT%'); $s.TargetPath='%VPY%'; $s.Arguments='""%CD%\main.py""'; $s.WorkingDirectory='%CD%'; $s.Description='Spatial Desktop Prototype'; if (Test-Path $env:ICON_ICO) { $s.IconLocation=$env:ICON_ICO + ',0' }; $s.Save()"
if errorlevel 1 goto :shortcut_error

echo.
echo ============================================================
echo INSTALLATION COMPLETE
echo Start Menu shortcut: Spatial Desktop Prototype
echo Project folder: %CD%
echo Local search uses SQLite and remains available without d1. d1 is an optional local decision model.
echo StreamTTS runs offline when its runtime and voice files are installed. d1 runs locally after its model is cached.
echo Face Welcome is optional and local-only; enroll a user through Settings to enable it.
echo ============================================================
pause
exit /b 0

:no_supported_python
echo ERROR: Automatic setup could not find a supported Python 3.10, 3.11, or 3.12 installation.
echo Install a supported Python version and rerun this installer.
goto :failed

:install_python
echo.
echo Python 3.10, 3.11, or 3.12 was not found. Attempting automatic installation...
where winget >nul 2>&1
if not errorlevel 1 (
  winget install --id Python.Python.3.12 --exact --silent --accept-package-agreements --accept-source-agreements
  if errorlevel 1 echo winget installation failed or was cancelled; trying the official Python installer...
)
rem If winget is unavailable or did not register the launcher, download the official 64-bit Python 3.12 installer.
if exist "%LocalAppData%\Programs\Python\Python312\python.exe" (
  set "PATH=%LocalAppData%\Programs\Python\Python312;%LocalAppData%\Programs\Python\Python312\Scripts;%LocalAppData%\Programs\Python\Launcher;%PATH%"
  "%LocalAppData%\Programs\Python\Python312\python.exe" --version
  exit /b 0
)
where py >nul 2>&1
if not errorlevel 1 (
  py -3.12 -c "import sys; sys.exit(0 if sys.version_info[:2] == (3,12) else 1)" >nul 2>&1
  if not errorlevel 1 exit /b 0
)
set "PY_INSTALLER=%TEMP%\python-3.12.10-amd64.exe"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; Invoke-WebRequest -UseBasicParsing 'https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe' -OutFile $env:PY_INSTALLER"
if errorlevel 1 (
  echo ERROR: Could not download Python automatically. Check your internet connection or install Python 3.12 from https://www.python.org/downloads/windows/.
  exit /b 1
)
"%PY_INSTALLER%" /quiet InstallAllUsers=0 PrependPath=1 Include_launcher=1 Include_pip=1 Include_test=0 SimpleInstall=1
if errorlevel 1 (
  echo ERROR: The Python installer failed. Install Python 3.12 from https://www.python.org/downloads/windows/ and rerun.
  exit /b 1
)
rem Refresh this process PATH because a child installer cannot update the current command prompt's environment.
set "PATH=%LocalAppData%\Programs\Python\Python312;%LocalAppData%\Programs\Python\Python312\Scripts;%LocalAppData%\Programs\Python\Launcher;%PATH%"
if exist "%LocalAppData%\Programs\Python\Python312\python.exe" (
  "%LocalAppData%\Programs\Python\Python312\python.exe" --version
  exit /b 0
)
where py >nul 2>&1
if not errorlevel 1 exit /b 0
echo ERROR: Python installation completed but no usable interpreter was found.
exit /b 1

:shortcut_error
echo ERROR: Package installation finished, but Start Menu shortcut creation failed.
goto :failed

:error
echo ERROR: A required setup command failed. Read the output above.
:failed
echo Installation failed.
pause
exit /b 1
