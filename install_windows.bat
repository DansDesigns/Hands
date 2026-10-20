@echo off
setlocal EnableExtensions
cd /d "%~dp0"
rem Accept installation folders where main.py lives at the root or in Hands.
if not exist "main.py" if exist "Hands\main.py" cd /d "Hands"
if not exist "main.py" (
  echo ERROR: main.py was not found beside this installer or in the Hands subfolder.
  goto :failed
)
rem Keep original installer directory for bundled add-ons if main.py is in Hands.
set "FREEHAND_INSTALLER_DIR=%~dp0"
rem Repair a missing Start Menu shortcut without rerunning model downloads.
rem Usage: install_windows.bat --shortcut-only
if /I "%~1"=="--shortcut-only" goto :shortcut_step

echo ============================================================
echo Freehand: Spatial Computing - Windows installer
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
echo [3b/9] Installing Freehand Search and Indexing modules...
rem The distributed package includes these files. Stage them next to main.py;
rem do not download or replace user search settings or the SQLite index.
for %%F in (freehand_search_index.py apply_freehand_search_settings.py) do (
  if not exist "%%F" if exist "%FREEHAND_INSTALLER_DIR%%%F" (
    copy /Y "%FREEHAND_INSTALLER_DIR%%%F" "%%F" >nul
    if errorlevel 1 goto :search_setup_error
  )
  if not exist "%%F" (
    echo ERROR: Required bundled Search and Indexing module is missing: %%F
    echo Extract all files from the Freehand installer package into the app folder.
    goto :search_setup_error
  )
)
"%VPY%" -u apply_freehand_search_settings.py --main main.py
if errorlevel 1 goto :search_setup_error
"%VPY%" -c "from freehand_search_index import SearchIndex; print('Search indexing module ready; roots default to user home')"
if errorlevel 1 goto :search_setup_error

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
echo [4c/9] Enabling optional local music tag recognition ^(Mutagen^)...
"%VPY%" -c "import mutagen" >nul 2>&1
if errorlevel 1 (
  "%VPY%" -m pip install mutagen
  if errorlevel 1 echo WARNING: Music indexing still works, but embedded artist/genre tags may not be available.
)

echo.
echo [5/9] Installing optional offline StreamTTS runtime...
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


:shortcut_step
echo.
echo [9/9] Preparing the windowless launcher and creating the Start Menu shortcut...
rem The installer embeds a current launcher so a missing/outdated VBS does not
rem prevent Start Menu installation. Existing up-to-date launchers are preserved.
set "FREEHAND_LAUNCHER=%CD%\launch_windows.vbs"
set "FREEHAND_LAUNCHER_CURRENT=0"
if exist "%FREEHAND_LAUNCHER%" (
  findstr /L /C:"Installer shortcut mode: cscript" "%FREEHAND_LAUNCHER%" >nul 2>&1
  if not errorlevel 1 set "FREEHAND_LAUNCHER_CURRENT=1"
)
if "%FREEHAND_LAUNCHER_CURRENT%"=="0" (
  echo Restoring/upgrading launch_windows.vbs...
  call :write_launcher
  if errorlevel 1 goto :shortcut_error
)
if not exist "%CD%\.venv\Scripts\pythonw.exe" (
  echo ERROR: Cannot create a working shortcut because .venv\Scripts\pythonw.exe is missing.
  goto :shortcut_error
)
if not exist "%CD%\.venv\Scripts\python.exe" (
  echo ERROR: Cannot create a working shortcut because .venv\Scripts\python.exe is missing.
  goto :shortcut_error
)
set "ICON_PNG=%CD%\Freehand_logo_small.png"
set "ICON_ICO=%CD%\Freehand_logo_small.ico"
rem Icon generation is optional; a missing PNG must not prevent shortcut creation.
if not exist "%ICON_ICO%" powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; try { if (Test-Path $env:ICON_PNG) { Add-Type -AssemblyName System.Drawing; $src=[System.Drawing.Image]::FromFile($env:ICON_PNG); try { $bmp=[System.Drawing.Bitmap]::new(256,256,[System.Drawing.Imaging.PixelFormat]::Format32bppArgb); try { $g=[System.Drawing.Graphics]::FromImage($bmp); try { $g.Clear([System.Drawing.Color]::Transparent); $scale=[Math]::Min(256.0/$src.Width,256.0/$src.Height); $w=[int]($src.Width*$scale); $h=[int]($src.Height*$scale); $g.DrawImage($src,[int]((256-$w)/2),[int]((256-$h)/2),$w,$h) } finally { $g.Dispose() }; $hicon=$bmp.GetHicon(); $icon=[System.Drawing.Icon]::FromHandle($hicon); $fs=[System.IO.File]::Create($env:ICON_ICO); try { $icon.Save($fs) } finally { $fs.Dispose() } } finally { $bmp.Dispose() } } finally { $src.Dispose() } } } catch { Write-Warning ('Could not prepare Freehand launcher icon: ' + $_.Exception.Message) }"
rem Delegate link creation to the same VBS that will launch Freehand. Unlike
rem the previous PowerShell shortcut command this checks the link on disk and
rem reads the real Windows shell Programs directory, not an assumed APPDATA path.
cscript.exe //nologo "%FREEHAND_LAUNCHER%" --install-shortcut
if errorlevel 1 goto :shortcut_error

echo.
echo Start Menu entry verified: Freehand - Spatial Computing
if /I "%~1"=="--shortcut-only" goto :shortcut_repaired

echo.
echo ============================================================
echo INSTALLATION COMPLETE
echo Start Menu shortcut: Freehand - Spatial Computing
echo Project folder: %CD%
echo Search and Indexing is installed: home folder by default; add folders in Glass Settings - SEARCH.
echo The index updates in the background when Freehand launches; existing indexed folders are preserved.
echo Local search uses SQLite and remains available without d1. d1 is an optional local decision model.
echo StreamTTS runs offline when its runtime and voice files are installed. d1 runs locally after its model is cached.
echo Face Welcome is optional and local-only; enroll a user through Settings to enable it.
echo ============================================================
pause
exit /b 0


:shortcut_repaired
echo.
echo ============================================================
echo START MENU SHORTCUT REPAIRED
echo Open Windows Start and search for Freehand - Spatial Computing.
echo ============================================================
pause
exit /b 0

:write_launcher
rem Rebuild the launcher from a bundled copy; no extra download is required.
rem The base64 is intentionally ASCII to avoid CMD/VBS encoding corruption.
powershell -NoProfile -ExecutionPolicy Bypass -Command "[System.IO.File]::WriteAllBytes($env:FREEHAND_LAUNCHER,[Convert]::FromBase64String('JyBGcmVlaGFuZCBXaW5kb3dzIGxhdW5jaGVyOiB3aW5kb3dsZXNzIGJ5IGRlZmF1bHQsIGNvbnNvbGUgb25seSBmb3IgZGVidWdnaW5nLgonIEluc3RhbGxlciBzaG9ydGN1dCBtb2RlOiBjc2NyaXB0IC8vbm9sb2dvIGxhdW5jaF93aW5kb3dzLnZicyAtLWluc3RhbGwtc2hvcnRjdXQKT3B0aW9uIEV4cGxpY2l0CgpDb25zdCBGb3JSZWFkaW5nID0gMQpEaW0gZnMsIHNoZWxsLCByb290LCBweURpciwgZXhlLCBndWlQeXRob24sIGNvbnNvbGVQeXRob24sIG1haW5TY3JpcHQKRGltIGFwcERhdGEsIHByZWZlcmVuY2VzRmlsZSwgZGVidWdFbmFibGVkLCBjb21tYW5kTGluZQoKU2V0IGZzID0gQ3JlYXRlT2JqZWN0KCJTY3JpcHRpbmcuRmlsZVN5c3RlbU9iamVjdCIpClNldCBzaGVsbCA9IENyZWF0ZU9iamVjdCgiV1NjcmlwdC5TaGVsbCIpCnJvb3QgPSBmcy5HZXRQYXJlbnRGb2xkZXJOYW1lKFdTY3JpcHQuU2NyaXB0RnVsbE5hbWUpCm1haW5TY3JpcHQgPSBmcy5CdWlsZFBhdGgocm9vdCwgIm1haW4ucHkiKQpweURpciA9IGZzLkJ1aWxkUGF0aChmcy5CdWlsZFBhdGgocm9vdCwgIi52ZW52IiksICJTY3JpcHRzIikKZ3VpUHl0aG9uID0gZnMuQnVpbGRQYXRoKHB5RGlyLCAicHl0aG9udy5leGUiKQpjb25zb2xlUHl0aG9uID0gZnMuQnVpbGRQYXRoKHB5RGlyLCAicHl0aG9uLmV4ZSIpCgonIFRoZSBpbnN0YWxsZXIgdXNlcyB0aGlzIG1vZGUgdG8gY3JlYXRlIGFuZCBWRVJJRlkgdGhlIHNob3J0Y3V0IHdpdGhvdXQKJyBsYXVuY2hpbmcgRnJlZWhhbmQuIE5vcm1hbCBkb3VibGUtY2xpY2sgc3RhcnR1cCByZW1haW5zIHVuY2hhbmdlZC4KSWYgV1NjcmlwdC5Bcmd1bWVudHMuQ291bnQgPiAwIFRoZW4KICAgIElmIExDYXNlKENTdHIoV1NjcmlwdC5Bcmd1bWVudHMoMCkpKSA9ICItLWluc3RhbGwtc2hvcnRjdXQiIFRoZW4KICAgICAgICBJZiBJbnN0YWxsU3RhcnRNZW51U2hvcnRjdXQoKSBUaGVuCiAgICAgICAgICAgIFdTY3JpcHQuRWNobyAiRnJlZWhhbmQgU3RhcnQgTWVudSBzaG9ydGN1dCBpbnN0YWxsZWQgc3VjY2Vzc2Z1bGx5LiIKICAgICAgICAgICAgV1NjcmlwdC5RdWl0IDAKICAgICAgICBFbHNlCiAgICAgICAgICAgIFdTY3JpcHQuRWNobyAiRVJST1I6IENvdWxkIG5vdCBjcmVhdGUvdmVyaWZ5IHRoZSBGcmVlaGFuZCBTdGFydCBNZW51IHNob3J0Y3V0LiIKICAgICAgICAgICAgV1NjcmlwdC5RdWl0IDEKICAgICAgICBFbmQgSWYKICAgIEVuZCBJZgpFbmQgSWYKCklmIE5vdCBmcy5GaWxlRXhpc3RzKG1haW5TY3JpcHQpIFRoZW4KICAgIE1zZ0JveCAiRnJlZWhhbmQgY291bGQgbm90IGZpbmQgbWFpbi5weSBuZXh0IHRvIGxhdW5jaF93aW5kb3dzLnZicy4iLCB2YkNyaXRpY2FsLCAiRnJlZWhhbmQgTGF1bmNoZXIiCiAgICBXU2NyaXB0LlF1aXQgMQpFbmQgSWYKSWYgTm90IGZzLkZpbGVFeGlzdHMoZ3VpUHl0aG9uKSBPciBOb3QgZnMuRmlsZUV4aXN0cyhjb25zb2xlUHl0aG9uKSBUaGVuCiAgICBNc2dCb3ggIkZyZWVoYW5kJ3MgUHl0aG9uIGVudmlyb25tZW50IGlzIG1pc3NpbmcuIFJ1biBpbnN0YWxsX3dpbmRvd3MuYmF0IGZpcnN0LiIsIHZiRXhjbGFtYXRpb24sICJGcmVlaGFuZCBMYXVuY2hlciIKICAgIFdTY3JpcHQuUXVpdCAxCkVuZCBJZgoKJyBSZXBhaXIgdGhlIHNob3J0Y3V0IHF1aWV0bHkgb24gbm9ybWFsIGxhdW5jaGVzIGlmIGEgcHJvZmlsZSB3YXMgY2hhbmdlZC4KQ2FsbCBJbnN0YWxsU3RhcnRNZW51U2hvcnRjdXQoKQoKJyBSZWFkIHRoZSBzYW1lIHBlci11c2VyIERlYnVnZ2luZyBzZXR0aW5nIGFzIG1haW4ucHkuCmFwcERhdGEgPSBzaGVsbC5FeHBhbmRFbnZpcm9ubWVudFN0cmluZ3MoIiVBUFBEQVRBJSIpCnByZWZlcmVuY2VzRmlsZSA9IGZzLkJ1aWxkUGF0aChmcy5CdWlsZFBhdGgoYXBwRGF0YSwgIkZyZWVoYW5kIiksICJ1aV9zZXR0aW5ncy5qc29uIikKSWYgTm90IGZzLkZpbGVFeGlzdHMocHJlZmVyZW5jZXNGaWxlKSBUaGVuCiAgICBwcmVmZXJlbmNlc0ZpbGUgPSBmcy5CdWlsZFBhdGgocm9vdCwgInVpX3NldHRpbmdzLmpzb24iKQpFbmQgSWYKCmRlYnVnRW5hYmxlZCA9IFJlYWREZWJ1Z1ByZWZlcmVuY2UocHJlZmVyZW5jZXNGaWxlKQpJZiBkZWJ1Z0VuYWJsZWQgVGhlbgogICAgZXhlID0gY29uc29sZVB5dGhvbgpFbHNlCiAgICBleGUgPSBndWlQeXRob24KRW5kIElmCgpzaGVsbC5DdXJyZW50RGlyZWN0b3J5ID0gcm9vdApjb21tYW5kTGluZSA9IFF1b3RlKGV4ZSkgJiAiICIgJiBRdW90ZShtYWluU2NyaXB0KQpzaGVsbC5SdW4gY29tbWFuZExpbmUsIDEsIEZhbHNlCgpGdW5jdGlvbiBRdW90ZSh2YWx1ZSkKICAgIFF1b3RlID0gQ2hyKDM0KSAmIFJlcGxhY2UoQ1N0cih2YWx1ZSksIENocigzNCksIENocigzNCkgJiBDaHIoMzQpKSAmIENocigzNCkKRW5kIEZ1bmN0aW9uCgpGdW5jdGlvbiBJbnN0YWxsU3RhcnRNZW51U2hvcnRjdXQoKQogICAgRGltIHByb2dyYW1zLCBzaG9ydGN1dEZpbGUsIGxpbmssIHZlcmlmeUxpbmssIHdzY3JpcHRFeGUsIGljb25GaWxlCiAgICBEaW0gb2xkVW5pY29kZSwgb2xkUHJvdG90eXBlCiAgICBJbnN0YWxsU3RhcnRNZW51U2hvcnRjdXQgPSBGYWxzZQogICAgT24gRXJyb3IgUmVzdW1lIE5leHQKICAgIEVyci5DbGVhcgogICAgJyBUaGUgV2luZG93cyBzaGVsbCByZXNvbHZlcyB0aGUgYWN0dWFsIHBlci11c2VyIFByb2dyYW1zIGRpcmVjdG9yeSwKICAgICcgaW5jbHVkaW5nIHJlZGlyZWN0ZWQvcm9hbWluZyBTdGFydCBNZW51IGxvY2F0aW9ucy4KICAgIHByb2dyYW1zID0gc2hlbGwuU3BlY2lhbEZvbGRlcnMoIlByb2dyYW1zIikKICAgIElmIEVyci5OdW1iZXIgPD4gMCBPciBMZW4ocHJvZ3JhbXMpID0gMCBUaGVuCiAgICAgICAgRXJyLkNsZWFyCiAgICAgICAgRXhpdCBGdW5jdGlvbgogICAgRW5kIElmCiAgICBJZiBOb3QgZnMuRm9sZGVyRXhpc3RzKHByb2dyYW1zKSBUaGVuCiAgICAgICAgRXJyLkNsZWFyCiAgICAgICAgRXhpdCBGdW5jdGlvbgogICAgRW5kIElmCgogICAgJyBVc2UgYSBwbGFpbiBBU0NJSSBmaWxlIG5hbWU6IGl0IHJlbGlhYmx5IGFwcGVhcnMgaW4gV2luZG93cyBTdGFydCBzZWFyY2guCiAgICAnIFdpbmRvd3MgcHJvaGliaXRzICc6JyBpbnNpZGUgYSBmaWxlIG5hbWUuCiAgICBzaG9ydGN1dEZpbGUgPSBmcy5CdWlsZFBhdGgocHJvZ3JhbXMsICJGcmVlaGFuZCAtIFNwYXRpYWwgQ29tcHV0aW5nLmxuayIpCiAgICB3c2NyaXB0RXhlID0gZnMuQnVpbGRQYXRoKHNoZWxsLkV4cGFuZEVudmlyb25tZW50U3RyaW5ncygiJVN5c3RlbVJvb3QlIiksICJTeXN0ZW0zMlx3c2NyaXB0LmV4ZSIpCiAgICBTZXQgbGluayA9IHNoZWxsLkNyZWF0ZVNob3J0Y3V0KHNob3J0Y3V0RmlsZSkKICAgIElmIEVyci5OdW1iZXIgPD4gMCBUaGVuCiAgICAgICAgRXJyLkNsZWFyCiAgICAgICAgRXhpdCBGdW5jdGlvbgogICAgRW5kIElmCiAgICBsaW5rLlRhcmdldFBhdGggPSB3c2NyaXB0RXhlCiAgICBsaW5rLkFyZ3VtZW50cyA9IFF1b3RlKFdTY3JpcHQuU2NyaXB0RnVsbE5hbWUpCiAgICBsaW5rLldvcmtpbmdEaXJlY3RvcnkgPSByb290CiAgICBsaW5rLkRlc2NyaXB0aW9uID0gIkZyZWVoYW5kOiBTcGF0aWFsIENvbXB1dGluZyIKICAgIGljb25GaWxlID0gZnMuQnVpbGRQYXRoKHJvb3QsICJGcmVlaGFuZF9sb2dvX3NtYWxsLmljbyIpCiAgICBJZiBmcy5GaWxlRXhpc3RzKGljb25GaWxlKSBUaGVuIGxpbmsuSWNvbkxvY2F0aW9uID0gaWNvbkZpbGUgJiAiLDAiCiAgICBFcnIuQ2xlYXIKICAgIGxpbmsuU2F2ZQogICAgSWYgRXJyLk51bWJlciA8PiAwIFRoZW4KICAgICAgICBFcnIuQ2xlYXIKICAgICAgICBFeGl0IEZ1bmN0aW9uCiAgICBFbmQgSWYKICAgIElmIE5vdCBmcy5GaWxlRXhpc3RzKHNob3J0Y3V0RmlsZSkgVGhlbiBFeGl0IEZ1bmN0aW9uCgogICAgRXJyLkNsZWFyCiAgICBTZXQgdmVyaWZ5TGluayA9IHNoZWxsLkNyZWF0ZVNob3J0Y3V0KHNob3J0Y3V0RmlsZSkKICAgIElmIEVyci5OdW1iZXIgPD4gMCBUaGVuCiAgICAgICAgRXJyLkNsZWFyCiAgICAgICAgRXhpdCBGdW5jdGlvbgogICAgRW5kIElmCiAgICBJZiBMQ2FzZShDU3RyKHZlcmlmeUxpbmsuQXJndW1lbnRzKSkgPD4gTENhc2UoUXVvdGUoV1NjcmlwdC5TY3JpcHRGdWxsTmFtZSkpIFRoZW4gRXhpdCBGdW5jdGlvbgogICAgSWYgTENhc2UoQ1N0cih2ZXJpZnlMaW5rLlRhcmdldFBhdGgpKSA8PiBMQ2FzZSh3c2NyaXB0RXhlKSBUaGVuIEV4aXQgRnVuY3Rpb24KICAgIEluc3RhbGxTdGFydE1lbnVTaG9ydGN1dCA9IFRydWUKCiAgICAnIFJlbW92ZSBzdXBlcnNlZGVkIG5hbWVzIG9ubHkgYWZ0ZXIgdGhlIG5ldyBsaW5rIGhhcyBiZWVuIHZlcmlmaWVkLgogICAgb2xkVW5pY29kZSA9IGZzLkJ1aWxkUGF0aChwcm9ncmFtcywgIkZyZWVoYW5kIiAmIENoclcoJkhGRjFBKSAmICIgU3BhdGlhbCBDb21wdXRpbmcubG5rIikKICAgIG9sZFByb3RvdHlwZSA9IGZzLkJ1aWxkUGF0aChwcm9ncmFtcywgIlNwYXRpYWwgRGVza3RvcCBQcm90b3R5cGUubG5rIikKICAgIElmIGZzLkZpbGVFeGlzdHMob2xkVW5pY29kZSkgVGhlbiBmcy5EZWxldGVGaWxlIG9sZFVuaWNvZGUsIFRydWUKICAgIElmIGZzLkZpbGVFeGlzdHMob2xkUHJvdG90eXBlKSBUaGVuIGZzLkRlbGV0ZUZpbGUgb2xkUHJvdG90eXBlLCBUcnVlCiAgICBFcnIuQ2xlYXIKICAgIE9uIEVycm9yIEdvVG8gMApFbmQgRnVuY3Rpb24KCkZ1bmN0aW9uIFJlYWREZWJ1Z1ByZWZlcmVuY2UoZmlsZVBhdGgpCiAgICBEaW0gaW5wdXRGaWxlLCBqc29uVGV4dCwgbWF0Y2hlcgogICAgUmVhZERlYnVnUHJlZmVyZW5jZSA9IEZhbHNlCiAgICBJZiBOb3QgZnMuRmlsZUV4aXN0cyhmaWxlUGF0aCkgVGhlbiBFeGl0IEZ1bmN0aW9uCiAgICBPbiBFcnJvciBSZXN1bWUgTmV4dAogICAgU2V0IGlucHV0RmlsZSA9IGZzLk9wZW5UZXh0RmlsZShmaWxlUGF0aCwgRm9yUmVhZGluZywgRmFsc2UpCiAgICBJZiBFcnIuTnVtYmVyIDw+IDAgVGhlbgogICAgICAgIEVyci5DbGVhcgogICAgICAgIEV4aXQgRnVuY3Rpb24KICAgIEVuZCBJZgogICAganNvblRleHQgPSBpbnB1dEZpbGUuUmVhZEFsbAogICAgaW5wdXRGaWxlLkNsb3NlCiAgICBJZiBFcnIuTnVtYmVyIDw+IDAgVGhlbgogICAgICAgIEVyci5DbGVhcgogICAgICAgIEV4aXQgRnVuY3Rpb24KICAgIEVuZCBJZgogICAgT24gRXJyb3IgR29UbyAwCiAgICBTZXQgbWF0Y2hlciA9IENyZWF0ZU9iamVjdCgiVkJTY3JpcHQuUmVnRXhwIikKICAgIG1hdGNoZXIuSWdub3JlQ2FzZSA9IEZhbHNlCiAgICBtYXRjaGVyLkdsb2JhbCA9IEZhbHNlCiAgICBtYXRjaGVyLlBhdHRlcm4gPSBDaHIoMzQpICYgImRlYnVnX2NvbnNvbGUiICYgQ2hyKDM0KSAmICJccyo6XHMqdHJ1ZVxiIgogICAgUmVhZERlYnVnUHJlZmVyZW5jZSA9IG1hdGNoZXIuVGVzdChqc29uVGV4dCkKRW5kIEZ1bmN0aW9uCg=='))"
if errorlevel 1 exit /b 1
if not exist "%FREEHAND_LAUNCHER%" exit /b 1
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

:search_setup_error
echo ERROR: Search and Indexing setup failed. Existing Freehand files were left as intact as possible.
echo Check that freehand_search_index.py and apply_freehand_search_settings.py were included.
goto :failed

:shortcut_error
echo ERROR: Package installation finished, but Start Menu shortcut creation failed.
goto :failed

:error
echo ERROR: A required setup command failed. Read the output above.
:failed
echo Installation failed.
pause
exit /b 1
