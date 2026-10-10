@echo off
REM Full inference onedir build. Default: isolated ROCm 10 gfx1102 environment.
REM Explicit legacy CUDA build: build_portable.bat cuda
setlocal
set "ROOT=%~dp0"
cd /d "%ROOT%"
set "PYTHON=%ROOT%venv-rocm10\Scripts\python.exe"
if /I "%~1"=="cuda" set "PYTHON=%ROOT%venv\Scripts\python.exe"
if not exist "%PYTHON%" goto :err
set "TEMP=%ROOT%temp\portable-build"
set "TMP=%TEMP%"
if not exist "%TEMP%" mkdir "%TEMP%"
set "PYTHONUTF8=1"

echo [1/5] Verify recorded model hashes
"%PYTHON%" scripts\download_models.py --verify || goto :err

echo [2/5] Build UI and CLI with all inference dependencies
"%PYTHON%" -m PyInstaller VocalStudioBackend.spec --noconfirm --clean || goto :err

echo [3/5] Assemble complete model inventory and FFmpeg shared runtime
"%PYTHON%" scripts\assemble_portable_assets.py || goto :err
"%PYTHON%" -m pip freeze > "%ROOT%dist\VocalStudioBackend\dependencies.lock.txt"
copy /Y "%ROOT%docs\training-extension-contract.md" "%ROOT%dist\VocalStudioBackend\training-extension-contract.md" >nul
copy /Y "%ROOT%docs\portable-readme.md" "%ROOT%dist\VocalStudioBackend\README.md" >nul
copy /Y "%ROOT%docs\denoise-model-provenance.json" "%ROOT%dist\VocalStudioBackend\denoise-model-provenance.json" >nul

echo [4/5] Window and CLI startup smoke
set "VOCALSTUDIO_SMOKE=1"
set "QT_QPA_PLATFORM=offscreen"
"%ROOT%dist\VocalStudioBackend\VocalStudioBackend.exe" > "%TEMP%\vs_ui_smoke.txt" 2>&1
if errorlevel 1 goto :err
"%ROOT%dist\VocalStudioBackend\VocalStudioBackend.exe" info > "%TEMP%\vs_cli_smoke.txt" 2>&1
if errorlevel 1 goto :err
set "VOCALSTUDIO_SMOKE="
set "QT_QPA_PLATFORM="
findstr /C:"SMOKE_FROZEN_PASS" "%TEMP%\vs_ui_smoke.txt" >nul || goto :err
findstr /C:"SMOKE_FROZEN_PASS" "%TEMP%\vs_cli_smoke.txt" >nul || goto :err

echo [5/5] Candidate assembled. Real inference and relocation acceptance REQUIRED.
echo Status remains UNVERIFIED until scripts\validate_amd_release.py succeeds.
exit /b 0
:err
echo BUILD FAILED. Inspect build output and temp\portable-build logs.
exit /b 1
