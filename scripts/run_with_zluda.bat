@echo off
REM Run VocalStudio GUI on the AMD dGPU through the ZLUDA launcher.
REM   Usage: run_with_zluda.bat [path\to\zluda.exe]
REM ZLUDA v6 emulates nvcuda/nvml + cublas/cuDNN, so both torch (cu128) and
REM onnxruntime-gpu see a normal "NVIDIA CUDA" device with no code changes.
setlocal
cd /d "%~dp0.."

if "%~1"=="" (
    set "ZLUDA="
    for %%i in ("zluda.exe" "%LOCALAPPDATA%\zluda\zluda.exe" "%ProgramFiles%\zluda\zluda.exe") do if exist "%%~i" set "ZLUDA=%%~i"
) else (
    set "ZLUDA=%~f1"
)

if not defined ZLUDA (
    echo ZLUDA launcher not found. Download the Windows build from https://github.com/zhilongpu/ZLUDA, extract zluda.exe somewhere, then:
    echo   run_with_zluda.bat C:\path\to\zluda.exe
    exit /b 1
)

REM HIP_PATH (AMD GPU SDK, nightly "therock" dist with MIOpen) must be set for full ML support.
if not defined HIP_PATH (
    echo WARNING: HIP_PATH is not set — CUDA-compatible inference works; cuDNN/MIOpen features may not.
)

REM Authoritative marker for core.device_utils.detect_zluda(): when both an AMD
REM and an NVIDIA GPU are physically connected, only this env var can tell a
REM ZLUDA launch apart from a plain CUDA (NVIDIA) session.
set "ZLUDA_ACTIVE=1"

call "%ZLUDA%" -- venv\Scripts\python gui\fluent_main.py %*
