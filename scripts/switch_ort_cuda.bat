@echo off
REM Switch the venv to the onnxruntime-gpu (CUDA cu12) build — default for this project.
setlocal
cd /d "%~dp0.."
venv\Scripts\python -m pip uninstall -y onnxruntime-directml 2>nul
venv\Scripts\python -m pip install --no-cache-dir "onnxruntime-gpu==1.22.0" || exit /b 1
echo Done: ORT flavor = GPU (CUDA EP). Torch engines unaffected.
