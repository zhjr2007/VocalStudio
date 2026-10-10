@echo off
REM Switch the venv to the onnxruntime-directml build (DirectML EP, runs on any DX12 GPU).
REM Note: both ORT flavors ship the same top-level package; only one can be installed at a time.
setlocal
cd /d "%~dp0.."
venv\Scripts\python -m pip uninstall -y onnxruntime-gpu 2>nul
venv\Scripts\python -m pip install --no-cache-dir "onnxruntime-directml==1.22.0" || exit /b 1
echo Done: ORT flavor = DirectML (DML EP). Torch engines still need their own backend choice.
