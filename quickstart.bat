@echo off
chcp 65001 >nul
setlocal EnableExtensions
title VocalStudio 快速开始
cd /d "%~dp0"

REM VocalStudio quickstart: create venv -> install deps -> fetch models -> launch UI.
REM Usage: quickstart.bat [cpu|cuda|rocm|check]   (default: cpu)

set "FLAVOR=%~1"
if "%FLAVOR%"=="" set "FLAVOR=cpu"

if /I "%FLAVOR%"=="check" goto :check

echo ============================================================
echo  VocalStudio 快速开始
echo  PyTorch 版本：%FLAVOR%（首次运行需联网下载依赖，请耐心等待）
echo ============================================================
echo.

REM ---- 1/5 查找 Python 3.13 ----
set "PYCMD="
py -3.13 --version >nul 2>nul && set "PYCMD=py -3.13"
if not defined PYCMD (
  python --version 2>nul | findstr /b /c:"Python 3.13" >nul && set "PYCMD=python"
)
if not defined PYCMD (
  echo [错误] 未找到 Python 3.13。
  echo   请到 https://www.python.org/downloads/ 安装 Python 3.13，
  echo   安装时勾选 "Add python.exe to PATH"，然后重新运行本脚本。
  pause
  exit /b 1
)
echo [1/5] Python 已就绪：
%PYCMD% --version
echo.

REM ---- 2/5 创建虚拟环境 ----
if exist "venv\Scripts\python.exe" (
  echo [2/5] 已存在 venv，跳过创建。
) else (
  echo [2/5] 创建虚拟环境 venv ...
  %PYCMD% -m venv venv || goto :err
)
echo.

REM ---- 3/5 安装 PyTorch ----
if /I "%FLAVOR%"=="rocm" (
  echo [3/5] AMD ROCm 环境请使用专用脚本：
  echo    powershell -ExecutionPolicy Bypass -File scripts\prepare_rocm10.ps1
  pause
  exit /b 0
)
venv\Scripts\python -m pip install --upgrade pip || goto :err
if /I "%FLAVOR%"=="cuda" (
  echo [3/5] 安装 PyTorch（CUDA 12.8）...
  venv\Scripts\python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128 || goto :err
) else (
  echo [3/5] 安装 PyTorch（CPU 兼容版）...
  venv\Scripts\python -m pip install torch torchaudio || goto :err
)
echo.

REM ---- 4/5 安装其余依赖 ----
echo [4/5] 安装其余依赖（funasr / GUI / 音频处理等）...
venv\Scripts\python -m pip install "funasr>=1.2.7,<2" modelscope librosa soundfile mido pyopenjtalk sounddevice einops PyQt-Fluent-Widgets loguru einx fast-langdetect split-lang opencc-python-reimplemented || goto :err
if /I "%FLAVOR%"=="cuda" (
  venv\Scripts\python -m pip install onnxruntime-gpu==1.22.0 || goto :err
) else (
  venv\Scripts\python -m pip install onnxruntime==1.22.0 || venv\Scripts\python -m pip install onnxruntime || goto :err
)
echo.

REM ---- 5/5 下载可自动获取的模型 ----
echo [5/5] 下载模型（sensevoice / campplus，如失败可稍后重试）...
venv\Scripts\python scripts\download_models.py --only sensevoice campplus
echo.
echo 提示：其余权重（人声分离 / MIDI / TTS）需要自行放置，方法见 README
echo       的“模型权重说明”一节；校验：venv\Scripts\python scripts\download_models.py --verify
echo.

echo 启动 VocalStudio 界面 ...
call run_ui.bat
exit /b 0

:err
echo.
echo [错误] 某一步安装失败。请检查网络连接后重新运行本脚本；
echo        若问题持续，请把上方报错信息发给作者。
pause
exit /b 1

:check
echo [check] 检测 Python 环境...
py -3.13 --version 2>nul || echo   py -3.13 未找到
python --version 2>nul || echo   python 未找到
echo [check] OK
exit /b 0
