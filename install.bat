@echo off
REM VocalStudio environment setup (CUDA 12.8 baseline)
py -3.13 -m venv venv || exit /b 1
venv\Scripts\python -m pip install --upgrade pip || exit /b 1
venv\Scripts\python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128 || exit /b 1
venv\Scripts\python -m pip install "funasr>=1.2.7,<2" modelscope "onnxruntime-gpu==1.22.0" librosa soundfile mido pyopenjtalk sounddevice einops PyQt-Fluent-Widgets || exit /b 1
REM GSV TTS reference chain runtime deps (v2Pro inference + g2pw frontend)
venv\Scripts\python -m pip install loguru einx fast-langdetect split-lang opencc-python-reimplemented || exit /b 1
echo Done. Activate with: venv\Scripts\activate
