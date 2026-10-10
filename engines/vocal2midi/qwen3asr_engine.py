"""Qwen3-ASR transcription engine (wave C2-2).

Thin wrapper around the vendored Qwen ASR runtime
(``Vocal2Midi/inference/API/asr_api.py`` -> ``qwen3asr_dml.runtime``). The
reference uses the unified ONNX (encoder) + llama.cpp GGUF (decoder) runtime.

We load the model *in-process* (``load_qwen_model``) rather than through the
reference ``batch_transcribe_asr(force_subprocess=True)`` worker: that worker
spawns a daemon ``multiprocessing.Process`` which then tries to start the Qwen
helper process, failing with "daemonic processes are not allowed to have
children" on this host. Loading in the current process (inside the GUI
JobRunner QThread) is the path verified on this machine.

Transcribed text is cleaned of Qwen special tokens (``<|...|>``) via the same
reference sanitizer the pipeline uses, so downstream lyric matching only sees
plain text.
"""

from __future__ import annotations

import pathlib
import sys
import threading
from typing import Any

from core.config import PROJECT_ROOT

_V2M_REF_DIR = PROJECT_ROOT / "Vocal2Midi"

_QWEN_DEFAULT_REL = pathlib.Path("Vocal2Midi") / "experiments" / "Qwen3-ASR-1.7B-dml"

_QWEN_SYNC_LOCK = threading.Lock()


def _ensure_ref_path() -> None:
    ref = str(_V2M_REF_DIR)
    if _V2M_REF_DIR.is_dir() and ref not in sys.path:
        sys.path.append(ref)


def resolve_asr_model_path(model_paths=None) -> pathlib.Path:
    """Resolve the Qwen3-ASR model directory from settings or repo default."""
    from core.config import ModelPaths

    mp = model_paths if model_paths is not None else ModelPaths.from_settings()
    value = getattr(mp, "asr", "") or ""
    if value:
        p = pathlib.Path(value)
        return p if p.is_absolute() else PROJECT_ROOT / p
    return PROJECT_ROOT / _QWEN_DEFAULT_REL


class Qwen3AsrEngine:
    """Loads the Qwen ASR runtime and transcribes mono audio chunks."""

    def __init__(self) -> None:
        self.model: Any = None
        self.device: str = "cpu"
        self.loaded = False

    def load(self, model_path=None, device: str = "cpu") -> None:
        """Load (or reuse the cached) Qwen ASR model; returns None."""
        _ensure_ref_path()
        from inference.API.asr_api import load_qwen_model

        model_dir = pathlib.Path(model_path) if model_path else resolve_asr_model_path()
        # Sync so concurrent jobs do not interleave the helper-process chdir.
        with _QWEN_SYNC_LOCK:
            self.model = load_qwen_model(str(model_dir), device=device, use_cache=True)
        self.device = device or "cpu"
        self.loaded = self.model is not None

    def unload(self) -> None:
        """Drop the reference to the cached model (cache may persist)."""
        self.model = None
        self.loaded = False

    def transcribe(self, audio: Any, sr: int, language: str = "auto", model_path=None) -> str:
        """Transcribe a mono waveform (or writable audio path) to clean text.

        Args:
            audio: mono np.ndarray (float32) or a path to a wav file.
            sr: sample rate (ignored when ``audio`` is a file path).
            language: ``"zh"``/``"en"``/``"ja"`` or ``"auto"``.
            model_path: override the model dir for this call.

        Returns:
            Cleaned transcription string (empty on failure / silence).
        """
        if not self.loaded:
            self.load(model_path=model_path, device=self.device)

        import tempfile

        import numpy as np
        import soundfile as sf

        # Always hand the runtime a wav path (the reference loads from disk).
        if isinstance(audio, (str, pathlib.Path)):
            wav_path = str(audio)
        else:
            arr = np.asarray(audio, dtype=np.float32)
            if arr.ndim > 1:
                arr = arr.mean(axis=1).astype(np.float32)
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            sf.write(tmp.name, arr, int(sr))
            wav_path = tmp.name

        lang = None if language in ("auto", "", None) else language
        with _QWEN_SYNC_LOCK:
            result = self.model.transcribe(wav_path, language=lang)

        text = self._clean_text(result)
        if not isinstance(audio, (str, pathlib.Path)):
            try:
                pathlib.Path(wav_path).unlink(missing_ok=True)
            except OSError:
                pass
        return text

    @staticmethod
    def _clean_text(result) -> str:
        """Extract + sanitize text from a TranscribeResult / dict / str."""
        raw = ""
        if result is None:
            raw = ""
        elif isinstance(result, dict):
            raw = result.get("text") or result.get("transcript") or ""
        else:
            text_attr = getattr(result, "text", None)
            raw = text_attr if text_attr is not None else result
        raw = str(raw or "")
        # Strip Qwen special tokens <|...|> and collapse whitespace.
        import re

        raw = re.sub(r"<\|[^|]*\|>", "", raw)
        raw = re.sub(r"\s+", " ", raw).strip()
        return raw
