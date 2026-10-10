"""HubertFA forced-alignment engine (wave C2-3).

Thin wrapper around the vendored HubertFA reference
(``Vocal2Midi/inference/API/hfa_api.py`` -> ``HubertFA/onnx_infer``). The
reference requires a ``.lab`` phoneme/text file next to each ``.wav`` chunk to
align; those are produced by the lyric composer from the ASR transcript / G2P
output. This engine runs the ONNX forced alignment and returns per-word time
intervals.

``textgrid`` is imported by the reference at load time but only used when
*exporting* TextGrid artifacts; :mod:`engines.vocal2midi.hfa_compat` installs
a stub so core inference is importable without the pip package.
"""

from __future__ import annotations

import pathlib
import sys
import tempfile
import threading
from typing import Any

from core.config import PROJECT_ROOT

_V2M_REF_DIR = PROJECT_ROOT / "Vocal2Midi"

_HFA_DEFAULT_REL = pathlib.Path("Vocal2Midi") / "experiments" / "1218_hfa_model_new_dict"
_INFERENCE_LOCK = threading.Lock()


def _ensure_ref() -> None:
    from .hfa_compat import install_textgrid_stub

    install_textgrid_stub()  # before the reference imports textgrid
    ref = str(_V2M_REF_DIR)
    if _V2M_REF_DIR.is_dir() and ref not in sys.path:
        sys.path.append(ref)


def resolve_hfa_model_path(model_paths=None) -> pathlib.Path:
    """Resolve the HubertFA model dir from settings or repo default."""
    from core.config import ModelPaths

    mp = model_paths if model_paths is not None else ModelPaths.from_settings()
    value = getattr(mp, "hfa", "") or ""
    if value:
        p = pathlib.Path(value)
        return p if p.is_absolute() else PROJECT_ROOT / p
    return PROJECT_ROOT / _HFA_DEFAULT_REL


class HfaEngine:
    """Loads HubertFA ONNX and aligns a waveform against a word/phone sequence."""

    def __init__(self) -> None:
        self.model: Any = None
        self.device: str = "cpu"
        self.loaded = False

    def load(self, model_path=None, device: str = "cpu") -> None:
        """Load (or reuse the cached) HubertFA model."""
        _ensure_ref()
        from inference.API.hfa_api import load_hfa_model

        model_dir = pathlib.Path(model_path) if model_path else resolve_hfa_model_path()
        with _INFERENCE_LOCK:
            self.model = load_hfa_model(str(model_dir), device=device, use_cache=True)
        self.device = device or "cpu"
        self.loaded = self.model is not None

    def unload(self) -> None:
        self.model = None
        self.loaded = False

    def align(self, audio: Any, sr: int, lab_text: str, language: str = "zh") -> dict[str, Any]:
        """Align ``audio`` against ``lab_text``; returns ``{stem: (start, end, words)}``.

        Args:
            audio: mono np.ndarray or wav path.
            sr: sample rate (ignored when ``audio`` is a path).
            lab_text: the phoneme/word sequence for the chunk (written as .lab).
            language: ``"zh"`` or ``"ja"`` (selects the dictionary).

        Returns:
            ``pred_dict`` from the reference: ``{stem: [wav_path, ph_seq, words]}``.
            Empty dict when ``lab_text`` is blank (documented behaviour).
        """
        if not self.loaded:
            self.load(device=self.device)
        if not lab_text or not str(lab_text).strip():
            return {}

        import numpy as np
        import soundfile as sf

        _ensure_ref()
        from inference.API.hfa_api import run_hubert_fa

        arr = np.asarray(audio, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=1).astype(np.float32)

        tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix="hfa_chunk_"))
        try:
            wav_path = tmp_dir / "chunk_0.wav"
            sf.write(str(wav_path), arr, int(sr))
            (tmp_dir / "chunk_0.lab").write_text(str(lab_text).strip(), encoding="utf-8")
            with _INFERENCE_LOCK:
                pred_dict = run_hubert_fa(self.model, tmp_dir, language=language, cancel_checker=None)
            return dict(pred_dict)
        finally:
            import shutil

            shutil.rmtree(tmp_dir, ignore_errors=True)

    @staticmethod
    def words_from_pred(pred: Any) -> list[dict[str, Any]]:
        """Convert a reference pred tuple to ``[{text, start, end}, ...]``."""
        if not pred or len(pred) < 3:
            return []
        words = pred[2]
        result = []
        for word in words or []:
            result.append(
                {
                    "text": getattr(word, "text", str(word)),
                    "start": float(getattr(word, "start", 0.0)),
                    "end": float(getattr(word, "end", 0.0)),
                }
            )
        return result
