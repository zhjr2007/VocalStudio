"""GAME pitch-extraction engine (wave C1-3).

Thin wrapper around the reference ONNX runtime
(``Vocal2Midi/inference/API/game_api.py`` + ``inference/game/onnx_runtime.py``).
Backend selection in :meth:`GameEngine.load`: PyTorch ``.pt`` weights are
preferred when present, otherwise the ONNX suite is used; the chosen backend
is logged and stored on ``self.backend`` (``"pytorch" | "onnx"``). The shipped
GAME-1.0.3-medium-onnx weights contain no .pt files, so this host runs with
backend="onnx".

The ONNX suite is loaded through a small subclass of the reference
``GameOnnxModel`` that resolves execution providers via our own
:func:`core.device_utils.ort_execution_providers` (CUDA EP first on this
host), so ``device="cuda"`` really uses the GPU when available.
"""

from __future__ import annotations

import gc
import pathlib
import sys
from typing import Any

from core.config import PROJECT_ROOT
from core.device_utils import ort_execution_providers, create_ort_session

_V2M_REF_DIR = PROJECT_ROOT / "Vocal2Midi"

_ONNX_FILES = ("encoder.onnx", "segmenter.onnx", "estimator.onnx", "dur2bd.onnx", "bd2dur.onnx")

# Reference D3PM sampling schedule (auto_lyric_hybrid __main__ defaults):
# t0=0.0, nsteps=8 -> ts = [0, 1/8, ..., 7/8].
DEFAULT_TS: list[float] = [i / 8 for i in range(8)]


def _ensure_ref_path() -> None:
    ref = str(_V2M_REF_DIR)
    if _V2M_REF_DIR.is_dir() and ref not in sys.path:
        sys.path.append(ref)


class GameEngine:
    """Loads a GAME model suite and extracts pitch notes from audio chunks."""

    def __init__(self) -> None:
        self.model: Any = None
        self.backend: str | None = None
        self.device: str = "cuda"
        self.provider_note: str = ""

    # -- lifecycle ---------------------------------------------------------

    def load(self, model_dir: str | pathlib.Path, device: str = "cuda") -> str:
        """Load the GAME suite from ``model_dir``; returns the backend name."""
        # Resolve providers (and register nvidia DLL dirs) BEFORE importing the
        # reference module: it imports onnxruntime at module level.
        providers, note = ort_execution_providers(device)
        self.provider_note = note
        _ensure_ref_path()
        from inference.game.onnx_runtime import GameOnnxModel

        model_dir = pathlib.Path(model_dir)
        if not model_dir.is_dir():
            raise FileNotFoundError(f"GAME model directory not found: {model_dir}")

        has_onnx = all((model_dir / name).exists() for name in _ONNX_FILES)
        has_pt = any(model_dir.glob("*.pt")) or any(model_dir.glob("*.pth"))
        if has_pt and has_onnx:
            # PyTorch .pt weights are preferred, but the PyTorch loader is not
            # ported yet; fall back to the ONNX suite (logged below).
            print(f"[GameEngine] .pt weights present in {model_dir}; PyTorch loader not ported yet, using ONNX fallback.")
        if has_pt and not has_onnx:
            self.backend = "pytorch"
            raise RuntimeError(
                f"PyTorch GAME weights found in {model_dir} but the PyTorch loader is not "
                "ported yet; provide the ONNX suite (GAME-1.0.3-medium-onnx)."
            )
        if not has_onnx:
            raise FileNotFoundError(f"GAME model files not found in {model_dir}")

        self.backend = "onnx"  # the runtime that actually runs (see above)
        self.device = device

        class _GameOnnxSuite(GameOnnxModel):
            @staticmethod
            def _resolve_providers(_device: str):  # noqa: N802 - reference signature
                name = "cuda" if providers[0] == "CUDAExecutionProvider" else (
                    "dml" if providers[0] in ("DmlExecutionProvider", "DMLExecutionProvider") else "cpu"
                )
                return name, list(providers)

            def _load_sessions(self, selected):
                return {pathlib.Path(name).stem: create_ort_session(self.model_dir/name, selected) for name in _ONNX_FILES}

        self.model = _GameOnnxSuite(model_dir, requested_device=device)
        print(
            f"[GameEngine] backend={self.backend} device={device} "
            f"providers={[p if isinstance(p, str) else p[0] for p in providers]}"
            + (f" ({note})" if note else "")
        )
        return self.backend

    def unload(self) -> None:
        """Release the ORT sessions and cached state."""
        if self.model is not None:
            try:
                for session in getattr(self.model, '_sessions', {}).values():
                    session.end_profiling()
                self.model.shutdown()
            finally:
                self.model = None
        gc.collect()

    # -- inference ---------------------------------------------------------

    def infer(
        self,
        chunks: list[dict],
        sr: int,
        *,
        ts: list[float] | None = None,
        seg_threshold: float = 0.2,
        seg_radius: float = 0.02,
        est_threshold: float = 0.2,
        batch_size: int = 4,
        language: str | None = None,
        cancel_checker=None,
    ) -> list[Any]:
        """Pitch-only extraction over ``chunks`` (see slicer output format).

        Each chunk dict needs ``waveform`` and a start position under either
        ``start`` or the reference key ``offset``. Returns a list of
        :class:`engines.vocal2midi.note_io.NoteInfo` with global onsets.
        """
        if self.model is None:
            raise RuntimeError("GameEngine.load() must be called before infer().")
        _ensure_ref_path()
        from inference.API.game_api import extract_pitches_only_torch

        from engines.vocal2midi.note_io import NoteInfo

        ref_chunks = [
            {"waveform": chunk["waveform"], "offset": float(chunk.get("start", chunk.get("offset", 0.0)))}
            for chunk in chunks
        ]
        raw_notes = extract_pitches_only_torch(
            ref_chunks,
            sr,
            self.model,
            self.device,
            list(ts) if ts is not None else DEFAULT_TS,
            seg_threshold,
            seg_radius,
            est_threshold,
            batch_size=batch_size,
            language=language,
            cancel_checker=cancel_checker,
        )
        return [
            NoteInfo(onset=float(n.onset), offset=float(n.offset), pitch=float(n.pitch), lyric=n.lyric or "")
            for n in raw_notes
        ]
