"""RMVPE f0 / MIDI-pitch engine (wave C1-4).

Thin wrapper around the reference transcriber
(``Vocal2Midi/inference/API/rmvpe_api.py``). The ONNX session is created via a
small subclass of ``RmvpeTranscriber`` that resolves execution providers with
:func:`core.device_utils.ort_execution_providers`, so ``device="cuda"`` uses
the CUDA EP when the venv's onnxruntime build provides it.

Time convention: audio is resampled internally to 16 kHz and processed with a
hop of 160 samples, so every output frame covers ``time_step_seconds = 0.01 s``
of source time; frame *i* starts at ``i * 0.01 s``.
"""

from __future__ import annotations

import ast
import gc
import pathlib
import sys
from typing import Any

import numpy as np

from core.config import PROJECT_ROOT
from core.device_utils import ort_execution_providers, create_ort_session

_V2M_REF_DIR = PROJECT_ROOT / "Vocal2Midi"

SAMPLE_RATE = 16000
HOP_LENGTH = 160
TIME_STEP_SECONDS = HOP_LENGTH / SAMPLE_RATE  # 0.01 s per frame


def _ensure_ref_path() -> None:
    ref = str(_V2M_REF_DIR)
    if _V2M_REF_DIR.is_dir() and ref not in sys.path:
        sys.path.append(ref)


_NESTED_METHOD_NAMES = (
    "_create_session",
    "_get_fixed_batch_size",
    "infer",
    "_inference_salience",
    "_pad_audio",
    "_en_frame",
    "_forward_in_mini_batch",
    "_run_single_segment",
    "_waveform_to_mel",
    "_de_frame",
    "_to_local_average_cents",
    "_salience_to_f0",
    "_f0_to_interpolated_midi",
    "shutdown",
)


def _rebind_nested_methods(cls) -> None:
    """Work around an upstream layout bug in rmvpe_api.py.

    The reference file defines the transcriber helper methods *inside* the
    module-level ``clear_rmvpe_transcriber_cache`` function instead of on the
    class, so a fresh ``RmvpeTranscriber`` instance has no ``_create_session``
    etc. We extract those nested definitions with ast and bind them onto our
    subclass (compiled against the reference module's globals). The reference
    file itself is left untouched.
    """
    import inference.API.rmvpe_api as rmvpe_mod

    missing = [name for name in _NESTED_METHOD_NAMES if not hasattr(cls, name)]
    if not missing:
        return
    source = pathlib.Path(rmvpe_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    clear_fn = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "clear_rmvpe_transcriber_cache"
    )
    namespace = vars(rmvpe_mod)
    for node in (n for n in clear_fn.body if isinstance(n, ast.FunctionDef)):
        code = compile(ast.Module(body=[node], type_ignores=[]), rmvpe_mod.__file__, "exec")
        exec(code, namespace)  # noqa: S102 - fixed source from the reference repo
    for name in missing:
        setattr(cls, name, namespace[name])


class RmvpeEngine:
    """Loads the RMVPE ONNX model and estimates f0 / MIDI pitch curves."""

    def __init__(self) -> None:
        self.model: Any = None
        self.device: str = "cuda"
        self.provider_note: str = ""

    # -- lifecycle ---------------------------------------------------------

    def load(self, model_path: str | pathlib.Path, device: str = "cuda") -> None:
        """Load the RMVPE ONNX weights from ``model_path``."""
        # Resolve providers (and register nvidia DLL dirs) BEFORE importing the
        # reference module: it imports onnxruntime at module level.
        providers, note = ort_execution_providers(device)
        self.provider_note = note
        _ensure_ref_path()
        from inference.API.rmvpe_api import RmvpeTranscriber

        model_path = pathlib.Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"RMVPE model not found: {model_path}")

        class _RmvpeTranscriber(RmvpeTranscriber):
            @staticmethod
            def _resolve_providers(_device: str):  # noqa: N802 - reference signature
                name = "cuda" if providers[0] == "CUDAExecutionProvider" else (
                    "dml" if providers[0] in ("DmlExecutionProvider", "DMLExecutionProvider") else "cpu"
                )
                return name, list(providers)

            def _create_session(self, selected):
                return create_ort_session(self.model_path, selected)

        _rebind_nested_methods(_RmvpeTranscriber)
        self.model = _RmvpeTranscriber(model_path, device=device)
        print(
            f"[RmvpeEngine] loaded {model_path.name} device={device} "
            f"providers={[p if isinstance(p, str) else p[0] for p in providers]}"
            + (f" ({note})" if note else "")
        )

    def unload(self) -> None:
        """Release the ORT session and cached state."""
        if self.model is not None:
            try:
                self.model.session.end_profiling()
                self.model.shutdown()
            finally:
                self.model = None
        gc.collect()

    # -- inference ---------------------------------------------------------

    def infer(self, waveform: np.ndarray, sr: int) -> dict[str, Any]:
        """Estimate the pitch curve for ``waveform``.

        Returns a dict with:
        - ``f0_hz``: float array in Hz (0 where unvoiced),
        - ``midi_pitch``: float MIDI numbers (interpolated across gaps; values
          lie within [0, 360] for voiced content),
        - ``voiced_mask``: bool array,
        - ``time_step_seconds``: 0.01 s — frame *i* starts at i * 0.01 s of the
          source audio (the input is resampled to 16 kHz internally).

        Empty input returns empty arrays (documented behaviour, no crash).
        """
        if self.model is None:
            raise RuntimeError("RmvpeEngine.load() must be called before infer().")
        _ensure_ref_path()
        from inference.API.rmvpe_api import RmvpeResult  # noqa: F401 (type hint)

        result = self.model.infer(np.asarray(waveform, dtype=np.float32), int(sr))
        midi_pitch = np.asarray(result.midi_pitch, dtype=np.float64)
        voiced_mask = np.asarray(result.voiced_mask, dtype=bool)
        # The reference only returns the interpolated MIDI curve; derive Hz
        # from it and zero out unvoiced frames per the voiced mask.
        f0_hz = np.zeros_like(midi_pitch)
        valid = voiced_mask & ~np.isnan(midi_pitch)
        f0_hz[valid] = 440.0 * np.power(2.0, (midi_pitch[valid] - 69.0) / 12.0)
        return {
            "f0_hz": f0_hz,
            "midi_pitch": midi_pitch,
            "voiced_mask": voiced_mask,
            "time_step_seconds": float(result.time_step_seconds),
        }
