"""Audio slicing for the vocal-to-MIDI pipeline (wave C1-2).

Thin wrapper around the reference RMS slicer
(``Vocal2Midi/inference/slicer/slicer2.py``, pure numpy) plus a fixed-interval
"grid" mode. Every mode returns a list of chunk dicts::

    {"waveform": np.ndarray, "sr": int, "start": float, "end": float}

``start``/``end`` are seconds on the original timeline; ``len(waveform)``
(last axis) equals ``(end - start) * sr`` up to rounding.

Modes (see :data:`SLICE_MODES`):
- ``"default"``: reference default slicer params — RMS threshold -30 dB,
  min segment length 5 s, keep at most 500 ms of silence per boundary.
- ``"smart"``: RMS threshold slicing with a configurable threshold
  (``threshold_db``, default -40 dB), min segment length 5 s, keep up to
  5 s of silence per boundary.
- ``"heuristic"``: two-stage — remove long silences first (-40 dB, min
  segment 800 ms, keep 100 ms), then split segments longer than
  ``max_len_sec`` into fixed windows; tiny fragments (<0.35 s) are merged
  into a neighbour instead of being dropped.
- ``"grid"``: fixed interval of ``length`` seconds (default 5 s); produces
  ``ceil(duration / length)`` chunks, the last one shorter when the audio
  does not divide evenly.

All-silent input never crashes; documented behaviour per mode:
- grid: empty audio -> ``[]``; non-empty silence still yields the fixed
  windows (silence is kept).
- default/smart: short all-silence (<= 5 s, the min segment length) -> a
  single chunk covering the whole audio; long all-silence -> ``[]`` (no
  voiced content to keep).
- heuristic: any all-silence longer than its pre-stage min segment
  (800 ms) is fully removed -> ``[]``.
"""

from __future__ import annotations

import sys

import numpy as np

from core.config import PROJECT_ROOT

_V2M_REF_DIR = PROJECT_ROOT / "Vocal2Midi"

SLICE_MODES = ("default", "smart", "heuristic", "grid")
_SLICE_ALIASES = {
    "auto": "default",
    "默认": "default",
    "智能": "smart",
    "启发式": "heuristic",
    "网格": "grid",
}

# Reference slicer parameters (see Vocal2Midi/inference/API/slicer_api.py).
_DEFAULT_PARAMS = dict(threshold_db=-30.0, min_length_ms=5000, max_sil_kept_ms=500)
_SMART_DEFAULTS = dict(min_length_ms=5000, max_sil_kept_ms=5000)
_HEURISTIC_PRE = dict(threshold_db=-40.0, min_length_ms=800, min_interval_ms=200, max_sil_kept_ms=100)
_TINY_CHUNK_SEC = 0.35


def _ensure_ref_path() -> None:
    ref = str(_V2M_REF_DIR)
    if _V2M_REF_DIR.is_dir() and ref not in sys.path:
        sys.path.append(ref)


def normalize_slice_mode(mode: str | None) -> str:
    """Map a user-facing slice mode (incl. Chinese labels) to SLICE_MODES."""
    if mode is None:
        return "default"
    key = str(mode).strip()
    normalized = _SLICE_ALIASES.get(key) or _SLICE_ALIASES.get(key.lower())
    if normalized is not None:
        return normalized
    lowered = key.lower()
    for candidate in SLICE_MODES:
        if lowered == candidate:
            return candidate
    raise ValueError(f"Unsupported slice mode: {mode!r}. Supported values: {', '.join(SLICE_MODES)}")


def _rms_chunks(waveform: np.ndarray, sr: int, threshold_db: float, min_length_ms: int, max_sil_kept_ms: int, min_interval_ms: int = 300) -> list[dict]:
    """Run the reference RMS Slicer and convert chunks to our dict format."""
    _ensure_ref_path()
    from inference.slicer.slicer2 import Slicer

    slicer = Slicer(
        sr=sr,
        threshold=threshold_db,
        min_length=min_length_ms,
        min_interval=min_interval_ms,
        max_sil_kept=max_sil_kept_ms,
    )
    chunks: list[dict] = []
    for raw in slicer.slice(waveform):
        wav = np.asarray(raw["waveform"])
        start = float(raw["offset"])
        end = start + (wav.shape[-1] / sr)
        chunks.append({"waveform": wav, "sr": int(sr), "start": start, "end": end})
    return chunks


def _grid_chunks(waveform: np.ndarray, sr: int, length_sec: float) -> list[dict]:
    total = waveform.shape[-1]
    if total <= 0 or length_sec <= 0:
        return []
    step = int(round(length_sec * sr))
    chunks: list[dict] = []
    for begin in range(0, total, step):
        end_sample = min(begin + step, total)
        if waveform.ndim > 1:
            wav = waveform[:, begin:end_sample]
        else:
            wav = waveform[begin:end_sample]
        chunks.append({
            "waveform": np.asarray(wav),
            "sr": int(sr),
            "start": begin / sr,
            "end": end_sample / sr,
        })
    return chunks


def _merge_tiny(chunks: list[dict], sr: int) -> list[dict]:
    """Merge fragments shorter than _TINY_CHUNK_SEC into a neighbour."""
    merged: list[dict] = []
    for chunk in chunks:
        dur = chunk["waveform"].shape[-1] / sr
        if dur < _TINY_CHUNK_SEC and merged:
            prev = merged[-1]
            gap = max(0, int(round((chunk["start"] - prev["end"]) * sr)))
            pieces = [prev["waveform"]]
            if gap > 0:
                shape = list(prev["waveform"].shape)
                shape[-1] = gap
                pieces.append(np.zeros(shape, dtype=prev["waveform"].dtype))
            pieces.append(chunk["waveform"])
            axis = -1 if prev["waveform"].ndim > 1 else 0
            merged[-1] = {**prev, "end": chunk["end"], "waveform": np.concatenate(pieces, axis=axis)}
        else:
            merged.append(dict(chunk))
    return merged


def _heuristic_chunks(waveform: np.ndarray, sr: int, max_len_sec: float) -> list[dict]:
    pre = _rms_chunks(
        waveform, sr,
        threshold_db=_HEURISTIC_PRE["threshold_db"],
        min_length_ms=_HEURISTIC_PRE["min_length_ms"],
        max_sil_kept_ms=_HEURISTIC_PRE["max_sil_kept_ms"],
        min_interval_ms=_HEURISTIC_PRE["min_interval_ms"],
    )
    final: list[dict] = []
    for chunk in pre:
        dur = chunk["waveform"].shape[-1] / sr
        if dur > max_len_sec:
            step = int(round(max_len_sec * sr))
            wav = chunk["waveform"]
            axis_last = wav.shape[-1]
            for i in range(0, axis_last, step):
                piece = wav[..., i:i + step] if wav.ndim > 1 else wav[i:i + step]
                final.append({
                    "waveform": np.asarray(piece),
                    "sr": int(sr),
                    "start": chunk["start"] + i / sr,
                    "end": chunk["start"] + min(i + step, axis_last) / sr,
                })
        else:
            final.append(chunk)
    return _merge_tiny(final, sr)


def slice(
    audio: np.ndarray,
    sr: int,
    mode: str = "default",
    *,
    length: float = 5.0,
    threshold_db: float = -40.0,
    max_len_sec: float = 22.0,
) -> list[dict]:
    """Slice ``audio`` into chunks (see module docstring for mode semantics).

    Args:
        audio: 1-D mono or 2-D multi-channel waveform (any numeric dtype).
        sr: sample rate of ``audio``.
        mode: one of :data:`SLICE_MODES` (Chinese labels accepted too).
        length: window length in seconds for the "grid" mode.
        threshold_db: RMS silence threshold in dB for the "smart" mode.
        max_len_sec: upper bound for "heuristic" segments (longer segments
            are split into windows of this size).
    """
    normalized = normalize_slice_mode(mode)
    waveform = np.asarray(audio, dtype=np.float32)
    if sr <= 0:
        raise ValueError(f"sr must be positive, got {sr}")
    if waveform.size == 0:
        return []
    if normalized == "grid":
        return _grid_chunks(waveform, int(sr), float(length))
    if normalized == "smart":
        p = _SMART_DEFAULTS
        return _rms_chunks(waveform, int(sr), float(threshold_db), p["min_length_ms"], p["max_sil_kept_ms"])
    if normalized == "default":
        p = _DEFAULT_PARAMS
        return _rms_chunks(waveform, int(sr), p["threshold_db"], p["min_length_ms"], p["max_sil_kept_ms"])
    # heuristic
    return _heuristic_chunks(waveform, int(sr), float(max_len_sec))
