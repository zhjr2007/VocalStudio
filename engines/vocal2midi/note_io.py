"""Note export for the vocal-to-MIDI pipeline (wave C1-6).

Thin wrapper around the reference exporters
(``Vocal2Midi/inference/io/note_io.py`` and ``inference/API/ustx_api.py``) so
output formats stay byte-compatible with the baseline artifacts in
``materials/v2m_diag_out``. Only the requested formats are written.

:class:`NoteInfo` mirrors the reference dataclass; every save function also
accepts any object exposing ``onset`` / ``offset`` / ``pitch`` (and optional
``lyric``) attributes.
"""

from __future__ import annotations

import pathlib
import sys
from dataclasses import dataclass
from typing import Any

from core.config import PROJECT_ROOT

_V2M_REF_DIR = PROJECT_ROOT / "Vocal2Midi"

OUTPUT_FORMATS = ("mid", "txt", "csv", "ustx")


@dataclass
class NoteInfo:
    """A single extracted note (times in seconds, pitch as MIDI number)."""

    onset: float
    offset: float
    pitch: float
    lyric: str = ""


def _ensure_ref_path() -> None:
    ref = str(_V2M_REF_DIR)
    if _V2M_REF_DIR.is_dir() and ref not in sys.path:
        sys.path.append(ref)


def normalize_output_formats(formats: Any) -> list[str]:
    """Lower-case, de-duplicated subset of OUTPUT_FORMATS (order preserved)."""
    if formats is None:
        return []
    if isinstance(formats, str):
        formats = [formats]
    seen: set[str] = set()
    result: list[str] = []
    for fmt in formats:
        key = str(fmt).strip().lower().lstrip(".")
        if not key:
            continue
        if key not in OUTPUT_FORMATS:
            raise ValueError(f"Unsupported output format: {fmt!r}. Supported values: {', '.join(OUTPUT_FORMATS)}")
        if key not in seen:
            seen.add(key)
            result.append(key)
    return result


def save_notes(
    notes: list[Any],
    out_dir: str | pathlib.Path,
    key: str,
    formats: Any = ("mid",),
    *,
    tempo: float = 120.0,
    pitch_format: str = "name",
    round_pitch: bool = True,
    rmvpe_result: Any | None = None,
) -> dict[str, pathlib.Path]:
    """Write ``notes`` to ``out_dir/<key>.<fmt>`` for each selected format.

    Returns a mapping of format -> written path (only the requested formats).
    ``rmvpe_result`` is only used by the "ustx" format (pitch-deviation curve);
    pass None to export notes without it.
    """
    _ensure_ref_path()
    from inference.io.note_io import _save_midi, _save_text

    selected = normalize_output_formats(formats)
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, pathlib.Path] = {}

    for fmt in selected:
        if fmt == "mid":
            path = out_dir / f"{key}.mid"
            _save_midi(list(notes), path, int(round(tempo)))
        elif fmt in ("txt", "csv"):
            path = out_dir / f"{key}.{fmt}"
            _save_text(list(notes), path, fmt, pitch_format, round_pitch)
        else:  # ustx (lazy import: pulls rmvpe_api -> onnxruntime/librosa)
            from inference.API.ustx_api import save_ustx

            path = out_dir / f"{key}.ustx"
            save_ustx(list(notes), path, tempo=float(tempo), rmvpe_result=rmvpe_result)
        written[fmt] = path
    return written
