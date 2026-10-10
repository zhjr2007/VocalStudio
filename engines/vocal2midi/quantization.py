"""Note quantization for the vocal-to-MIDI pipeline (wave C1-5).

Thin wrapper around the reference implementation
(``Vocal2Midi/inference/quant/quantization.py``, pure numpy, no model deps).
Notes are plain objects with mutable ``.onset`` / ``.offset`` attributes in
seconds (see :class:`engines.vocal2midi.note_io.NoteInfo`); quantization is
applied in place and the same list is returned.

Grid convention: ticks = seconds * tempo * 8, i.e. 480 ticks per beat, so a
``step`` of N ticks snaps to (N/480)-beat boundaries; BPM therefore changes
the physical size of the snap grid.

Modes (:data:`QUANTIZATION_MODES`):
- ``"simple"``: round each onset/offset to the nearest grid step, then fix
  ordering so onsets stay strictly increasing.
- ``"smart"``: DP over duration candidates (preferred note values) plus
  simple onset snapping.
- ``"bayes"``: segment-wise DP with metrical position / motif priors.
- ``"dp"``: phrase-hybrid DP (asymmetric costs, per-segment phase centers).
"""

from __future__ import annotations

import sys
from typing import Any

from core.config import PROJECT_ROOT

_V2M_REF_DIR = PROJECT_ROOT / "Vocal2Midi"

QUANTIZATION_MODES = ("simple", "smart", "bayes", "dp")


def _ensure_ref_path() -> None:
    ref = str(_V2M_REF_DIR)
    if _V2M_REF_DIR.is_dir() and ref not in sys.path:
        sys.path.append(ref)


def quantize_notes(notes: list[Any], tempo: float = 120.0, step: int = 30, mode: str = "simple") -> list[Any]:
    """Quantize ``notes`` in place onto the (tempo, step) grid; returns them."""
    if not notes or tempo <= 0:
        return notes
    _ensure_ref_path()
    from inference.quant.quantization import quantize_notes as _ref_quantize

    _ref_quantize(list(notes), float(tempo), int(step), mode)
    return notes


def should_apply_quantization(mode: str, step: int) -> bool:
    """Mirror of the reference rule: "dp" always applies; others need step > 0."""
    _ensure_ref_path()
    from inference.quant.quantization import should_apply_quantization as _ref_should

    return bool(_ref_should(mode, int(step or 0)))
