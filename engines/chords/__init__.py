"""Chord / key / meter analysis engine package (wave C1-7, madmom-based).

Importing this package installs the py3.13/numpy2/scipy compatibility shims
(:mod:`engines.chords.compat`) before any ``madmom`` module is imported, so
both ``from engines.chords import ChordEngine`` and direct submodule imports
work unchanged.
"""

from __future__ import annotations

from .compat import install_madmom_compat

install_madmom_compat()  # must run before any `import madmom` in this package

from .chord_engine import (  # noqa: E402
    BEAT_FPS,
    CHORD_FPS,
    LEGAL_CHORD_LABELS,
    SAMPLE_RATE,
    ChordEngine,
    write_chord_outputs,
)

__all__ = [
    "BEAT_FPS",
    "CHORD_FPS",
    "LEGAL_CHORD_LABELS",
    "SAMPLE_RATE",
    "ChordEngine",
    "install_madmom_compat",
    "write_chord_outputs",
]
