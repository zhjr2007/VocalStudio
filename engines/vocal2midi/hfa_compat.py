"""HubertFA textgrid shim (wave C2).

The vendored HubertFA reference (``Vocal2Midi/inference/HubertFA``) imports
``textgrid`` at module load time, but only ever *uses* it to export/evaluate
TextGrid artifacts. Core ONNX inference + forced alignment never touches a
real ``textgrid`` object. The ``textgrid`` pip package is not installed and the
network is blocked on this host, so we install a minimal import stub into
``sys.modules`` before the reference module is imported.

Only the class names referenced by
``inference/HubertFA/tools/export_tool.py`` / ``metrics.py`` are provided.
They are never instantiated in the C2 inference path; a real export would
raise a clear ``NotImplementedError`` instead of a confusing module error.
"""

from __future__ import annotations

import sys
import types

_STUB_CLASSES = {"TextGrid", "IntervalTier", "PointTier", "Interval", "Point"}


def install_textgrid_stub() -> None:
    """Install a module-level ``textgrid`` import stub (idempotent)."""
    existing = sys.modules.get("textgrid")
    if existing is not None:
        return

    tg = types.ModuleType("textgrid")

    class _StubBase:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError(
                "textgrid is not installed; HFA inference does not need it, "
                "but TextGrid export/evaluation is unavailable."
            )

    for name in _STUB_CLASSES:
        setattr(tg, name, _StubBase)

    sys.modules["textgrid"] = tg
