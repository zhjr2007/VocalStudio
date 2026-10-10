"""Tests for the frozen entry point and the packaging gate.

Small module, but the failure it guards against is expensive: the packaging entry
is only exercised by ``build_portable.bat``, which takes tens of minutes and a
machine with the weights. A launcher that dispatches the wrong way, or a smoke
marker the build script no longer greps for, would be found *after* a full build —
or not at all, on a build nobody runs.

Three things are pinned down:

* ``vstudio_launcher.main`` sends arguments to the CLI and no arguments to the UI;
* the new UI prints ``SMOKE_FROZEN_PASS`` when ``VOCALSTUDIO_SMOKE=1``, which is
  the marker the build gate greps for — the same one the CLI prints, so the gate
  does not have to know which front end it built;
* the spec that is actually built points at the launcher, and does not exclude Qt.
"""

from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# the launcher
# ---------------------------------------------------------------------------


def test_launcher_starts_the_ui_when_given_no_arguments(monkeypatch):
    """Double-clicking the exe must reach the window, not the help text.

    The old behaviour was ``main.py`` with no argv: argparse printed help and
    returned 1, so a double-click produced a flash and nothing else.
    """
    import vstudio_launcher
    import vstudio_ui.app as ui_app

    seen: list[list] = []

    def fake_ui(argv):
        seen.append(list(argv))
        return 0

    monkeypatch.setattr(ui_app, "main", fake_ui)

    assert vstudio_launcher.main([]) == 0
    assert seen == [[]]


def test_launcher_sends_arguments_to_the_cli(monkeypatch):
    import main as cli_module
    import vstudio_launcher
    import vstudio_ui.app as ui_app

    seen: list[list] = []

    def fake_cli(argv):
        seen.append(list(argv))
        return 0

    def explode(argv):  # pragma: no cover - only runs if dispatch is wrong
        raise AssertionError("the UI was started for a CLI invocation")

    monkeypatch.setattr(cli_module, "main", fake_cli)
    monkeypatch.setattr(ui_app, "main", explode)

    assert vstudio_launcher.main(["info"]) == 0
    assert seen == [["info"]]


# ---------------------------------------------------------------------------
# the smoke marker the build gate greps for
# ---------------------------------------------------------------------------


def test_smoke_mode_reads_both_variables(monkeypatch):
    from vstudio_ui.app import _smoke_mode

    monkeypatch.delenv("VOCALSTUDIO_SMOKE", raising=False)
    monkeypatch.delenv("VOCALSTUDIO_UI_SMOKE", raising=False)
    assert _smoke_mode() == ""

    monkeypatch.setenv("VOCALSTUDIO_UI_SMOKE", "1")
    assert _smoke_mode() == "ui"

    # The frozen marker wins: build_portable.bat sets exactly this one, and the
    # gate greps stdout for the marker it produces.
    monkeypatch.setenv("VOCALSTUDIO_SMOKE", "1")
    assert _smoke_mode() == "frozen"


def test_build_gate_greps_the_marker_the_ui_prints():
    """One marker for both front ends, checked against the source that prints it.

    If the UI ever stops printing ``SMOKE_FROZEN_PASS`` the build fails with "see
    vs_ui_smoke.txt", which is a confusing place to learn about a missing print.
    """
    app_source = (PROJECT_ROOT / "vstudio_ui" / "app.py").read_text(encoding="utf-8")
    batch = (PROJECT_ROOT / "build_portable.bat").read_text(encoding="utf-8")

    assert 'print("SMOKE_FROZEN_PASS")' in app_source
    markers = re.findall(r'findstr /C:"([A-Z_]+)"', batch)
    assert markers, "build_portable.bat no longer checks any smoke marker"
    assert set(markers) == {"SMOKE_FROZEN_PASS"}


# ---------------------------------------------------------------------------
# the spec that is actually built
# ---------------------------------------------------------------------------


def test_the_built_spec_uses_the_launcher_and_bundles_qt():
    """``VocalStudioBackend.spec`` is the only spec on the build path now.

    Two ways to get this wrong, both silent until a full build finishes: pointing
    the entry back at ``main.py`` (so the double-clickable exe is the CLI), and
    putting ``PyQt5`` back into ``excludes`` (so the UI cannot import Qt).
    """
    spec = (PROJECT_ROOT / "VocalStudioBackend.spec").read_text(encoding="utf-8")
    batch = (PROJECT_ROOT / "build_portable.bat").read_text(encoding="utf-8")

    assert re.search(r'Analysis\(\s*\[\s*"vstudio_launcher\.py"', spec)
    excludes = re.search(r"excludes=\[([^\]]*)\]", spec)
    assert excludes is not None
    assert '"PyQt5"' not in excludes.group(1)

    # The archived one-file frontend is kept for the record but must not be built:
    # it excludes torch, and the new UI runs the orchestrator in process.
    assert "VocalStudio.spec" not in batch
    assert "VocalStudioBackend.spec" in batch


def test_archived_frontend_spec_is_marked_as_not_built():
    spec = (PROJECT_ROOT / "VocalStudio.spec").read_text(encoding="utf-8")
    assert "ARCHIVED" in spec
    assert "no longer built" in spec.lower()
