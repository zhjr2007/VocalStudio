"""Entry point for the new VocalStudio UI.

    python -m vstudio_ui.app
    python vstudio_ui/app.py

torch is imported before Qt on Windows so the native libraries load in the order
Qt expects; getting this wrong produces a hard crash on some driver stacks.

Startup reads ``settings/user.json`` for the theme, language and font scale, then
hands the window to the event loop. Writing it back is the settings page's job
(through ``AppShell``), not this module's. The first hardware probe is kicked off
from here rather than from ``OverviewPage.__init__`` so that building a shell
stays cheap for tests and screenshots — see ``OverviewPage.auto_start``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:  # keep the native DLL load order Qt-friendly
    import torch  # noqa: F401
    from vstudio_runtime import configure_runtime
    configure_runtime()
except Exception:
    pass

from PyQt5.QtCore import QTimer, Qt  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402
if QApplication.instance() is None:
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

from vstudio_ui import services, strings  # noqa: E402
from vstudio_ui.pages import (  # noqa: E402
    AsrPage,
    MidiPage,
    OverviewPage,
    PipelinePage,
    SeparationPage,
    SettingsPage,
    TrainingPage,
    TtsPage,
)
from vstudio_ui.shell import AppShell  # noqa: E402

#: (nav key, page factory). Order defines the navigation order, and each key must
#: match the page's own ``key`` attribute — that is what ``TaskPage`` uses to look
#: up its strings and to name the default output subdirectory.
PAGE_SPECS = (
    ("overview", lambda shell: OverviewPage(shell)),
    ("separation", lambda shell: SeparationPage(shell)),
    ("midi", lambda shell: MidiPage(shell)),
    ("asr", lambda shell: AsrPage(shell)),
    ("tts", lambda shell: TtsPage(shell)),
    ("pipeline", lambda shell: PipelinePage(shell)),
    ("training", lambda shell: TrainingPage(shell)),
    ("settings", lambda shell: SettingsPage(shell)),
)


def build_shell(theme: str | None = None, font_scale: float | None = None) -> AppShell:
    """Construct the shell with every page registered.

    Language is *not* a parameter: ``strings`` is a module-level singleton that
    the caller sets with ``strings.set_language()`` before building, because the
    page factories below read their titles during construction.
    """
    shell = AppShell(theme=theme, font_scale=font_scale)
    for key, factory in PAGE_SPECS:
        shell.add_page(key, strings.NAV_LABELS[key], factory(shell))
    return shell


def shutdown_pages(shell: AppShell) -> None:
    """Ask every page with a job to stop.

    Iterates the pages rather than holding a reference to Overview: there are now
    seven pages that can own a worker thread, and the one that gets forgotten is
    the one that produces "QThread: Destroyed while thread is still running" on
    exit — a warning that is easy to ignore and hides a real dangling thread.
    """
    for key in shell.page_keys():
        page = shell.page(key)
        hook = getattr(page, "shutdown", None)
        if callable(hook):
            hook()


def _smoke_mode() -> str:
    """Which smoke run, if any, was requested.

    ``"frozen"`` is the packaging gate. ``build_portable.bat`` greps the exe's
    stdout for ``SMOKE_FROZEN_PASS`` — the same marker the CLI prints — so the
    build script does not have to know which front end it just built. ``"ui"`` is
    the interactive variant used while developing the UI; it stays silent.
    """
    if os.environ.get("VOCALSTUDIO_SMOKE") == "1":
        return "frozen"
    if os.environ.get("VOCALSTUDIO_UI_SMOKE") == "1":
        return "ui"
    return ""


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    app = QApplication(argv)

    preferences = services.load_preferences()
    strings.set_language(preferences.language)

    shell = build_shell(theme=preferences.theme, font_scale=preferences.font_scale)
    shell.show()

    overview = shell.page("overview")
    mode = _smoke_mode()

    if mode:
        # No probe here: the smoke run only has to prove the window paints and the
        # event loop survives. A WMI round trip that outlives app.quit() would
        # turn a clean exit into "QThread: Destroyed while thread is still running".

        def _finish() -> None:
            if mode == "frozen":
                print("SMOKE_FROZEN_PASS")
            app.quit()

        QTimer.singleShot(1200, _finish)
        return app.exec_()

    if overview is not None:
        overview.auto_start()
    app.aboutToQuit.connect(lambda: shutdown_pages(shell))
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
