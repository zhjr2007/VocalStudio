"""UI snapshot tool — render every page and save PNGs for before/after comparison.

Usage
-----
    # all three themes, one subprocess each
    python scripts/ui_snapshot.py

    # a single theme, in this process
    python scripts/ui_snapshot.py --theme rhine

    # custom output directory / window size / page subset
    python scripts/ui_snapshot.py --theme rhine --out temp/shots --pages 0,7

Why one process per theme
-------------------------
qfluentwidgets applies theme changes lazily. Switching themes inside a single
process leaves widgets in a half-restyled state, so a snapshot taken after a
switch is not trustworthy. `--theme all` therefore re-invokes this file as a
child process per theme.

Why the real display platform
-----------------------------
Under QT_QPA_PLATFORM=offscreen, `QWidget.grab()` returns layout geometry with
no rendered text. These snapshots are meant to be *looked at*, so the script
refuses to run offscreen unless --allow-offscreen is passed.

This is the acceptance tool referenced by ui-design/refactor-plan.md §5.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PAGE_NAMES = [
    "overview",
    "separation",
    "midi",
    "asr",
    "tts",
    "pipeline",
    "training",
    "settings",
]

THEMES = ("light", "dark", "rhine")


def _settle(app, seconds: float) -> None:
    """Pump the event loop for *seconds* so animations actually advance."""
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.02)


def capture(theme: str, out_dir: Path, pages: list[int], size: tuple[int, int]) -> int:
    """Render *pages* under *theme* into *out_dir*. Returns the exit code."""
    # Same native-library load order as gui/fluent_main.py: torch before Qt.
    try:
        import torch  # noqa: F401
    except Exception:
        pass

    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv[:1])

    from gui.fluent_main import MainWindow
    from gui.theme import set_app_theme

    window = MainWindow()
    window.resize(*size)
    window.show()

    set_app_theme(theme)
    _settle(app, 1.5)  # let qfluentwidgets finish its lazy restyle

    # model_status styles its own list and must be told to re-read the theme
    refresh = getattr(getattr(window, "settings_page", None), "refresh_model_status", None)
    if callable(refresh):
        try:
            refresh()
        except Exception as exc:  # pragma: no cover - cosmetic only
            print(f"  (refresh_model_status skipped: {exc})", file=sys.stderr)
    _settle(app, 0.6)

    out_dir.mkdir(parents=True, exist_ok=True)
    for index in pages:
        window.stackedWidget.setCurrentIndex(index)
        _settle(app, 0.6)  # cover the 240 ms page fade
        path = out_dir / f"{theme}_{index}_{PAGE_NAMES[index]}.png"
        window.grab().save(str(path))
        print(f"  {path.name}")

    print(f"SNAPSHOT_OK {theme} ({len(pages)} pages) -> {out_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--theme", default="all", choices=(*THEMES, "all"))
    parser.add_argument("--out", default=str(PROJECT_ROOT / "temp" / "ui_snapshots"))
    parser.add_argument("--pages", default="", help="comma-separated page indices, e.g. 0,7")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--allow-offscreen", action="store_true",
                        help="permit QT_QPA_PLATFORM=offscreen (text will NOT render)")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if os.environ.get("QT_QPA_PLATFORM") == "offscreen" and not args.allow_offscreen:
        print(
            "refusing to run under offscreen: text does not render there.\n"
            "unset QT_QPA_PLATFORM, or pass --allow-offscreen if you only want geometry.",
            file=sys.stderr,
        )
        return 2

    pages = (
        [int(p) for p in args.pages.split(",") if p.strip()]
        if args.pages.strip()
        else list(range(len(PAGE_NAMES)))
    )
    for index in pages:
        if not 0 <= index < len(PAGE_NAMES):
            print(f"page index out of range: {index}", file=sys.stderr)
            return 2

    out_dir = Path(args.out)

    # Single theme: do the work here.
    if args.theme != "all" or args.worker:
        return capture(args.theme, out_dir, pages, (args.width, args.height))

    # All themes: one child process each, so no theme state leaks across.
    failures = []
    for theme in THEMES:
        print(f"=== {theme} ===")
        cmd = [
            sys.executable, str(Path(__file__).resolve()),
            "--theme", theme,
            "--out", str(out_dir),
            "--width", str(args.width),
            "--height", str(args.height),
            "--worker",
        ]
        if args.pages.strip():
            cmd += ["--pages", args.pages]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            failures.append(theme)

    if failures:
        print(f"FAILED themes: {', '.join(failures)}", file=sys.stderr)
        return 1
    print(f"ALL OK -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
