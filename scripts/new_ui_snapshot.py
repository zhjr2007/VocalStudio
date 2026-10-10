"""Snapshot the NEW interface (``vstudio_ui``) — one PNG per page.

Usage
-----
    # both themes, all 8 pages, real display
    python scripts/new_ui_snapshot.py

    # one theme, two pages, into a custom directory
    python scripts/new_ui_snapshot.py --theme dark --pages 0,7 --out temp/shots

    # wait for the real hardware probe so the Overview page shows live values
    python scripts/new_ui_snapshot.py --probe --probe-timeout 25

Why this is separate from ``scripts/ui_snapshot.py``
---------------------------------------------------
That one drives the archived first-generation GUI through qfluentwidgets, which
needs a settle window per theme and a "refresh_model_status" nudge. The new UI has
no third-party widget library and re-styles synchronously, so it needs none of
that — and pretending otherwise would hide a regression if it ever crept back.

Why one process per theme
-------------------------
Kept anyway, because it is the honest way to compare two themes: no state from
the first capture can leak into the second.

Why the real display platform
-----------------------------
Under ``QT_QPA_PLATFORM=offscreen``, ``QWidget.grab()`` returns layout geometry
with no rendered text. These snapshots exist to be *looked at*, so the script
refuses to run offscreen unless ``--allow-offscreen`` is passed.

By default the Overview page is captured in its "pending" state (dashes) — that
is what a user sees for the first few seconds, and it is worth checking. Pass
``--probe`` to wait for the real probe instead.
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

THEMES = ("light", "dark")


def page_keys() -> list[str]:
    from vstudio_ui.app import PAGE_SPECS

    return [key for key, _factory in PAGE_SPECS]


def settle(app, seconds: float) -> None:
    """Pump the event loop for *seconds* so animations and paint actually advance."""
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.02)


def capture(
    theme: str,
    out_dir: Path,
    pages: list[int],
    size: tuple[int, int],
    probe: bool,
    probe_timeout: float,
    selfcheck: bool = False,
    font_scale: float = 1.0,
) -> int:
    # Same native-library load order as vstudio_ui/app.py: torch before Qt.
    try:
        import torch  # noqa: F401
    except Exception:
        pass

    from PyQt5.QtWidgets import QApplication

    from vstudio_ui.app import build_shell
    app = QApplication.instance() or QApplication(sys.argv[:1])

    keys = page_keys()
    # font_scale is passed rather than applied afterwards so the very first paint
    # is already scaled — a scaled-then-rescaled window would hide a chrome size
    # that is only computed at construction.
    shell = build_shell(theme=theme, font_scale=font_scale)
    shell.resize(*size)
    shell.show()
    settle(app, 0.8)

    overview = shell.page("overview")
    if overview is not None and probe:
        overview.auto_start()
        deadline = time.time() + probe_timeout
        while overview._probe.running and time.time() < deadline:
            app.processEvents()
            time.sleep(0.02)
        settle(app, 0.4)
        if overview._probe.running:
            print(f"  (probe still running after {probe_timeout:.0f}s)", file=sys.stderr)

    if overview is not None and selfcheck:
        overview.run_self_check()
        deadline = time.time() + probe_timeout
        while overview._check.running and time.time() < deadline:
            app.processEvents()
            time.sleep(0.02)
        settle(app, 0.3)
        if overview._check.running:
            print(f"  (self-check still running after {probe_timeout:.0f}s)", file=sys.stderr)

    out_dir.mkdir(parents=True, exist_ok=True)
    for index in pages:
        shell.navigate(index)
        settle(app, 0.6)
        path = out_dir / f"{theme}_{index}_{keys[index]}.png"
        shell.grab().save(str(path))
        print(f"  {path.name}")

    print(f"NEW_UI_SNAPSHOT_OK {theme} ({len(pages)} pages) -> {out_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--theme", default="all", choices=(*THEMES, "all"))
    parser.add_argument('--language', default='zh', choices=('zh', 'en'))
    parser.add_argument("--out", default=str(PROJECT_ROOT / "temp" / "new_ui_snapshots"))
    parser.add_argument("--pages", default="", help="comma-separated page indices, e.g. 0,7")
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--probe", action="store_true", help="wait for the live hardware probe")
    parser.add_argument(
        "--selfcheck", action="store_true", help="run the Overview self-check and wait for it"
    )
    parser.add_argument("--probe-timeout", type=float, default=20.0)
    parser.add_argument(
        "--font-scale",
        type=float,
        default=1.0,
        help="type scale to render at, e.g. 1.5 — the point of this flag is to see "
        "whether the fixed-size chrome (header, nav rail, run log) really grows with it",
    )
    parser.add_argument(
        "--allow-offscreen",
        action="store_true",
        help="permit QT_QPA_PLATFORM=offscreen (text will NOT render)",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    from vstudio_ui import strings
    strings.set_language(args.language)

    if os.environ.get("QT_QPA_PLATFORM") == "offscreen" and not args.allow_offscreen:
        print(
            "refusing to run under offscreen: text does not render there.\n"
            "unset QT_QPA_PLATFORM, or pass --allow-offscreen if you only want geometry.",
            file=sys.stderr,
        )
        return 2

    total = len(page_keys())
    pages = (
        [int(p) for p in args.pages.split(",") if p.strip()]
        if args.pages.strip()
        else list(range(total))
    )
    for index in pages:
        if not 0 <= index < total:
            print(f"page index out of range: {index}", file=sys.stderr)
            return 2

    out_dir = Path(args.out)

    if args.theme != "all" or args.worker:
        return capture(
            args.theme,
            out_dir,
            pages,
            (args.width, args.height),
            args.probe,
            args.probe_timeout,
            args.selfcheck,
            args.font_scale,
        )

    failures = []
    for theme in THEMES:
        print(f"=== {theme} ===")
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--theme",
            theme,
            '--language', args.language,
            "--out",
            str(out_dir),
            "--width",
            str(args.width),
            "--height",
            str(args.height),
            "--probe-timeout",
            str(args.probe_timeout),
            "--font-scale",
            str(args.font_scale),
            "--worker",
        ]
        if args.pages.strip():
            cmd += ["--pages", args.pages]
        if args.probe:
            cmd.append("--probe")
        if args.selfcheck:
            cmd.append("--selfcheck")
        if subprocess.run(cmd).returncode != 0:
            failures.append(theme)

    if failures:
        print(f"FAILED themes: {', '.join(failures)}", file=sys.stderr)
        return 1
    print(f"ALL OK -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
