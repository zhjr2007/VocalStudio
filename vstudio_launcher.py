"""Frozen entry point: one bundle, two front ends.

    VocalStudioBackend.exe                 -> the new interface (vstudio_ui)
    VocalStudioBackend.exe separate ...    -> the CLI, exactly as before
    VocalStudioBackend.exe info            -> the CLI's environment report

Why a dispatcher instead of two executables: the new UI runs the pipeline
orchestrator **in process**, so it needs the same torch CUDA build and native
libraries the CLI does — over 4 GiB. A second one-dir bundle would double the
portable distribution to no benefit, and a single-file bundle of that size cannot
be produced at all (PyInstaller's CArchive offsets are 32-bit; see
``VocalStudioBackend.spec``).

Why this file instead of a branch inside ``main.py``: ``main.py`` is inside the
frozen-contract gate — ``git diff --stat core/ engines/ main.py`` has to stay
empty — and a new "if no arguments, start a GUI" branch in the CLI entry point is
precisely the kind of change that gate exists to prevent.

Note that the *dispatch* is on "were arguments given", not on which front end is
"primary". A CLI user types a subcommand and gets the CLI; a user who
double-clicks the exe gets the window.
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    from vstudio_runtime import configure_runtime
    configure_runtime()

    if argv:
        # Imported lazily so that the GUI path never pays for argparse and the CLI
        # path never pays for Qt.
        from main import main as cli_main

        return cli_main(argv)

    from vstudio_ui.app import main as ui_main

    return ui_main([])


if __name__ == "__main__":
    raise SystemExit(main())
