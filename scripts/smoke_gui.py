"""Offscreen GUI smoke test: build MainWindow and visit all pages."""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

EXPECTED_PAGES = 8


def main() -> int:
    from gui.fluent_main import MainWindow  # loads torch before Qt (Windows DLL order)

    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv[:1])
    window = MainWindow()

    stack = window.stackedWidget
    if stack.count() != EXPECTED_PAGES:
        print(f"FAIL: expected {EXPECTED_PAGES} pages, found {stack.count()}")
        return 1

    for index in range(stack.count()):
        stack.setCurrentIndex(index)
        for _ in range(3):
            app.processEvents()
        name = stack.currentWidget().objectName()
        if not name:
            print(f"FAIL: page {index} has empty objectName")
            return 1

    window.switch_language("en")
    window.switch_language("zh")
    for _ in range(3):
        app.processEvents()

    print(f"SMOKE_GUI_PASS pages={stack.count()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
