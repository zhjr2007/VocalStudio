"""GUI usability checks + offscreen screenshot capture.

Run with:
    venv\\Scripts\\python -m pytest tests/test_gui_usability.py -q

Verifies that the main window is freely resizable, that the Settings page has
a user-adjustable font-size setting persisted through ``core.app_settings``,
and captures proof screenshots into ``temp/gui_diag/``.
"""

from __future__ import annotations

import os
import sys

# The offscreen QPA on Windows loads no fonts by default, so point Qt at the
# Windows font directory before the first QApplication is created. This makes
# text render in the captured screenshots. QT_QPA_PLATFORM is fixed to
# "offscreen" as required, so nothing ever opens a real window.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

import json
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCREENSHOT_DIR = REPO_ROOT / "temp" / "gui_diag"


@pytest.fixture(scope="module")
def screenshots_dir():
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    return SCREENSHOT_DIR


@pytest.fixture(scope="module")
def main_window(qapp):
    """Build the real MainWindow once (it is heavy: imports torch first)."""
    from gui.fluent_main import MainWindow

    window = MainWindow()
    window.show()
    for _ in range(5):
        qapp.processEvents()
    yield window
    window.close()


def _page_scroll(page, name):
    from PyQt5.QtWidgets import QFrame, QScrollArea

    scroll = QScrollArea()
    scroll.setObjectName(name)
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.NoFrame)
    scroll.setWidget(page)
    return scroll


def test_main_window_resizable(main_window, qapp):
    win = main_window
    # The window must actually shrink below its natural size and not be
    # pinned to a large minimum by the page content.
    minimum = win.minimumSizeHint()
    assert minimum.width() <= 900, f"minimum width too large: {minimum.width()}"
    assert minimum.height() <= 600, f"minimum height too large: {minimum.height()}"

    win.resize(800, 500)
    qapp.processEvents()
    assert win.width() == 800 and win.height() == 500, "window did not resize to 800x500"

    win.resize(1600, 900)
    qapp.processEvents()
    assert win.width() == 1600 and win.height() == 900, "window did not resize to 1600x900"

    # Every stack page is a scroll container with a non-empty object name.
    stack = win.stackedWidget
    assert stack.count() == 8
    for i in range(stack.count()):
        widget = stack.widget(i)
        assert widget.objectName(), f"stack page {i} has empty objectName"


def test_font_scale_setting_roundtrip(qapp):
    from core.app_settings import AppSettings
    from gui.pages.settings_page import SettingsPage
    from gui.theme import apply_font_scale

    tmp_settings = Path(tempfile.mkdtemp()) / "user.json"

    page = SettingsPage(settings_file=tmp_settings)
    page.set_font_scale(1.0)
    page.save_settings()
    # Capture the pixel size at the neutral scale.
    small_px = page._hardware_title_label.font().pixelSize()

    page.set_font_scale(1.5)
    page.save_settings()
    apply_font_scale(1.5)
    qapp.processEvents()
    large_px = page._hardware_title_label.font().pixelSize()
    assert large_px > small_px, f"font did not grow: {small_px} -> {large_px}"

    # The value must persist to user.json and round-trip through AppSettings.
    data = json.loads(tmp_settings.read_text(encoding="utf-8"))
    assert abs(float(data["font_scale"]) - 1.5) < 1e-6, data

    loaded = AppSettings.load(tmp_settings)
    assert abs(loaded.font_scale - 1.5) < 1e-6

    # A fresh SettingsPage reading the same file must show the stored scale.
    page2 = SettingsPage(settings_file=tmp_settings)
    assert abs(page2.font_scale - 1.5) < 1e-6


def test_screenshot_capture(main_window, qapp, screenshots_dir):
    from gui.i18n import i18n
    from gui.theme import apply_font_scale, set_app_theme

    i18n.set_language("en")
    set_app_theme("light")
    apply_font_scale(1.0)
    qapp.processEvents()

    # -- full window resizability proofs ----------------------------------
    win = main_window
    win.resize(800, 500)
    qapp.processEvents()
    win.grab().save(str(screenshots_dir / "window_800x500.png"))
    win.resize(1600, 900)
    qapp.processEvents()
    win.grab().save(str(screenshots_dir / "window_1600x900.png"))

    # -- Settings page at two window sizes --------------------------------
    from gui.pages.settings_page import SettingsPage

    tmp_settings = Path(tempfile.mkdtemp()) / "user.json"
    settings_page = SettingsPage(settings_file=tmp_settings)
    settings_page.setObjectName("settings_page")
    settings_page.set_theme("light")
    settings_page.set_font_scale(1.0)
    settings_page.save_settings()
    settings_scroll = _page_scroll(settings_page, "settings_page")

    for (width, height), tag in (((800, 500), "800x500"), ((1600, 900), "1600x900")):
        settings_scroll.resize(width, height)
        settings_scroll.show()
        qapp.processEvents()
        settings_scroll.grab().save(str(screenshots_dir / f"settings_{tag}.png"))

    # -- font scale normal vs large ---------------------------------------
    settings_page.set_font_scale(1.0)
    settings_page.save_settings()
    apply_font_scale(1.0)
    settings_scroll.resize(800, 500)
    settings_scroll.show()
    qapp.processEvents()
    settings_scroll.grab().save(str(screenshots_dir / "settings_font_normal.png"))

    settings_page.set_font_scale(1.5)
    settings_page.save_settings()
    apply_font_scale(1.5)
    settings_scroll.resize(800, 500)
    settings_scroll.show()
    qapp.processEvents()
    settings_scroll.grab().save(str(screenshots_dir / "settings_font_large.png"))

    # -- Separation page at two window sizes --------------------------------
    from gui.pages.separation_page import SeparationPage

    separation_page = SeparationPage()
    separation_page.setObjectName("separation_page")
    separation_scroll = _page_scroll(separation_page, "separation_page")
    for (width, height), tag in (((800, 500), "800x500"), ((1600, 900), "1600x900")):
        separation_scroll.resize(width, height)
        separation_scroll.show()
        qapp.processEvents()
        separation_scroll.grab().save(str(screenshots_dir / f"separation_{tag}.png"))

    # -- assert every screenshot exists and is non-trivial -----------------
    expected = [
        "window_800x500.png",
        "window_1600x900.png",
        "settings_800x500.png",
        "settings_1600x900.png",
        "settings_font_normal.png",
        "settings_font_large.png",
        "separation_800x500.png",
        "separation_1600x900.png",
    ]
    for name in expected:
        path = screenshots_dir / name
        assert path.exists(), f"missing screenshot {path}"
        assert path.stat().st_size > 5000, f"screenshot too small: {path}"
