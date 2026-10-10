"""Theme application for the new UI.

Three lessons from the archived first-generation UI are baked in here, because
each of them cost real debugging time:

1. **Never touch ``QApplication.setStyleSheet``.** qfluentwidgets-style libraries
   own the application sheet and re-emit it on every theme change; overwriting it
   from outside produced an access violation as soon as the event loop ran. The
   sheet therefore rides on a **registered root widget** (``set_qss_root``) — a
   widget-level sheet cascades to children just the same.
2. **Set the QPalette too.** QSS does not reach Fusion-drawn internals
   (drop-down arrows, scrollbar arrows, text cursors), so the palette has to be
   pinned in lockstep or stray platform colours leak through.
3. **Re-capture font baselines when the family changes**, otherwise the next
   font-scale pass restores the old family.

This module is the only place in ``vstudio_ui`` that knows about themes, and —
because both end up on the same two objects, the application font and the
stylesheet — the only place that applies a font scale either.
"""

from __future__ import annotations

import os

from PyQt5.QtCore import QEasingCurve, QPointF
from PyQt5.QtGui import QColor, QFont, QPalette
from PyQt5.QtWidgets import QApplication

from vstudio_ui import tokens
from vstudio_ui.tokens import (
    BORDER_WIDTH,
    BORDER_WIDTH_STRONG,
    FOCUS,
    FONT_FAMILY,
    MOTION,
    RADIUS,
    THEMES,
    THEME_ORDER,
)

_active_theme: str = "light"
_qss_root = None


# ---------------------------------------------------------------------------
# Colour
# ---------------------------------------------------------------------------


def current_theme() -> str:
    return _active_theme


def is_dark() -> bool:
    return _active_theme == "dark"


def table(theme: str | None = None) -> dict[str, str]:
    """Token table for *theme* (defaults to the active one)."""
    return THEMES.get(theme or _active_theme, THEMES["light"])


def color(name: str) -> QColor:
    """QColor for token *name* under the active theme.

    Raises ``KeyError`` for an unknown name: the token table is the contract, and
    a missing row means the table needs a row — not that a widget should invent a
    colour.
    """
    values = table()
    if name not in values:
        raise KeyError(f"{name!r} is not a design token")
    return QColor(values[name])


def color_hex(name: str) -> str:
    return color(name).name()


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------


def _fuse(palette: QPalette, role, value: QColor) -> None:
    palette.setColor(QPalette.Active, role, value)
    palette.setColor(QPalette.Inactive, role, value)
    palette.setColor(QPalette.Disabled, role, value.darker(160))


def build_palette(theme: str) -> QPalette:
    values = table(theme)
    p = QPalette()
    for role, key in (
        (QPalette.Window, "background"),
        (QPalette.WindowText, "foreground"),
        (QPalette.Base, "card"),
        (QPalette.AlternateBase, "card_alt"),
        (QPalette.Text, "foreground"),
        (QPalette.Button, "background_alt"),
        (QPalette.ButtonText, "foreground"),
        (QPalette.BrightText, "destructive"),
        (QPalette.Highlight, "primary"),
        (QPalette.HighlightedText, "primary_foreground"),
        (QPalette.ToolTipBase, "card"),
        (QPalette.ToolTipText, "foreground"),
        (QPalette.PlaceholderText, "muted_foreground"),
        (QPalette.Link, "secondary"),
    ):
        _fuse(p, role, QColor(values[key]))
    # frame roles: hairlines only, never a blurred edge
    p.setColor(QPalette.Light, QColor(values["card"]))
    p.setColor(QPalette.Midlight, QColor(values["border_soft"]))
    p.setColor(QPalette.Mid, QColor(values["border"]))
    p.setColor(QPalette.Dark, QColor(values["neutral"]))
    p.setColor(QPalette.Shadow, QColor(values["border_strong"]))
    return p


# ---------------------------------------------------------------------------
# Stylesheet
# ---------------------------------------------------------------------------


def build_qss(theme: str) -> str:
    """Global sheet for *theme*. Applied to the registered root widget."""
    p = table(theme)
    return f"""
/* NB: no font-family here. QSS accepts a single family and would drop the
   fallback chain, so CJK text fell back to a serif face. The family is applied
   at QApplication level instead (see _apply_font_family), where a families list
   is supported. */
QWidget {{
    color: {p['foreground']};
}}
QMainWindow, QDialog {{
    background: {p['background']};
}}
QScrollArea {{
    background: transparent;
    border: none;
}}
QScrollArea > QWidget > QWidget {{
    background: transparent;
}}
QSplitter::handle {{
    background: {p['border_soft']};
}}

QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {p['card']};
    color: {p['foreground']};
    border: {BORDER_WIDTH}px solid {p['border']};
    border-radius: {RADIUS}px;
    padding: 4px 6px;
    selection-background-color: {p['primary']};
    selection-color: {p['primary_foreground']};
}}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus,
QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border: {FOCUS['ring_width']}px solid {p['border_strong']};
}}
QLineEdit:disabled, QPlainTextEdit:disabled, QTextEdit:disabled,
QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{
    background: {p['background_alt']};
    color: {p['muted_foreground']};
}}
QComboBox QAbstractItemView {{
    background: {p['card']};
    color: {p['foreground']};
    border: {BORDER_WIDTH}px solid {p['border']};
    selection-background-color: {p['primary']};
    selection-color: {p['primary_foreground']};
    outline: none;
}}

QPushButton {{
    background: {p['card']};
    color: {p['foreground']};
    border: {BORDER_WIDTH}px solid {p['border']};
    border-radius: {RADIUS}px;
    padding: 5px 14px;
}}
QPushButton:hover {{
    background: {p['background_alt']};
    border-color: {p['border_strong']};
}}
QPushButton:pressed {{
    background: {p['primary']};
    border-color: {p['border_strong']};
    color: {p['primary_foreground']};
}}
QPushButton:disabled {{
    color: {p['muted_foreground']};
    border-color: {p['border_soft']};
}}
QPushButton[primary="true"] {{
    background: {p['primary']};
    color: {p['primary_foreground']};
    border: none;
    border-radius: 18px;
    padding: 10px 28px;
    font-weight: 700;
}}
QPushButton[primary="true"]:disabled {{
    background: {p['border']};
    color: {p['muted_foreground']};
}}
QScrollBar:vertical {{ width: 8px; background: transparent; }}
QScrollBar::handle:vertical {{ background: {p['border']}; min-height: 24px; border-radius: 4px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}

QSlider::groove:horizontal {{
    height: 2px;
    background: {p['border']};
}}
QSlider::sub-page:horizontal {{
    height: 2px;
    background: {p['secondary']};
}}
QSlider::handle:horizontal {{
    width: 10px;
    margin: -5px 0;
    border-radius: 5px;
    background: {p['foreground']};
}}
QProgressBar {{
    background: {p['background_alt']};
    border: {BORDER_WIDTH}px solid {p['border']};
    border-radius: {RADIUS}px;
    color: {p['foreground']};
    text-align: center;
}}
QProgressBar::chunk {{
    background: {p['primary']};
}}
QCheckBox, QRadioButton {{
    color: {p['foreground']};
    spacing: 6px;
}}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 14px;
    height: 14px;
    border: {BORDER_WIDTH}px solid {p['border_strong']};
    background: {p['card']};
}}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
    background: {p['primary']};
}}

QListWidget, QListView, QTreeView, QTableView {{
    background: {p['card']};
    color: {p['foreground']};
    border: {BORDER_WIDTH}px solid {p['border']};
    border-radius: {RADIUS}px;
    outline: none;
}}
QListWidget::item:selected, QListView::item:selected,
QTreeView::item:selected, QTableView::item:selected {{
    background: {p['primary']};
    color: {p['primary_foreground']};
}}
QHeaderView::section {{
    background: {p['background_alt']};
    color: {p['foreground']};
    border: none;
    border-bottom: {BORDER_WIDTH}px solid {p['border']};
    padding: 4px 6px;
}}
QTabBar::tab {{
    background: transparent;
    color: {p['muted_foreground']};
    border: none;
    border-bottom: {BORDER_WIDTH_STRONG}px solid transparent;
    padding: 6px 12px;
}}
QTabBar::tab:selected {{
    color: {p['foreground']};
    border-bottom-color: {p['secondary']};
}}

QToolTip {{
    background: {p['card']};
    color: {p['foreground']};
    border: {BORDER_WIDTH}px solid {p['border_strong']};
    padding: 4px 6px;
}}
QMenu {{
    background: {p['card']};
    color: {p['foreground']};
    border: {BORDER_WIDTH}px solid {p['border']};
}}
QMenu::item:selected {{
    background: {p['primary']};
    color: {p['primary_foreground']};
}}
"""


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


def set_qss_root(widget) -> None:
    """Register the widget that carries the theme stylesheet (see module doc)."""
    global _qss_root
    _qss_root = widget


def _apply_qss(theme: str) -> None:
    if _qss_root is None:
        return
    try:
        _qss_root.setStyleSheet(build_qss(theme))
    except RuntimeError:
        # root destroyed (window closed) — nothing left to style
        pass


def apply_theme(theme: str) -> None:
    """Set the active theme: palette + stylesheet + application font, in lockstep."""
    global _active_theme
    _active_theme = theme if theme in THEMES else "light"
    app = QApplication.instance()
    if app is not None:
        if app.style().objectName().lower() != "fusion":
            app.setStyle("Fusion")
        app.setPalette(build_palette(_active_theme))
    _apply_app_font()
    _apply_qss(_active_theme)


# ---------------------------------------------------------------------------
# Application font (family + size)
# ---------------------------------------------------------------------------

_original_family: str | None = None
_base_point_size: float | None = None
_active_font_scale: float = tokens.DEFAULT_FONT_SCALE


def current_font_scale() -> float:
    """The font scale currently applied to the application font."""
    return _active_font_scale


def apply_font_scale(scale: float) -> float:
    """Rescale the type scale and the application font. Returns the scale used.

    Two things have to move together, and skipping either one produces a UI that
    is only half-scaled — which is much harder to notice than not scaling at all:

    * ``tokens.TYPE_SCALE``, which every stylesheet reads. The kit sets explicit
      ``font-size`` in QSS, so the application font cannot reach those widgets.
    * the **application font** itself, which is what the widgets the kit does
      *not* style with a size use: ``QLineEdit``, ``QComboBox``, and the run log
      (``QPlainTextEdit``). Without this the log box keeps 9pt text at a 2x
      scale.

    Callers still have to call ``refresh_theme()`` on the widget tree afterwards;
    this function cannot know which tree it was called for.
    """
    global _active_font_scale
    _active_font_scale = tokens.set_font_scale(scale)
    _apply_app_font()
    return _active_font_scale


def app_font(point_size: float | None = None) -> QFont:
    """A font to **measure** with, based on the application font.

    Widgets must not measure with ``self.font()``. Qt updates a widget's font
    through font-change propagation, and a ``refresh_theme`` pass triggered by a
    scale change can run before that has happened — so ``self.font()`` may still
    report the previous size. Measuring against the stale font is precisely how
    the nav rail fails to widen and the run log keeps its old minimum height,
    i.e. the two failures that are invisible until a label is clipped.
    """
    app = QApplication.instance()
    font = QFont(app.font()) if app is not None else QFont()
    if point_size is not None:
        font.setPointSizeF(float(point_size))
    return font


def _apply_app_font() -> None:
    """Put the token font family and the current scale on the application font.

    Three Qt 5 traps live here:

    * ``QFont.setFamilies()`` populates the substitution list but leaves
      ``family()`` — and therefore the *resolved* face — untouched. On a machine
      whose default is SimSun the UI then renders CJK in a serif face even though
      ``families()`` looks correct.
    * A single ``setFamily`` has no fallback, so a missing face silently degrades.
      ``insertSubstitutions`` supplies the chain.
    * The baseline point size must be captured **before** the first ``setFont``.
      Reading it back afterwards returns the size this function just wrote, so
      every subsequent scale would compound instead of replacing.
    """
    global _original_family, _base_point_size
    app = QApplication.instance()
    if app is None:
        return
    font = QFont(app.font())
    if _original_family is None:
        _original_family = font.family()
    if _base_point_size is None:
        # pointSizeF() is -1 when the font was specified in pixels; 9pt is the
        # Windows default and a sane fallback for any platform that does that.
        _base_point_size = float(font.pointSizeF())
        if _base_point_size <= 0:
            _base_point_size = 9.0
    font.setFamily(FONT_FAMILY[0])
    QFont.insertSubstitutions(FONT_FAMILY[0], list(FONT_FAMILY[1:]))
    font.setPointSizeF(max(1.0, _base_point_size * _active_font_scale))
    app.setFont(font)


# ---------------------------------------------------------------------------
# Motion
# ---------------------------------------------------------------------------


def easing_curve(name: str = "enter") -> QEasingCurve:
    """Build a QEasingCurve from ``MOTION`` (Qt has no CSS cubic-bezier syntax)."""
    x1, y1, x2, y2 = MOTION[f"{name}_curve"]  # type: ignore[index]
    curve = QEasingCurve(QEasingCurve.BezierSpline)
    curve.addCubicBezierSegment(QPointF(x1, y1), QPointF(x2, y2), QPointF(1.0, 1.0))
    return curve


def motion_duration(name: str = "enter") -> int:
    return int(MOTION[f"{name}_ms"])  # type: ignore[index]


def prefers_reduced_motion() -> bool:
    """Qt 5 has no ``prefers-reduced-motion``; read an env override instead."""
    return os.environ.get("VOCALSTUDIO_REDUCED_MOTION", "").strip().lower() in ("1", "true", "yes")


__all__ = [
    "THEME_ORDER",
    "app_font",
    "apply_font_scale",
    "apply_theme",
    "build_palette",
    "build_qss",
    "color",
    "color_hex",
    "current_font_scale",
    "current_theme",
    "easing_curve",
    "is_dark",
    "motion_duration",
    "prefers_reduced_motion",
    "set_qss_root",
    "table",
]
