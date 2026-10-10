"""App-level theme bridge.

qfluentwidgets themes only its own widgets (via per-widget QSS). Plain PyQt
widgets (QComboBox, QLineEdit, QPushButton, QSlider, QProgressBar, QCheckBox,
QListWidget ...) follow the application QPalette, which stays light unless we
swap it. This module keeps the two in lockstep and forces the Fusion style so
native controls honour the palette on every platform.

Font scaling lives here too: qfluentwidgets controls create their own
pixel-sized QFonts, so raising the QApplication default font alone does not
scale them. Each widget stores its "base" font (the font it would have at
scale == 1.0) as a Python attribute and receives a scaled copy whenever
:func:`apply_font_scale` is called, so repeated calls never compound.
"""

from __future__ import annotations

import os

from PyQt5.QtCore import QEasingCurve, QPointF
from PyQt5.QtGui import QColor, QFont, QPalette
from PyQt5.QtWidgets import QApplication

try:
    from qfluentwidgets import Theme, setTheme as _set_fluent_theme, setThemeColor as _set_fluent_color
except ModuleNotFoundError:  # renamed distribution
    from qfluent_widgets import Theme, setTheme as _set_fluent_theme, setThemeColor as _set_fluent_color

from gui.design_tokens import (
    FONT_FAMILY,
    MOTION,
    RHINE,
    THEMES,
    build_qss,
    tokens_for,
)

#: Maps the theme registry's string base names to qfluentwidgets' enum. Kept
#: here (not in design_tokens) so the token module stays Qt-free.
_FLUENT_BASE = {
    "light": Theme.LIGHT,
    "dark": Theme.DARK,
}


def _fuse(cls, role, color):
    cls.setColor(QPalette.Active, role, color)
    cls.setColor(QPalette.Inactive, role, color)
    cls.setColor(QPalette.Disabled, role, color.darker(160))


def _dark_palette() -> QPalette:
    p = QPalette()
    window = QColor(0x20, 0x20, 0x20)
    base = QColor(0x1e, 0x1e, 0x1e)
    button = QColor(0x3c, 0x3c, 0x3c)
    text = QColor(0xff, 0xff, 0xff)
    highlight = QColor(0x00, 0x9f, 0xaa)
    for role, color in (
        (QPalette.Window, window),
        (QPalette.WindowText, text),
        (QPalette.Base, base),
        (QPalette.AlternateBase, QColor(0x2a, 0x2a, 0x2a)),
        (QPalette.Text, text),
        (QPalette.Button, button),
        (QPalette.ButtonText, text),
        (QPalette.BrightText, QColor(255, 60, 60)),
        (QPalette.Highlight, highlight),
        (QPalette.HighlightedText, QColor(0xff, 0xff, 0xff)),
        (QPalette.ToolTipBase, QColor(0x3c, 0x3c, 0x3c)),
        (QPalette.ToolTipText, text),
        (QPalette.PlaceholderText, QColor(0x9a, 0x9a, 0x9a)),
        (QPalette.Link, highlight),
    ):
        _fuse(p, role, color)
    # border / shadow roles so sunken frames stay dark
    p.setColor(QPalette.Light, QColor(0x3c, 0x3c, 0x3c))
    p.setColor(QPalette.Midlight, QColor(0x30, 0x30, 0x30))
    p.setColor(QPalette.Mid, QColor(0x28, 0x28, 0x28))
    p.setColor(QPalette.Dark, QColor(0x14, 0x14, 0x14))
    p.setColor(QPalette.Shadow, QColor(0x00, 0x00, 0x00))
    return p


def _light_palette() -> QPalette:
    app = QApplication.instance()
    if app is None:
        return QPalette()
    return app.style().standardPalette()


def _rhine_palette() -> QPalette:
    """Warm-white palette for the rhine theme.

    Same contract as :func:`_dark_palette`: every role is pinned so Fusion-drawn
    native widgets cannot fall back to the stock (cool, blue-tinted) greys.
    """
    p = QPalette()
    window = QColor(RHINE["background"])
    base = QColor(RHINE["card"])
    button = QColor(RHINE["background_alt"])
    text = QColor(RHINE["foreground"])
    accent = QColor(RHINE["primary"])
    accent_ink = QColor(RHINE["primary_foreground"])
    for role, color in (
        (QPalette.Window, window),
        (QPalette.WindowText, text),
        (QPalette.Base, base),
        (QPalette.AlternateBase, QColor(RHINE["card_alt"])),
        (QPalette.Text, text),
        (QPalette.Button, button),
        (QPalette.ButtonText, text),
        (QPalette.BrightText, QColor(RHINE["destructive"])),
        (QPalette.Highlight, accent),
        (QPalette.HighlightedText, accent_ink),
        (QPalette.ToolTipBase, base),
        (QPalette.ToolTipText, text),
        (QPalette.PlaceholderText, QColor(RHINE["muted_foreground"])),
        (QPalette.Link, QColor(RHINE["secondary"])),
    ):
        _fuse(p, role, color)
    # border / shadow roles: hairlines only, no blur anywhere in this theme
    p.setColor(QPalette.Light, QColor(RHINE["card"]))
    p.setColor(QPalette.Midlight, QColor(RHINE["border_soft"]))
    p.setColor(QPalette.Mid, QColor(RHINE["border"]))
    p.setColor(QPalette.Dark, QColor(RHINE["neutral"]))
    p.setColor(QPalette.Shadow, QColor(RHINE["border_strong"]))
    return p


def _is_dark(theme: str) -> bool:
    spec = THEMES.get(theme)
    if spec is not None and theme != "auto":
        return spec.is_dark
    # "auto" (or an unknown value) -> follow the system preference
    try:
        import darkdetect

        return darkdetect.theme() == "Dark"
    except Exception:
        return False


def apply_app_palette(theme: str) -> None:
    """Swap the application QPalette so native Qt widgets match the fluent theme."""
    app = QApplication.instance()
    if app is None:
        return
    if app.style().objectName().lower() != "fusion":
        app.setStyle("Fusion")
    if theme == "rhine":
        app.setPalette(_rhine_palette())
    else:
        app.setPalette(_dark_palette() if _is_dark(theme) else _light_palette())


_qss_root = None


def set_qss_root(widget) -> None:
    """Register the widget that carries the rhine stylesheet.

    Deliberately **not** ``QApplication.setStyleSheet``: qfluentwidgets owns the
    application-level sheet and re-emits it on every theme / accent change.
    Overwriting it from here produced an access violation (Windows fatal
    exception in ``apply_app_qss``) as soon as the event loop ran. A
    widget-level sheet cascades to children just the same, so the root window
    carries it and the application sheet is left alone.
    """
    global _qss_root
    _qss_root = widget


def apply_app_qss(theme: str) -> None:
    """Apply (or clear) the rhine stylesheet on the registered root widget.

    Stock light/dark get an empty sheet, so adding the rhine theme cannot change
    how they look. All rhine styling lives in ``design_tokens.build_qss``.
    """
    if _qss_root is None:
        return
    try:
        _qss_root.setStyleSheet(build_qss(theme))
    except RuntimeError:
        # root widget already destroyed (window closed) — nothing to style
        pass


def set_app_theme(theme: str) -> None:
    """Single entry point: fluent theme + native palette + QSS in lockstep.

    Everything theme-specific comes from the :data:`gui.design_tokens.THEMES`
    registry, so adding a theme is a one-table change.
    """
    global _active_theme
    spec = THEMES.get(theme) or THEMES["auto"]
    _active_theme = spec.name
    _set_fluent_theme(_FLUENT_BASE.get(spec.fluent_base, Theme.AUTO))
    try:
        _set_fluent_color(spec.fluent_accent, save=False)
    except Exception:
        # Accent colour is cosmetic; never let it break a theme switch.
        pass
    apply_app_palette(spec.name)
    apply_app_qss(spec.name)
    # Only themes with a token table carry a font family. Stock light/dark keep
    # the platform default so they are not visually changed by this layer.
    apply_font_family(use_tokens=bool(tokens_for(spec.name)))


#: The application's original font family, captured before we ever override it.
_original_font_family: str | None = None


def apply_font_family(use_tokens: bool) -> None:
    """Switch the application font family between the token family and the
    platform default.

    The token family (``FONT_FAMILY``) is what ``design.md`` specifies. It is
    applied only for themes that carry a token table — ``light`` / ``dark`` must
    keep whatever the system gave them, otherwise this layer would visibly change
    themes it is not supposed to touch.
    """
    global _original_font_family
    app = QApplication.instance()
    if app is None:
        return

    current = QFont(app.font())
    if _original_font_family is None:
        _original_font_family = current.family()

    if use_tokens:
        families = list(FONT_FAMILY)
    else:
        families = [_original_font_family]

    if current.family() == families[0]:
        return

    if hasattr(current, "setFamilies"):
        current.setFamilies(families)
    else:  # pragma: no cover - Qt < 5.13
        current.setFamily(families[0])
    app.setFont(current)
    # Cached base fonts (both the scaler's and every widget's) still carry the
    # old family; drop them so the next apply_font_scale() re-captures.
    _font_scaler.invalidate()


# Widgets that style themselves (see gui/widgets/model_status.py) cannot be
# reached by the application stylesheet, so they need to know which theme is
# active in order to pick the matching token instead of a hard-coded grey.
_active_theme: str = "auto"


def current_theme() -> str:
    """Return the theme last handed to :func:`set_app_theme`."""
    return _active_theme


def is_rhine_theme() -> bool:
    return _active_theme == "rhine"


# ---------------------------------------------------------------------------
# Motion helpers
# ---------------------------------------------------------------------------


def easing_curve(name: str = "enter") -> QEasingCurve:
    """Build a QEasingCurve from the ``MOTION`` token table.

    Qt has no CSS ``cubic-bezier()`` syntax, so the four control points from the
    token table are fed to a bezier spline. ``name`` selects the ``<name>_curve``
    token, e.g. ``"enter"`` / ``"exit"``.
    """
    x1, y1, x2, y2 = MOTION[f"{name}_curve"]
    curve = QEasingCurve(QEasingCurve.BezierSpline)
    curve.addCubicBezierSegment(QPointF(x1, y1), QPointF(x2, y2), QPointF(1.0, 1.0))
    return curve


def motion_duration(name: str = "enter") -> int:
    """Duration in ms for the ``<name>_ms`` token."""
    return int(MOTION[f"{name}_ms"])


def prefers_reduced_motion() -> bool:
    """Whether the user asked for reduced motion.

    Qt 5 exposes no ``prefers-reduced-motion`` equivalent, so this reads the
    ``VOCALSTUDIO_REDUCED_MOTION`` environment variable and otherwise returns
    False. Animations built in ``gui/ui_kit`` must consult this before starting;
    when the project moves to Qt 6, replace the body with the real style hint.
    """
    return os.environ.get("VOCALSTUDIO_REDUCED_MOTION", "").strip().lower() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# Font scaling
# ---------------------------------------------------------------------------

_FONT_SCALE_MIN = 0.5
_FONT_SCALE_MAX = 3.0

_BASE_FONT_ATTR = "_vocalstudio_base_font"


def _scale_font(base: QFont, scale: float) -> QFont:
    """Return a copy of *base* scaled by *scale* (point or pixel, whichever is set)."""
    font = QFont(base)
    if base.pixelSize() > 0:
        font.setPixelSize(max(6, round(base.pixelSize() * scale)))
        font.setPointSize(-1)
    elif base.pointSizeF() > 0:
        font.setPointSizeF(base.pointSizeF() * scale)
    elif base.pointSize() > 0:
        font.setPointSize(max(6, round(base.pointSize() * scale)))
    return font


class _FontScaler:
    """Scale every widget's font by a multiplier, without compounding.

    Each widget caches its *base* font (the natural font at scale 1.0) as a
    Python attribute so repeated calls with different scales always start from
    a known baseline (and so a destroyed widget / reused object id cannot
    poison a shared registry).
    """

    def __init__(self) -> None:
        self._scale: float = 1.0
        self._base_app_font: QFont | None = None

    def invalidate(self) -> None:
        """Forget every cached base font.

        Called when the application font family changes: both the scaler's own
        baseline and each widget's cached base carry the *old* family, so leaving
        them in place would make the next apply() restore the old font.
        """
        self._base_app_font = None
        app = QApplication.instance()
        if app is None:
            return
        for widget in QApplication.allWidgets():
            if hasattr(widget, _BASE_FONT_ATTR):
                delattr(widget, _BASE_FONT_ATTR)

    def apply(self, scale: float) -> None:
        app = QApplication.instance()
        if app is None:
            return
        try:
            scale = float(scale)
        except (TypeError, ValueError):
            scale = 1.0
        scale = max(_FONT_SCALE_MIN, min(_FONT_SCALE_MAX, scale))
        self._scale = scale
        if self._base_app_font is None:
            self._base_app_font = QFont(app.font())

        # 1) Reset the application default font to its natural size so native
        #    widgets that previously inherited a scaled font are recaptured at
        #    scale 1.0 (avoiding double-scaling on later calls).
        app.setFont(_scale_font(self._base_app_font, 1.0))

        # 2) Capture base fonts for any widget not seen before, then re-emit a
        #    scaled copy. A single pass avoids chasing widgets created as a side
        #    effect of the reset above.
        for widget in QApplication.allWidgets():
            base = getattr(widget, _BASE_FONT_ATTR, None)
            if base is None:
                base = QFont(widget.font())
                setattr(widget, _BASE_FONT_ATTR, base)
            widget.setFont(_scale_font(base, scale))

        # 3) Finally set the application default font to the scaled size so
        #    widgets/dialogs created after this call inherit it.
        app.setFont(_scale_font(self._base_app_font, scale))

    @property
    def scale(self) -> float:
        return self._scale


_font_scaler = _FontScaler()


def apply_font_scale(scale: float) -> None:
    """Apply a global font-size multiplier to the whole application."""
    _font_scaler.apply(scale)


def current_font_scale() -> float:
    """Return the currently-applied font-scale multiplier."""
    return _font_scaler.scale
