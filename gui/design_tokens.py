"""Design tokens for VocalStudio.

Single source of truth for colours, typography, spacing, motion and component
shape. ``gui/theme.py`` turns these tokens into a QPalette + a QSS sheet, so a
new page that imports nothing from here still inherits the look, and a page that
needs an explicit colour has somewhere authoritative to read it from.

Themes
------
``light`` / ``dark``
    The pre-existing Fluent themes. They keep using qfluentwidgets' own palettes
    and carry no token table — only their accent is pinned (see :data:`THEMES`),
    recorded so a rhine accent can never leak back into them.
``rhine``
    The warm-white theme derived from the RHINE LAB reference deck (see
    ``ui-design/design.md``). Warm off-white canvas, near-black "ink", hairline
    rules, zero corner radius, zero shadow, lime + amber accents.

Naming
------
The palette follows the shadcn/ui vocabulary that the reference project
(cc-switch, MIT) uses, because that vocabulary is already familiar to anyone who
has touched a modern web UI:

    background / background_alt     page canvas, secondary canvas
    card / card_alt                 panels and their interiors
    muted                           subtle fill (same value as card_alt)
    foreground / foreground_2 / muted_foreground   text ramp
    border / border_soft / border_strong           hairlines and emphasis rules
    primary / primary_foreground    lime — the highlight *field*
    secondary / secondary_soft      amber — the status *dot* / accent rule
    destructive / ok / error        status colours
    neutral / neutral_dark          decorative greys

Deliberately **not** adopted from shadcn: ``popover`` (Qt menus/tooltips reuse
``card``) and ``input`` (inputs reuse ``border``). Fewer names, same coverage.

Every colour is either taken directly from the reference deck (marked "ref") or
derived from it (marked "derived"); see ``ui-design/design.md`` for the full
extraction record.

This module is intentionally **Qt-free** so it can be imported and unit-tested
without a QApplication.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# RHINE — warm-white theme
# ---------------------------------------------------------------------------

RHINE: dict[str, str] = {
    # -- surfaces ---------------------------------------------------------
    "background": "#F4F2EE",       # ref  (window / page canvas)
    "background_alt": "#EBE9E4",   # ref  (content canvas, inset strips)
    "card": "#FFFEFF",             # ref  (panels)
    "card_alt": "#F6F4F2",         # ref  (panel interior, table zebra)
    # -- text ramp --------------------------------------------------------
    "foreground": "#050C0A",       # ref  (near-black, faint green cast)
    "foreground_2": "#4A4F4B",     # derived
    "muted_foreground": "#8E8E8E",  # ref
    # -- rules ------------------------------------------------------------
    "border": "#D7D2CB",           # ref
    "border_soft": "#E0DEDC",      # ref
    "border_strong": "#050C0A",    # ref  (emphasis rule, HUD line)
    # -- accents ----------------------------------------------------------
    "primary": "#BFE252",          # ref  (lime — highlight *field*)
    "primary_foreground": "#050C0A",  # derived (text on lime)
    "secondary": "#FCA11C",        # ref  (amber — status *dot*, accent rule)
    "secondary_soft": "#FFE2C1",   # ref
    "destructive": "#FF7C00",      # ref
    # -- status -----------------------------------------------------------
    # The reference deck has no red: its alert colour *is* the amber/orange.
    # So `error` is a deepened `destructive`, not a traffic-light red.
    "ok": "#6E9B1F",               # derived (lime darkened ~35%, legible on warm white)
    "error": "#B65800",            # derived (destructive darkened ~30%)
    # -- neutral ramp -----------------------------------------------------
    "neutral": "#D1CEC9",          # ref
    "neutral_dark": "#DADAD7",     # ref
}

#: Subtle fill. Same value as :data:`RHINE` ``card_alt`` — kept as its own name
#: because it reads differently at the call site (hover / muted row vs. panel
#: interior), and because shadcn users expect the name to exist.
RHINE["muted"] = RHINE["card_alt"]

# Accent colour handed to qfluentwidgets' own widgets (buttons, sliders,
# switches, focus rings). Must be a colour the Fluent widgets can render text
# on top of, which is why the lime is paired with near-black ink.
RHINE_FLUENT_ACCENT = RHINE["primary"]
RHINE_FLUENT_ACCENT_INK = RHINE["primary_foreground"]

#: Renames applied in the S2 token alignment. Kept as data (not prose) so the
#: mapping can be asserted in tests and grepped by anyone reading old code.
LEGACY_TOKEN_NAMES: dict[str, str] = {
    "bg": "background",
    "bg_alt": "background_alt",
    "surface": "card",
    "surface_alt": "card_alt",
    "ink": "foreground",
    "ink_2": "foreground_2",
    "ink_muted": "muted_foreground",
    "border_ink": "border_strong",
    "accent": "primary",
    "accent_ink": "primary_foreground",
    "accent_2": "secondary",
    "accent_2_soft": "secondary_soft",
    "warn": "destructive",
    "grey": "neutral",
    "grey_dark": "neutral_dark",
}

# ---------------------------------------------------------------------------
# Theme registry
# ---------------------------------------------------------------------------


class ThemeSpec:
    """Static description of one theme. See :data:`THEMES`.

    Before this table existed, the same facts were scattered across three files
    (`core/app_settings.THEME_CHOICES`, a `mapping` dict and an accent dict in
    `gui/theme.py`, and `THEME_ITEMS` in the settings page). Adding a theme meant
    touching all of them, and forgetting one failed *silently* — a value missing
    from `mapping` fell through to `Theme.AUTO` and looked like "the option does
    nothing". One table, one place to add.

    ``fluent_base`` / ``fluent_accent`` are plain strings on purpose: this module
    must stay Qt-free so it can be imported (and unit-tested) without a
    QApplication. `gui/theme.py` maps the strings to qfluentwidgets enums.
    """

    __slots__ = ("name", "is_dark", "fluent_base", "fluent_accent", "has_custom_qss", "label_key")

    def __init__(
        self,
        name: str,
        *,
        is_dark: bool,
        fluent_base: str,
        fluent_accent: str,
        has_custom_qss: bool,
        label_key: str,
    ) -> None:
        self.name = name
        self.is_dark = is_dark
        #: "light" | "dark" | "auto" — which qfluentwidgets base theme to set.
        self.fluent_base = fluent_base
        #: Accent handed to qfluentwidgets' own widgets via setThemeColor().
        self.fluent_accent = fluent_accent
        #: True when this theme ships its own stylesheet (see build_qss).
        self.has_custom_qss = has_custom_qss
        #: i18n key for the settings dropdown label.
        self.label_key = label_key

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ThemeSpec({self.name!r}, is_dark={self.is_dark}, qss={self.has_custom_qss})"


THEMES: dict[str, ThemeSpec] = {}
THEMES.update(
    {
        # The stock Fluent themes. Their palettes come from qfluentwidgets /
        # Fusion, so they carry no token table — only the accent is pinned here
        # (it is qfluentwidgets' own default, recorded so a rhine accent can
        # never leak back into them).
        "auto": ThemeSpec(
            "auto", is_dark=False, fluent_base="auto",
            fluent_accent="#009faa", has_custom_qss=False, label_key="theme.auto",
        ),
        "light": ThemeSpec(
            "light", is_dark=False, fluent_base="light",
            fluent_accent="#009faa", has_custom_qss=False, label_key="theme.light",
        ),
        "dark": ThemeSpec(
            "dark", is_dark=True, fluent_base="dark",
            fluent_accent="#29f1ff", has_custom_qss=False, label_key="theme.dark",
        ),
        # Warm-white theme built from the RHINE LAB reference deck.
        # Based on Fluent LIGHT because its canvas is warm off-white; the
        # near-black "ink" is carried by the palette + QSS, not by dark mode.
        "rhine": ThemeSpec(
            "rhine", is_dark=False, fluent_base="light",
            fluent_accent=RHINE_FLUENT_ACCENT, has_custom_qss=True, label_key="theme.rhine",
        ),
    }
)

#: Theme values in dropdown order.
THEME_ORDER: tuple[str, ...] = ("auto", "light", "dark", "rhine")

# ---------------------------------------------------------------------------
# Typography
# ---------------------------------------------------------------------------

# Ordered fallbacks; Qt picks the first family that exists.
FONT_FAMILY = (
    "Microsoft YaHei UI",
    "Microsoft YaHei",
    "Segoe UI",
    "Arial",
)

# Point sizes. The reference deck leans on a small set of steps; these are the
# ones worth naming. (Extracted scale: 7/8/9/9.5/10/10.5/12/13/14/16/18/20/22/
# 24/28/32/50/54/60 — collapsed to the steps we actually need.)
TYPE_SCALE = {
    "display": 32,
    "h1": 24,
    "h2": 20,
    "h3": 16,
    "body": 13,
    "body_small": 12,
    "label": 10,   # all-caps micro label
    "caption": 9,
}

# The reference sets all-caps micro labels with positive tracking. Qt cannot do
# this in QSS — apply it with QFont.setLetterSpacing(QFont.AbsoluteSpacing, pt).
LETTER_SPACING_LABEL = 0.2     # pt
LETTER_SPACING_DISPLAY = -0.4  # pt (large type is tightened)

# ---------------------------------------------------------------------------
# Spacing
# ---------------------------------------------------------------------------

# The reference's outer margin is 25pt on a 960pt canvas; the rest of the rhythm
# is multiples of that. Mapped to pixels at 1x for widget layout.
SPACE = {
    "xs": 4,
    "sm": 8,
    "md": 14,
    "lg": 25,
    "xl": 40,
}

# ---------------------------------------------------------------------------
# Component shape
# ---------------------------------------------------------------------------

# The reference deck contains 173 rect / 35 line / 0 roundRect and zero
# drop-shadows: corners are square and depth comes from hairlines, never from
# blur. Keep it that way — it is also what keeps Qt cheap here.
RADIUS = 0
RADIUS_PILL = 999          # status dots only
BORDER_WIDTH = 1
BORDER_WIDTH_STRONG = 2

# ---------------------------------------------------------------------------
# Motion
# ---------------------------------------------------------------------------

# Durations in ms and cubic-bezier control points (x1, y1, x2, y2). The curves
# are taken from the cc-switch reference (MIT): an ease-out with a slight
# overshoot for entrances, a plain ease-in for exits. Qt has no bezier string
# syntax — `gui.theme.easing_curve()` converts these to a QEasingCurve.
MOTION = {
    "enter_ms": 260,
    "enter_curve": (0.16, 1.0, 0.3, 1.0),
    "exit_ms": 150,
    "exit_curve": (0.4, 0.0, 1.0, 1.0),
    "hover_ms": 120,
}

#: Whether the user's system asks for reduced motion. Qt exposes this via
#: QStyleHints on some platforms only, so the check lives behind a helper
#: (`gui.theme.prefers_reduced_motion`) that degrades to False.
FOCUS = {
    "ring_width": 2,
    "ring_offset": 2,
}

# ---------------------------------------------------------------------------


def tokens_for(theme: str) -> dict:
    """Return the token dict for *theme* (``rhine`` is the only custom one)."""
    return RHINE if theme == "rhine" else {}


def build_qss(theme: str) -> str:
    """QSS sheet for *theme*.

    Returns an empty string for the stock ``light`` / ``dark`` themes so their
    appearance is exactly what it was before the rhine theme was added.

    Scope is deliberately narrow: only plain-Qt widgets and container backgrounds
    are touched. qfluentwidgets widgets style themselves and are left alone apart
    from the accent colour applied via ``setThemeColor``.
    """
    if theme != "rhine":
        return ""

    p = RHINE
    return f"""
/* ---- containers ------------------------------------------------------- */
QMainWindow, QDialog, QWidget#VocalStudioRoot {{
    background: {p['background']};
}}
QScrollArea {{
    background: {p['background_alt']};
    border: none;
}}
QScrollArea > QWidget > QWidget {{
    background: transparent;
}}
QSplitter::handle {{
    background: {p['border_soft']};
}}

/* ---- plain Qt inputs (Fusion-rendered) -------------------------------- */
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
    border: {BORDER_WIDTH_STRONG}px solid {p['border_strong']};
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
    border-radius: {RADIUS}px;
    selection-background-color: {p['primary']};
    selection-color: {p['primary_foreground']};
    outline: none;
}}

/* ---- buttons ---------------------------------------------------------- */
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

/* ---- sliders, progress, checks ---------------------------------------- */
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

/* ---- lists, tables, tabs ---------------------------------------------- */
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

/* ---- misc ------------------------------------------------------------- */
QToolTip {{
    background: {p['card']};
    color: {p['foreground']};
    border: {BORDER_WIDTH}px solid {p['border_strong']};
    border-radius: {RADIUS}px;
    padding: 4px 6px;
}}
QGroupBox {{
    border: {BORDER_WIDTH}px solid {p['border']};
    border-radius: {RADIUS}px;
    margin-top: 10px;
    padding-top: 6px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
    color: {p['foreground']};
}}
QMenu {{
    background: {p['card']};
    color: {p['foreground']};
    border: {BORDER_WIDTH}px solid {p['border']};
    border-radius: {RADIUS}px;
}}
QMenu::item:selected {{
    background: {p['primary']};
    color: {p['primary_foreground']};
}}

/* ---- focus ring ------------------------------------------------------- */
/* QSS has no `outline`; a border swap is the closest equivalent that works
   across Fusion-drawn widgets. See design.md §6. */
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus,
QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus,
QPushButton:focus, QListWidget:focus, QListView:focus,
QTreeView:focus, QTableView:focus {{
    border: {FOCUS['ring_width']}px solid {p['border_strong']};
}}
"""
