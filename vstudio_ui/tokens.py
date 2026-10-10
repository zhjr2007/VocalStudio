"""VocalStudio launcher tokens. The active v2 design is documented in ui-design/launcher-2026.md; legacy gui tokens remain separate."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

LIGHT: dict[str, str] = {
    # surfaces
    "background": "#E6EDF3",
    "background_alt": "#F0F5F8",
    "card": "#FFFEFF",              # panels
    "card_alt": "#F6F4F2",          # panel interior, zebra rows
    "muted": "#F6F4F2",
    # text ramp
    "foreground": "#050C0A",        # near-black with a faint green cast
    "foreground_2": "#4A4F4B",
    "muted_foreground": "#8E8E8E",
    # rules
    "border": "#D7D2CB",
    "border_soft": "#E0DEDC",
    "border_strong": "#050C0A",
    # accents
    "primary": "#BFE252",           # lime — highlight *field*
    "primary_foreground": "#050C0A",
    "secondary": "#FCA11C",         # amber — status *dot*, accent rule
    "secondary_soft": "#FFE2C1",
    "destructive": "#FF7C00",
    # status
    "ok": "#6E9B1F",
    "error": "#B65800",
    # neutrals
    "neutral": "#D1CEC9",
    "neutral_dark": "#DADAD7",
}

DARK: dict[str, str] = {
    # surfaces — the warm cast is kept, just inverted in value
    "background": "#090F1B",
    "background_alt": "#101A29",
    "card": "#142131",
    "card_alt": "#1A2B3E",
    "muted": "#1A2B3E",
    # text ramp
    "foreground": "#F0F5FA",
    "foreground_2": "#BACBDC",
    "muted_foreground": "#899EB4",
    # rules — hairlines are lighter than the surface, never a shadow
    "border": "#314458",
    "border_soft": "#223449",
    "border_strong": "#F2F4F0",
    # accents carry over unchanged: they are the brand
    "primary": "#D5ED51",
    "primary_foreground": "#050C0A",
    "secondary": "#FCA11C",
    "secondary_soft": "#4A3416",
    "destructive": "#FF7C00",
    # status
    "ok": "#A8D24A",
    "error": "#E08A3C",
    # neutrals
    "neutral": "#2E3531",
    "neutral_dark": "#3A423D",
}

THEMES: dict[str, dict[str, str]] = {"light": LIGHT, "dark": DARK}
THEME_ORDER: tuple[str, ...] = ("light", "dark")

# ---------------------------------------------------------------------------
# Typography
# ---------------------------------------------------------------------------

FONT_FAMILY = (
    "Microsoft YaHei UI",
    "Microsoft YaHei",
    "Segoe UI",
    "Arial",
)

#: The design sizes, in points, at a font scale of 1.0. **Do not read these**:
#: read :data:`TYPE_SCALE`, which has the user's scale applied.
BASE_TYPE_SCALE: dict[str, int] = {
    "display": 36,
    "h1": 23,
    "h2": 17,
    "h3": 13,
    "body": 11,
    "body_small": 10,
    "label": 9,
    "caption": 8,
}

#: Scales the settings page offers. Discrete steps rather than a free number:
#: the sizes are rounded to 0.1pt, so neighbouring values would produce the same
#: layout while pretending to be different settings.
FONT_SCALE_STEPS: tuple[float, ...] = (0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0)
DEFAULT_FONT_SCALE = 1.0

#: Mirrors ``core.app_settings.FONT_SCALE_MIN/MAX``. Duplicated instead of
#: imported because this module is pure design data and must not reach into the
#: business layer; the test suite asserts the two pairs stay equal, so a change
#: on either side fails loudly rather than silently clamping to a stale range.
FONT_SCALE_MIN = 0.5
FONT_SCALE_MAX = 3.0

#: The live type scale. Mutated in place by :func:`set_font_scale` so that the
#: ``from vstudio_ui.tokens import TYPE_SCALE`` every widget already holds keeps
#: working — rebinding the name would leave every import site pointing at the old
#: table, and the UI would scale exactly once (on the first change) and then stop.
TYPE_SCALE: dict[str, float] = dict(BASE_TYPE_SCALE)

_font_scale: float = DEFAULT_FONT_SCALE


def current_font_scale() -> float:
    """The scale currently applied to :data:`TYPE_SCALE`."""
    return _font_scale


def set_font_scale(scale: float) -> float:
    """Rescale every type size. Returns the scale actually applied.

    Never raises: a corrupt value falls back to 1.0 and an out-of-range one is
    clamped, because this runs while the user drags a control and a raised
    exception there would leave the UI half-scaled.
    """
    global _font_scale
    try:
        value = float(scale)
    except (TypeError, ValueError):
        value = DEFAULT_FONT_SCALE
    if value != value:  # NaN compares unequal to itself
        value = DEFAULT_FONT_SCALE
    _font_scale = min(FONT_SCALE_MAX, max(FONT_SCALE_MIN, value))
    for name, points in BASE_TYPE_SCALE.items():
        TYPE_SCALE[name] = _scaled_points(points)
    return _font_scale


def _scaled_points(points: float) -> float:
    """One scaled type size.

    Returns an ``int`` when the result is whole, which keeps the stylesheet text
    identical at scale 1.0 (``font-size: 24pt``, not ``24.0pt``) and keeps
    ``QFont.setPointSize`` usable without a cast.
    """
    value = round(max(4.0, points * _font_scale), 1)
    return int(value) if float(value).is_integer() else value

# Qt cannot express letter-spacing in QSS; apply with
# QFont.setLetterSpacing(QFont.AbsoluteSpacing, pt).
LETTER_SPACING_LABEL = 0.2
LETTER_SPACING_DISPLAY = -0.4

# ---------------------------------------------------------------------------
# Spacing
# ---------------------------------------------------------------------------

SPACE: dict[str, int] = {
    "xs": 4,
    "sm": 8,
    "md": 14,
    "lg": 25,
    "xl": 40,
}

# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------

RADIUS = 10
RADIUS_PILL = 999
BORDER_WIDTH = 1
BORDER_WIDTH_STRONG = 2

# ---------------------------------------------------------------------------
# Motion
# ---------------------------------------------------------------------------

MOTION: dict[str, object] = {
    "enter_ms": 260,
    "enter_curve": (0.16, 1.0, 0.3, 1.0),
    "exit_ms": 150,
    "exit_curve": (0.4, 0.0, 1.0, 1.0),
    "hover_ms": 120,
}

FOCUS = {"ring_width": 2, "ring_offset": 2}

# ---------------------------------------------------------------------------
# Layout constants shared by the shell
# ---------------------------------------------------------------------------

NAV_WIDTH = 88
NAV_ITEM_HEIGHT = 60
HEADER_HEIGHT = 56
