"""Colour resolution for ``gui/ui_kit``.

``ui_kit`` widgets must never hard-code a colour and must never import
qfluentwidgets. They ask this module for a *token name* and get a QColor back.

Resolution order
----------------
1. The active theme's token table (only ``rhine`` has one today).
2. A QPalette role, so the controls stay legible under ``light`` / ``dark``
   even though those themes carry no token table (see refactor-plan D3).

An unknown token name raises ``KeyError`` on purpose: the token table is the
contract, and a missing entry means the table needs a new row — not that the
widget should invent a colour. The ui_kit unit tests construct every control, so
a typo fails in CI rather than silently painting black.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import QApplication

from gui.design_tokens import tokens_for
from gui.theme import current_theme

#: Every token name ui_kit uses, mapped to the QPalette role that stands in for
#: it under the stock themes. Deliberately exhaustive: a name that is missing
#: here *and* missing from the active token table is a programming error.
_PALETTE_FALLBACK: dict[str, QPalette.ColorRole] = {
    "background": QPalette.Window,
    "background_alt": QPalette.AlternateBase,
    "card": QPalette.Base,
    "card_alt": QPalette.AlternateBase,
    "muted": QPalette.AlternateBase,
    "foreground": QPalette.WindowText,
    "foreground_2": QPalette.Text,
    "muted_foreground": QPalette.PlaceholderText,
    "border": QPalette.Mid,
    "border_soft": QPalette.Midlight,
    "border_strong": QPalette.Dark,
    "primary": QPalette.Highlight,
    "primary_foreground": QPalette.HighlightedText,
    "secondary": QPalette.Link,
    "secondary_soft": QPalette.AlternateBase,
    "destructive": QPalette.BrightText,
    "ok": QPalette.Highlight,
    "error": QPalette.BrightText,
    "neutral": QPalette.Mid,
    "neutral_dark": QPalette.Dark,
}

#: Names ui_kit may ask for. Kept as a tuple so tests can assert coverage.
KNOWN_TOKENS: tuple[str, ...] = tuple(_PALETTE_FALLBACK)


def color(name: str) -> QColor:
    """Return the QColor for token *name* under the active theme.

    Raises ``KeyError`` when the name is not a token — see the module docstring.
    """
    if name not in _PALETTE_FALLBACK:
        raise KeyError(f"{name!r} is not a design token (add it to design_tokens.RHINE)")

    table = tokens_for(current_theme())
    value = table.get(name)
    if value:
        return QColor(value)

    app = QApplication.instance()
    if app is not None:
        return app.palette().color(_PALETTE_FALLBACK[name])
    # No QApplication (unit tests / headless import): black is a valid QColor and
    # keeps this function total. Callers always run inside a QApplication.
    return QColor(Qt.black)


def color_hex(name: str) -> str:
    """Same as :func:`color`, as a ``#rrggbb`` string (handy for QSS)."""
    return color(name).name()


def is_square_theme() -> bool:
    """Whether the active theme wants square corners and no shadows.

    True for rhine (0 radius / 0 shadow — the reference deck has no roundRect and
    no outerShdw). Stock light/dark keep their Fluent look, so this is False and
    ui_kit falls back to a small radius.
    """
    from gui.design_tokens import RADIUS

    return bool(tokens_for(current_theme())) and RADIUS == 0
