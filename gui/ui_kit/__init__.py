"""``gui.ui_kit`` — VocalStudio's own widget kit.

Hard constraint (refactor-plan §3): everything here depends on **PyQt5 +
``gui.design_tokens`` only**. No qfluentwidgets import is allowed, so the look is
fully under our control and a Fluent theme change can never move these pixels.

Colours always come from :func:`gui.ui_kit.tokens.color`; a hard-coded hex in
this package is a bug (``scripts/check_hardcoded_colors.py`` will flag it).

Controls are expected to expose ``refresh_theme()``. ``AppShell`` walks its tree
after a theme switch and calls it; widgets that only paint just call ``update()``.
"""

from __future__ import annotations

from gui.ui_kit.card import Card, CardHeader, Hairline, ThemeAware
from gui.ui_kit.empty_state import EmptyState
from gui.ui_kit.field_row import DEFAULT_LABEL_WIDTH, FieldRow
from gui.ui_kit.hud_ring import HudRing
from gui.ui_kit.page_scaffold import PageScaffold
from gui.ui_kit.status_dot import STATE_TOKENS, StatusDot
from gui.ui_kit.tokens import KNOWN_TOKENS, color, color_hex, is_square_theme

__all__ = [
    "Card",
    "CardHeader",
    "DEFAULT_LABEL_WIDTH",
    "EmptyState",
    "FieldRow",
    "Hairline",
    "HudRing",
    "KNOWN_TOKENS",
    "PageScaffold",
    "STATE_TOKENS",
    "StatusDot",
    "ThemeAware",
    "color",
    "color_hex",
    "is_square_theme",
    "refresh_theme_tree",
]


def refresh_theme_tree(root) -> None:
    """Call ``refresh_theme()`` on *root* and every descendant that has it.

    Used after ``gui.theme.set_app_theme`` so a live window restyles without
    being rebuilt.
    """
    for widget in [root, *root.findChildren(object)]:
        hook = getattr(widget, "refresh_theme", None)
        if callable(hook):
            hook()
