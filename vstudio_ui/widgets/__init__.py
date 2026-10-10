"""``vstudio_ui.widgets`` — the widget kit for the new UI.

Constraint: these modules import **only PyQt5, ``vstudio_ui.tokens`` and
``vstudio_ui.theme``**. No third-party widget library, no import from the archived
``gui`` package. That is what keeps the look under our control; the test suite
enforces it.

Colours always come from :func:`vstudio_ui.theme.color`. A hex literal in this
package is a bug.

Every control exposes ``refresh_theme()``; :func:`refresh_theme_tree` walks a
widget tree after a theme switch.
"""

from __future__ import annotations

from PyQt5 import sip
from PyQt5.QtWidgets import QWidget

from vstudio_ui.widgets.card import Card, CardHeader, Hairline
from vstudio_ui.widgets.indicators import STATE_TOKENS, HudRing, StatusDot
from vstudio_ui.widgets.layout import (
    DEFAULT_LABEL_WIDTH,
    EmptyState,
    FieldRow,
    PageScaffold,
    ValueLabel,
)
from vstudio_ui.widgets.nav_rail import NavRail

__all__ = [
    "Card",
    "CardHeader",
    "DEFAULT_LABEL_WIDTH",
    "EmptyState",
    "FieldRow",
    "Hairline",
    "HudRing",
    "NavRail",
    "PageScaffold",
    "STATE_TOKENS",
    "StatusDot",
    "ValueLabel",
    "refresh_theme_tree",
]


def refresh_theme_tree(root, *, include_root: bool = True) -> None:
    """Call ``refresh_theme()`` on *root* and every descendant that defines it.

    Three details matter here, all learned the hard way:

    * Walk **QWidget** children, not ``object``. ``findChildren(object)`` also
      returns non-widget QObjects and — worse — wrappers whose C++ side has
      already been destroyed; touching one of those takes the process down with a
      native crash and no traceback.
    * Skip anything ``sip.isdeleted`` reports as gone, for the same reason.
    * Pass ``include_root=False`` from a widget whose own ``refresh_theme``
      delegates back here. Otherwise the walk calls it again, which calls the
      walk again — infinite recursion that ends in a silent native crash rather
      than a ``RecursionError``.
    """
    targets: list[object] = [root] if include_root else []
    if isinstance(root, QWidget):
        targets.extend(root.findChildren(QWidget))

    for widget in targets:
        if sip.isdeleted(widget):
            continue
        hook = getattr(widget, "refresh_theme", None)
        if callable(hook):
            hook()
