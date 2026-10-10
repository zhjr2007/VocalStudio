"""StatusDot — the amber "status anchor" from the reference deck.

The deck uses amber as a *dot*, never as a field: a small round marker that says
"this item is in this state". It reads faster than a text badge and costs no
horizontal space, which matters on the dense pages (separation, pipeline).

Shape rule: this is the one place in the design language where a circle is
correct — ``RADIUS_PILL`` exists for it.
"""

from __future__ import annotations

from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QColor, QPainter
from PyQt5.QtWidgets import QWidget

from gui.ui_kit.tokens import color

#: state -> token name. Kept as data so a page can enumerate valid states.
STATE_TOKENS: dict[str, str] = {
    "idle": "neutral",
    "ready": "secondary",   # amber
    "running": "primary",   # lime
    "done": "ok",
    "failed": "error",
}


class StatusDot(QWidget):
    """A small filled circle whose colour encodes a state."""

    def __init__(self, state: str = "idle", parent=None, *, diameter: int = 8):
        super().__init__(parent)
        self._state = state if state in STATE_TOKENS else "idle"
        self._diameter = diameter
        self.setFixedSize(QSize(diameter + 4, diameter + 4))
        self.setToolTip(self._state)

    # -- API --------------------------------------------------------------

    @property
    def state(self) -> str:
        return self._state

    def set_state(self, state: str) -> None:
        self._state = state if state in STATE_TOKENS else "idle"
        self.setToolTip(self._state)
        self.update()

    def refresh_theme(self) -> None:
        self.update()

    # -- painting ---------------------------------------------------------

    def paintEvent(self, event):  # noqa: N802 (Qt naming)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(Qt.NoPen)
        painter.setBrush(color(STATE_TOKENS[self._state]))
        offset = (self.width() - self._diameter) // 2
        painter.drawEllipse(offset, offset, self._diameter, self._diameter)
        painter.end()
