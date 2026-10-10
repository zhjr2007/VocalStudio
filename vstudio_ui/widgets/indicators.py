"""Indicators: StatusDot and HudRing.

``StatusDot`` is the amber "status anchor" — a state marker that reads faster than
a text badge and costs no horizontal space.

``HudRing`` is the task-running indicator, reconstructed from ``media1.mp4`` in
the reference deck: a gapped ink arc, a light inner arc, two short centre arcs,
and two amber dots riding the outer ring. Entrance is an elastic scale overshoot
(not a fade); the steady state is a slow rotation.

Both are pure ``QPainter`` work — no images, no blur, no shadow effects, because
those stack badly with the window's fade animations.
"""

from __future__ import annotations

import math

from PyQt5.QtCore import QEasingCurve, QPointF, QPropertyAnimation, QRectF, Qt, pyqtProperty
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtWidgets import QWidget

from vstudio_ui.theme import color, easing_curve, motion_duration, prefers_reduced_motion
from vstudio_ui.tokens import BORDER_WIDTH

#: state -> token name. Enumerated so a page can list the valid states.
STATE_TOKENS: dict[str, str] = {
    "idle": "neutral",
    "ready": "secondary",
    "running": "primary",
    "done": "ok",
    "failed": "error",
}


class StatusDot(QWidget):
    """A small filled circle whose colour encodes a state."""

    def __init__(self, state: str = "idle", parent=None, *, diameter: int = 8):
        super().__init__(parent)
        self._state = state if state in STATE_TOKENS else "idle"
        self._diameter = diameter
        self.setFixedSize(diameter + 4, diameter + 4)
        self.setToolTip(self._state)

    @property
    def state(self) -> str:
        return self._state

    def set_state(self, state: str) -> None:
        self._state = state if state in STATE_TOKENS else "idle"
        self.setToolTip(self._state)
        self.update()

    def refresh_theme(self) -> None:
        self.update()

    def paintEvent(self, event):  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(Qt.NoPen)
        painter.setBrush(color(STATE_TOKENS[self._state]))
        offset = (self.width() - self._diameter) // 2
        painter.drawEllipse(offset, offset, self._diameter, self._diameter)
        painter.end()


_OUTER_SPAN = 300 * 16
_INNER_SPAN = 210 * 16
_SHORT_SPAN = 60 * 16


class HudRing(QWidget):
    """A rotating HUD ring. ``start()`` / ``stop()`` drive it."""

    def __init__(self, parent=None, *, diameter: int = 96):
        super().__init__(parent)
        self._diameter = diameter
        self.setFixedSize(diameter, diameter)
        self._angle = 0.0
        self._scale = 1.0

        self._spin = QPropertyAnimation(self, b"angle", self)
        self._spin.setStartValue(0.0)
        self._spin.setEndValue(360.0)
        self._spin.setLoopCount(-1)
        self._spin.setDuration(2400)
        self._spin.setEasingCurve(QEasingCurve.Linear)

        self._enter = QPropertyAnimation(self, b"scale", self)
        self._enter.setStartValue(1.28)
        self._enter.setEndValue(1.0)
        self._enter.setDuration(motion_duration("enter"))
        self._enter.setEasingCurve(easing_curve("enter"))

    def _get_angle(self) -> float:
        return self._angle

    def _set_angle(self, value: float) -> None:
        self._angle = float(value)
        self.update()

    angle = pyqtProperty(float, fget=_get_angle, fset=_set_angle)

    def _get_scale(self) -> float:
        return self._scale

    def _set_scale(self, value: float) -> None:
        self._scale = float(value)
        self.update()

    scale = pyqtProperty(float, fget=_get_scale, fset=_set_scale)

    def start(self) -> None:
        if prefers_reduced_motion():
            self._scale = 1.0
            self._angle = 0.0
            self.update()
            return
        self._enter.start()
        if self._spin.state() != QPropertyAnimation.Running:
            self._spin.start()

    def stop(self) -> None:
        self._spin.stop()
        self._enter.stop()
        self._angle = 0.0
        self._scale = 1.0
        self.update()

    def is_running(self) -> bool:
        return self._spin.state() == QPropertyAnimation.Running

    def refresh_theme(self) -> None:
        self.update()

    def paintEvent(self, event):  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        side = min(self.width(), self.height()) * self._scale
        rect = QRectF((self.width() - side) / 2, (self.height() - side) / 2, side, side)

        ink = color("foreground")
        paper = color("card")
        amber = color("secondary")

        outer = QPen(ink)
        outer.setWidth(BORDER_WIDTH)
        painter.setPen(outer)
        painter.drawArc(rect, int((90 - self._angle) * 16), _OUTER_SPAN)

        inner_rect = rect.adjusted(side * 0.14, side * 0.14, -side * 0.14, -side * 0.14)
        paper_pen = QPen(paper)
        paper_pen.setWidth(BORDER_WIDTH + 1)
        painter.setPen(paper_pen)
        painter.drawArc(inner_rect, int((90 + self._angle) * 16), _INNER_SPAN)

        short_rect = rect.adjusted(side * 0.30, side * 0.30, -side * 0.30, -side * 0.30)
        painter.setPen(outer)
        painter.drawArc(short_rect, int((90 - self._angle * 1.4) * 16), _SHORT_SPAN)
        painter.drawArc(short_rect, int((270 - self._angle * 1.4) * 16), _SHORT_SPAN)

        painter.setPen(Qt.NoPen)
        painter.setBrush(amber)
        radius = side / 2
        centre = rect.center()
        dot = max(2.0, side * 0.035)
        for phase in (0.0, 180.0):
            theta = math.radians(self._angle + phase)
            painter.drawEllipse(
                QPointF(centre.x() + radius * 0.98 * math.cos(theta),
                        centre.y() + radius * 0.98 * math.sin(theta)),
                dot,
                dot,
            )
        painter.end()
