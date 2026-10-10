"""HudRing — the task-running indicator lifted from the reference deck.

Source: ``media1.mp4`` embedded in ``RHINE LAB_1.pptx`` (1920x1080, 60fps, 169
frames). Frame analysis gives the construction:

* a thin dark outer arc with a gap,
* a white inner arc, offset in phase,
* two short dark arcs near the centre,
* two amber dots riding the outer ring.

Motion: an elastic entrance (scale overshoot, not a fade), then a slow rotation
with the amber dots travelling around the ring.

Everything is drawn with ``QPainter.drawArc``. No images, no blur, no shadow —
``QGraphicsDropShadowEffect`` / ``QGraphicsBlurEffect`` stack badly with the page
fade's ``QGraphicsOpacityEffect`` (see framework-audit P4), so ui_kit avoids them
entirely.

Use it where the app currently shows an indeterminate progress bar: separating,
ASR, MIDI and TTS are all long-running with no honest percentage early on.
"""

from __future__ import annotations

from PyQt5.QtCore import QEasingCurve, QPointF, QPropertyAnimation, QRectF, Qt, pyqtProperty
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtWidgets import QWidget

from gui.design_tokens import BORDER_WIDTH
from gui.theme import easing_curve, motion_duration, prefers_reduced_motion
from gui.ui_kit.tokens import color

#: Arc geometry, in 1/16th of a degree (Qt's unit), copied from the frames.
_OUTER_SPAN = 300 * 16
_INNER_SPAN = 210 * 16
_SHORT_SPAN = 60 * 16


class HudRing(QWidget):
    """A rotating HUD ring. ``start()`` / ``stop()`` drive the animation."""

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

    # -- animated properties ---------------------------------------------

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

    # -- API --------------------------------------------------------------

    def start(self) -> None:
        """Run the entrance, then spin until :meth:`stop`."""
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

    # -- painting ---------------------------------------------------------

    def paintEvent(self, event):  # noqa: N802 (Qt naming)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        side = min(self.width(), self.height()) * self._scale
        rect = QRectF(
            (self.width() - side) / 2,
            (self.height() - side) / 2,
            side,
            side,
        )

        ink = color("foreground")
        paper = color("card")
        amber = color("secondary")

        # outer ring: thin ink arc with a gap, rotating
        outer = QPen(ink)
        outer.setWidth(BORDER_WIDTH)
        painter.setPen(outer)
        start = int((90 - self._angle) * 16)
        painter.drawArc(rect, start, _OUTER_SPAN)

        # inner ring: light arc, phase-shifted so the gap travels independently
        inner_rect = rect.adjusted(side * 0.14, side * 0.14, -side * 0.14, -side * 0.14)
        paper_pen = QPen(paper)
        paper_pen.setWidth(BORDER_WIDTH + 1)
        painter.setPen(paper_pen)
        painter.drawArc(inner_rect, int((90 + self._angle) * 16), _INNER_SPAN)

        # two short centre arcs
        short_rect = rect.adjusted(side * 0.30, side * 0.30, -side * 0.30, -side * 0.30)
        painter.setPen(outer)
        painter.drawArc(short_rect, int((90 - self._angle * 1.4) * 16), _SHORT_SPAN)
        painter.drawArc(short_rect, int((270 - self._angle * 1.4) * 16), _SHORT_SPAN)

        # amber dots riding the outer ring
        painter.setPen(Qt.NoPen)
        painter.setBrush(amber)
        radius = side / 2
        centre = rect.center()
        dot = max(2.0, side * 0.035)
        for phase in (0.0, 180.0):
            theta = (self._angle + phase) * 3.141592653589793 / 180.0
            x = centre.x() + radius * 0.98 * _cos(theta)
            y = centre.y() + radius * 0.98 * _sin(theta)
            painter.drawEllipse(QPointF(x, y), dot, dot)

        painter.end()


def _cos(theta: float) -> float:
    import math

    return math.cos(theta)


def _sin(theta: float) -> float:
    import math

    return math.sin(theta)
