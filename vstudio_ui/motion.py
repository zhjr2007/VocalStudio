"""Non-blocking Qt motion. Live forms and worker ownership never move or reset."""
from PyQt5.QtCore import QObject, QEvent, Qt, QVariantAnimation, QRectF, QPointF, QTimer
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtWidgets import QWidget, QApplication, QAbstractButton, QLineEdit, QComboBox, QPlainTextEdit

from .theme import color, easing_curve, prefers_reduced_motion


class PageMotion(QWidget):
    """Brief snapshot transition above persistent pages; input passes through."""
    def __init__(self, stack):
        super().__init__(stack)
        self.stack = stack
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setFocusPolicy(Qt.NoFocus)
        self.frames = None
        self.progress = 0.0
        self.generation = 0
        self.animation = QVariantAnimation(self)
        self.animation.setDuration(320)
        self.animation.setEasingCurve(easing_curve())
        self.animation.valueChanged.connect(self._frame)
        self.animation.finished.connect(self.clear)
        stack.installEventFilter(self)
        self.hide()

    def clear(self):
        self.generation += 1
        self.animation.stop()
        self.hide()
        self.frames = None

    def switch_to(self, index):
        self.clear()
        if index == self.stack.currentIndex():
            return
        animate = self.stack.isVisible() and not prefers_reduced_motion()
        before = self._snapshot() if animate else None
        self.stack.setCurrentIndex(index)
        if not animate:
            return
        self.stack.currentWidget().layout().activate() if self.stack.currentWidget().layout() else None
        self.frames = (before, before)
        self.setGeometry(self.stack.rect())
        self.progress = 0.0
        self.show()
        self.raise_()
        generation = self.generation
        # Responsive forms schedule their first reflow when shown. Capture on
        # the next event-loop turn so the transition uses their settled layout.
        QTimer.singleShot(32, lambda: self._start_transition(before, generation))

    def _snapshot(self):
        # Rendering a transparent child alone omits the acoustic ancestor on
        # some Windows Qt builds. Capture the real composited window then crop
        # at native DPI, keeping the backdrop fixed throughout the transition.
        root = self.stack.window()
        shot = root.grab()
        origin = self.stack.mapTo(root, self.stack.rect().topLeft())
        dpr = shot.devicePixelRatioF()
        return shot.copy(round(origin.x()*dpr), round(origin.y()*dpr),
                         round(self.stack.width()*dpr), round(self.stack.height()*dpr))

    def _start_transition(self, before, generation):
        if generation != self.generation or not self.stack.isVisible():
            return
        self.hide()
        self.frames = (before, self._snapshot())
        self.show()
        self.raise_()
        self.animation.setStartValue(0.0)
        self.animation.setEndValue(1.0)
        self.animation.start()

    def _frame(self, value):
        self.progress = float(value)
        self.update()

    def eventFilter(self, obj, event):
        if event.type() in (QEvent.Resize, QEvent.Hide):
            self.clear()
        return False

    def paintEvent(self, event):
        if not self.frames:
            return
        p = QPainter(self)
        t = self.progress
        p.setOpacity(1 - t)
        p.drawPixmap(QPointF(-14 * t, 0), self.frames[0])
        p.setOpacity(t)
        p.drawPixmap(QPointF(30 * (1 - t), 8 * (1 - t)), self.frames[1])


class Feedback(QWidget):
    """Paint-only hover/focus glow and click ripple, with no layout mutation."""
    def __init__(self, target):
        super().__init__(target)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setFocusPolicy(Qt.NoFocus)
        self.amount = 0.0
        self.ripple = 1.0
        self.origin = QPointF()
        self.glow = QVariantAnimation(self)
        self.glow.setDuration(150)
        self.glow.valueChanged.connect(self._glow)
        self.pulse = QVariantAnimation(self)
        self.pulse.setDuration(380)
        self.pulse.setEasingCurve(easing_curve())
        self.pulse.valueChanged.connect(self._pulse)

    def _glow(self, value):
        self.amount = float(value)
        self.update()

    def _pulse(self, value):
        self.ripple = float(value)
        self.update()

    def respond(self, event):
        self.setGeometry(self.parentWidget().rect())
        self.show()
        self.raise_()
        if event.type() in (QEvent.MouseButtonPress, QEvent.KeyPress):
            if event.type() == QEvent.KeyPress and event.key() not in (Qt.Key_Space, Qt.Key_Return, Qt.Key_Enter):
                return
            self.origin = QPointF(event.pos()) if event.type() == QEvent.MouseButtonPress else QPointF(self.rect().center())
            self.pulse.stop()
            self.pulse.setStartValue(0.0)
            self.pulse.setEndValue(1.0)
            self.pulse.start()
        target = self.parentWidget()
        self.glow.stop()
        self.glow.setStartValue(self.amount)
        self.glow.setEndValue(1.0 if target.underMouse() or target.hasFocus() else 0.0)
        self.glow.start()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        bounds = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        tint = color('primary')
        tint.setAlpha(int(110 * self.amount))
        p.setPen(QPen(tint, 1.2))
        tint.setAlpha(int(8 * self.amount))
        p.setBrush(tint)
        p.drawRoundedRect(bounds, 9, 9)
        if self.ripple < 1:
            from PyQt5.QtGui import QPainterPath
            clip = QPainterPath()
            clip.addRoundedRect(bounds, 9, 9)
            p.setClipPath(clip)
            tint.setAlpha(int(38 * (1 - self.ripple)))
            p.setPen(Qt.NoPen)
            p.setBrush(tint)
            radius = max(self.width(), self.height()) * self.ripple
            p.drawEllipse(self.origin, radius, radius)


class InteractionMotion(QObject):
    def __init__(self, root):
        super().__init__(root)
        self.root = root
        QApplication.instance().installEventFilter(self)

    def eventFilter(self, obj, event):
        if event.type() not in (QEvent.Enter, QEvent.Leave, QEvent.FocusIn, QEvent.FocusOut, QEvent.MouseButtonPress, QEvent.KeyPress, QEvent.Resize):
            return False
        if not isinstance(obj, (QAbstractButton, QLineEdit, QComboBox, QPlainTextEdit)):
            return False
        if obj.window() is not self.root or not obj.isVisible() or not obj.isEnabled() or prefers_reduced_motion():
            return False
        feedback = getattr(obj, '_motion_feedback', None)
        if event.type() == QEvent.Resize:
            if feedback:
                feedback.setGeometry(obj.rect())
            return False
        if feedback is None:
            feedback = obj._motion_feedback = Feedback(obj)
        feedback.respond(event)
        return False


class RevealMotion(QObject):
    """Reversible disclosure animation, restoring the original height limits."""
    def __init__(self, widget):
        super().__init__(widget)
        self.widget = widget
        self.limits = (widget.minimumHeight(), widget.maximumHeight())
        self.open = not widget.isHidden()
        self.animation = QVariantAnimation(self)
        self.animation.setDuration(240)
        self.animation.setEasingCurve(easing_curve())
        self.animation.valueChanged.connect(lambda value: widget.setFixedHeight(int(value)))
        self.animation.finished.connect(self._finish)

    def _finish(self):
        self.widget.setVisible(self.open)
        self.widget.setMinimumHeight(self.limits[0])
        self.widget.setMaximumHeight(self.limits[1])

    def set_visible(self, visible):
        self.animation.stop()
        self.open = bool(visible)
        if not self.widget.window().isVisible() or prefers_reduced_motion():
            self._finish()
            return
        current = self.widget.height() if not self.widget.isHidden() else 0
        natural = max(self.limits[0], self.widget.sizeHint().height())
        natural = min(natural, self.limits[1])
        self.widget.setFixedHeight(current)
        self.widget.show()
        self.animation.setStartValue(current)
        self.animation.setEndValue(natural if visible else 0)
        self.animation.start()


def reveal(widget, visible):
    controller = getattr(widget, '_reveal_motion', None)
    if controller is None:
        controller = widget._reveal_motion = RevealMotion(widget)
    controller.set_visible(visible)
