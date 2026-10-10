"""Accessible icon rail, with original resolution independent vector marks."""
from PyQt5.QtCore import Qt, QRectF, pyqtSignal, QVariantAnimation, QEasingCurve
from PyQt5.QtGui import QPainter, QPen, QPainterPath
from PyQt5.QtWidgets import QVBoxLayout, QWidget, QAbstractButton
from vstudio_ui.theme import color, prefers_reduced_motion
from vstudio_ui.tokens import NAV_WIDTH, NAV_ITEM_HEIGHT


def draw_mark(p, index):
    if index == 0:
        path = QPainterPath(); path.moveTo(2, 11); path.lineTo(12, 3); path.lineTo(22, 11)
        p.drawPath(path); p.drawRoundedRect(QRectF(5, 11, 14, 11), 2, 2)
    elif index == 1:
        for x, y in ((4, 7), (9, 2), (14, 5), (19, 9)): p.drawLine(x, y, x, 24-y)
    elif index == 2:
        p.drawLine(9, 18, 9, 4); p.drawLine(9, 4, 21, 2); p.drawLine(21, 2, 21, 16)
        p.drawEllipse(QRectF(2, 16, 7, 5)); p.drawEllipse(QRectF(14, 14, 7, 5))
    elif index == 3:
        p.drawRoundedRect(QRectF(7, 2, 10, 14), 5, 5)
        p.drawArc(QRectF(3, 8, 18, 12), 180*16, 180*16); p.drawLine(12, 20, 12, 23)
    elif index == 4:
        p.drawRoundedRect(QRectF(2, 3, 20, 15), 3, 3)
        p.drawLine(5, 18, 5, 23); p.drawLine(5, 23, 11, 18)
        for x in (7, 12, 17): p.drawLine(x, 8, x, 13)
    elif index == 5:
        for x, y in ((2, 2), (15, 2), (15, 15)): p.drawRoundedRect(QRectF(x, y, 7, 7), 1, 1)
        p.drawLine(9, 5, 15, 5); p.drawLine(18, 9, 18, 15)
        p.drawLine(5, 9, 5, 18); p.drawLine(5, 18, 15, 18)
    elif index == 6:
        for x, y, h in ((3, 14, 8), (10, 8, 14), (17, 2, 20)): p.drawRoundedRect(QRectF(x, y, 4, h), 1, 1)
    else:
        p.drawEllipse(QRectF(4, 4, 16, 16)); p.drawEllipse(QRectF(9, 9, 6, 6))
        for x, y, u, v in ((12, 0, 12, 4), (12, 20, 12, 24), (0, 12, 4, 12), (20, 12, 24, 12)): p.drawLine(x, y, u, v)


class _NavItem(QAbstractButton):
    clicked_index = pyqtSignal(int)

    def __init__(self, index, label, parent=None):
        super().__init__(parent)
        self._index = index; self._selected = False
        self._selection = 0.0
        self._selection_motion = QVariantAnimation(self)
        self._selection_motion.setDuration(300)
        self._selection_motion.setEasingCurve(QEasingCurve.OutBack)
        self._selection_motion.valueChanged.connect(self._selection_frame)
        self.set_label(label); self.setFixedSize(60, NAV_ITEM_HEIGHT)
        self.setCursor(Qt.PointingHandCursor); self.setFocusPolicy(Qt.StrongFocus)
        self.clicked.connect(lambda: self.clicked_index.emit(index))

    @property
    def label(self): return self._label
    def set_label(self, text):
        self._label = text; self.setToolTip(text); self.setAccessibleName(text); self.update()
    def _selection_frame(self, value):
        self._selection = float(value); self.update()
    def set_selected(self, value):
        if self._selected == value: return
        self._selected = value
        self._selection_motion.stop()
        if not self.isVisible() or prefers_reduced_motion():
            self._selection_frame(float(value)); return
        self._selection_motion.setStartValue(self._selection)
        self._selection_motion.setEndValue(float(value))
        self._selection_motion.start()
    def refresh_theme(self): self.update()
    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Up, Qt.Key_Down):
            self.parent().focus_item(self._index + (1 if event.key() == Qt.Key_Down else -1))
        else: super().keyPressEvent(event)
    def paintEvent(self, event):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        active = self._selected or self.hasFocus()
        p.setPen(QPen(color('primary' if active else 'border_soft'), 1))
        p.setBrush(color('card_alt') if self.underMouse() or active else Qt.NoBrush)
        inset = 2 * (1 - self._selection)
        p.drawRoundedRect(QRectF(6 + inset, 6 + inset, 48 - inset * 2, 48 - inset * 2), 13, 13)
        p.translate(30, 30)
        scale = 1 + 0.05 * self._selection
        p.scale(scale, scale); p.translate(-12, -12)
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(color('primary' if active else 'muted_foreground'), 1.7, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        draw_mark(p, self._index)


class NavRail(QWidget):
    navigate = pyqtSignal(int)
    def __init__(self, parent=None):
        super().__init__(parent); self.setFixedWidth(NAV_WIDTH)
        self._items = []; self._current = 0
        self._column = QVBoxLayout(self)
        self._column.setContentsMargins(14, 66, 14, 20); self._column.setSpacing(6); self._column.addStretch(1)
    def add_item(self, label):
        index = len(self._items); item = _NavItem(index, label, self)
        item.clicked_index.connect(self.set_current); self._items.append(item)
        if index == 7: self._column.addWidget(item)
        else: self._column.insertWidget(index, item)
        item.set_selected(index == self._current); return index
    def count(self): return len(self._items)
    def current_index(self): return self._current
    def rail_width(self): return NAV_WIDTH
    def add_stretch(self): pass
    def focus_item(self, index): self._items[index % len(self._items)].setFocus()
    def set_label(self, index, text): self._items[index].set_label(text)
    def refresh_theme(self):
        for item in self._items: item.update()
        self.update()
    def set_current(self, index):
        if not 0 <= index < len(self._items): return
        for i, item in enumerate(self._items): item.set_selected(i == index)
        changed = self._current != index; self._current = index
        if changed: self.navigate.emit(index)
    def paintEvent(self, event):
        p = QPainter(self); fill = color('background_alt'); fill.setAlpha(240); p.fillRect(self.rect(), fill)
        p.setPen(color('border_soft')); p.drawLine(self.width()-1, 0, self.width()-1, self.height())
        p.translate(32, 23); p.setPen(QPen(color('primary'), 3, Qt.SolidLine, Qt.RoundCap))
        for x, y in ((0, 6), (7, 0), (14, 3), (21, 8)): p.drawLine(x, y, x, 24-y)

