"""Panel primitives: Card, CardHeader, Hairline.

Painted in ``paintEvent`` rather than styled with a widget-level QSS sheet: the
sheet would shadow the root theme sheet, and painting gives exact control over the
1px hairline this design language is built on.

No shadows anywhere — depth is a colour step plus a rule.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from vstudio_ui.theme import color, color_hex
from vstudio_ui.tokens import BORDER_WIDTH, SPACE, TYPE_SCALE, RADIUS


class Hairline(QFrame):
    """A 1px rule. ``strong`` doubles the weight for section breaks."""

    def __init__(self, parent=None, *, vertical: bool = False, strong: bool = False):
        super().__init__(parent)
        self._vertical = vertical
        self._strong = strong
        weight = BORDER_WIDTH + (1 if strong else 0)
        if vertical:
            self.setFixedWidth(weight)
        else:
            self.setFixedHeight(weight)
        self.refresh_theme()

    def refresh_theme(self) -> None:
        name = "border_strong" if self._strong else "border_soft"
        axis = "border-left" if self._vertical else "border-top"
        weight = self.width() if self._vertical else self.height()
        self.setStyleSheet(f"Hairline {{ {axis}: {max(weight, 1)}px solid {color_hex(name)}; }}")


class Card(QFrame):
    """A square-cornered panel with a hairline border."""

    def __init__(self, parent=None, *, padding: int = 0, alt: bool = False):
        super().__init__(parent)
        self._alt = alt
        if padding:
            self.setContentsMargins(padding, padding, padding, padding)

    def refresh_theme(self) -> None:
        self.update()

    def paintEvent(self, event):  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect().adjusted(0, 0, -1, -1)
        painter.setPen(Qt.NoPen)
        fill = color("card_alt" if self._alt else "card")
        fill.setAlpha(225)
        painter.setBrush(fill)
        painter.drawRoundedRect(QRectF(rect), RADIUS, RADIUS)
        pen = QPen(color("border"))
        pen.setWidth(BORDER_WIDTH)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(QRectF(rect), RADIUS, RADIUS)
        painter.end()


class CardHeader(QWidget):
    """Title (+ optional caption) on the left, an optional widget on the right."""

    def __init__(self, title: str = "", parent=None, *, caption: str = "", trailing: QWidget | None = None):
        super().__init__(parent)
        self._title = QLabel(title, self)
        self._caption = QLabel(caption, self)
        self._trailing = trailing

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(SPACE["sm"])

        column = QVBoxLayout()
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(2)
        column.addWidget(self._title)
        if caption:
            column.addWidget(self._caption)
        outer.addLayout(column, 1)
        if trailing is not None:
            outer.addWidget(trailing, 0, Qt.AlignTop)

        self.refresh_theme()

    def set_title(self, text: str) -> None:
        self._title.setText(text)

    def set_caption(self, text: str) -> None:
        self._caption.setText(text)
        self._caption.setVisible(bool(text))

    def refresh_theme(self) -> None:
        self._title.setStyleSheet(
            f"color: {color_hex('foreground')}; font-size: {TYPE_SCALE['h3']}pt; font-weight: 600;"
        )
        self._caption.setStyleSheet(
            f"color: {color_hex('muted_foreground')}; font-size: {TYPE_SCALE['caption']}pt;"
        )
        hook = getattr(self._trailing, "refresh_theme", None)
        if callable(hook):
            hook()
