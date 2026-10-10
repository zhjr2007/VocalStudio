"""Card / CardHeader / Hairline — the panel primitives.

Everything is painted in ``paintEvent`` rather than styled with QSS on purpose:
a widget-level stylesheet would shadow the root sheet and make the theme harder
to reason about, and painting gives exact control over the hairline weight that
this design language depends on.

No shadows. The reference deck contains zero ``outerShdw``; depth is expressed
with a 1px rule and a colour step.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPainter, QPen
from PyQt5.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from gui.design_tokens import BORDER_WIDTH, SPACE
from gui.ui_kit.tokens import color, color_hex


class Hairline(QFrame):
    """A 1px rule. Horizontal by default; pass ``vertical=True`` to rotate."""

    def __init__(self, parent=None, *, vertical: bool = False, strong: bool = False):
        super().__init__(parent)
        self._vertical = vertical
        self._strong = strong
        thickness = BORDER_WIDTH + (1 if strong else 0)
        if vertical:
            self.setFixedWidth(thickness)
        else:
            self.setFixedHeight(thickness)
        self.refresh_theme()

    def refresh_theme(self) -> None:
        name = "border_strong" if self._strong else "border_soft"
        axis = "border-left" if self._vertical else "border-top"
        self.setStyleSheet(f"Hairline {{ {axis}: {self.width() or BORDER_WIDTH}px solid {color_hex(name)}; }}")


class Card(QFrame):
    """A square-cornered panel with a hairline border.

    ``elevated=False`` (default) uses the plain card fill. The class exists so a
    page never hand-rolls a QFrame + stylesheet again.
    """

    def __init__(self, parent=None, *, padding: int = SPACE["md"], alt: bool = False):
        super().__init__(parent)
        self._alt = alt
        self._radius = 0
        self.setAttribute(Qt.WA_StyledBackground, False)
        if padding:
            self.setContentsMargins(padding, padding, padding, padding)

    # -- painting ---------------------------------------------------------

    def refresh_theme(self) -> None:
        self._radius = 0 if _square() else 4
        self.update()

    def paintEvent(self, event):  # noqa: N802 (Qt naming)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)
        rect = self.rect().adjusted(0, 0, -1, -1)

        painter.setPen(Qt.NoPen)
        painter.setBrush(color("card_alt" if self._alt else "card"))
        painter.drawRect(rect)

        pen = QPen(color("border"))
        pen.setWidth(BORDER_WIDTH)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(rect)
        painter.end()


class CardHeader(QWidget):
    """Title (+ optional caption) on the left, an optional widget on the right."""

    def __init__(
        self,
        title: str = "",
        parent=None,
        *,
        caption: str = "",
        trailing: QWidget | None = None,
    ):
        super().__init__(parent)
        self._title = QLabel(title, self)
        self._caption = QLabel(caption, self)
        self._trailing = trailing

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACE["sm"])

        text_column = QVBoxLayout()
        text_column.setContentsMargins(0, 0, 0, 0)
        text_column.setSpacing(SPACE["xs"] // 2)
        text_column.addWidget(self._title)
        if caption:
            text_column.addWidget(self._caption)
        layout.addLayout(text_column, 1)
        if trailing is not None:
            layout.addWidget(trailing, 0, Qt.AlignTop)

        self.refresh_theme()

    # -- API --------------------------------------------------------------

    def set_title(self, text: str) -> None:
        self._title.setText(text)

    def set_caption(self, text: str) -> None:
        self._caption.setText(text)
        self._caption.setVisible(bool(text))

    def refresh_theme(self) -> None:
        from gui.design_tokens import TYPE_SCALE

        self._title.setStyleSheet(
            f"color: {color_hex('foreground')};"
            f"font-size: {TYPE_SCALE['h3']}pt; font-weight: 600;"
        )
        self._caption.setStyleSheet(
            f"color: {color_hex('muted_foreground')};"
            f"font-size: {TYPE_SCALE['caption']}pt;"
        )
        if isinstance(self._trailing, ThemeAware):
            self._trailing.refresh_theme()


class ThemeAware:
    """Mixin for ui_kit controls that re-read tokens on a theme switch.

    ``AppShell`` walks its children and calls ``refresh_theme()`` after
    ``gui.theme.set_app_theme``. Controls that paint in ``paintEvent`` only need
    to ``update()``; controls that carry labels must restyle them.
    """

    def refresh_theme(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError


def _square() -> bool:
    from gui.ui_kit.tokens import is_square_theme

    return is_square_theme()
