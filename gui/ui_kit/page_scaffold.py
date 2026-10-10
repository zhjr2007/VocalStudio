"""PageScaffold — the shared page skeleton.

Every page in the old UI built its own ``QVBoxLayout`` + header + cards, which is
how 8 pages drifted into 8 slightly different margins (refactor-plan U2). This
gives one structure:

    +------------------------------------------+
    | title                          [trailing] |   <- header
    | caption                                   |
    +------------------------------------------+
    | [toolbar]                                 |   <- optional
    +------------------------------------------+
    |                                           |
    | content  (pages add widgets here)         |
    |                                           |
    +------------------------------------------+

Margins come from ``SPACE``, so the 25px reference rhythm is applied in one
place instead of eight.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from gui.design_tokens import SPACE, TYPE_SCALE
from gui.ui_kit.card import Hairline
from gui.ui_kit.tokens import color_hex


class PageScaffold(QWidget):
    """Header + optional toolbar + content area, with consistent spacing."""

    def __init__(
        self,
        title: str = "",
        parent=None,
        *,
        caption: str = "",
        margin: int = SPACE["lg"],
    ):
        super().__init__(parent)
        self._margin = margin

        root = QVBoxLayout(self)
        root.setContentsMargins(margin, margin, margin, margin)
        root.setSpacing(SPACE["md"])

        # -- header -------------------------------------------------------
        self._title = QLabel(title, self)
        self._caption = QLabel(caption, self)
        self._trailing_row = QHBoxLayout()
        self._trailing_row.setContentsMargins(0, 0, 0, 0)
        self._trailing_row.setSpacing(SPACE["sm"])

        title_column = QVBoxLayout()
        title_column.setContentsMargins(0, 0, 0, 0)
        title_column.setSpacing(SPACE["xs"] // 2)
        title_column.addWidget(self._title)
        if caption:
            title_column.addWidget(self._caption)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(SPACE["md"])
        header.addLayout(title_column, 1)
        header.addLayout(self._trailing_row, 0)
        root.addLayout(header)

        root.addWidget(Hairline(strong=True))

        # -- toolbar (hidden until a page puts something in it) -----------
        self._toolbar = QHBoxLayout()
        self._toolbar.setContentsMargins(0, 0, 0, 0)
        self._toolbar.setSpacing(SPACE["sm"])
        self._toolbar_host = QWidget(self)
        self._toolbar_host.setLayout(self._toolbar)
        self._toolbar_host.setVisible(False)
        root.addWidget(self._toolbar_host)

        # -- content ------------------------------------------------------
        self._content = QVBoxLayout()
        self._content.setContentsMargins(0, 0, 0, 0)
        self._content.setSpacing(SPACE["md"])
        root.addLayout(self._content, 1)

        self.refresh_theme()

    # -- API --------------------------------------------------------------

    @property
    def content_layout(self) -> QVBoxLayout:
        """Pages add their widgets here."""
        return self._content

    @property
    def toolbar_layout(self) -> QHBoxLayout:
        """Pages add toolbar widgets here; the row appears automatically."""
        return self._toolbar

    def add_toolbar_widget(self, widget: QWidget, stretch: int = 0) -> None:
        self._toolbar.addWidget(widget, stretch)
        self._toolbar_host.setVisible(True)

    def add_trailing_widget(self, widget: QWidget) -> None:
        self._trailing_row.addWidget(widget, 0, Qt.AlignTop)

    def set_title(self, text: str) -> None:
        self._title.setText(text)

    def set_caption(self, text: str) -> None:
        self._caption.setText(text)
        self._caption.setVisible(bool(text))

    def refresh_theme(self) -> None:
        self._title.setStyleSheet(
            f"color: {color_hex('foreground')};"
            f"font-size: {TYPE_SCALE['h1']}pt; font-weight: 700;"
        )
        self._caption.setStyleSheet(
            f"color: {color_hex('muted_foreground')};"
            f"font-size: {TYPE_SCALE['body_small']}pt;"
        )
