"""EmptyState — what a page shows before it has anything to show.

The old pages left a blank scroll area when idle, which reads as "broken" rather
than "waiting". A one-line title plus a muted hint is enough; no illustration, no
shadow (the design language has neither).
"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget

from gui.design_tokens import SPACE, TYPE_SCALE
from gui.ui_kit.tokens import color_hex


class EmptyState(QWidget):
    """Centred title + hint. ``setVisible`` it when the page is idle."""

    def __init__(self, title: str = "", hint: str = "", parent=None):
        super().__init__(parent)
        self._title = QLabel(title, self)
        self._hint = QLabel(hint, self)
        self._title.setAlignment(Qt.AlignCenter)
        self._hint.setAlignment(Qt.AlignCenter)
        self._hint.setWordWrap(True)

        column = QVBoxLayout(self)
        column.setContentsMargins(SPACE["xl"], SPACE["xl"], SPACE["xl"], SPACE["xl"])
        column.setSpacing(SPACE["sm"])
        column.addStretch(1)
        column.addWidget(self._title)
        column.addWidget(self._hint)
        column.addStretch(1)

        self.refresh_theme()

    # -- API --------------------------------------------------------------

    def set_title(self, text: str) -> None:
        self._title.setText(text)

    def set_hint(self, text: str) -> None:
        self._hint.setText(text)
        self._hint.setVisible(bool(text))

    def refresh_theme(self) -> None:
        self._title.setStyleSheet(
            f"color: {color_hex('foreground')};"
            f"font-size: {TYPE_SCALE['h3']}pt; font-weight: 600;"
        )
        self._hint.setStyleSheet(
            f"color: {color_hex('muted_foreground')};"
            f"font-size: {TYPE_SCALE['body_small']}pt;"
        )
