"""Reusable titled info card: a heading plus wrapped caption lines (i18n keys)."""

from __future__ import annotations

from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QVBoxLayout, QWidget

try:
    from qfluentwidgets import BodyLabel as FluentLabel, CardWidget, CaptionLabel
except ModuleNotFoundError:  # renamed distribution
    from qfluent_widgets import BodyLabel as FluentLabel, CardWidget, CaptionLabel

try:
    from qfluentwidgets import TitleLabel
except Exception:  # renamed distribution without the export
    class TitleLabel(FluentLabel):
        pass


class InfoCard(CardWidget):
    """Prominent usage/model-intro card used on every module page."""

    def __init__(self, title_key: str, line_keys: tuple[str, ...], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._title_key = title_key
        self._line_keys = tuple(line_keys)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 14)
        layout.setSpacing(8)

        self._title_label = TitleLabel()
        font = QFont(self._title_label.font())
        font.setBold(True)
        self._title_label.setFont(font)
        layout.addWidget(self._title_label)

        self._line_labels: list[CaptionLabel] = []
        for _key in self._line_keys:
            label = CaptionLabel()
            label.setWordWrap(True)
            self._line_labels.append(label)
            layout.addWidget(label)
        layout.addStretch(1)

        self.retranslate()

    def retranslate(self) -> None:
        from gui.i18n import tr

        self._title_label.setText(tr(self._title_key))
        for label, key in zip(self._line_labels, self._line_keys):
            label.setText(tr(key))

    def line_texts(self) -> list[str]:
        return [label.text() for label in self._line_labels]
