"""FieldRow — a labelled control row.

The old pages each rebuilt ``QHBoxLayout() + label + control + addStretch`` with
their own label width, which is why labels never line up across pages. This fixes
the label column once.

    [ label        ] [ control ...................... ]

``label_width`` is a **minimum**, not a fixed width: at a 2.0x font scale a fixed
column would clip the text, so the label is allowed to grow. Rows that should
share a column call :meth:`FieldRow.align`, which pins them to the widest label.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QWidget

from gui.design_tokens import SPACE, TYPE_SCALE
from gui.ui_kit.tokens import color_hex

DEFAULT_LABEL_WIDTH = 132


class FieldRow(QWidget):
    """A label plus one or more controls, aligned on a shared label column."""

    def __init__(
        self,
        label: str = "",
        control: QWidget | None = None,
        parent=None,
        *,
        label_width: int = DEFAULT_LABEL_WIDTH,
        hint: str = "",
    ):
        super().__init__(parent)
        self._label = QLabel(label, self)
        self._hint = QLabel(hint, self)
        self._hint.setVisible(bool(hint))
        self._controls: list[QWidget] = []

        if label_width:
            # minimum, not fixed — a fixed column clips as soon as the user
            # raises the font scale
            self._label.setMinimumWidth(label_width)
        self._label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(SPACE["sm"])
        row.addWidget(self._label, 0)
        if control is not None:
            row.addWidget(control, 1)
        row.addWidget(self._hint, 0)

        self.refresh_theme()

    # -- API --------------------------------------------------------------

    @staticmethod
    def align(rows, width: int | None = None) -> int:
        """Pin a group of FieldRows to one label width and return it.

        Defaults to the widest ``sizeHint`` in *rows*, so the column is as narrow
        as it can be while still fitting every label.
        """
        rows = [row for row in rows if isinstance(row, FieldRow)]
        if not rows:
            return 0
        target = width or max(row._label.sizeHint().width() for row in rows)
        for row in rows:
            row._label.setMinimumWidth(target)
            row._label.setMaximumWidth(target)
        return target

    def add_control(self, widget: QWidget, stretch: int = 0) -> None:
        """Append another control to the row (e.g. a unit label after a spinbox)."""
        layout = self.layout()
        insert_at = layout.count() - 1 if self._hint.isVisible() or self._hint.text() else layout.count()
        layout.insertWidget(insert_at, widget, stretch)
        self._controls.append(widget)

    @property
    def label(self) -> QLabel:
        return self._label

    def set_label(self, text: str) -> None:
        self._label.setText(text)

    def set_hint(self, text: str) -> None:
        self._hint.setText(text)
        self._hint.setVisible(bool(text))

    def refresh_theme(self) -> None:
        self._label.setStyleSheet(
            f"color: {color_hex('foreground')}; font-size: {TYPE_SCALE['body']}pt;"
        )
        self._hint.setStyleSheet(
            f"color: {color_hex('muted_foreground')}; font-size: {TYPE_SCALE['caption']}pt;"
        )
