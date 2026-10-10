"""Layout primitives: PageScaffold, FieldRow, ValueLabel, EmptyState.

These exist because the archived UI let all eight pages build their own
``QVBoxLayout`` + header + cards, which is how eight pages ended up with eight
slightly different margins. Here the skeleton and the label column are decided
once.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFontMetrics
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget, QBoxLayout

from vstudio_ui.theme import color_hex
from vstudio_ui.tokens import SPACE, TYPE_SCALE
from vstudio_ui.widgets.card import Hairline

DEFAULT_LABEL_WIDTH = 132


class PageScaffold(QWidget):
    """Header + optional toolbar + content area, with one set of margins."""

    def __init__(self, title: str = "", parent=None, *, caption: str = "", margin: int = SPACE["lg"]):
        super().__init__(parent)

        root = QVBoxLayout(self)
        root.setContentsMargins(margin, margin, margin, margin)
        root.setSpacing(SPACE["md"])

        self._title = QLabel(title, self)
        self._caption = QLabel(caption, self)
        self._caption.setWordWrap(True)
        self._caption.setVisible(bool(caption))

        self._trailing_row = QHBoxLayout()
        self._trailing_row.setContentsMargins(0, 0, 0, 0)
        self._trailing_row.setSpacing(SPACE["sm"])

        title_column = QVBoxLayout()
        title_column.setContentsMargins(0, 0, 0, 0)
        title_column.setSpacing(2)
        title_column.addWidget(self._title)
        title_column.addWidget(self._caption)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(SPACE["md"])
        header.addLayout(title_column, 1)
        header.addLayout(self._trailing_row, 0)
        root.addLayout(header)

        self._toolbar = QHBoxLayout()
        self._toolbar.setContentsMargins(0, 0, 0, 0)
        self._toolbar.setSpacing(SPACE["sm"])
        self._toolbar_host = QWidget(self)
        self._toolbar_host.setLayout(self._toolbar)
        self._toolbar_host.setVisible(False)
        root.addWidget(self._toolbar_host)

        self._content = QVBoxLayout()
        self._content.setContentsMargins(0, 0, 0, 0)
        self._content.setSpacing(SPACE["md"])
        root.addLayout(self._content, 1)

        # Deliberately the *unbound* call, not ``self.refresh_theme()``: a virtual
        # dispatch here runs the subclass's override before the subclass has built
        # anything, and every page that styles its own widgets then dies on an
        # AttributeError during ``super().__init__``. Calling the base method makes
        # the invariant explicit — a subclass's ``refresh_theme`` may assume its own
        # widgets exist, because it is only ever called once they do.
        PageScaffold.refresh_theme(self)

    @property
    def content_layout(self) -> QVBoxLayout:
        return self._content

    @property
    def toolbar_layout(self) -> QHBoxLayout:
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
            f"color: {color_hex('foreground')}; font-size: {TYPE_SCALE['h1']}pt; font-weight: 700;"
        )
        self._caption.setStyleSheet(
            f"color: {color_hex('muted_foreground')}; font-size: {TYPE_SCALE['body_small']}pt;"
        )


class FieldRow(QWidget):
    """A label plus one or more controls on a shared label column.

    ``label_width`` is a **minimum**, not a fixed width: at a 2.0x font scale a
    fixed column clips the text. Rows that must line up call :meth:`align`.
    """

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

        if label_width:
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

    @staticmethod
    def align(rows, width: int | None = None) -> int:
        """Pin a group of FieldRows to one label width; returns the width used."""
        rows = [row for row in rows if isinstance(row, FieldRow)]
        if not rows:
            return 0
        target = width or max(row._label.sizeHint().width() for row in rows)
        for row in rows:
            row._label.setMinimumWidth(target)
            row._label.setMaximumWidth(target)
        return target

    def add_control(self, widget: QWidget, stretch: int = 0) -> None:
        layout = self.layout()
        layout.insertWidget(layout.count() - 1, widget, stretch)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        compact = self.width() < 580
        self.layout().setDirection(QBoxLayout.TopToBottom if compact else QBoxLayout.LeftToRight)
        self._label.setAlignment((Qt.AlignLeft if compact else Qt.AlignRight) | Qt.AlignVCenter)

    @property
    def label(self) -> QLabel:
        return self._label

    @property
    def hint(self) -> QLabel:
        return self._hint

    def set_label(self, text: str) -> None:
        self._label.setText(text)

    def set_hint(self, text: str) -> None:
        self._hint.setText(text)
        self._hint.setVisible(bool(text))

    def refresh_theme(self) -> None:
        self._label.setStyleSheet(f"color: {color_hex('foreground')}; font-size: {TYPE_SCALE['body']}pt;")
        self._hint.setStyleSheet(
            f"color: {color_hex('muted_foreground')}; font-size: {TYPE_SCALE['caption']}pt;"
        )


class ValueLabel(QLabel):
    """The right-hand value of a :class:`FieldRow`.

    Three behaviours that a bare ``QLabel`` gets wrong in this layout:

    * **Elides instead of pushing the layout.** A GPU name is arbitrarily long;
      a wrapping or expanding label silently widens the whole card and knocks the
      page out of alignment. The full text stays available as a tooltip.
    * **Knows when it has no value yet.** An empty or pending value renders in the
      muted token, so "not probed" never looks like a fact.
    * **Restyles itself.** Everything in the kit exposes ``refresh_theme()``, and
      the elision has to be recomputed after a font-size change.
    """

    def __init__(self, text: str = "", parent=None, *, muted: bool = False):
        super().__init__(parent)
        self._full = ""
        self._muted = muted
        self.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.setText(text)
        self.refresh_theme()

    @property
    def muted(self) -> bool:
        return self._muted

    def set_muted(self, muted: bool) -> None:
        muted = bool(muted)
        if muted != self._muted:
            self._muted = muted
            self.refresh_theme()

    @property
    def full_text(self) -> str:
        """The un-elided text (what the tooltip shows)."""
        return self._full

    def set_value(self, text: str, *, muted: bool | None = None) -> None:
        """Set the value; ``muted=None`` infers "no value" from an empty string."""
        self.setText(text)
        self.set_muted((not text) if muted is None else muted)

    def setText(self, text: str) -> None:  # noqa: N802 - Qt naming
        self._full = "" if text is None else str(text)
        self.setToolTip(self._full)
        self._elide()

    def resizeEvent(self, event):  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        self._elide()

    def refresh_theme(self) -> None:
        token = "muted_foreground" if self._muted else "foreground_2"
        self.setStyleSheet(f"color: {color_hex(token)}; font-size: {TYPE_SCALE['body']}pt;")
        self._elide()

    def _elide(self) -> None:
        full = getattr(self, "_full", "")
        available = max(0, self.width() - 2)
        if not available or not full:
            super().setText(full)
            return
        metrics = QFontMetrics(self.font())
        super().setText(metrics.elidedText(full, Qt.ElideRight, available))


class EmptyState(QWidget):
    """Centred title + hint, shown when a page has nothing to display yet."""

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

    def set_title(self, text: str) -> None:
        self._title.setText(text)

    def set_hint(self, text: str) -> None:
        self._hint.setText(text)
        self._hint.setVisible(bool(text))

    def refresh_theme(self) -> None:
        self._title.setStyleSheet(
            f"color: {color_hex('foreground')}; font-size: {TYPE_SCALE['h3']}pt; font-weight: 600;"
        )
        self._hint.setStyleSheet(
            f"color: {color_hex('muted_foreground')}; font-size: {TYPE_SCALE['body_small']}pt;"
        )
