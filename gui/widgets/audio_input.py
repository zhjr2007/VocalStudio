"""AudioInputArea: drag & drop audio queue with whitelist filtering (wave B-1)."""

from __future__ import annotations

import os

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush, QColor, QFont
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from gui.i18n import tr

ACCEPTED_EXTENSIONS = ("wav", "mp3", "flac", "m4a", "ogg")
REJECTED_GRAY = QColor(0x9A, 0x9A, 0x9A)


class AudioInputArea(QWidget):
    """Drop zone + single-selection file queue; rejected rows stay greyed out."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 14)
        layout.setSpacing(8)

        header_row = QHBoxLayout()
        self._title_label = QLabel(self)
        title_font = QFont(self._title_label.font())
        title_font.setBold(True)
        self._title_label.setFont(title_font)
        header_row.addWidget(self._title_label)
        header_row.addStretch(1)
        self._add_button = QPushButton(self)
        header_row.addWidget(self._add_button)
        layout.addLayout(header_row)

        self._list_widget = QListWidget(self)
        self._list_widget.setSelectionMode(QAbstractItemView.SingleSelection)
        self._list_widget.setMinimumHeight(120)
        layout.addWidget(self._list_widget, 1)

        self._hint_label = QLabel(self)
        layout.addWidget(self._hint_label)

        self._add_button.clicked.connect(self.add_from_dialog)
        self.retranslate()

    @staticmethod
    def extension_ok(path: str) -> bool:
        suffix = os.path.splitext(str(path))[1].lower().lstrip(".")
        return suffix in ACCEPTED_EXTENSIONS

    def add_files(self, paths: list[str]) -> int:
        accepted = 0
        for raw in list(paths or []):
            path = str(raw)
            item = QListWidgetItem(os.path.basename(path) or path)
            item.setData(Qt.UserRole, path)
            if self.extension_ok(path):
                accepted += 1
            else:
                item.setForeground(QBrush(REJECTED_GRAY))
                item.setToolTip(f"{path} {tr('input_rejected', '（扩展名不支持，未加入分离队列）')}")
            self._list_widget.addItem(item)
        return accepted

    def rows(self) -> list[tuple[str, bool]]:
        result = []
        for i in range(self._list_widget.count()):
            path = str(self._list_widget.item(i).data(Qt.UserRole))
            result.append((path, self.extension_ok(path)))
        return result

    def accepted_count(self) -> int:
        return sum(1 for _path, ok in self.rows() if ok)

    def selected_index(self) -> int:
        return self._list_widget.currentRow()

    def select(self, index: int) -> None:
        if 0 <= index < self._list_widget.count():
            self._list_widget.setCurrentRow(index)

    def selected_path(self) -> str | None:
        index = self.selected_index()
        if index == -1:
            return None
        return self.rows()[index][0]

    def queue(self) -> QListWidget:
        return self._list_widget

    def clear_queue(self) -> None:
        self._list_widget.clear()

    def add_from_dialog(self) -> int:
        patterns = " ".join(f"*.{ext}" for ext in ACCEPTED_EXTENSIONS)
        files, _all = QFileDialog.getOpenFileNames(
            self, tr("input_add_button", "添加音频…"), "", f"Audio ({patterns}) | All (*)"
        )
        if files:
            return self.add_files(files)
        return 0

    def dragEnterEvent(self, event) -> None:
        if self._drag_paths(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:
        paths = self._drag_paths(event)
        if paths:
            self.add_files(paths)
            event.acceptProposedAction()

    @staticmethod
    def _drag_paths(event):
        mime = event.mimeData()
        if not mime.hasUrls():
            return []
        paths = [url.toLocalFile() for url in mime.urls()]
        return [path for path in paths if os.path.isfile(path)]

    def retranslate(self) -> None:
        self._title_label.setText(tr("audio_input", "音频输入"))
        self._add_button.setText(tr("input_add_button", "添加音频…"))
        self._hint_label.setText(
            tr("input_drop_hint", "拖入音频文件（wav/mp3/flac/m4a/ogg），多文件自动排队；不支持的扩展名置灰显示")
        )
        for i in range(self._list_widget.count()):
            item = self._list_widget.item(i)
            path = str(item.data(Qt.UserRole))
            if not self.extension_ok(path):
                item.setToolTip(f"{path} {tr('input_rejected', '（扩展名不支持，未加入分离队列）')}")
