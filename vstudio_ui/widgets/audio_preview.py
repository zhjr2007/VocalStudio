"""In-app playback with explicit errors, including unavailable audio devices."""
from pathlib import Path
from PyQt5.QtCore import QUrl
from PyQt5.QtWidgets import QWidget, QPushButton, QSlider, QVBoxLayout, QHBoxLayout, QLabel
from PyQt5.QtCore import Qt
from vstudio_ui import strings


class AudioPreview(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.path = ''
        self.play = QPushButton(self); self.stop = QPushButton(self)
        self.seek = QSlider(Qt.Horizontal, self); self.label = QLabel(self)
        self.label.setWordWrap(True)
        column = QVBoxLayout(self); column.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout(); row.addWidget(self.play); row.addWidget(self.stop); row.addWidget(self.seek, 1)
        column.addWidget(self.label); column.addLayout(row)
        self.player = None
        self._loaded_path = ''
        self.play.clicked.connect(self._play)
        self.stop.clicked.connect(lambda: self.player.stop() if self.player else None)
        self.seek.sliderMoved.connect(lambda n: self.player.setPosition(n) if self.player else None)
        self.retranslate(); self.set_path('')

    def _ensure_player(self):
        if self.player is not None: return True
        try:
            from PyQt5.QtMultimedia import QMediaPlayer
            self.player = QMediaPlayer(self)
            self.player.positionChanged.connect(self.seek.setValue)
            self.player.durationChanged.connect(lambda n: self.seek.setRange(0, n))
            self.player.error.connect(self._playback_error)
            return True
        except ImportError as exc:
            self.label.setText(str(exc)); self.play.setEnabled(False)
            return False

    def _playback_error(self, _error):
        if self.player: self.label.setText(self.player.errorString())

    def _play(self):
        if not self.path or not self._ensure_player(): return
        if self._loaded_path != self.path:
            from PyQt5.QtMultimedia import QMediaContent
            self.player.setMedia(QMediaContent(QUrl.fromLocalFile(self.path)))
            self._loaded_path = self.path
        self.seek.setRange(0, self.player.duration())
        self.player.play()
        self.stop.setEnabled(True)

    def set_path(self, path):
        if self.player: self.player.stop()
        self.path = path
        playable = bool(path and Path(path).suffix.lower() in ('.wav', '.mp3', '.flac', '.m4a', '.ogg'))
        self.play.setEnabled(playable)
        self.stop.setEnabled(False)
        self.seek.setValue(0); self.seek.setRange(0, 0)
        self.label.setText(Path(path).name if playable else strings.tr('task.audio_empty'))
        # Selecting inputs/results must not start DirectShow or decode a file.
        # Windows native backends can crash when many unused players are torn
        # down while their asynchronous media loading is still pending.

    def retranslate(self):
        self.play.setText(strings.tr('task.play')); self.stop.setText(strings.tr('task.stop'))
