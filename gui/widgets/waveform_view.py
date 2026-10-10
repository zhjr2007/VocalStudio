"""WaveformView: paint-only multi-track waveform widget with time marks.

Supports any number of waveform lanes (one per track), vertical time marks,
a click-to-seek ``seek_requested`` signal and an optional loop region band.
"""

from __future__ import annotations

import numpy as np

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPalette, QPainter
from PyQt5.QtWidgets import QLabel, QWidget

from gui.i18n import tr


def compute_peaks(audio: np.ndarray, sr: int, num_buckets: int) -> np.ndarray:
    """Max-abs per bucket of a 1-D signal, normalized to the range 0..1."""
    if num_buckets <= 0:
        return np.zeros(0, dtype="float32")
    audio = np.asarray(audio).ravel()
    n = int(audio.size)
    if n == 0 or sr <= 0:
        return np.zeros(num_buckets, dtype="float32")
    peaks = np.empty(num_buckets, dtype=np.float64)
    for i in range(num_buckets):
        start = i * n // num_buckets
        end = (i + 1) * n // num_buckets if i < num_buckets - 1 else n
        chunk = audio[start:end]
        peaks[i] = float(np.max(np.abs(chunk))) if chunk.size else 0.0
    peak = float(peaks.max())
    if peak > 0:
        peaks /= peak
    return peaks.astype("float32")


TRACK_COLORS = ("#4C8DFF", "#3DD68C", "#F7B32B", "#B26EE6", "#FF6B6B")
MARK_COLOR = QColor("#F75C49")
LOOP_COLOR = QColor("#F7B32B")


class WaveformView(QWidget):
    """Canvas for any number of waveform lanes plus vertical time marks.

    Clicking (or dragging) on the canvas emits ``seek_requested(ms)`` so a
    consumer can locate playback. ``set_loop(start_ms, end_ms)`` draws a
    translucent loop band for practice/segment-loop workflows.
    """

    seek_requested = pyqtSignal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._tracks: list[dict] = []
        self._marks: list[dict] = []
        self._duration_ms: int = 0
        self._loop_start_ms: int | None = None
        self._loop_end_ms: int | None = None
        self._max_value: int = 0
        self.setMinimumSize(240, 120)

        self._time_label = QLabel("0:00", self)
        hint_font = self.font()
        hint_font.setPointSize(max(8, hint_font.pointSize() - 1))
        self._hint = QLabel(tr("waveform.empty", "No waveform yet"), self)
        self._hint.setFont(hint_font)
        self._position_labels()

    def set_waveforms(self, tracks: list[dict]) -> None:
        """tracks: dicts with keys label, peaks (np.ndarray), color."""
        self._tracks = [track for track in (tracks or []) if isinstance(track, dict)]
        self._position_labels()
        self.update()

    def set_marks(self, marks: list[dict]) -> None:
        """marks: dicts with keys position_ms, label and optional duration_ms."""
        self._marks = [mark for mark in (marks or []) if isinstance(mark, dict)]
        self.update()

    def set_duration(self, ms: int) -> None:
        try:
            self._duration_ms = max(0, int(ms))
        except (TypeError, ValueError):
            self._duration_ms = 0
        self._time_label.setText(self._format_ms(self._duration_ms))
        if self._loop_start_ms is not None and self._loop_start_ms > self._duration_ms:
            self.clear_loop()

    def duration_ms(self) -> int:
        return self._duration_ms

    # -- loop region --------------------------------------------------------

    def set_loop(self, start_ms: int, end_ms: int) -> None:
        try:
            start = max(0, int(start_ms))
            end = max(0, int(end_ms))
        except (TypeError, ValueError):
            return
        if end <= start:
            self.clear_loop()
            return
        if self._duration_ms and end > self._duration_ms:
            return
        self._loop_start_ms = start
        self._loop_end_ms = end
        self.update()

    def clear_loop(self) -> None:
        if self._loop_start_ms is None and self._loop_end_ms is None:
            return
        self._loop_start_ms = None
        self._loop_end_ms = None
        self.update()

    def loop_bounds(self) -> tuple[int | None, int | None]:
        return self._loop_start_ms, self._loop_end_ms

    # -- seek ----------------------------------------------------------------

    def _emit_seek(self, x: int) -> None:
        if self._duration_ms <= 0 or self.width() <= 1:
            return
        value = max(0, min(self.width() - 1, int(x)))
        ms = value * self._duration_ms // self.width()
        ms = min(ms, self._duration_ms)
        self.seek_requested.emit(ms)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._max_value = self._duration_ms
            self._emit_seek(event.pos().x())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if event.buttons() & Qt.LeftButton:
            self._emit_seek(event.pos().x())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._emit_seek(event.pos().x())
        super().mouseReleaseEvent(event)

    def retranslate(self) -> None:
        self._hint.setText(tr("waveform.empty", "No waveform yet"))

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        rect = self.rect()
        bg = self.palette().color(QPalette.Base)
        painter.fillRect(rect, bg)

        n_tracks = len(self._tracks)
        lane_count = max(2, n_tracks)  # keep the original dual-lane look for <=2 tracks
        lane_h = max(1, rect.height() // lane_count)
        for slot in range(lane_count):
            track = self._tracks[slot] if slot < len(self._tracks) else None
            color = self._track_color(track, slot)
            y0 = slot * lane_h
            y1 = rect.height() if slot == lane_count - 1 else y0 + lane_h
            peaks = track.get("peaks") if track is not None else None
            self._draw_track(painter, peaks, int(y0), int(min(y1, rect.height())), color)

        self._draw_marks(painter, rect.width(), rect.height())
        self._draw_loop(painter, rect.width(), rect.height())

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._position_labels()

    @staticmethod
    def _track_color(track: dict | None, slot: int) -> QColor:
        fallback = QColor(TRACK_COLORS[slot % len(TRACK_COLORS)])
        value = track.get("color") if isinstance(track, dict) else None
        color: QColor | None = None
        if isinstance(value, str):
            candidate = QColor(value)
            color = candidate if candidate.isValid() else None
        elif isinstance(value, (tuple, list)) and len(value) == 3:
            color = QColor(int(value[0]), int(value[1]), int(value[2]))
        if color is None or not color.isValid():
            return fallback
        return color

    @staticmethod
    def _draw_track(
        painter: QPainter, peaks: np.ndarray | None, y0: int, y1: int, color: QColor
    ) -> None:
        width = max(painter.viewport().width(), 1)
        if peaks is None or len(peaks) == 0 or y1 <= y0:
            return
        n = len(peaks)
        center_y = (y0 + y1) // 2
        max_bar = max(1, int((y1 - y0) * 0.7))
        painter.setPen(Qt.NoPen)
        for x in range(width):
            index = min(x * n // width, n - 1)
            bar_height = int(round(float(peaks[index]) * max_bar))
            if bar_height <= 0:
                continue
            top = center_y - (bar_height // 2)
            painter.fillRect(x, top, 1, max(1, bar_height), color)

    def _draw_marks(self, painter: QPainter, width: int, height: int) -> None:
        duration_ms = self._duration_ms
        if width <= 0 or not self._marks or duration_ms <= 0:
            return

        def x_of(ms_value: float) -> int:
            return max(0, min(width - 1, int(ms_value * (width - 1)) // duration_ms))

        for mark in self._marks:
            try:
                position_ms = float(mark.get("position_ms", 0))
            except (TypeError, ValueError):
                continue
            x = x_of(position_ms)
            painter.fillRect(x, 0, 2, height, MARK_COLOR)

            duration = mark.get("duration_ms")
            if isinstance(duration, (int, float)) and duration > 0:
                x_end = x_of(position_ms + float(duration))
                band_color = QColor(MARK_COLOR)
                band_color.setAlpha(64)
                painter.fillRect(x, 0, max(1, x_end - x), height, band_color)

            label = str(mark.get("label", ""))
            if label:
                painter.setPen(MARK_COLOR)
                painter.drawText(x + 4, 2, width - x - 6, 18, Qt.AlignLeft | Qt.AlignVCenter, label)

    def _draw_loop(self, painter: QPainter, width: int, height: int) -> None:
        duration_ms = self._duration_ms
        if self._loop_start_ms is None or self._loop_end_ms is None or duration_ms <= 0:
            return
        start = max(0, min(self._loop_start_ms, duration_ms))
        end = max(0, min(self._loop_end_ms, duration_ms))
        if end <= start:
            return
        x0 = start * width // duration_ms
        x1 = max(x0 + 1, end * width // duration_ms)
        band = QColor(LOOP_COLOR)
        band.setAlpha(56)
        painter.fillRect(x0, 0, x1 - x0, height, band)
        painter.fillRect(x0, 0, 2, height, LOOP_COLOR)
        painter.fillRect(x1 - 1, 0, 2, height, LOOP_COLOR)

    def _position_labels(self) -> None:
        self._time_label.move(8, 4)
        if self._tracks:
            self._hint.hide()
            return
        self._hint.show()
        hint_size = self._hint.sizeHint()
        x = max(0, (self.width() - hint_size.width()) // 2)
        y = max(0, (self.height() - hint_size.height()) // 2)
        self._hint.move(x, y)

    @staticmethod
    def _format_ms(ms: int) -> str:
        total_seconds = max(0, int(ms)) // 1000
        return f"{total_seconds // 60}:{total_seconds % 60:02d}"
