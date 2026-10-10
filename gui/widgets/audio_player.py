"""AudioPlayerWidget: sounddevice-based audio player with seek, volume,
transpose (pitch-shift by resampling) and a selectable segment loop."""

from __future__ import annotations

import threading

import numpy as np
import sounddevice as sd
import soundfile as sf

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QCloseEvent
from PyQt5.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from gui.i18n import tr


class AudioPlayerWidget(QWidget):
    position_changed = pyqtSignal(int)
    finished = pyqtSignal()
    transpose_changed = pyqtSignal(int)
    loop_changed = pyqtSignal()

    TICK_MS = 40
    SLIDER_STEPS = 1000

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pcm: np.ndarray | None = None
        self._samplerate: int = 0
        self._n_frames: int = 0
        self._duration_ms: int = 0
        self._pos_frames: int | float = 0
        self._state: str = "stopped"
        self._volume: float = 0.8
        self._seeking: bool = False
        self._stream: sd.OutputStream | None = None
        self._lock = threading.Lock()
        self._transpose: int = 0
        self._playback_ratio: float = 1.0
        self._loop_enabled: bool = False
        self._loop_start_ms: int = 0
        self._loop_end_ms: int = 0
        self._loop_start_frames: int = 0
        self._loop_end_frames: int = 0

        card = QWidget(self)
        card.setObjectName("PlayerCard")
        card.setStyleSheet(
            "QWidget#PlayerCard {"
            " background: palette(base);"
            " border: 1px solid palette(midlight);"
            " border-radius: 8px;"
            "}"
        )
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(10, 8, 10, 8)

        top_row = QHBoxLayout()
        self._play_pause_btn = QPushButton(tr("player.play", "Play"))
        self._stop_btn = QPushButton(tr("player.stop", "Stop"))
        self._progress_bar = QSlider(Qt.Horizontal)
        self._progress_bar.setRange(0, 1)
        top_row.addWidget(self._play_pause_btn)
        top_row.addWidget(self._stop_btn)
        top_row.addWidget(self._progress_bar, 1)

        bottom_row = QHBoxLayout()
        time_font = self.font()
        time_font.setPointSize(max(8, time_font.pointSize() - 1))
        self._time_label = QLabel("0:00 / 0:00")
        self._time_label.setFont(time_font)
        self._volume_slider = QSlider(Qt.Horizontal)
        self._volume_slider.setRange(0, 100)
        self._volume_slider.setValue(int(round(self._volume * 100)))
        bottom_row.addWidget(self._time_label)
        bottom_row.addStretch(1)
        self._volume_label = QLabel(tr("player.volume", "Volume"))
        bottom_row.addWidget(self._volume_label)
        bottom_row.addWidget(self._volume_slider)

        transpose_row = QHBoxLayout()
        self._transpose_label = QLabel(tr("m4_transpose", "Transpose"))
        self._transpose_slider = QSlider(Qt.Horizontal)
        self._transpose_slider.setRange(-12, 12)
        self._transpose_slider.setValue(0)
        self._transpose_value = QLabel("0 st")
        transpose_row.addWidget(self._transpose_label)
        transpose_row.addWidget(self._transpose_slider, 1)
        transpose_row.addWidget(self._transpose_value)

        loop_row = QHBoxLayout()
        self._loop_check = QCheckBox(tr("m4_loop", "Loop"))
        self._loop_start_spin = QSpinBox()
        self._loop_start_spin.setRange(0, 0)
        self._loop_start_spin.setSuffix(" ms")
        self._loop_start_spin.setSingleStep(250)
        self._loop_end_spin = QSpinBox()
        self._loop_end_spin.setRange(0, 0)
        self._loop_end_spin.setSuffix(" ms")
        self._loop_end_spin.setSingleStep(250)
        self._set_a_btn = QPushButton(tr("m4_set_a", "Set A"))
        self._set_b_btn = QPushButton(tr("m4_set_b", "Set B"))
        loop_row.addWidget(self._loop_check)
        loop_row.addWidget(self._loop_start_spin)
        loop_row.addWidget(self._loop_end_spin)
        loop_row.addWidget(self._set_a_btn)
        loop_row.addWidget(self._set_b_btn)
        loop_row.addStretch(1)

        card_layout.addLayout(top_row)
        card_layout.addLayout(bottom_row)
        card_layout.addLayout(transpose_row)
        card_layout.addLayout(loop_row)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(card)

        self._tick_timer = QTimer(self)
        self._tick_timer.setInterval(self.TICK_MS)
        self._tick_timer.timeout.connect(self._on_tick)

        self._play_pause_btn.clicked.connect(self._on_play_pause_clicked)
        self._stop_btn.clicked.connect(self.stop)
        self._progress_bar.sliderPressed.connect(self._on_slider_pressed)
        self._progress_bar.sliderReleased.connect(self._on_slider_released)
        self._progress_bar.valueChanged.connect(self._on_progress_changed)
        self._volume_slider.valueChanged.connect(self._on_volume_changed)
        self._transpose_slider.valueChanged.connect(self._on_transpose_changed)
        self._loop_check.toggled.connect(self.set_loop_enabled)
        self._loop_start_spin.valueChanged.connect(self._on_loop_spin_changed)
        self._loop_end_spin.valueChanged.connect(self._on_loop_spin_changed)
        self._set_a_btn.clicked.connect(self._on_set_a_clicked)
        self._set_b_btn.clicked.connect(self._on_set_b_clicked)

    # -- properties ---------------------------------------------------------

    @property
    def state(self) -> str:
        return self._state

    @property
    def is_playing(self) -> bool:
        return self._state == "playing"

    @property
    def duration_ms(self) -> int:
        return self._duration_ms

    @property
    def position_ms(self) -> int:
        with self._lock:
            frames = self._pos_frames
        if self._samplerate <= 0 or self._n_frames <= 0:
            return 0
        ms = int(frames * 1000 // self._samplerate)
        return min(ms, self._duration_ms)

    @property
    def volume(self) -> float:
        return self._volume

    @volume.setter
    def volume(self, value: float) -> None:
        clamped = max(0.0, min(1.0, float(value)))
        self._volume = clamped
        slider_value = int(round(clamped * 100))
        if self._volume_slider.value() != slider_value:
            self._volume_slider.blockSignals(True)
            try:
                self._volume_slider.setValue(slider_value)
            finally:
                self._volume_slider.blockSignals(False)

    @property
    def transpose(self) -> int:
        return self._transpose

    @transpose.setter
    def transpose(self, semitones: int) -> None:
        self.set_transpose(int(semitones))

    @property
    def loop_enabled(self) -> bool:
        return self._loop_enabled

    @property
    def loop_start_ms(self) -> int:
        return self._loop_start_ms

    @property
    def loop_end_ms(self) -> int:
        return self._loop_end_ms

    def loop_bounds(self) -> tuple[int, int]:
        return self._loop_start_ms, self._loop_end_ms

    # -- set file / audio ---------------------------------------------------

    def set_file(self, path: str) -> None:
        self.stop()
        data, samplerate = sf.read(str(path), dtype="float32", always_2d=True)
        pcm = np.ascontiguousarray(data)
        n_frames = int(pcm.shape[0])
        sr = int(samplerate)
        self._pcm = pcm
        self._samplerate = sr
        self._n_frames = n_frames
        self._duration_ms = int(n_frames * 1000 / sr) if sr > 0 else 0
        with self._lock:
            self._pos_frames = 0
        self._state = "stopped"
        self._reset_loop()
        self._progress_bar.setRange(0, max(self.SLIDER_STEPS, 1))
        self._progress_bar.setValue(0)
        self._sync_time_label()
        self._update_play_button_text()

    # -- transpose ----------------------------------------------------------

    def set_transpose(self, semitones: int) -> None:
        clamped = max(-12, min(12, int(semitones)))
        self._transpose = clamped
        self._playback_ratio = float(2.0 ** (clamped / 12.0))
        if self._transpose_slider.value() != clamped:
            self._transpose_slider.blockSignals(True)
            try:
                self._transpose_slider.setValue(clamped)
            finally:
                self._transpose_slider.blockSignals(False)
        self._transpose_value.setText(self._format_semitones(clamped))
        self.transpose_changed.emit(clamped)

    # -- loop ---------------------------------------------------------------

    def set_loop(self, start_ms: int, end_ms: int) -> None:
        start = max(0, min(int(start_ms), self._duration_ms or int(start_ms)))
        end = max(0, min(int(end_ms), self._duration_ms or int(end_ms)))
        if end <= start:
            self._loop_start_ms = 0
            self._loop_end_ms = 0
            self._loop_start_frames = 0
            self._loop_end_frames = 0
        else:
            self._loop_start_ms = start
            self._loop_end_ms = end
            self._loop_start_frames = start * self._samplerate // 1000
            self._loop_end_frames = end * self._samplerate // 1000
        self._sync_loop_spins()
        self.loop_changed.emit()

    def set_loop_enabled(self, enabled: bool) -> None:
        self._loop_enabled = bool(enabled)
        if enabled and self._loop_end_frames <= self._loop_start_frames:
            self.set_loop(0, self._duration_ms)
        self.loop_changed.emit()

    def _reset_loop(self) -> None:
        self._loop_enabled = False
        self._loop_start_ms = 0
        self._loop_end_ms = 0
        self._loop_start_frames = 0
        self._loop_end_frames = 0
        self._loop_start_spin.blockSignals(True)
        self._loop_end_spin.blockSignals(True)
        try:
            self._loop_start_spin.setRange(0, max(self._duration_ms, 1))
            self._loop_end_spin.setRange(0, max(self._duration_ms, 1))
            self._loop_start_spin.setValue(0)
            self._loop_end_spin.setValue(self._duration_ms)
        finally:
            self._loop_start_spin.blockSignals(False)
            self._loop_end_spin.blockSignals(False)
        self._loop_check.setChecked(False)

    def _sync_loop_spins(self) -> None:
        self._loop_start_spin.blockSignals(True)
        self._loop_end_spin.blockSignals(True)
        try:
            self._loop_start_spin.setRange(0, max(self._duration_ms, 1))
            self._loop_end_spin.setRange(0, max(self._duration_ms, 1))
            self._loop_start_spin.setValue(self._loop_start_ms)
            self._loop_end_spin.setValue(self._loop_end_ms)
        finally:
            self._loop_start_spin.blockSignals(False)
            self._loop_end_spin.blockSignals(False)

    # -- transport ----------------------------------------------------------

    def play(self) -> None:
        if self._pcm is None or self._state == "playing":
            return
        if self._duration_ms > 0 and self.position_ms >= self._duration_ms:
            with self._lock:
                self._pos_frames = 0
        if self._loop_enabled and self._loop_end_frames > self._loop_start_frames:
            with self._lock:
                if self._pos_frames >= self._loop_end_frames or self._pos_frames < self._loop_start_frames:
                    self._pos_frames = self._loop_start_frames
        try:
            stream = sd.OutputStream(
                samplerate=self._samplerate,
                channels=int(self._pcm.shape[1]),
                dtype="float32",
                callback=self._callback,
            )
            stream.start()
        except sd.PortAudioError:
            self._state = "stopped"
            self._update_play_button_text()
            return
        self._stream = stream
        self._state = "playing"
        self._update_play_button_text()
        self._tick_timer.start()

    def pause(self) -> None:
        if self._state != "playing":
            return
        self._stop_stream()
        self._tick_timer.stop()
        self._state = "paused"
        self._update_play_button_text()

    def stop(self) -> None:
        self._stop_stream()
        self._tick_timer.stop()
        with self._lock:
            self._pos_frames = 0
        self._state = "stopped"
        if not self._seeking:
            self._progress_bar.blockSignals(True)
            try:
                self._progress_bar.setValue(0)
            finally:
                self._progress_bar.blockSignals(False)
        self._sync_time_label()
        self._update_play_button_text()

    def set_position(self, ms: int) -> None:
        if self._pcm is None or self._samplerate <= 0:
            return
        clamped_ms = max(0, min(int(ms), self._duration_ms))
        frames = clamped_ms * self._samplerate // 1000
        with self._lock:
            self._pos_frames = frames
        if not self._seeking:
            value = self._ms_to_value(clamped_ms)
            self._progress_bar.blockSignals(True)
            try:
                self._progress_bar.setValue(value)
            finally:
                self._progress_bar.blockSignals(False)
        self._sync_time_label()
        self.position_changed.emit(clamped_ms)

    def closeEvent(self, event: QCloseEvent) -> None:
        self.stop()
        super().closeEvent(event)

    def _stop_stream(self) -> None:
        stream = self._stream
        self._stream = None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except sd.PortAudioError:
                pass

    # -- callback -----------------------------------------------------------

    def _callback(
        self, outdata: np.ndarray, frames: int, time_info: tuple, status: sd.CallbackFlags
    ) -> None:
        pcm = self._pcm
        if pcm is None or pcm.size == 0:
            outdata[:] = 0.0
            return
        ratio = self._playback_ratio
        sr = self._samplerate
        with self._lock:
            start = float(self._pos_frames)
            n_frames = self._n_frames
            volume = self._volume
            if self._loop_enabled and self._loop_end_frames > self._loop_start_frames:
                if start >= self._loop_end_frames or start < self._loop_start_frames:
                    start = float(self._loop_start_frames)
                advance = frames * ratio
                new_pos = start + advance
                span = self._loop_end_frames - self._loop_start_frames
                if span > 0 and new_pos >= self._loop_end_frames:
                    new_pos = self._loop_start_frames + (new_pos - self._loop_end_frames) % span
                self._pos_frames = new_pos
            else:
                self._pos_frames = start + frames * ratio

        channels = int(pcm.shape[1])
        if ratio == 1.0:
            start_int = int(start)
            remaining = max(n_frames - start_int, 0)
            n = min(frames, remaining)
            if n > 0:
                outdata[0:n] = pcm[start_int : start_int + n] * volume
            if n < frames:
                outdata[n:] = 0.0
            return

        outdata[:] = 0.0
        idx = start + np.arange(frames, dtype=np.float64) * ratio
        i0 = np.floor(idx).astype(np.int64)
        frac = (idx - i0).astype(np.float32)
        for c in range(channels):
            col = pcm[:, c]
            i0c = np.clip(i0, 0, n_frames - 1)
            i1c = np.clip(i0 + 1, 0, n_frames - 1)
            vals = col[i0c] * (1.0 - frac) + col[i1c] * frac
            valid = (i0 >= 0) & (i0 < n_frames - 1)
            vals = np.where(valid, vals, 0.0)
            outdata[:, c] = vals * volume

    # -- tick / finish ------------------------------------------------------

    def _on_tick(self) -> None:
        if self._state != "playing":
            return
        pos = self.position_ms
        if self._duration_ms > 0 and pos >= self._duration_ms and not self._loop_enabled:
            self._finish_playback()
            return
        if not self._seeking:
            value = self._ms_to_value(pos)
            self._progress_bar.blockSignals(True)
            try:
                self._progress_bar.setValue(value)
            finally:
                self._progress_bar.blockSignals(False)
        self._sync_time_label()
        self.position_changed.emit(int(pos))

    def _finish_playback(self) -> None:
        self._stop_stream()
        self._tick_timer.stop()
        with self._lock:
            self._pos_frames = self._n_frames
        self._state = "stopped"
        if not self._seeking:
            self._progress_bar.blockSignals(True)
            try:
                self._progress_bar.setValue(self.SLIDER_STEPS)
            finally:
                self._progress_bar.blockSignals(False)
        self._sync_time_label()
        self._update_play_button_text()
        self.position_changed.emit(int(self._duration_ms))
        self.finished.emit()

    # -- ui helpers ---------------------------------------------------------

    def _ms_to_value(self, ms: int) -> int:
        if self._duration_ms <= 0:
            return 0
        value = ms * self.SLIDER_STEPS // self._duration_ms
        return max(0, min(value, self.SLIDER_STEPS))

    def _value_to_ms(self, value: int) -> int:
        if self._duration_ms <= 0:
            return 0
        ms = value * self._duration_ms // self.SLIDER_STEPS
        return max(0, min(ms, self._duration_ms))

    def _sync_time_label(self) -> None:
        current = f"{self._format_ms(self.position_ms)}"
        total = f"{self._format_ms(self._duration_ms)}"
        self._time_label.setText(f"{current} / {total}")

    @staticmethod
    def _format_semitones(semitones: int) -> str:
        if semitones > 0:
            return f"+{semitones} st"
        return f"{semitones} st"

    def retranslate(self) -> None:
        self._volume_label.setText(tr("player.volume", "Volume"))
        self._transpose_label.setText(tr("m4_transpose", "Transpose"))
        self._loop_check.setText(tr("m4_loop", "Loop"))
        self._set_a_btn.setText(tr("m4_set_a", "Set A"))
        self._set_b_btn.setText(tr("m4_set_b", "Set B"))
        self._update_play_button_text()

    def _update_play_button_text(self) -> None:
        if self._state == "playing":
            self._play_pause_btn.setText(tr("player.pause", "Pause"))
        else:
            self._play_pause_btn.setText(tr("player.play", "Play"))

    @staticmethod
    def _format_ms(ms: int) -> str:
        total_seconds = max(0, int(ms)) // 1000
        return f"{total_seconds // 60}:{total_seconds % 60:02d}"

    def _on_play_pause_clicked(self) -> None:
        if self._state == "playing":
            self.pause()
        else:
            self.play()

    def _on_slider_pressed(self) -> None:
        self._seeking = True

    def _on_slider_released(self, value: int) -> None:
        self.set_position(self._value_to_ms(int(value)))
        self._seeking = False

    def _on_progress_changed(self, value: int) -> None:
        if not self._seeking or self._pcm is None:
            return
        self.set_position(self._value_to_ms(int(value)))

    def _on_volume_changed(self, value: int) -> None:
        self.volume = int(value) / 100.0

    def _on_transpose_changed(self, value: int) -> None:
        self.set_transpose(int(value))

    def _on_loop_spin_changed(self, value: int) -> None:
        del value
        if not self._loop_check.isChecked():
            return
        self.set_loop(self._loop_start_spin.value(), self._loop_end_spin.value())

    def _on_set_a_clicked(self) -> None:
        pos = self.position_ms
        self._loop_start_spin.setValue(pos)

    def _on_set_b_clicked(self) -> None:
        pos = self.position_ms
        self._loop_end_spin.setValue(pos)
