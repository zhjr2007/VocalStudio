"""StemMixerWidget: live per-stem mixer for separated stems.

Shows one waveform lane per stem (click-to-seek), per-stem Mute / Solo /
volume fader that update the live mix, a karaoke follow-along "You" recording
track and a mixdown (``WAV``) download.

Playback is a single ``sounddevice`` output stream whose callback mixes every
audible stem in real time (mute / solo / gain applied on the fly).  Recording
uses a ``sounddevice`` input stream that accumulates the take in a background
callback thread; the take then becomes a new "You" track aligned to the mix
timeline.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QCloseEvent
from PyQt5.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from gui.i18n import tr
from gui.widgets.waveform_view import TRACK_COLORS, WaveformView, compute_peaks


def _to_stereo(data: np.ndarray) -> np.ndarray:
    """Return a float32 (n, 2) array from any channel layout."""
    arr = np.asarray(data, dtype="float32")
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[1] == 1:
        return np.repeat(arr, 2, axis=1)
    if arr.shape[1] == 2:
        return arr
    return arr[:, :2].copy()


def _resample(data: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Linear resample along time; mono/stereo safe."""
    if src_sr == dst_sr or src_sr <= 0 or dst_sr <= 0:
        return data.astype("float32", copy=False)
    n_in = int(data.shape[0])
    if n_in == 0:
        return data.astype("float32", copy=False)
    n_out = max(1, int(round(n_in * dst_sr / src_sr)))
    old_idx = np.arange(n_in, dtype=np.float64)
    new_idx = np.linspace(0.0, n_in - 1, n_out)
    out = np.empty((n_out, data.shape[1]), dtype="float32")
    for c in range(data.shape[1]):
        out[:, c] = np.interp(new_idx, old_idx, data[:, c])
    return out


class StemMixerWidget(QWidget):
    position_changed = pyqtSignal(int)
    finished = pyqtSignal()
    you_track_ready = pyqtSignal(str)
    mixdown_written = pyqtSignal(str)

    TICK_MS = 40
    SLIDER_STEPS = 1000

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._names: list[str] = []
        self._buffers: list[np.ndarray] = []
        self._gains: list[float] = []
        self._muted: list[bool] = []
        self._solo: list[bool] = []
        self._samplerate: int = 0
        self._n_frames: int = 0
        self._duration_ms: int = 0
        self._pos_frames: int = 0
        self._state: str = "stopped"
        self._volume: float = 0.8
        self._seeking: bool = False
        self._stream: sd.OutputStream | None = None
        self._lock = threading.Lock()

        self._recording: bool = False
        self._record_chunks: list[np.ndarray] = []
        self._record_lock = threading.Lock()
        self._record_stream: sd.InputStream | None = None
        self._record_error: str = ""

        card = QWidget(self)
        card.setObjectName("StemMixerCard")
        card.setStyleSheet(
            "QWidget#StemMixerCard {"
            " background: palette(base);"
            " border: 1px solid palette(midlight);"
            " border-radius: 8px;"
            "}"
        )
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(10, 8, 10, 8)

        self._title_label = QLabel(tr("m4_stem_mixer", "Live stem mixer"), card)
        title_font = self._title_label.font()
        title_font.setBold(True)
        self._title_label.setFont(title_font)
        card_layout.addWidget(self._title_label)

        self._waveform = WaveformView(card)
        self._waveform.setMinimumHeight(120)
        card_layout.addWidget(self._waveform, 2)
        self._waveform.seek_requested.connect(self.set_position)

        self._rows_host = QWidget(card)
        self._rows_layout = QVBoxLayout(self._rows_host)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        card_layout.addWidget(self._rows_host)

        record_row = QHBoxLayout()
        self._record_button = QPushButton(tr("m4_record_you", "Record You"), card)
        self._mixdown_button = QPushButton(tr("m4_mixdown", "Download Mix"), card)
        self._record_status = QLabel("", card)
        record_row.addWidget(self._record_button)
        record_row.addWidget(self._mixdown_button)
        record_row.addWidget(self._record_status)
        record_row.addStretch(1)
        card_layout.addLayout(record_row)

        transport_row = QHBoxLayout()
        self._play_pause_btn = QPushButton(tr("player.play", "Play"), card)
        self._stop_btn = QPushButton(tr("player.stop", "Stop"), card)
        self._progress_bar = QSlider(Qt.Horizontal, card)
        self._progress_bar.setRange(0, 1)
        transport_row.addWidget(self._play_pause_btn)
        transport_row.addWidget(self._stop_btn)
        transport_row.addWidget(self._progress_bar, 1)
        card_layout.addLayout(transport_row)

        time_row = QHBoxLayout()
        time_font = self.font()
        time_font.setPointSize(max(8, time_font.pointSize() - 1))
        self._time_label = QLabel("0:00 / 0:00", card)
        self._time_label.setFont(time_font)
        self._volume_label = QLabel(tr("player.volume", "Volume"), card)
        self._volume_slider = QSlider(Qt.Horizontal, card)
        self._volume_slider.setRange(0, 100)
        self._volume_slider.setValue(int(round(self._volume * 100)))
        time_row.addWidget(self._time_label)
        time_row.addStretch(1)
        time_row.addWidget(self._volume_label)
        time_row.addWidget(self._volume_slider)
        card_layout.addLayout(time_row)

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
        self._record_button.clicked.connect(self._on_record_clicked)
        self._mixdown_button.clicked.connect(self._on_mixdown_clicked)

        self._rows: list[dict] = []

    # -- public API ---------------------------------------------------------

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
        self._volume = max(0.0, min(2.0, float(value)))
        slider_value = int(round(self._volume * 100))
        if self._volume_slider.value() != slider_value:
            self._volume_slider.blockSignals(True)
            try:
                self._volume_slider.setValue(slider_value)
            finally:
                self._volume_slider.blockSignals(False)

    @property
    def recording(self) -> bool:
        return self._recording

    @property
    def stem_names(self) -> list[str]:
        return list(self._names)

    def track_gains(self) -> dict[str, float]:
        """Return the effective audible gain per stem name."""
        gains = self._effective_gains()
        return {name: float(gain) for name, gain in zip(self._names, gains)}

    def set_stems(self, stems: list[dict]) -> None:
        """Load stems. Each dict: name, path and optional pcm/sr."""
        self.stop()
        self._names = []
        self._buffers = []
        self._gains = []
        self._muted = []
        self._solo = []
        self._samplerate = 0

        for meta in stems or []:
            if not isinstance(meta, dict):
                continue
            name = str(meta.get("name") or "stem")
            pcm = meta.get("pcm")
            sr = int(meta.get("sr") or 0)
            if pcm is None and meta.get("path"):
                pcm, sr = sf.read(str(meta["path"]), dtype="float32", always_2d=True)
            if pcm is None:
                continue
            pcm = np.asarray(pcm, dtype="float32")
            if self._samplerate == 0:
                self._samplerate = sr if sr > 0 else 44100
            elif sr > 0 and sr != self._samplerate:
                pcm = _resample(pcm, sr, self._samplerate)
            pcm = _to_stereo(pcm)
            self._names.append(name)
            self._buffers.append(pcm)
            self._gains.append(1.0)
            self._muted.append(False)
            self._solo.append(False)

        if self._buffers:
            self._n_frames = max(int(b.shape[0]) for b in self._buffers)
            padded = []
            for b in self._buffers:
                if b.shape[0] < self._n_frames:
                    pad = np.zeros((self._n_frames - b.shape[0], b.shape[1]), dtype="float32")
                    b = np.concatenate([b, pad], axis=0)
                padded.append(b)
            self._buffers = padded
            self._duration_ms = int(self._n_frames * 1000 / self._samplerate)
            self._rebuild_waveform_and_rows()
            self._reset_transport()
        else:
            self._n_frames = 0
            self._duration_ms = 0
            self._waveform.set_waveforms([])
            self._waveform.set_duration(0)
            self._clear_rows()
            self._reset_transport()

    def waveform_view(self) -> WaveformView:
        return self._waveform

    def row_controls(self) -> list[dict]:
        """Return the per-stem control rows (for tests / introspection)."""
        return list(self._rows)

    def set_mute(self, index: int, muted: bool) -> None:
        if 0 <= index < len(self._muted):
            self._muted[index] = bool(muted)

    def set_solo(self, index: int, solo: bool) -> None:
        if 0 <= index < len(self._solo):
            self._solo[index] = bool(solo)

    def set_gain(self, index: int, gain: float) -> None:
        if 0 <= index < len(self._gains):
            self._gains[index] = max(0.0, float(gain))

    def set_position(self, ms: int) -> None:
        if self._samplerate <= 0 or self._n_frames <= 0:
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

    # -- recording ----------------------------------------------------------

    def record_start(self) -> None:
        if self._recording or not self._buffers:
            return
        self.set_position(0)
        if not self.is_playing:
            self.play()
        self._recording = True
        self._record_error = ""
        self._record_chunks = []
        try:
            self._record_stream = sd.InputStream(
                samplerate=self._samplerate,
                channels=1,
                dtype="float32",
                callback=self._record_callback,
            )
            self._record_stream.start()
        except Exception as exc:  # noqa: BLE001
            self._recording = False
            self._record_error = str(exc)
            self._record_status.setText(tr("m4_record_error", "Recording unavailable"))
        self._record_button.setText(
            tr("m4_stop_recording", "Stop Recording") if self._recording
            else tr("m4_record_you", "Record You")
        )

    def record_stop(self) -> None:
        if not self._recording:
            return
        self._recording = False
        stream = self._record_stream
        self._record_stream = None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
        with self._record_lock:
            chunks = self._record_chunks
            self._record_chunks = []
        if chunks:
            rec = np.concatenate(chunks, axis=0).reshape(-1)
            buf = self._recording_to_buffer(rec)
            self._add_track(str(tr("m4_you_track", "You")), buf)
            self.you_track_ready.emit(self._names[-1])
        self._record_button.setText(tr("m4_record_you", "Record You"))
        if self._record_error:
            self._record_status.setText(tr("m4_record_error", "Recording unavailable"))
        else:
            self._record_status.setText("")

    # -- mixdown ------------------------------------------------------------

    def mixdown(self, path: str) -> str | None:
        if not self._buffers or not path:
            return None
        gains = self._effective_gains()
        mixed = np.zeros((self._n_frames, 2), dtype="float32")
        for buf, gain in zip(self._buffers, gains):
            if gain > 0:
                mixed += buf * gain
        np.clip(mixed, -1.0, 1.0, out=mixed)
        sf.write(str(path), mixed, self._samplerate)
        self.mixdown_written.emit(str(path))
        return str(path)

    # -- transport ----------------------------------------------------------

    def play(self) -> None:
        if self._state == "playing":
            return
        if self._samplerate <= 0 or self._n_frames <= 0 or not self._buffers:
            return
        if self.position_ms >= self._duration_ms:
            with self._lock:
                self._pos_frames = 0
        try:
            stream = sd.OutputStream(
                samplerate=self._samplerate,
                channels=2,
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

    def closeEvent(self, event: QCloseEvent) -> None:
        self.stop()
        if self._recording:
            self.record_stop()
        super().closeEvent(event)

    def retranslate(self) -> None:
        self._title_label.setText(tr("m4_stem_mixer", "Live stem mixer"))
        self._record_button.setText(
            tr("m4_stop_recording", "Stop Recording") if self._recording
            else tr("m4_record_you", "Record You")
        )
        self._mixdown_button.setText(tr("m4_mixdown", "Download Mix"))
        self._volume_label.setText(tr("player.volume", "Volume"))
        self._update_play_button_text()
        for index, row in enumerate(self._rows):
            row["mute"].setText(tr("m4_mute", "Mute") if not row["mute"].isChecked() else tr("m4_unmute", "Unmute"))
            row["solo"].setText(tr("m4_solo", "Solo") if not row["solo"].isChecked() else tr("m4_unsolo", "Un-solo"))
        self._waveform.retranslate()

    # -- internals ----------------------------------------------------------

    def _add_track(self, name: str, buffer: np.ndarray) -> None:
        if not self._buffers:
            self._samplerate = self._samplerate or 44100
            self._n_frames = int(buffer.shape[0])
            self._duration_ms = int(self._n_frames * 1000 / self._samplerate)
        if buffer.shape[0] < self._n_frames:
            pad = np.zeros((self._n_frames - buffer.shape[0], buffer.shape[1]), dtype="float32")
            buffer = np.concatenate([buffer, pad], axis=0)
        self._names.append(name)
        self._buffers.append(buffer)
        self._gains.append(1.0)
        self._muted.append(False)
        self._solo.append(False)
        self._rebuild_waveform_and_rows()

    def _recording_to_buffer(self, rec: np.ndarray) -> np.ndarray:
        rec_stereo = np.repeat(rec[:, None], 2, axis=1) if rec.ndim == 1 else _to_stereo(rec)
        total = self._n_frames if self._n_frames > 0 else int(len(rec_stereo))
        buf = np.zeros((total, 2), dtype="float32")
        n = min(len(rec_stereo), total)
        buf[:n] = rec_stereo[:n]
        return buf

    def _record_callback(self, indata, frames, time_info, status) -> None:
        del frames, time_info, status
        with self._record_lock:
            self._record_chunks.append(np.asarray(indata, dtype="float32").copy())

    def _effective_gains(self) -> list[float]:
        solo_any = any(self._solo)
        gains: list[float] = []
        for i in range(len(self._buffers)):
            if self._muted[i]:
                gains.append(0.0)
            elif solo_any and not self._solo[i]:
                gains.append(0.0)
            else:
                gains.append(self._gains[i] * self._volume)
        return gains

    def _callback(self, outdata, frames, time_info, status) -> None:
        del time_info, status
        with self._lock:
            start = self._pos_frames
            remaining = max(self._n_frames - start, 0)
            n = min(frames, remaining)
            gains = self._effective_gains()
            self._pos_frames += frames
        outdata[:] = 0.0
        if n <= 0:
            return
        for buf, gain in zip(self._buffers, gains):
            if gain <= 0:
                continue
            chunk = buf[start : start + n]
            if chunk.shape[0] > 0:
                outdata[0 : chunk.shape[0]] += chunk * gain

    def _on_tick(self) -> None:
        if self._state != "playing":
            return
        pos = self.position_ms
        if self._duration_ms > 0 and pos >= self._duration_ms:
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

    def _stop_stream(self) -> None:
        stream = self._stream
        self._stream = None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except sd.PortAudioError:
                pass

    def _reset_transport(self) -> None:
        with self._lock:
            self._pos_frames = 0
        self._state = "stopped"
        self._progress_bar.setRange(0, max(self.SLIDER_STEPS, 1))
        self._progress_bar.setValue(0)
        self._sync_time_label()
        self._update_play_button_text()

    def _rebuild_waveform_and_rows(self) -> None:
        tracks: list[dict] = []
        for i, name in enumerate(self._names):
            buf = self._buffers[i]
            mono = buf.mean(axis=1)
            peaks = compute_peaks(mono, self._samplerate, 600)
            tracks.append(
                {"label": name, "peaks": peaks, "color": TRACK_COLORS[i % len(TRACK_COLORS)]}
            )
        self._waveform.set_waveforms(tracks)
        self._waveform.set_duration(self._duration_ms)
        self._clear_rows()
        for i, name in enumerate(self._names):
            self._rows.append(self._build_row(i, name))

    def _build_row(self, index: int, name: str) -> dict:
        row = QWidget(self._rows_host)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 2, 0, 2)
        label = QLabel(name, row)
        label.setMinimumWidth(110)
        mute = QPushButton(tr("m4_mute", "Mute"), row)
        mute.setCheckable(True)
        solo = QPushButton(tr("m4_solo", "Solo"), row)
        solo.setCheckable(True)
        gain = QSlider(Qt.Horizontal, row)
        gain.setRange(0, 150)
        gain.setValue(100)
        gain.setMaximumWidth(220)
        gain_value = QLabel("1.00x", row)
        layout.addWidget(label)
        layout.addWidget(mute)
        layout.addWidget(solo)
        layout.addWidget(gain, 1)
        layout.addWidget(gain_value)
        self._rows_layout.addWidget(row)

        mute.toggled.connect(lambda checked, idx=index: self._on_mute_toggled(idx, checked))
        solo.toggled.connect(lambda checked, idx=index: self._on_solo_toggled(idx, checked))
        gain.valueChanged.connect(
            lambda value, idx=index, lbl=gain_value: self._on_gain_changed(idx, value, lbl)
        )
        return {"index": index, "mute": mute, "solo": solo, "gain": gain, "label": label}

    def _clear_rows(self) -> None:
        while self._rows_layout.count():
            item = self._rows_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._rows = []

    # -- handlers -----------------------------------------------------------

    def _on_mute_toggled(self, index: int, checked: bool) -> None:
        self.set_mute(index, checked)
        self._rows[index]["mute"].setText(
            tr("m4_unmute", "Unmute") if checked else tr("m4_mute", "Mute")
        )

    def _on_solo_toggled(self, index: int, checked: bool) -> None:
        self.set_solo(index, checked)
        self._rows[index]["solo"].setText(
            tr("m4_unsolo", "Un-solo") if checked else tr("m4_solo", "Solo")
        )

    def _on_gain_changed(self, index: int, value: int, label: QLabel) -> None:
        gain = int(value) / 100.0
        self.set_gain(index, gain)
        label.setText(f"{gain:.2f}x")

    def _on_record_clicked(self) -> None:
        if self._recording:
            self.record_stop()
        else:
            self.record_start()

    def _on_mixdown_clicked(self) -> None:
        default = str(Path.home() / "vocalstudio_mix.wav")
        chosen, _ = QFileDialog.getSaveFileName(
            self, tr("m4_save_mix", "Save mixed WAV"), default, "WAV (*.wav)"
        )
        if chosen:
            self.mixdown(chosen)

    def _update_play_button_text(self) -> None:
        if self._state == "playing":
            self._play_pause_btn.setText(tr("player.pause", "Pause"))
        else:
            self._play_pause_btn.setText(tr("player.play", "Play"))

    def _sync_time_label(self) -> None:
        current = self._format_ms(self.position_ms)
        total = self._format_ms(self._duration_ms)
        self._time_label.setText(f"{current} / {total}")

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
        if not self._seeking:
            return
        self.set_position(self._value_to_ms(int(value)))

    def _on_volume_changed(self, value: int) -> None:
        self.volume = int(value) / 100.0
