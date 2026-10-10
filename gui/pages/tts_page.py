"""Voice clone / TTS page (GPT-SoVITS). E-1 controls; E-5 synthesis flow."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import soundfile as sf

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QBrush
from PyQt5.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QProgressBar, QPushButton, QSlider,
    QTextEdit, QVBoxLayout, QWidget,
)

from gui.i18n import tr
from gui.widgets.audio_input import AudioInputArea, REJECTED_GRAY
from gui.widgets.audio_player import AudioPlayerWidget
from gui.widgets.info_card import InfoCard
from gui.widgets.job_runner import JobRunner
from gui.widgets.waveform_view import TRACK_COLORS, WaveformView, compute_peaks

USAGE_KEYS = ("tts_usage_1", "tts_usage_2", "tts_usage_3")
MODEL_KEYS = (
    "tts_models_1",
    "tts_models_2",
    "tts_models_3",
)

LANG_CHOICES = (("zh", "中文"), ("en", "English"), ("ja", "日本語"))
VERSION_CHOICES = (
    ("v2Pro", "GPT-SoVITS v2Pro"),
    ("v2ProPlus", "GPT-SoVITS v2ProPlus"),
)
REF_MIN_SEC, REF_MAX_SEC = 3.0, 10.0
SPEED_MIN, SPEED_MAX, SPEED_DEFAULT = 0.5, 1.5, 1.0

JOB_STAGE_LABEL_KEYS = {
    "load": "tts_stage_load",
    "preprocess": "tts_stage_preprocess",
    "t2s": "tts_stage_t2s",
    "sovits": "tts_stage_sovits",
    "vocoder": "tts_stage_vocoder",
    "save": "tts_stage_save",
}


def audio_duration_sec(path: str) -> float | None:
    """Duration of *path* in seconds, or ``None`` when it cannot be probed."""
    try:
        import soundfile as sf

        return float(sf.info(str(path)).duration)
    except Exception:
        return None


class RefAudioArea(AudioInputArea):
    """Reference-audio queue that also enforces the 3~10 s duration rule.

    Files whose probed duration falls outside [REF_MIN_SEC, REF_MAX_SEC] are
    greyed with a tooltip and not counted as accepted; one ``warning`` signal
    per rejected file. Unprobeable durations (None) pass the extension check only.
    """

    warning = pyqtSignal(str)

    def add_files(self, paths: list[str]) -> int:
        before = self.queue().count()
        accepted = super().add_files(paths)
        for i in range(before, self.queue().count()):
            item = self.queue().item(i)
            path = str(item.data(Qt.UserRole))
            if not self.extension_ok(path):
                continue  # already greyed by the base class
            duration = audio_duration_sec(path)
            if duration is None:
                continue
            if not (REF_MIN_SEC <= duration <= REF_MAX_SEC):
                item.setForeground(QBrush(REJECTED_GRAY))
                reason = tr("tts_ref_bad_duration").format(sec=duration)
                item.setToolTip(f"{path} {reason}")
                self.warning.emit(f"{os.path.basename(path)}: {reason}")
                accepted -= 1
        return max(accepted, 0)

    def ref_valid(self, path: str | None) -> bool:
        if not path or not self.extension_ok(path):
            return False
        duration = audio_duration_sec(path)
        return duration is None or (REF_MIN_SEC <= duration <= REF_MAX_SEC)

    def retranslate(self) -> None:
        super().retranslate()
        self._title_label.setText(tr("ref_audio"))
        self._hint_label.setText(tr("tts_ref_hint"))


class TtsJobCfg:
    """Job specification consumed by :func:`run_tts_job` (JobRunner wiring)."""

    def __init__(self, text="", lang="zh", version="v2Pro", speed=1.0,
                 ref_audio="", output_dir="", device="auto") -> None:
        self.text = str(text)
        self.lang = str(lang or "zh")
        self.version = str(version or "v2Pro")
        self.speed = float(speed or 1.0)
        self.ref_audio = str(ref_audio or "")
        self.output_dir = str(output_dir or "")
        self.device = str(device or "auto")
        self.log_cb = None
        self.cancel_checker = lambda: False


def run_tts_job(ctx: dict, cfg: TtsJobCfg, progress_cb) -> dict:
    """Job body for JobRunner: clones the target text (see voice_clone)."""
    from engines.tts.voice_clone import run_tts_job as _engine_job

    return _engine_job(ctx, cfg, progress_cb)


class TtsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self._title_label = QLabel(self)
        title_font = self._title_label.font()
        title_font.setBold(True)
        self._title_label.setFont(title_font)
        layout.addWidget(self._title_label)

        # -- reference audio (E-1: 3~10 s duration rule) -------------------
        self._ref_area = RefAudioArea(self)
        layout.addWidget(self._ref_area, 2)

        # -- target text ----------------------------------------------------
        self._target_label = QLabel(self)
        layout.addWidget(self._target_label)
        self._target_edit = QTextEdit(self)
        self._target_edit.setMinimumHeight(90)
        layout.addWidget(self._target_edit, 2)

        # -- options row (E-1: language / version / speed) ------------------
        opts = QHBoxLayout()
        self._lang_label = QLabel(self)
        opts.addWidget(self._lang_label)
        self._lang_combo = QComboBox(self)
        for value, label in LANG_CHOICES:
            self._lang_combo.addItem(label, userData=value)
        opts.addWidget(self._lang_combo, 1)

        self._version_label = QLabel(self)
        opts.addWidget(self._version_label)
        self._version_combo = QComboBox(self)
        for value, label in VERSION_CHOICES:
            self._version_combo.addItem(label, userData=value)
        opts.addWidget(self._version_combo, 1)

        self._speed_label = QLabel(self)
        opts.addWidget(self._speed_label)
        self._speed_slider = QSlider(Qt.Horizontal, self)
        self._speed_slider.setRange(int(SPEED_MIN * 20), int(SPEED_MAX * 20))
        self._speed_slider.setValue(int(SPEED_DEFAULT * 20))
        opts.addWidget(self._speed_slider, 1)
        self._speed_value_label = QLabel(self)
        opts.addWidget(self._speed_value_label)
        layout.addLayout(opts)
        self._speed_slider.valueChanged.connect(self._update_speed_label)
        self._update_speed_label()

        # -- synthesis flow (wave E-5) ---------------------------------------
        start_row = QHBoxLayout()
        start_row.addStretch(1)
        self._start_button = QPushButton(self)
        self._start_button.clicked.connect(self.start)
        start_row.addWidget(self._start_button)
        layout.addLayout(start_row)

        progress_row = QHBoxLayout()
        self._progress_label = QLabel(self)
        progress_row.addWidget(self._progress_label)
        self._progress_bar = QProgressBar(self)
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        progress_row.addWidget(self._progress_bar, 1)
        self._status_stage_label = QLabel(self)
        progress_row.addWidget(self._status_stage_label)
        layout.addLayout(progress_row)

        self._result_title = QLabel(self)
        result_font = self._result_title.font()
        result_font.setBold(True)
        self._result_title.setFont(result_font)
        layout.addWidget(self._result_title)

        self._waveform = WaveformView(self)
        layout.addWidget(self._waveform, 3)
        self._player = AudioPlayerWidget(self)
        layout.addWidget(self._player, 2)

        self._job_runner: JobRunner | None = None
        self._progress_events: list[tuple[float, str]] = []
        self._last_progress: float = 0.0
        self._stage_raw = ""
        self._failure_text = ""
        self._output_path = ""
        self._ref_area.warning.connect(lambda message: self._set_status(message))

        self._usage_card = InfoCard("info_usage", USAGE_KEYS, self)
        self._models_card = InfoCard("info_models", MODEL_KEYS, self)
        layout.addWidget(self._usage_card, 1)
        layout.addWidget(self._models_card, 1)

        self.retranslate()

    # -- public API ------------------------------------------------------

    def retranslate(self) -> None:
        self._title_label.setText(tr("tts"))
        self._ref_area.retranslate()
        self._target_label.setText(tr("tts_target_text"))
        self._lang_label.setText(tr("language"))
        self._version_label.setText(tr("tts_version"))
        self._speed_label.setText(tr("speed"))
        self._start_button.setText(tr("tts_synthesize"))
        self._progress_label.setText(tr("progress"))
        self._result_title.setText(tr("tts_result_title"))
        if not self.is_running():
            key = JOB_STAGE_LABEL_KEYS.get(self._stage_raw)
            if key:
                self._status_stage_label.setText(tr(key))
            elif self._stage_raw == "done":
                self._status_stage_label.setText(tr("tts_done"))
            elif self._stage_raw == "failed":
                text = tr("tts_failed") + (f": {self._failure_text}" if self._failure_text else "")
                self._status_stage_label.setText(text)
            else:
                self._status_stage_label.setText(tr("tts_stage_idle"))
        self._waveform.retranslate()
        self._player.retranslate()
        self._usage_card.retranslate()
        self._models_card.retranslate()

    def ref_area(self) -> RefAudioArea:
        return self._ref_area

    def ref_path(self) -> str:
        path = self._ref_area.selected_path()
        return path or ""

    def target_text(self) -> str:
        return self._target_edit.toPlainText().strip()

    def set_target_text(self, text: str) -> None:
        self._target_edit.setPlainText(str(text))

    def lang(self) -> str:
        return str(self._lang_combo.currentData() or "zh")

    def set_lang(self, value: str) -> None:
        for i in range(self._lang_combo.count()):
            if self._lang_combo.itemData(i) == value:
                self._lang_combo.setCurrentIndex(i)
                return

    def version(self) -> str:
        return str(self._version_combo.currentData() or "v2Pro")

    def set_version(self, value: str) -> None:
        for i in range(self._version_combo.count()):
            if self._version_combo.itemData(i) == value:
                self._version_combo.setCurrentIndex(i)
                return

    def speed(self) -> float:
        return self._speed_slider.value() / 20.0

    def set_speed(self, value: float) -> None:
        self._speed_slider.setValue(int(round(float(value) * 20)))

    def config(self) -> dict:
        """Duck-typed settings matching core.config.PipelineConfig TTS fields."""
        return {
            "tts_text": self.target_text(),
            "tts_lang": self.lang(),
            "tts_version": self.version(),
            "tts_speed": self.speed(),
            "ref_audio": self.ref_path(),
            "ref_text": "",
            "ref_lang": self.lang(),
        }

    # -- synthesis flow (wave E-5) ------------------------------------------

    def waveform_view(self) -> WaveformView:
        return self._waveform

    def audio_player(self) -> AudioPlayerWidget:
        return self._player

    def can_start(self) -> bool:
        return bool(self.ref_path()) and bool(self.target_text())

    def is_running(self) -> bool:
        runner = self._job_runner
        return bool(runner is not None and runner.isRunning())

    def progress_events(self) -> list[tuple[float, str]]:
        return list(self._progress_events)

    def last_progress(self) -> float:
        return self._last_progress

    def stage_sequence(self) -> list[str]:
        seen: list[str] = []
        for _value, stage in self._progress_events:
            if stage and stage not in seen:
                seen.append(stage)
        return seen

    def output_path(self) -> str:
        return self._output_path

    def start(self) -> JobRunner | None:
        """Kicks off the voice-clone job; one JobRunner per page run."""
        if self.is_running():
            return self._job_runner
        ref = self.ref_path()
        if not ref or not Path(ref).is_file():
            self._set_status(tr("tts_ref_hint"))
            return None
        text = self.target_text()
        if not text:
            self._set_status(tr("tts_target_text"))
            return None

        from core.app_settings import get_settings
        from core.config import PROJECT_ROOT

        output_dir = str(PROJECT_ROOT / "outputs" / "tts")
        cfg = TtsJobCfg(
            text=text,
            lang=self.lang(),
            version=self.version(),
            speed=self.speed(),
            ref_audio=ref,
            output_dir=output_dir,
            device=get_settings().device or "auto",
        )
        ctx: dict = {"ref": ref, "text": text, "lang": cfg.lang}
        runner = JobRunner(run_tts_job, ctx=ctx, cfg=cfg, parent=self)
        self._job_runner = runner
        self._progress_events = []
        self._last_progress = 0.0
        self._stage_raw = ""
        self._failure_text = ""
        self._output_path = ""
        runner.progress.connect(self._on_job_progress)
        runner.log_signal.connect(self._on_job_log)
        runner.done.connect(self._on_job_done)
        runner.failed.connect(self._on_job_failed)
        runner.start_job()
        self._start_button.setEnabled(False)
        self._progress_bar.setValue(0)
        return runner

    def _set_status(self, message: str) -> None:
        self._status_stage_label.setText(message)

    def _on_job_progress(self, value: float, stage: str) -> None:
        fraction = max(0.0, min(1.0, float(value)))
        self._progress_events.append((fraction, str(stage)))
        self._last_progress = fraction
        self._progress_bar.setValue(int(round(fraction * 100)))
        key = JOB_STAGE_LABEL_KEYS.get(str(stage))
        self._stage_raw = str(stage) if key else self._stage_raw
        self._status_stage_label.setText(tr(key, str(stage)) if key else (str(stage) or ""))

    def _on_job_log(self, message: str) -> None:
        text = str(message)
        if text.startswith("Stage "):
            self._status_stage_label.setText(text)

    def _on_job_done(self, result, cancelled: bool) -> None:
        self._start_button.setEnabled(True)
        if not result or cancelled:
            self._stage_raw = "cancelled"
            return
        out_path = str((result or {}).get("output") or "")
        if not out_path or not Path(out_path).is_file():
            self._stage_raw = "failed"
            self._status_stage_label.setText(tr("tts_failed"))
            return
        data, rate = sf.read(str(out_path), dtype="float32", always_2d=True)
        duration_ms = int(data.shape[0] * 1000 / int(rate)) if int(rate) > 0 else 0
        mono = np.mean(data, axis=1).astype("float32") if data.shape[1] > 1 else data[:, 0].copy()
        peaks = compute_peaks(mono, int(rate), 512)
        self._waveform.set_waveforms(
            [{"label": "tts", "peaks": peaks, "color": TRACK_COLORS[0]}]
        )
        self._waveform.set_duration(duration_ms)
        self._player.set_file(out_path)
        self._output_path = out_path
        self._stage_raw = "done"
        self._status_stage_label.setText(tr("tts_done"))

    def _on_job_failed(self, exc) -> None:
        self._start_button.setEnabled(True)
        self._failure_text = str(exc)
        self._stage_raw = "failed"
        self._status_stage_label.setText(tr("tts_failed") + (f": {exc}" if str(exc) else ""))

    # -- internals -------------------------------------------------------

    def _update_speed_label(self) -> None:
        self._speed_value_label.setText(f"{self.speed():.2f}x")
