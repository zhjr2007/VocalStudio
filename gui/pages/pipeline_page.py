"""PipelinePage: full chained workflow (wave F-1/F-3).

Step cards mirror ``core.orchestrator.STEP_ORDER``; each card is a collapsible
group with an enable checkbox and its own parameters. Start() runs the whole
chain through :class:`core.orchestrator.VocalStudioOrchestrator` on a
JobRunner thread, mapping per-step progress onto one monotone page progress.
"""

from __future__ import annotations

import time as _time
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from gui.i18n import tr
from gui.widgets.audio_input import AudioInputArea
from gui.widgets.info_card import InfoCard
from gui.widgets.job_runner import JobRunner

USAGE_KEYS = ("pipeline_usage_1", "pipeline_usage_2", "pipeline_usage_3")
MODEL_KEYS = (
    "pipeline_models_1",
    "pipeline_models_2",
    "pipeline_models_3",
)

STEP_DEFS = (
    ("separation", "pl_step_separation", "pl_enable", True),
    ("diarization", "pl_step_diarization", "pl_enable", False),
    ("content_detection", "pl_step_content", "pl_auto_detect", True),
    ("vocal2midi", "pl_step_v2midi", "pl_enable", True),
    ("speech_asr", "pl_step_speech_asr", "pl_enable", False),
    ("tts", "pl_step_tts", "pl_enable", False),
)
PRESET_ORDER = ("Karaoke", "FullVocal", "Band4", "Producer5")
SLICE_CHOICES = ("default", "smart", "heuristic", "grid")
QUANT_CHOICES = ("simple", "smart", "bayes", "dp")


class StepCard(QGroupBox):
    """Collapsible pipeline step card: checkbox (enable) + param area."""

    def __init__(self, key: str, title_key: str, check_key: str, checked: bool, parent=None):
        super().__init__(parent)
        self.key = key
        self._title_key = title_key
        self._check_key = check_key
        outer = QVBoxLayout(self)
        head = QHBoxLayout()
        self._check = QCheckBox(self)
        self._check.setChecked(checked)
        head.addWidget(self._check)
        self._title = QLabel(self)
        font = self._title.font()
        font.setBold(True)
        self._title.setFont(font)
        head.addWidget(self._title, 1)
        self._toggle = QToolButton(self)
        self._toggle.setText("▼")
        self._toggle.setCheckable(True)
        self._toggle.setChecked(True)
        self._toggle.setAutoRaise(True)
        head.addWidget(self._toggle)
        outer.addLayout(head)

        self._body = QWidget(self)
        self._body_layout = QVBoxLayout(self._body)
        outer.addWidget(self._body)
        self._toggle.toggled.connect(self._body.setVisible)

    # -- API -------------------------------------------------------------

    def set_title(self, text: str) -> None:
        self._title.setText(text)

    def title(self) -> QLabel:
        return self._title

    def check(self) -> QCheckBox:
        return self._check

    def body(self) -> QWidget:
        return self._body

    def enabled(self) -> bool:
        return self._check.isChecked()

    def set_enabled(self, enabled: bool) -> None:
        self._check.setChecked(bool(enabled))

    def retranslate(self, title_text: str = "", check_text: str = "") -> None:
        self._title.setText(tr(self._title_key) if not title_text else title_text)
        self._check.setText(tr(self._check_key) if not check_text else check_text)


def _hinted(combo: QComboBox, text: str) -> None:
    combo.setToolTip(text)


class PipelinePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self._title_label = QLabel(self)
        title_font = self._title_label.font()
        title_font.setBold(True)
        self._title_label.setFont(title_font)
        layout.addWidget(self._title_label)

        self._input_area = AudioInputArea(self)
        layout.addWidget(self._input_area, 2)

        self._cards: dict[str, StepCard] = {}
        self._params: dict[str, dict] = {}
        for key, title_key, check_key, checked in STEP_DEFS:
            self._cards[key] = StepCard(key, title_key, check_key, checked, self)
            card = self._cards[key]
            p = self._build_step_body(key, card.body())
            self._params[key] = p
            layout.addWidget(card)

        self._run_row = QHBoxLayout()
        self._run_row.addStretch(1)
        self._start_button = QPushButton(self)
        self._start_button.clicked.connect(self.start)
        self._run_row.addWidget(self._start_button)
        self._cancel_button = QPushButton(self)
        self._cancel_button.setEnabled(False)
        self._cancel_button.clicked.connect(self.cancel)
        self._run_row.addWidget(self._cancel_button)
        layout.addLayout(self._run_row)

        progress_row = QHBoxLayout()
        self._progress_label = QLabel(self)
        progress_row.addWidget(self._progress_label)
        self._progress_bar = QProgressBar(self)
        self._progress_bar.setRange(0, 100)
        progress_row.addWidget(self._progress_bar, 1)
        self._status_label = QLabel(self)
        progress_row.addWidget(self._status_label)
        layout.addLayout(progress_row)

        self._log = QPlainTextEdit(self)
        self._log.setReadOnly(True)
        layout.addWidget(self._log, 3)

        self._usage_card = InfoCard("info_usage", USAGE_KEYS, self)
        self._models_card = InfoCard("info_models", MODEL_KEYS, self)
        layout.addWidget(self._usage_card, 1)
        layout.addWidget(self._models_card, 1)

        self._job_runner: JobRunner | None = None
        self._last_progress: float = 0.0
        self._last_message: str = ""
        self._stage_raw: str = ""

        for card in self._cards.values():
            card.check().toggled.connect(self.refresh_step_enabled)

        self.retranslate()
        self.refresh_step_enabled()

    # -- step bodies -------------------------------------------------------

    def _build_step_body(self, key: str, body: QWidget) -> dict:
        p: dict = {}

        def add_row(widget: QWidget) -> None:
            row = QHBoxLayout()
            row.addWidget(widget, 1)
            body.layout().addLayout(row)

        if key == "separation":
            p["preset"] = QComboBox(body)
            for preset in PRESET_ORDER:
                p["preset"].addItem(preset, userData=preset)
            _hinted(p["preset"], tr("stage_label"))
            add_row(p["preset"])
            p["agg"] = QSlider(Qt.Horizontal, body)
            p["agg"].setRange(0, 20)
            add_row(p["agg"])
            p["denoise"] = QCheckBox(body)
            add_row(p["denoise"])
        elif key == "diarization":
            p["spk"] = QSpinBox(body)
            p["spk"].setRange(1, 16)
            p["spk"].setValue(3)
            add_row(p["spk"])
        elif key == "content_detection":
            p["note"] = QLabel(tr("pl_content_note"), body)
            p["note"].setWordWrap(True)
            body.layout().addWidget(p["note"])
        elif key == "vocal2midi":
            p["slicing"] = QComboBox(body)
            for value in SLICE_CHOICES:
                p["slicing"].addItem(value, userData=value)
            _hinted(p["slicing"], tr("midi_slice_mode"))
            add_row(p["slicing"])
            p["quant"] = QComboBox(body)
            for value in QUANT_CHOICES:
                p["quant"].addItem(value, userData=value)
            _hinted(p["quant"], tr("midi_quant_mode"))
            add_row(p["quant"])
            p["bpm"] = QSpinBox(body)
            p["bpm"].setRange(60, 240)
            p["bpm"].setValue(120)
            _hinted(p["bpm"], tr("midi_bpm_label"))
            add_row(p["bpm"])
            p["chords"] = QCheckBox(body)
            add_row(p["chords"])
        elif key == "speech_asr":
            p["engine"] = QComboBox(body)
            for value, tip in (("sensevoice", tr("asr_engine_sensevoice")), ("paraformer", tr("asr_engine_paraformer"))):
                p["engine"].addItem(value, userData=value)
                p["engine"].setItemData(p["engine"].count() - 1, tip, Qt.ToolTipRole)
            add_row(p["engine"])
            p["punc"] = QCheckBox(body)
            p["punc"].setChecked(True)
            add_row(p["punc"])
        elif key == "tts":
            p["version"] = QComboBox(body)
            for value, label in (("v2Pro", "v2Pro"), ("v2ProPlus", "v2ProPlus")):
                p["version"].addItem(label, userData=value)
            add_row(p["version"])
            ref_row = QHBoxLayout()
            p["ref_audio"] = QLineEdit(body)
            ref_row.addWidget(p["ref_audio"], 1)
            p["browse_ref"] = QPushButton(tr("browse"), body)
            p["browse_ref"].clicked.connect(self._on_browse_ref)
            ref_row.addWidget(p["browse_ref"])
            body.layout().addLayout(ref_row)
            p["text"] = QLineEdit(body)
            add_row(p["text"])
            p["speed"] = QSpinBox(body)
            p["speed"].setRange(5, 15)
            p["speed"].setSingleStep(1)
            p["speed"].setValue(10)
            add_row(p["speed"])
        return p


    def _on_browse_ref(self) -> None:
        chosen = QFileDialog.getOpenFileName(
            self, tr("browse"), str(Path.home()), "Audio (*.wav *.mp3 *.flac *.m4a *.ogg)"
        )
        if chosen and chosen[0]:
            card = self._params["tts"]
            card["ref_audio"].setText(chosen[0])

    # -- config --------------------------------------------------------------

    def input_area(self) -> AudioInputArea:
        return self._input_area

    def step_cards(self) -> dict[str, StepCard]:
        return self._cards

    def enabled_steps(self) -> list[str]:
        return [key for key, *_ in STEP_DEFS if self._cards[key].enabled()]

    def set_step_enabled(self, key: str, enabled: bool) -> None:
        self._cards[key].set_enabled(enabled)

    def refresh_step_enabled(self) -> None:
        cards = self._cards
        cards["content_detection"].set_enabled(
            cards["vocal2midi"].enabled() or cards["speech_asr"].enabled()
        )

    def config(self) -> dict:
        p = self._params
        return {
            "enable_separation": p["separation"]["preset"] is not None and self._cards["separation"].enabled(),
            "separation_model": str(p["separation"]["preset"].currentData() or "Karaoke"),
            "separation_agg": p["separation"]["agg"].value(),
            "enable_denoise": p["separation"]["denoise"].isChecked(),
            "enable_diarization": self._cards["diarization"].enabled(),
            "num_speakers": p["diarization"]["spk"].value(),
            "auto_detect_content": self._cards["content_detection"].enabled(),
            "enable_vocal2midi": self._cards["vocal2midi"].enabled(),
            "slicing_method": str(p["vocal2midi"]["slicing"].currentData() or "default"),
            "quantization_mode": str(p["vocal2midi"]["quant"].currentData() or "bayes"),
            "tempo": float(p["vocal2midi"]["bpm"].value()),
            "enable_chords": bool(p["vocal2midi"]["chords"].isChecked()),
            "enable_speech_asr": self._cards["speech_asr"].enabled(),
            "asr_engine": str(p["speech_asr"]["engine"].currentData() or "sensevoice"),
            "enable_punctuation": bool(p["speech_asr"]["punc"].isChecked()),
            "enable_tts": self._cards["tts"].enabled(),
            "tts_version": str(p["tts"]["version"].currentData() or "v2Pro"),
            "ref_audio": p["tts"]["ref_audio"].text().strip(),
            "tts_text": p["tts"]["text"].text().strip(),
            "tts_speed": float(p["tts"]["speed"].value()) / 10.0,
        }

    def validate(self) -> bool:
        cfg = self.config()
        messages: list[str] = []
        if not self.input_area().accepted_count() and not self.input_area().selected_path():
            messages.append(tr("pl_source_missing"))
        if cfg["enable_tts"] and (not cfg["ref_audio"] or not Path(cfg["ref_audio"]).is_file()):
            messages.append(tr("pl_tts_ref_missing"))
        if cfg["enable_tts"] and not cfg["tts_text"]:
            messages.append(tr("pl_tts_text_missing"))
        for message in messages:
            self._append_log(f"error: {message}")
        return not messages

    # -- job flow ------------------------------------------------------------

    def start(self) -> JobRunner | None:
        if self._job_runner is not None and self._job_runner.isRunning():
            return self._job_runner
        path = self.input_area().selected_path() or (
            self.input_area().rows()[0][0] if self.input_area().rows() else ""
        )
        if not path or not Path(path).is_file():
            self._append_log("error: " + tr("pl_source_missing"))
            return None
        if not self.validate():
            return None

        from core.config import PipelineConfig
        from core.orchestrator import VocalStudioOrchestrator

        values = self.config()
        cfg_kwargs = dict(values)
        chord_flag = bool(cfg_kwargs.pop("enable_chords", False))
        cfg = PipelineConfig(
            audio_path=str(path),
            output_dir=str(Path(path).parent / "outputs" / "pipeline"),
            device="auto",
            **cfg_kwargs,
        )
        cfg.enable_chords = chord_flag
        from core.app_settings import get_settings

        cfg.device = get_settings().device or "auto"

        def job(pctx, cfg_object, progress_cb):
            orchestrator = VocalStudioOrchestrator(cfg_object)
            orchestrator.with_progress(progress_cb)
            return orchestrator.execute(pctx)

        runner = JobRunner(job, ctx=None, cfg=cfg, parent=self)
        self._job_runner = runner
        self._last_progress = 0.0
        self._last_message = ""
        self._stage_raw = ""
        self._log.clear()
        runner.progress.connect(self._on_progress)
        runner.log_signal.connect(self._on_log)
        runner.done.connect(self._on_done)
        runner.failed.connect(self._on_failed)
        runner.start_job()
        self._start_button.setEnabled(False)
        self._cancel_button.setEnabled(True)
        return runner

    def cancel(self) -> None:
        if self._job_runner is not None:
            self._job_runner.cancel()
            self._append_log(tr("cancel"))

    def is_running(self) -> bool:
        runner = self._job_runner
        return bool(runner is not None and runner.isRunning())

    def last_progress(self) -> float:
        return self._last_progress

    def last_message(self) -> str:
        return self._last_message

    def log_text(self) -> str:
        return self._log.toPlainText()

    # -- slot handlers --------------------------------------------------------

    def _on_progress(self, value: float, message: str) -> None:
        fraction = max(0.0, min(1.0, float(value)))
        self._last_progress = fraction
        self._progress_bar.setValue(int(round(fraction * 100)))
        if message:
            self._last_message = str(message)
            self._status_label.setText(self._last_message)

    def _append_log(self, text: str) -> None:
        self._log.appendPlainText(str(text))

    def _on_log(self, message: str) -> None:
        self._append_log(message)

    def _on_done(self, result, cancelled: bool) -> None:
        self._start_button.setEnabled(True)
        self._cancel_button.setEnabled(False)
        if cancelled:
            self._append_log("cancelled")
            return
        self._append_log("done: " + tr("pipeline_done"))
        self._progress_bar.setValue(100)

    def _on_failed(self, exc) -> None:
        self._start_button.setEnabled(True)
        self._cancel_button.setEnabled(False)
        self._append_log(f"error: {exc}")

    # -- translate -------------------------------------------------------------

    def retranslate(self) -> None:
        self._title_label.setText(tr("pipeline"))
        self._input_area.retranslate()
        for key, title_key, check_key, _checked in STEP_DEFS:
            card = self._cards[key]
            card.retranslate()
        self._start_button.setText(tr("start"))
        self._cancel_button.setText(tr("cancel"))
        self._progress_label.setText(tr("progress"))
        self._usage_card.retranslate()
        self._models_card.retranslate()
