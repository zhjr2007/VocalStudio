"""Vocal-to-MIDI page (wave C1).

S1.2 (front half) controls: audio input, language / slicing method /
quantization dropdowns + BPM slider, an output-format checkbox row (MIDI /
TXT / CSV / USTX), an optional chord-analysis card (HPSS + min chord beats),
a start button with progress bar and a run log.

The job runs :func:`engines.vocal2midi.pipeline.run` (pitch-only, optionally
+ chord analysis) on a worker thread through :class:`gui.widgets.job_runner.JobRunner`.
"""

from __future__ import annotations

from pathlib import Path

import soundfile as sf

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from gui.i18n import tr
from gui.widgets.audio_input import AudioInputArea
from gui.widgets.audio_player import AudioPlayerWidget
from gui.widgets.info_card import InfoCard
from gui.widgets.job_runner import JobRunner

USAGE_KEYS = ("midi_usage_1", "midi_usage_2", "midi_usage_3")
MODEL_KEYS = (
    "midi_models_1",
    "midi_models_2",
    "midi_models_3",
    "midi_models_4",
)

LANGUAGE_CHOICES = (("auto", "auto"), ("zh", "zh"), ("en", "en"), ("ja", "ja"))
SLICE_CHOICES = ("default", "smart", "heuristic", "grid")
QUANT_CHOICES = ("simple", "smart", "bayes", "dp")
FORMAT_CHOICES = (("mid", "MIDI"), ("txt", "TXT"), ("csv", "CSV"), ("ustx", "USTX"))

JOB_STAGE_LABEL_KEYS = {
    "load": "midi_stage_load",
    "slice": "midi_stage_slice",
    "pitch": "midi_stage_pitch",
    "quant": "midi_stage_quant",
    "chord": "midi_stage_chord",
    "export": "midi_stage_export",
}


class V2MidiJobCfg:
    """Duck-typed config consumed by ``engines.vocal2midi.pipeline.run``.

    JobRunner wraps the settable ``log_cb`` / ``cancel_checker`` before calling
    the job body; pipeline defaults fill any attribute left as None.
    """

    def __init__(
        self,
        audio_path: str = "",
        output_dir: str = "",
        language: str = "auto",
        slicing_method: str = "default",
        quantization_mode: str = "bayes",
        tempo: float = 120.0,
        output_formats: tuple = ("mid", "txt"),
        enable_chords: bool = False,
        chord_hpss: bool = False,
        min_chord_beats: float = 0.0,
        output_lyrics: bool = True,
        reference_lyrics: str = "",
        output_pitch_curve: bool = False,
        device: str = "auto",
    ) -> None:
        self.audio_path = str(audio_path or "")
        self.out_dir = str(output_dir or "")
        self.key = Path(audio_path).stem if audio_path else "vocal2midi"
        self.language = str(language or "auto")
        self.slicing_method = str(slicing_method or "default")
        self.quantization_mode = str(quantization_mode or "bayes")
        self.tempo = float(tempo or 120.0)
        self.output_formats = tuple(output_formats or ("mid", "txt"))
        self.enable_chords = bool(enable_chords)
        self.chord_hpss = bool(chord_hpss)
        self.min_chord_beats = float(min_chord_beats or 0.0)
        self.output_lyrics = bool(output_lyrics)
        self.reference_lyrics = str(reference_lyrics or "")
        self.output_pitch_curve = bool(output_pitch_curve)
        self.device = str(device or "auto")
        self.log_cb = None
        self.cancel_checker = lambda: False


def run_v2midi_job(ctx, cfg: V2MidiJobCfg, progress_cb):
    """Job body: runs the vocal-to-MIDI pitch (optionally + chord) pipeline.

    Reads ``ctx["input"]``, calls :func:`engines.vocal2midi.pipeline.run`, and
    stores ``notes`` / ``files`` / ``chord_files`` / ``chords`` on ``ctx`` before
    returning it. Emitted progress fractions are monotone and end at 1.0.
    """
    from core.exceptions import CancellationError

    input_path = str(ctx.get("input") or "")
    if not input_path or not Path(input_path).exists():
        raise FileNotFoundError(f"input audio not found: {input_path!r}")

    def check_cancelled() -> None:
        checker = getattr(cfg, "cancel_checker", None)
        if callable(checker) and checker():
            raise CancellationError("vocal2midi cancelled")

    from engines.vocal2midi.pipeline import run as run_pipeline

    frames, rate = sf.read(str(input_path), dtype="float32", always_2d=True)
    waveform = frames.mean(axis=1).astype("float32") if frames.shape[1] > 1 else frames[:, 0].copy()
    ctx["duration_ms"] = int(frames.shape[0] * 1000 / int(rate)) if rate else 0
    check_cancelled()

    out_dir = str(getattr(cfg, "out_dir", "") or (Path(input_path).parent / "outputs" / "vocal2midi"))
    result = run_pipeline(waveform, int(rate), cfg, progress_cb)

    ctx["notes"] = list(result.get("notes") or [])
    ctx["files"] = {fmt: str(path) for fmt, path in (result.get("files") or {}).items()}
    ctx["chord_files"] = {name: str(path) for name, path in (result.get("chord_files") or {}).items()}
    ctx["chords"] = result.get("chords") or None
    ctx["n_chunks"] = result.get("n_chunks") or 0
    ctx["backend"] = result.get("backend") or ""
    return ctx


class V2MidiPage(QWidget):
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

        # -- options row ----------------------------------------------------
        opts = QVBoxLayout()
        self._lang_label = QLabel(self)
        self._lang_combo = QComboBox(self)
        for value, _label in LANGUAGE_CHOICES:
            self._lang_combo.addItem(value, userData=value)

        self._slice_label = QLabel(self)
        self._slice_combo = QComboBox(self)
        for value in SLICE_CHOICES:
            self._slice_combo.addItem(value, userData=value)

        self._quant_label = QLabel(self)
        self._quant_combo = QComboBox(self)
        for value in QUANT_CHOICES:
            self._quant_combo.addItem(value, userData=value)
        self._quant_combo.setCurrentIndex(QUANT_CHOICES.index("bayes"))

        bpm_row = QHBoxLayout()
        self._bpm_label = QLabel(self)
        bpm_row.addWidget(self._bpm_label)
        self._bpm_slider = QSlider(Qt.Horizontal, self)
        self._bpm_slider.setRange(60, 240)
        self._bpm_slider.setValue(120)
        bpm_row.addWidget(self._bpm_slider, 1)
        self._bpm_value_label = QLabel(self)
        bpm_row.addWidget(self._bpm_value_label)
        opts.addLayout(bpm_row)

        opt_row = QHBoxLayout()
        opt_row.addWidget(self._lang_label)
        opt_row.addWidget(self._lang_combo, 1)
        opt_row.addWidget(self._slice_label)
        opt_row.addWidget(self._slice_combo, 1)
        opt_row.addWidget(self._quant_label)
        opt_row.addWidget(self._quant_combo, 1)
        opts.addLayout(opt_row)

        self._formats_label = QLabel(self)
        opts.addWidget(self._formats_label)
        fmt_row = QHBoxLayout()
        self._format_checks: dict[str, QCheckBox] = {}
        for value, label in FORMAT_CHOICES:
            check = QCheckBox(label, self)
            self._format_checks[value] = check
            check.setChecked(value in ("mid", "txt"))
            fmt_row.addWidget(check)
        fmt_row.addStretch(1)
        opts.addLayout(fmt_row)
        layout.addLayout(opts)


        # -- lyric alignment options (C2-1) -----------------------------------
        lyric_row = QHBoxLayout()
        self._lyrics_align_check = QCheckBox(self)
        self._lyrics_align_check.setChecked(True)
        lyric_row.addWidget(self._lyrics_align_check)
        self._pitch_curve_check = QCheckBox(self)
        lyric_row.addWidget(self._pitch_curve_check)
        lyric_row.addStretch(1)
        layout.addLayout(lyric_row)

        self._reference_lyrics_label = QLabel(self)
        layout.addWidget(self._reference_lyrics_label)
        self._reference_lyrics_edit = QPlainTextEdit(self)
        self._reference_lyrics_edit.setPlaceholderText("")
        self._reference_lyrics_edit.setFixedHeight(64)
        layout.addWidget(self._reference_lyrics_edit)

        # -- chord analysis card ---------------------------------------------
        chord_card = QWidget(self)
        chord_card.setObjectName("ChordCard")
        chord_layout = QVBoxLayout(chord_card)
        chord_title = QLabel(self)
        chord_title_font = chord_title.font()
        chord_title_font.setBold(True)
        chord_title.setFont(chord_title_font)
        chord_layout.addWidget(chord_title)
        self._chord_title = chord_title

        chord_opts = QHBoxLayout()
        self._chord_enable = QCheckBox(self)
        chord_opts.addWidget(self._chord_enable)
        self._chord_hpss = QCheckBox(self)
        chord_opts.addWidget(self._chord_hpss)
        self._chord_beats_label = QLabel(self)
        chord_opts.addWidget(self._chord_beats_label)
        self._chord_beats_slider = QSlider(Qt.Horizontal, self)
        self._chord_beats_slider.setRange(0, 16)
        self._chord_beats_slider.setValue(0)
        chord_opts.addWidget(self._chord_beats_slider, 1)
        self._chord_beats_value = QLabel(self)
        chord_opts.addWidget(self._chord_beats_value)
        chord_layout.addLayout(chord_opts)
        layout.addWidget(chord_card, 2)

        # -- output dir + start ----------------------------------------------
        dir_row = QHBoxLayout()
        self._dir_label = QLabel(self)
        dir_row.addWidget(self._dir_label)
        self._dir_edit = QLineEdit(self)
        dir_row.addWidget(self._dir_edit, 1)
        self._browse_button = QPushButton(self)
        dir_row.addWidget(self._browse_button)
        layout.addLayout(dir_row)

        self._start_button = QPushButton(self)
        layout.addWidget(self._start_button)

        # -- progress + log ------------------------------------------------
        progress_row = QHBoxLayout()
        self._progress_label = QLabel(self)
        progress_row.addWidget(self._progress_label)
        self._progress_bar = QProgressBar(self)
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        progress_row.addWidget(self._progress_bar, 1)
        self._status_label = QLabel(self)
        progress_row.addWidget(self._status_label)
        layout.addLayout(progress_row)

        self._log_title = QLabel(self)
        layout.addWidget(self._log_title)
        self._log_view = QPlainTextEdit(self)
        self._log_view.setReadOnly(True)
        self._log_view.setMinimumHeight(150)
        layout.addWidget(self._log_view, 3)

        # -- result player (transpose + segment loop) -----------------------
        self._result_title = QLabel(self)
        result_font = self._result_title.font()
        result_font.setBold(True)
        self._result_title.setFont(result_font)
        layout.addWidget(self._result_title)
        self._player = AudioPlayerWidget(self)
        layout.addWidget(self._player, 2)

        self._usage_card = InfoCard("info_usage", USAGE_KEYS, self)
        self._models_card = InfoCard("info_models", MODEL_KEYS, self)
        layout.addWidget(self._usage_card, 3)
        layout.addWidget(self._models_card, 4)

        self._job_runner: JobRunner | None = None
        self._progress_events: list[tuple[float, str]] = []
        self._last_progress: float = 0.0
        self._result_ctx: dict | None = None
        self._stage_raw: str = ""
        self._failure_text: str = ""

        self._bpm_slider.valueChanged.connect(
            lambda value: self._bpm_value_label.setText(str(value))
        )
        self._chord_beats_slider.valueChanged.connect(
            lambda value: self._chord_beats_value.setText(str(value))
        )
        self._chord_enable.toggled.connect(self._update_start_enabled)
        self._browse_button.clicked.connect(self._on_browse_clicked)
        self._start_button.clicked.connect(self.start)
        self._dir_edit.textChanged.connect(lambda _t: self._update_start_enabled())
        self._update_start_enabled()

        self.retranslate()

    # -- public API ------------------------------------------------------

    def retranslate(self) -> None:
        self._title_label.setText(tr("midi"))
        self._input_area.retranslate()
        self._lang_label.setText(tr("midi_lang"))
        self._slice_label.setText(tr("midi_slice_mode"))
        self._quant_label.setText(tr("midi_quant_mode"))
        self._bpm_label.setText(tr("midi_bpm_label"))
        self._bpm_value_label.setText(str(self.bpm()))
        self._formats_label.setText(tr("midi_output_formats"))
        self._chord_title.setText(tr("midi_chord_card_title"))
        self._chord_enable.setText(tr("midi_chord_enable"))
        self._chord_hpss.setText(tr("midi_chord_hpss"))
        self._chord_beats_label.setText(tr("midi_chord_min_beats"))
        self._chord_beats_value.setText(str(self._chord_beats_slider.value()))
        self._dir_label.setText(tr("output_dir"))
        self._browse_button.setText(tr("browse"))
        self._start_button.setText(tr("start"))
        self._progress_label.setText(tr("progress"))
        self._lyrics_align_check.setText(tr("midi_lyrics_align"))
        self._pitch_curve_check.setText(tr("midi_pitch_curve"))
        self._reference_lyrics_label.setText(tr("midi_reference_lyrics"))
        self._log_title.setText(tr("midi_log_title"))
        self._result_title.setText(tr("m4_v2midi_result_title", "Result player"))
        self._player.retranslate()
        if not self.is_running():
            key = JOB_STAGE_LABEL_KEYS.get(self._stage_raw)
            if key:
                self._status_label.setText(tr(key))
            elif self._stage_raw == "done":
                self._status_label.setText(tr("midi_done"))
            elif self._stage_raw == "failed":
                self._status_label.setText(tr("midi_failed"))
            else:
                self._status_label.setText(tr("midi_stage_idle"))
        self._usage_card.retranslate()
        self._models_card.retranslate()

    def input_area(self) -> AudioInputArea:
        return self._input_area

    def audio_player(self) -> AudioPlayerWidget:
        return self._player

    def add_files(self, paths: list[str]) -> int:
        return self._input_area.add_files(paths)

    def rows(self) -> list[tuple[str, bool]]:
        return self._input_area.rows()

    def selected_path(self) -> str | None:
        return self._input_area.selected_path()

    def language(self) -> str:
        return str(self._lang_combo.currentData() or "auto")

    def set_language(self, value: str) -> None:
        for i in range(self._lang_combo.count()):
            if self._lang_combo.itemData(i) == value:
                self._lang_combo.setCurrentIndex(i)
                return

    def slicing_method(self) -> str:
        return str(self._slice_combo.currentData() or "default")

    def set_slicing_method(self, value: str) -> None:
        for i in range(self._slice_combo.count()):
            if self._slice_combo.itemData(i) == value:
                self._slice_combo.setCurrentIndex(i)
                return

    def quantization_mode(self) -> str:
        return str(self._quant_combo.currentData() or "bayes")

    def set_quantization_mode(self, value: str) -> None:
        for i in range(self._quant_combo.count()):
            if self._quant_combo.itemData(i) == value:
                self._quant_combo.setCurrentIndex(i)
                return

    def bpm(self) -> int:
        return int(self._bpm_slider.value())

    def set_bpm(self, value: int) -> None:
        self._bpm_slider.setValue(int(value))

    def output_formats(self) -> tuple[str, ...]:
        return tuple(value for value, _label in FORMAT_CHOICES if self._format_checks[value].isChecked())

    def set_output_formats(self, values) -> None:
        if isinstance(values, str):
            values = [values]
        wanted = {str(v).lower().lstrip(".") for v in (values or [])}
        for value, _label in FORMAT_CHOICES:
            self._format_checks[value].setChecked(value in wanted)

    def chords_enabled(self) -> bool:
        return self._chord_enable.isChecked()

    def set_chords_enabled(self, enabled: bool) -> None:
        self._chord_enable.setChecked(bool(enabled))

    def chord_hpss(self) -> bool:
        return self._chord_hpss.isChecked()

    def set_chord_hpss(self, enabled: bool) -> None:
        self._chord_hpss.setChecked(bool(enabled))

    def min_chord_beats(self) -> int:
        return int(self._chord_beats_slider.value())

    def set_min_chord_beats(self, value: int) -> None:
        self._chord_beats_slider.setValue(int(value))

    def lyrics_align_enabled(self) -> bool:
        return self._lyrics_align_check.isChecked()

    def set_lyrics_align(self, enabled: bool) -> None:
        self._lyrics_align_check.setChecked(bool(enabled))

    def reference_lyrics(self) -> str:
        return self._reference_lyrics_edit.toPlainText()

    def set_reference_lyrics(self, text: str) -> None:
        self._reference_lyrics_edit.setPlainText(str(text))

    def pitch_curve_enabled(self) -> bool:
        return self._pitch_curve_check.isChecked()

    def set_pitch_curve(self, enabled: bool) -> None:
        self._pitch_curve_check.setChecked(bool(enabled))

    def output_dir(self) -> str:
        return self._dir_edit.text()

    def set_output_dir(self, path: str) -> None:
        self._dir_edit.setText(str(path))

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

    def result_paths(self) -> dict[str, str]:
        return dict(self._result_ctx.get("files") or {}) if self._result_ctx else {}

    def chord_paths(self) -> dict[str, str]:
        return dict(self._result_ctx.get("chord_files") or {}) if self._result_ctx else {}

    def config(self) -> dict:
        return {
            "language": self.language(),
            "slicing_method": self.slicing_method(),
            "quantization_mode": self.quantization_mode(),
            "bpm": self.bpm(),
            "output_formats": self.output_formats(),
            "enable_chords": self.chords_enabled(),
            "chord_hpss": self.chord_hpss(),
            "min_chord_beats": self.min_chord_beats(),
            "output_lyrics": self.lyrics_align_enabled(),
            "reference_lyrics": self.reference_lyrics(),
            "output_pitch_curve": self.pitch_curve_enabled(),
            "output_dir": self.output_dir(),
        }

    # -- internals --------------------------------------------------------

    def _update_start_enabled(self, *_args) -> None:
        self._start_button.setEnabled(bool(self.output_dir().strip()) and not self.is_running())

    def _on_browse_clicked(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, tr("browse"), self.output_dir() or str(Path.home())
        )
        if chosen:
            self.set_output_dir(chosen)

    def start(self) -> JobRunner | None:
        if self.is_running():
            return self._job_runner
        rows = [path for path, ok in self.input_area().rows() if ok]
        selected = self.input_area().selected_path()
        audio_path = ""
        if selected and Path(selected).is_file():
            audio_path = str(selected)
        elif rows:
            audio_path = str(rows[0])
        if not audio_path:
            return None

        from core.app_settings import get_settings

        cfg = V2MidiJobCfg(
            audio_path=audio_path,
            output_dir=self.output_dir(),
            language=self.language(),
            slicing_method=self.slicing_method(),
            quantization_mode=self.quantization_mode(),
            tempo=self.bpm(),
            output_formats=self.output_formats(),
            enable_chords=self.chords_enabled(),
            chord_hpss=self.chord_hpss(),
            min_chord_beats=self.min_chord_beats(),
            output_lyrics=self.lyrics_align_enabled(),
            reference_lyrics=self.reference_lyrics(),
            output_pitch_curve=self.pitch_curve_enabled(),
            device=get_settings().device or "auto",
        )
        ctx: dict = {"input": audio_path}
        runner = JobRunner(run_v2midi_job, ctx=ctx, cfg=cfg, parent=self)
        self._job_runner = runner
        self._progress_events = []
        self._last_progress = 0.0
        self._result_ctx = None
        self._stage_raw = ""
        self._failure_text = ""
        self._log_view.clear()
        runner.progress.connect(self._on_job_progress)
        runner.log_signal.connect(self._append_log)
        runner.done.connect(self._on_job_done)
        runner.failed.connect(self._on_job_failed)
        runner.start_job()
        self._update_start_enabled()
        return runner

    def _append_log(self, message: str) -> None:
        self._log_view.appendPlainText(str(message))

    def _on_job_progress(self, value: float, stage: str) -> None:
        fraction = max(0.0, min(1.0, float(value)))
        self._progress_events.append((fraction, str(stage)))
        self._last_progress = fraction
        self._progress_bar.setValue(int(round(fraction * 100)))
        key = JOB_STAGE_LABEL_KEYS.get(str(stage))
        if key:
            self._stage_raw = str(stage)
            self._status_label.setText(tr(key))

    def _on_job_done(self, result, cancelled: bool) -> None:
        self._update_start_enabled()
        if not result or cancelled:
            if cancelled:
                self._status_label.setText(tr("midi_stage_idle"))
            return
        self._result_ctx = dict(result or {})
        input_path = str((result or {}).get("input") or "")
        if input_path and Path(input_path).is_file():
            self._player.set_file(input_path)
        self._stage_raw = "done"
        self._status_label.setText(tr("midi_done"))
        self._progress_bar.setValue(100)

    def _on_job_failed(self, exc) -> None:
        self._update_start_enabled()
        self._failure_text = str(exc)
        self._stage_raw = "failed"
        self._status_label.setText(tr("midi_failed") + (f": {exc}" if str(exc) else ""))
        self._append_log(str(exc))
