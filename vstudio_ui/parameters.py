"""Reusable parameter editors shared by standalone pages and the pipeline."""
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QComboBox, QCheckBox, QSpinBox,
    QDoubleSpinBox, QLineEdit, QPlainTextEdit)
from vstudio_ui import services, strings
from vstudio_ui.widgets import FieldRow
DEFAULT_MODEL = 'HP5_only_main_vocal'

class ParameterEditor(QWidget):
    def __init__(self, audio_edit=None, parent=None):
        super().__init__(parent)
        self.audio_edit = audio_edit if audio_edit is not None else QLineEdit()
        self._option_rows = []
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(10)
        self.build_options(column)
        FieldRow.align(self._option_rows)

    def add_option_row(self, column, row):
        self._option_rows.append(row)
        column.addWidget(row)

class SeparationParameters(ParameterEditor):
    def build_options(self, column: QVBoxLayout) -> None:
        self.model_combo = QComboBox()
        for name in services.SEPARATION_MODELS:
            self.model_combo.addItem(name, userData=name)
        index = self.model_combo.findData(DEFAULT_MODEL)
        if index >= 0:
            self.model_combo.setCurrentIndex(index)

        self.intensity_spin = QSpinBox()
        self.intensity_spin.setRange(0, 20)
        self.intensity_spin.setValue(0)

        self.denoise_check = QCheckBox(strings.tr("separation.denoise"))

        self.model_row = FieldRow(strings.tr("separation.model"), self.model_combo)
        self.intensity_row = FieldRow(strings.tr("separation.intensity"), self.intensity_spin)
        # Empty label: the checkbox carries its own text, but the row still takes
        # part in the label-column alignment so nothing shifts left.
        self.denoise_row = FieldRow("", self.denoise_check)
        for row in (self.model_row, self.intensity_row, self.denoise_row):
            self.add_option_row(column, row)

    def collect_options(self) -> tuple[tuple[str, object], ...]:
        return (
            ("separation_model", str(self.model_combo.currentData() or DEFAULT_MODEL)),
            ("separation_agg", int(self.intensity_spin.value())),
            ("enable_denoise", bool(self.denoise_check.isChecked())),
        )

    def retranslate_extra(self) -> None:
        self.model_row.set_label(strings.tr("separation.model"))
        self.intensity_row.set_label(strings.tr("separation.intensity"))
        self.denoise_check.setText(strings.tr("separation.denoise"))



class MidiParameters(ParameterEditor):
    def build_options(self, column: QVBoxLayout) -> None:
        self.slicing_combo = QComboBox()
        for name in services.SLICING_METHODS:
            self.slicing_combo.addItem(strings.tr(f"slicing.{name}", name), userData=name)

        self.quant_combo = QComboBox()
        for name in services.QUANTIZATION_MODES:
            self.quant_combo.addItem(strings.tr(f"quantization.{name}", name), userData=name)
        index = self.quant_combo.findData("bayes")
        if index >= 0:
            self.quant_combo.setCurrentIndex(index)

        self.quant_step_spin = QSpinBox()
        self.quant_step_spin.setRange(1, 480)
        self.quant_step_spin.setValue(16)

        self.tempo_spin = QDoubleSpinBox()
        self.tempo_spin.setRange(1.0, 400.0)
        self.tempo_spin.setDecimals(1)
        self.tempo_spin.setValue(120.0)

        self.lyrics_edit = QLineEdit()
        self.lyrics_edit.setPlaceholderText(strings.tr("midi.lyrics_placeholder"))

        self.slicing_row = FieldRow(strings.tr("midi.slicing"), self.slicing_combo)
        self.quant_row = FieldRow(strings.tr("midi.quantization"), self.quant_combo)
        self.quant_step_row = FieldRow(strings.tr("midi.quantization_step"), self.quant_step_spin)
        self.tempo_row = FieldRow(strings.tr("midi.tempo"), self.tempo_spin)
        self.lyrics_row = FieldRow(strings.tr("midi.lyrics"), self.lyrics_edit)
        for row in (
            self.slicing_row,
            self.quant_row,
            self.quant_step_row,
            self.tempo_row,
            self.lyrics_row,
        ):
            self.add_option_row(column, row)

    def collect_options(self) -> tuple[tuple[str, object], ...]:
        return (
            ("slicing_method", str(self.slicing_combo.currentData() or "default")),
            ("quantization_mode", str(self.quant_combo.currentData() or "bayes")),
            ("quantization_step", int(self.quant_step_spin.value())),
            ("tempo", float(self.tempo_spin.value())),
            ("original_lyrics", self.lyrics_edit.text()),
        )

    def retranslate_extra(self) -> None:
        self.slicing_row.set_label(strings.tr("midi.slicing"))
        self.quant_row.set_label(strings.tr("midi.quantization"))
        self.quant_step_row.set_label(strings.tr("midi.quantization_step"))
        self.tempo_row.set_label(strings.tr("midi.tempo"))
        self.lyrics_row.set_label(strings.tr("midi.lyrics"))
        self.lyrics_edit.setPlaceholderText(strings.tr("midi.lyrics_placeholder"))
        for index, name in enumerate(services.SLICING_METHODS):
            self.slicing_combo.setItemText(index, strings.tr(f"slicing.{name}", name))
        for index, name in enumerate(services.QUANTIZATION_MODES):
            self.quant_combo.setItemText(index, strings.tr(f"quantization.{name}", name))



class AsrParameters(ParameterEditor):
    def build_options(self, column: QVBoxLayout) -> None:
        self.engine_combo = QComboBox()
        for name in services.ASR_ENGINES:
            self.engine_combo.addItem(name, userData=name)

        self.lang_combo = QComboBox()
        for name in services.LANGUAGE_CHOICES:
            self.lang_combo.addItem(strings.tr(f"language.{name}", name), userData=name)

        self.punct_check = QCheckBox(strings.tr("asr.punctuation"))
        self.punct_check.setChecked(True)
        self.srt_check = QCheckBox(strings.tr("asr.srt"))
        self.json_check = QCheckBox(strings.tr("asr.json"))
        self.diarize_check = QCheckBox(strings.tr("asr.diarization"))

        self.speakers_spin = QSpinBox()
        self.speakers_spin.setRange(0, 20)
        self.speakers_spin.setValue(0)

        self.engine_row = FieldRow(strings.tr("asr.engine"), self.engine_combo)
        self.lang_row = FieldRow(strings.tr("asr.lang"), self.lang_combo)
        self.punct_row = FieldRow("", self.punct_check)
        self.srt_row = FieldRow("", self.srt_check)
        self.json_row = FieldRow("", self.json_check)
        self.diarize_row = FieldRow("", self.diarize_check)
        self.speakers_row = FieldRow(strings.tr("asr.num_speakers"), self.speakers_spin)
        for row in (
            self.engine_row,
            self.lang_row,
            self.punct_row,
            self.srt_row,
            self.json_row,
            self.diarize_row,
            self.speakers_row,
        ):
            self.add_option_row(column, row)

    def language(self) -> str:
        return str(self.lang_combo.currentData() or "auto")

    def collect_options(self) -> tuple[tuple[str, object], ...]:
        return (
            ("asr_engine", str(self.engine_combo.currentData() or "sensevoice")),
            ("enable_punctuation", bool(self.punct_check.isChecked())),
            ("output_srt", bool(self.srt_check.isChecked())),
            ("output_json", bool(self.json_check.isChecked())),
            # The flag alone would be ignored (see the module docstring); it is set
            # as well so the config is self-consistent for anyone reading it back.
            ("enable_diarization", bool(self.diarize_check.isChecked())),
            ("num_speakers", int(self.speakers_spin.value())),
        )

    def retranslate_extra(self) -> None:
        self.engine_row.set_label(strings.tr("asr.engine"))
        self.lang_row.set_label(strings.tr("asr.lang"))
        self.speakers_row.set_label(strings.tr("asr.num_speakers"))
        self.punct_check.setText(strings.tr("asr.punctuation"))
        self.srt_check.setText(strings.tr("asr.srt"))
        self.json_check.setText(strings.tr("asr.json"))
        self.diarize_check.setText(strings.tr("asr.diarization"))
        for index, name in enumerate(services.LANGUAGE_CHOICES):
            self.lang_combo.setItemText(index, strings.tr(f"language.{name}", name))



class TtsParameters(ParameterEditor):
    def build_options(self, column: QVBoxLayout) -> None:
        self.version_combo = QComboBox()
        for name in services.TTS_VERSIONS:
            self.version_combo.addItem(name, userData=name)

        self.lang_combo = QComboBox()
        for name in services.LANGUAGE_CHOICES:
            if name == "auto":
                continue  # TTS needs a concrete language
            self.lang_combo.addItem(strings.tr(f"language.{name}", name), userData=name)

        self.speed_spin = QDoubleSpinBox()
        # The config's own bound: PipelineConfig.validate() rejects anything else.
        self.speed_spin.setRange(0.5, 1.5)
        self.speed_spin.setSingleStep(0.05)
        self.speed_spin.setDecimals(2)
        self.speed_spin.setValue(1.0)

        self.text_edit = QPlainTextEdit()
        self.text_edit.setPlaceholderText(strings.tr("tts.text_placeholder"))
        self.text_edit.setMinimumHeight(72)

        self.ref_text_edit = QPlainTextEdit()
        self.ref_text_edit.setPlaceholderText(strings.tr("tts.ref_text_placeholder"))
        self.ref_text_edit.setMinimumHeight(56)

        self.version_row = FieldRow(strings.tr("tts.version"), self.version_combo)
        self.lang_row = FieldRow(strings.tr("tts.lang"), self.lang_combo)
        self.speed_row = FieldRow(strings.tr("tts.speed"), self.speed_spin)
        self.text_row = FieldRow(strings.tr("tts.text"), self.text_edit)
        self.ref_text_row = FieldRow(strings.tr("tts.ref_text"), self.ref_text_edit)
        for row in (
            self.version_row,
            self.lang_row,
            self.speed_row,
            self.text_row,
            self.ref_text_row,
        ):
            self.add_option_row(column, row)

    def language(self) -> str:
        return str(self.lang_combo.currentData() or "zh")

    def collect_options(self) -> tuple[tuple[str, object], ...]:
        reference = self.audio_edit.text().strip()
        return (
            ("tts_version", str(self.version_combo.currentData() or "v2Pro")),
            ("tts_text", self.text_edit.toPlainText().strip()),
            ("tts_lang", self.language()),
            ("tts_speed", float(self.speed_spin.value())),
            ("ref_text", self.ref_text_edit.toPlainText().strip()),
            ("ref_lang", self.language()),
            # Same file as audio_path: see the module docstring.
            ("ref_audio", reference),
        )

    def retranslate_extra(self) -> None:
        self.version_row.set_label(strings.tr("tts.version"))
        self.lang_row.set_label(strings.tr("tts.lang"))
        self.speed_row.set_label(strings.tr("tts.speed"))
        self.text_row.set_label(strings.tr("tts.text"))
        self.ref_text_row.set_label(strings.tr("tts.ref_text"))
        self.text_edit.setPlaceholderText(strings.tr("tts.text_placeholder"))
        self.ref_text_edit.setPlaceholderText(strings.tr("tts.ref_text_placeholder"))
        for index, name in enumerate(services.LANGUAGE_CHOICES):
            if name == "auto":
                continue
            item = self.lang_combo.findData(name)
            if item >= 0:
                self.lang_combo.setItemText(item, strings.tr(f"language.{name}", name))

