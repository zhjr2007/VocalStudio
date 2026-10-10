"""Speech ASR page (wave D).

D-1 controls: engine combo (SenseVoice / Paraformer) with per-item tooltips,
speaker-separation checkbox and a 0~16 speaker-count spinbox that stays
disabled while separation is off.
"""
from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from gui.i18n import tr
from gui.widgets.info_card import InfoCard

USAGE_KEYS = ("asr_usage_1", "asr_usage_2", "asr_usage_3")
MODEL_KEYS = (
    "asr_models_1",
    "asr_models_2",
    "asr_models_3",
)

ENGINE_CHOICES = (("sensevoice", "SenseVoice"), ("paraformer", "Paraformer"))
ENGINE_TOOLTIP_KEYS = {
    "sensevoice": "asr_engine_sensevoice",
    "paraformer": "asr_engine_paraformer",
}
FORMAT_CHOICES = (("txt", "fmt_txt"), ("srt", "fmt_srt"), ("json", "fmt_json"))


class AsrPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self._title_label = QLabel(self)
        title_font = self._title_label.font()
        title_font.setBold(True)
        self._title_label.setFont(title_font)
        layout.addWidget(self._title_label)

        # -- options row (D-1: engine + speaker separation) ------------------
        opts = QVBoxLayout()
        opt_row = QHBoxLayout()
        self._engine_label = QLabel(self)
        opt_row.addWidget(self._engine_label)
        self._engine_combo = QComboBox(self)
        for value, label in ENGINE_CHOICES:
            self._engine_combo.addItem(label, userData=value)
        # PyQt5 QComboBox has no setItemToolTip; use the model role. The
        # internal model only exists once the first item is added.
        combo_model = self._engine_combo.model()
        for i in range(self._engine_combo.count()):
            value = str(self._engine_combo.itemData(i))
            combo_model.setData(
                combo_model.index(i, 0), tr(ENGINE_TOOLTIP_KEYS[value]), Qt.ToolTipRole
            )
        opt_row.addWidget(self._engine_combo, 1)

        self._speaker_sep_check = QCheckBox(self)
        opt_row.addWidget(self._speaker_sep_check)

        self._spk_num_label = QLabel(self)
        opt_row.addWidget(self._spk_num_label)
        self._spk_num_spin = QSpinBox(self)
        self._spk_num_spin.setRange(0, 16)
        self._spk_num_spin.setValue(3)
        opt_row.addWidget(self._spk_num_spin)
        opts.addLayout(opt_row)

        # -- punctuation + output formats (D-2) ----------------------------
        self._punct_check = QCheckBox(self)
        self._punct_check.setChecked(True)  # punctuation restore on by default
        opts.addWidget(self._punct_check)

        fmt_row = QHBoxLayout()
        self._formats_label = QLabel(self)
        fmt_row.addWidget(self._formats_label)
        self._format_checks: dict[str, QCheckBox] = {}
        for value, _key in FORMAT_CHOICES:
            check = QCheckBox(self)
            self._format_checks[value] = check
            check.setChecked(value == "txt")  # TXT is the default output
            fmt_row.addWidget(check)
        fmt_row.addStretch(1)
        opts.addLayout(fmt_row)
        layout.addLayout(opts)

        self._speaker_sep_check.toggled.connect(self._update_spk_enabled)
        self._update_spk_enabled()  # initial state (checkbox defaults off)

        self._usage_card = InfoCard("info_usage", USAGE_KEYS, self)
        self._models_card = InfoCard("info_models", MODEL_KEYS, self)
        layout.addWidget(self._usage_card, 3)
        layout.addWidget(self._models_card, 4)

        self.retranslate()

    # -- public API ------------------------------------------------------

    def retranslate(self) -> None:
        self._title_label.setText(tr("asr"))
        combo_model = self._engine_combo.model()
        for i in range(combo_model.rowCount()):
            value = str(self._engine_combo.itemData(i))
            combo_model.setData(
                combo_model.index(i, 0), tr(ENGINE_TOOLTIP_KEYS[value]), Qt.ToolTipRole
            )
        self._engine_label.setText(tr("engine"))
        self._speaker_sep_check.setText(tr("speaker_sep"))
        self._spk_num_label.setText(tr("num_speakers"))
        self._punct_check.setText(tr("punctuation"))
        self._formats_label.setText(tr("asr_output_formats"))
        for value, key in FORMAT_CHOICES:
            self._format_checks[value].setText(tr(key))
        self._usage_card.retranslate()
        self._models_card.retranslate()

    def engine(self) -> str:
        return str(self._engine_combo.currentData() or "sensevoice")

    def set_engine(self, value: str) -> None:
        for i in range(self._engine_combo.count()):
            if self._engine_combo.itemData(i) == value:
                self._engine_combo.setCurrentIndex(i)
                return

    def speaker_sep_enabled(self) -> bool:
        return self._speaker_sep_check.isChecked()

    def set_speaker_sep(self, enabled: bool) -> None:
        self._speaker_sep_check.setChecked(bool(enabled))

    def spk_num(self) -> int:
        return int(self._spk_num_spin.value())

    def set_spk_num(self, value: int) -> None:
        self._spk_num_spin.setValue(int(value))

    def punctuation_enabled(self) -> bool:
        return self._punct_check.isChecked()

    def set_punctuation(self, enabled: bool) -> None:
        self._punct_check.setChecked(bool(enabled))

    def output_formats(self) -> tuple[str, ...]:
        return tuple(
            value for value, _key in FORMAT_CHOICES if self._format_checks[value].isChecked()
        )

    def set_output_formats(self, values) -> None:
        if isinstance(values, str):
            values = [values]
        wanted = {str(v).lower().lstrip(".") for v in (values or [])}
        for value, _key in FORMAT_CHOICES:
            self._format_checks[value].setChecked(value in wanted)

    def config(self) -> dict:
        """Duck-typed settings matching core.config.PipelineConfig ASR fields."""
        formats = set(self.output_formats())
        return {
            "asr_engine": self.engine(),
            "enable_punctuation": self.punctuation_enabled(),
            "output_srt": "srt" in formats,
            "output_json": "json" in formats,
            "num_speakers": self.spk_num() if self.speaker_sep_enabled() else 0,
        }

    # -- internals -------------------------------------------------------

    def _update_spk_enabled(self) -> None:
        self._spk_num_spin.setEnabled(self.speaker_sep_enabled())
