"""Separation page — UVR5 vocals / accompaniment split.

The simplest of the task pages, and therefore the one that shows the skeleton
works: three knobs, one step, three output paths.
"""

from __future__ import annotations

from PyQt5.QtWidgets import QCheckBox, QComboBox, QSpinBox, QVBoxLayout

from vstudio_ui import services, strings
from vstudio_ui.pages.task_page import TaskPage
from vstudio_ui.widgets import FieldRow
from vstudio_ui.parameters import SeparationParameters

#: The engine's own default, and the one the CLI documents.
DEFAULT_MODEL = "HP5_only_main_vocal"


class SeparationPage(TaskPage):
    key = "separation"
    steps = ("separation",)
    result_spec = (
        ("vocals", "separation.result_vocals"),
        ("accompaniment", "separation.result_accompaniment"),
        ("denoised", "separation.result_denoised"),
    )

    def build_options(self, column):
        self._parameters = SeparationParameters(self.audio_edit)
        column.addWidget(self._parameters)
        for name, value in vars(self._parameters).items():
            if name not in ("audio_edit",): setattr(self, name, value)

    def collect_options(self):
        return self._parameters.collect_options()

    def render_result(self, result: services.RunResult) -> None:
        self.set_result("vocals", result.vocals_path)
        self.set_result("accompaniment", result.accompaniment_path)
        self.set_result("denoised", result.denoised_vocals_path)

    def retranslate_extra(self):
        return self._parameters.retranslate_extra()
