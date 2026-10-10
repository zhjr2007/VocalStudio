"""Voice-clone page.

Step set mirrors ``main.py tts``: ``tts`` alone.

Two things here differ from the other task pages and both come from the
orchestrator's own rules:

* the input file is a **reference clip**, not the thing being processed, so the
  page sets ``ref_audio`` to the same path as ``audio_path``. Without ``ref_audio``
  the orchestrator silently *skips* the TTS step and the run "succeeds" having done
  nothing.
* ``tts_text`` is required for the same reason, so it is validated here rather than
  discovered as a skipped step afterwards.
"""

from __future__ import annotations

from PyQt5.QtWidgets import QComboBox, QDoubleSpinBox, QPlainTextEdit, QVBoxLayout

from vstudio_ui import services, strings
from vstudio_ui.pages.task_page import TaskPage
from vstudio_ui.widgets import FieldRow
from vstudio_ui.parameters import TtsParameters

_TEXT_MIN_LINES = 4


class TtsPage(TaskPage):
    key = "tts"
    steps = ("tts",)
    result_spec = (
        ("audio", "tts.result_audio"),
        ("language", "tts.result_language"),
    )

    def build_options(self, column):
        self._parameters = TtsParameters(self.audio_edit)
        column.addWidget(self._parameters)
        for name, value in vars(self._parameters).items():
            if name not in ("audio_edit",): setattr(self, name, value)

    def validate_request(self) -> str:
        problem = super().validate_request()
        if problem:
            return problem
        if not self.text_edit.toPlainText().strip():
            return strings.tr("tts.missing_text")
        return ""

    def language(self):
        return self._parameters.language()

    def collect_options(self):
        return self._parameters.collect_options()

    def render_result(self, result: services.RunResult) -> None:
        self.set_result("audio", result.tts_audio_path)
        self.set_result("language", result.detected_language)

    def retranslate_extra(self):
        return self._parameters.retranslate_extra()
