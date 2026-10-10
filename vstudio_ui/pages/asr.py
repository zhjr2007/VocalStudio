"""Speech-ASR page.

Step set mirrors ``main.py asr``: ``speech_asr``, with ``diarization`` inserted
before it when the speaker count is asked for.

The step list is computed in :meth:`steps_for_run` rather than toggled with
``enable_diarization``: an explicit ``pipeline_steps`` list wins over the
orchestrator's ``enable_*`` flags, so setting the flag alone would do nothing.
"""

from __future__ import annotations

from PyQt5.QtWidgets import QCheckBox, QComboBox, QSpinBox, QVBoxLayout

from vstudio_ui import services, strings
from vstudio_ui.pages.task_page import TaskPage
from vstudio_ui.widgets import FieldRow
from vstudio_ui.parameters import AsrParameters


class AsrPage(TaskPage):
    key = "asr"
    steps = ("speech_asr",)
    result_spec = (
        ("language", "asr.result_language"),
        ("speakers", "asr.result_speakers"),
        ("srt", "asr.result_srt"),
        ("text", "asr.result_text"),
    )

    def build_options(self, column):
        self._parameters = AsrParameters(self.audio_edit)
        column.addWidget(self._parameters)
        for name, value in vars(self._parameters).items():
            if name not in ("audio_edit",): setattr(self, name, value)

    def language(self):
        return self._parameters.language()

    def steps_for_run(self) -> tuple[str, ...]:
        steps: list[str] = []
        if self.diarize_check.isChecked():
            steps.append("diarization")
        steps.append("speech_asr")
        return tuple(steps)

    def collect_options(self):
        return self._parameters.collect_options()

    def render_result(self, result: services.RunResult) -> None:
        self.set_result("language", result.detected_language)
        self.set_result("speakers", " · ".join(result.speakers))
        self.set_result("srt", result.srt_path)
        # The transcript is arbitrarily long, and ValueLabel elides rather than
        # wrapping — the tooltip carries the full text, which is the point.
        self.set_result("text", result.asr_text)

    def retranslate_extra(self):
        return self._parameters.retranslate_extra()
