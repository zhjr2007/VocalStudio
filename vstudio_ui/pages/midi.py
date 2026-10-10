"""Vocal-to-MIDI page.

Step set mirrors ``main.py transcribe``: ``vocal2midi`` alone. Separation is
available on its own page, so this page does not re-offer it — a user who wants
the split runs it first and feeds the vocal in.
"""

from __future__ import annotations

from PyQt5.QtWidgets import QComboBox, QDoubleSpinBox, QLineEdit, QSpinBox, QVBoxLayout

from vstudio_ui import services, strings
from vstudio_ui.pages.task_page import TaskPage
from vstudio_ui.widgets import FieldRow
from vstudio_ui.parameters import MidiParameters


class MidiPage(TaskPage):
    key = "midi"
    steps = ("vocal2midi",)
    result_spec = (
        ("audio_type", "midi.result_audio_type"),
        ("language", "midi.result_language"),
        ("notes", "midi.result_notes"),
    )

    def build_options(self, column):
        self._parameters = MidiParameters(self.audio_edit)
        column.addWidget(self._parameters)
        for name, value in vars(self._parameters).items():
            if name not in ("audio_edit",): setattr(self, name, value)

    def collect_options(self):
        return self._parameters.collect_options()

    def render_result(self, result: services.RunResult) -> None:
        self.set_result("audio_type", self.audio_type_label(result.audio_type))
        self.set_result("language", result.detected_language)
        # A note count of 0 is only meaningful if the MIDI step ran.
        self.set_result(
            "notes", str(result.notes_count) if result.did_run("vocal2midi") else ""
        )

    def retranslate_extra(self):
        return self._parameters.retranslate_extra()
