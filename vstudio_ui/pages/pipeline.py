"""Pipeline page — run several steps in one pass.

The only page whose step set is the user's choice, so it is also the only one that
has to guard against an empty selection: ``execute()`` raises
``VocalStudioError("No pipeline steps are enabled")``, and finding that out after
pressing Start is worse than being told before.
"""

from __future__ import annotations

from PyQt5.QtWidgets import QCheckBox, QVBoxLayout, QWidget, QToolButton, QHBoxLayout, QLineEdit, QPushButton, QSpinBox, QFileDialog
from PyQt5.QtCore import Qt
from pathlib import Path

from vstudio_ui import services, strings
from vstudio_ui.motion import reveal
from vstudio_ui.pages.task_page import TaskPage
from vstudio_ui.widgets import FieldRow
from vstudio_ui.parameters import SeparationParameters, MidiParameters, AsrParameters, TtsParameters

#: What ``main.py pipeline`` enables when no steps are given: detect the content,
#: then run both branches and let the orchestrator skip the inapplicable one.
DEFAULT_STEPS = ("content_detection", "vocal2midi", "speech_asr")


class PipelinePage(TaskPage):
    key = "pipeline"
    steps = DEFAULT_STEPS
    result_spec = (
        ("audio_type", "pipeline.result_audio_type"),
        ("vocals", "pipeline.result_vocals"),
        ("notes", "pipeline.result_notes"),
        ("text", "pipeline.result_text"),
        ("tts", "pipeline.result_tts"),
    )

    def build_options(self, column: QVBoxLayout) -> None:
        self.step_checks: dict[str, QCheckBox] = {}
        self.step_editors = {}
        self._step_toggles = {}
        editors = {'separation': SeparationParameters, 'vocal2midi': MidiParameters,
                   'speech_asr': AsrParameters, 'tts': TtsParameters}
        self.ref_audio_edit = QLineEdit()
        for name in services.PIPELINE_STEPS:
            check = QCheckBox(strings.tr(f"step.{name}", name))
            check.setChecked(name in DEFAULT_STEPS)
            self.step_checks[name] = check
            line = QHBoxLayout(); line.addWidget(check, 1)
            toggle = QToolButton(); toggle.setText('⌄'); toggle.setCheckable(True)
            toggle.setAccessibleName(strings.tr('pipeline.parameters'))
            toggle.setToolTip(strings.tr('pipeline.parameters')); line.addWidget(toggle)
            self._step_toggles[name] = toggle
            column.addLayout(line)
            host = QWidget(); panel = QVBoxLayout(host); panel.setContentsMargins(12, 4, 0, 12)
            if name in editors:
                editor = editors[name](self.ref_audio_edit if name == 'tts' else self.audio_edit)
                self.step_editors[name] = editor
                if name == 'speech_asr':
                    editor.diarize_row.hide(); editor.speakers_row.hide()
                if name == 'tts':
                    self._reference_row = FieldRow(strings.tr('pipeline.reference'), self.ref_audio_edit)
                    self._reference_browse = QPushButton(strings.tr('task.browse'))
                    self._reference_browse.clicked.connect(self._browse_reference)
                    self._reference_row.add_control(self._reference_browse)
                    panel.addWidget(self._reference_row)
                panel.addWidget(editor)
            elif name == 'diarization':
                self.speakers_spin = QSpinBox(); self.speakers_spin.setRange(0, 20)
                self._speaker_row = FieldRow(strings.tr('asr.num_speakers'), self.speakers_spin)
                panel.addWidget(self._speaker_row)
            else:
                from PyQt5.QtWidgets import QLabel
                label = QLabel(strings.tr('pipeline.detect_hint')); label.setWordWrap(True)
                self._detect_hint = label; panel.addWidget(label)
            toggle.toggled.connect(lambda visible, target=host: reveal(target, visible))
            check.toggled.connect(lambda enabled, t=toggle: t.setChecked(enabled))
            check.toggled.connect(host.setEnabled)
            column.addWidget(host)
            host.setEnabled(check.isChecked()); toggle.setChecked(check.isChecked())
            host.setVisible(check.isChecked())

    def _browse_reference(self):
        path, _ = QFileDialog.getOpenFileName(self, strings.tr('pipeline.reference'), '', 'Audio (*.wav *.mp3 *.flac *.m4a *.ogg)')
        if path: self.ref_audio_edit.setText(path)

    def collect_options(self):
        values = {}
        selected = self.steps_for_run()
        for name, editor in self.step_editors.items():
            if name in selected: values.update(editor.collect_options())
        values['enable_diarization'] = 'diarization' in selected
        if 'diarization' in selected: values['num_speakers'] = self.speakers_spin.value()
        return tuple(values.items())

    def language(self):
        if self.step_checks['speech_asr'].isChecked(): return self.step_editors['speech_asr'].language()
        return 'auto'

    def steps_for_run(self) -> tuple[str, ...]:
        # services.PIPELINE_STEPS is already in the orchestrator's canonical order,
        # which build_steps() would re-impose anyway — but building it in order here
        # keeps the tuple meaningful on its own.
        return tuple(
            name for name in services.PIPELINE_STEPS if self.step_checks[name].isChecked()
        )

    def validate_request(self) -> str:
        problem = super().validate_request()
        if problem:
            return problem
        if not self.steps_for_run():
            return strings.tr("pipeline.no_steps")
        if 'tts' in self.steps_for_run():
            if not self.step_editors['tts'].text_edit.toPlainText().strip(): return strings.tr('tts.missing_text')
            if not Path(self.ref_audio_edit.text().strip()).is_file(): return strings.tr('pipeline.missing_reference')
        return ""

    def render_result(self, result: services.RunResult) -> None:
        self.set_result("audio_type", self.audio_type_label(result.audio_type))
        self.set_result("vocals", result.vocals_path)
        # Only report a note count when the MIDI step actually produced one: a
        # hard 0 from a run that never did note detection reads as a finding.
        self.set_result(
            "notes",
            str(result.notes_count) if result.did_run("vocal2midi") else "",
        )
        self.set_result("text", result.asr_text)
        self.set_result("tts", result.tts_audio_path)

    def retranslate_extra(self) -> None:
        for name, check in self.step_checks.items():
            check.setText(strings.tr(f"step.{name}", name))
            self._step_toggles[name].setAccessibleName(strings.tr('pipeline.parameters'))
        for editor in self.step_editors.values(): editor.retranslate_extra()
        self._reference_row.set_label(strings.tr('pipeline.reference'))
        self._reference_browse.setText(strings.tr('task.browse'))
        self._speaker_row.set_label(strings.tr('asr.num_speakers'))
        self._detect_hint.setText(strings.tr('pipeline.detect_hint'))
