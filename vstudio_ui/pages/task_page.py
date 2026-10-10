"""Shared skeleton for the single-step task pages.

Five of the six remaining pages are the same shape: pick an input file, pick where
the output goes, set a few knobs, press Run, watch progress, read the result back.
Written once here instead of five times, because the parts that are easy to get
subtly wrong — cancel, re-run, who owns the progress bar, what happens when the job
raises — are exactly the parts that would have been copy-pasted five times.

The three rules from ``overview.py`` carry over unchanged:

* **the page never imports ``core``.** Everything goes through
  :mod:`vstudio_ui.services`, which is where the frozen contract is enforced.
* **nothing slow runs on the GUI thread.** A run is handed to
  :class:`~vstudio_ui.jobs.JobHandle`; the page only ever touches the frozen
  :class:`~vstudio_ui.services.RunRequest` and
  :class:`~vstudio_ui.services.RunResult`.
* **a page never holds a ``JobRunner``** — that is ``JobHandle``'s job.

Subclasses declare what makes them different and nothing else::

    class SeparationPage(TaskPage):
        key = "separation"
        steps = ("separation",)
        result_spec = (("vocals", "separation.result_vocals"),)

        def build_options(self, column): ...
        def collect_options(self): ...
        def render_result(self, result): ...

Why the content sits in a scroll area: a task page is four cards, and the base
font is an application setting this package does not own (``settings.font_scale``
is not wired yet, but it is planned). Without the wrapper, raising the font scale
clips the run button off the bottom of the window instead of adding a scrollbar.
"""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtGui import QFontMetrics
from PyQt5.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
    QSplitter, QListWidget, QApplication, QToolButton,
)

from vstudio_ui import services, strings
from vstudio_ui.jobs import JobHandle
from vstudio_ui.theme import app_font
from vstudio_ui.tokens import SPACE
from vstudio_ui.widgets import (
    Card,
    CardHeader,
    FieldRow,
    PageScaffold,
    StatusDot,
    ValueLabel,
)

#: What the file picker offers. Same filter the first-generation GUI used, so no
#: file a user could pick before became unpickable.
AUDIO_FILTER = "Audio (*.wav *.mp3 *.flac *.m4a *.ogg)"

_CARD_PADDING = 14
_CARD_SPACING = 10

#: Lines of log the run card guarantees before it starts compressing. Kept small
#: on purpose: the card grows into whatever space is left, and a larger minimum
#: would push the result card out of the window on a short screen.
_LOG_MIN_LINES = 5

_PENDING = "—"


def _pair(edit: QWidget, button: QWidget) -> QWidget:
    """A line edit with its button pinned to the right, as one FieldRow control."""
    host = QWidget()
    row = QHBoxLayout(host)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(SPACE["xs"])
    row.addWidget(edit, 1)
    row.addWidget(button, 0)
    return host


class TaskPage(PageScaffold):
    """Input card + options card + run card + result card, with a job behind Run."""

    #: Page key. Drives the ``<key>.*`` strings, the nav label and the default
    #: output subdirectory, so it must match the key in ``app.PAGE_SPECS``.
    key: str = ""
    #: Pipeline steps this page runs, in canonical order.
    steps: tuple[str, ...] = ()
    #: Options applied on every run. Subclasses normally add to this from
    #: :meth:`collect_options` instead of overriding it.
    base_options: tuple[tuple[str, object], ...] = ()
    #: ``(row key, strings key)`` for the result card, in display order.
    result_spec: tuple[tuple[str, str], ...] = ()
    #: Mirrors ``PipelineConfig``'s own default so a page that does not expose the
    #: setting does not silently change it.
    output_formats: tuple[str, ...] = ("mid", "txt")
    #: Shown in the result card before the first run.
    result_hint_key: str = "task.result_empty"
    #: True when the input is a directory (training takes a dataset folder, not a
    #: file). Only affects which dialog Browse opens and how validation checks it.
    input_is_dir: bool = False
    #: Devices offered in the combo. ``None`` means all of
    #: ``services.DEVICE_CHOICES``; training narrows it because its adapters treat
    #: anything that is not CUDA as "train on CPU", and offering rocm/zluda/directml
    #: there would silently mean something the label does not say.
    device_choices: tuple[str, ...] | None = None

    def __init__(self, shell=None, parent=None):
        super().__init__(
            strings.tr(f"{self.key}.title"), parent, caption=strings.tr(f"{self.key}.caption")
        )
        self._shell = shell

        self._status = StatusDot("idle")
        self.add_trailing_widget(self._status)

        self._job = JobHandle(self)
        self._job.progress.connect(self._on_progress)
        self._job.log.connect(self._append_log)
        self._job.done.connect(self._on_done)
        self._job.failed.connect(self._on_failed)

        self._auto_output = ""  # last value we filled in ourselves

        host = QWidget()
        self._cards = QVBoxLayout(host)
        self._cards.setContentsMargins(0, 0, SPACE["sm"], 0)
        self._cards.setSpacing(SPACE["md"])

        self._build_input_card()
        self._build_options_card()
        self._build_run_card()
        self._build_result_card()
        self._cards.addStretch(1)
        self._cards.removeWidget(self._run_card)
        self._cards.removeWidget(self._result_card)

        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setWidget(host)
        self._right_host = QWidget()
        self._right_column = QVBoxLayout(self._right_host)
        self._right_column.setContentsMargins(0, 0, 4, 0)
        self._right_column.addWidget(self._result_card)
        self._right_column.addStretch(1)
        self._result_scroll = QScrollArea(self)
        self._result_scroll.setWidgetResizable(True)
        self._result_scroll.setWidget(self._right_host)
        self._result_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._splitter = QSplitter(Qt.Horizontal, self)
        self._splitter.setChildrenCollapsible(False)
        self._splitter.addWidget(self._scroll)
        self._splitter.addWidget(self._result_scroll)
        self._splitter.setSizes([620, 480])
        self._compact = False
        self.content_layout.addWidget(self._splitter, 1)
        self.content_layout.addWidget(self._run_card)
        if shell is not None and hasattr(shell, 'tasks'):
            shell.tasks.changed.connect(self._sync_task_lock)
            shell.capabilities_changed.connect(self._show_capabilities)

        self._set_status("idle")
        self.set_stage(strings.tr("task.idle"))

    # -- construction -------------------------------------------------------

    def _build_input_card(self) -> None:
        card = Card()
        column = QVBoxLayout(card)
        column.setContentsMargins(_CARD_PADDING, _CARD_PADDING, _CARD_PADDING, _CARD_PADDING)
        column.setSpacing(_CARD_SPACING)

        self._input_header = CardHeader(
            strings.tr("task.input_card"),
            caption=strings.tr(f"{self.key}.input_hint", strings.tr("task.input_hint")),
        )
        column.addWidget(self._input_header)

        self.audio_edit = QLineEdit(self)
        self.audio_edit.setPlaceholderText(
            strings.tr(f"{self.key}.input_placeholder", strings.tr("task.input_placeholder"))
        )
        self.audio_edit.textChanged.connect(self._on_audio_changed)
        self.audio_button = QPushButton(strings.tr("task.browse"), self)
        self.audio_button.setCursor(Qt.PointingHandCursor)
        self.audio_button.clicked.connect(self.browse_audio)

        self.output_edit = QLineEdit(self)
        self.output_edit.setPlaceholderText(strings.tr("task.output_placeholder"))
        self.output_button = QPushButton(strings.tr("task.browse"), self)
        self.output_button.setCursor(Qt.PointingHandCursor)
        self.output_button.clicked.connect(self.browse_output)

        self.device_combo = QComboBox(self)
        for name in self.device_options():
            self.device_combo.addItem(strings.tr(f"device.{name}", name), userData=name)

        self._audio_row = FieldRow(
            strings.tr(f"{self.key}.input", strings.tr("task.input")),
            _pair(self.audio_edit, self.audio_button),
        )
        self._output_row = FieldRow(
            strings.tr("task.output"), _pair(self.output_edit, self.output_button)
        )
        self._device_row = FieldRow(strings.tr("task.device"), self.device_combo)

        self._input_rows = [self._audio_row, self._output_row, self._device_row]
        FieldRow.align(self._input_rows)
        for row in self._input_rows:
            column.addWidget(row)

        self._cards.addWidget(card)
        self._input_card = card
        self._capability_hint = ValueLabel('', muted=True)
        self._capability_hint.hide()
        column.addWidget(self._capability_hint)

    def _show_capabilities(self):
        statuses = [s for s in self._shell.capability_snapshot if s.step in self.steps_for_run()]
        self._capability_hint.setVisible(bool(statuses))
        self._capability_hint.set_value(' · '.join(f"{strings.tr('step.'+s.step)}: {strings.tr('capability.'+s.state, s.state)}" for s in statuses), muted=True)
        self._capability_hint.setToolTip('\n'.join(f'{s.step}: {s.detail}' for s in statuses))

    def _build_options_card(self) -> None:
        """Build the page's knobs. Subclasses override; the default adds nothing."""
        card = Card()
        column = QVBoxLayout(card)
        column.setContentsMargins(_CARD_PADDING, _CARD_PADDING, _CARD_PADDING, _CARD_PADDING)
        column.setSpacing(_CARD_SPACING)
        self._options_header = CardHeader(strings.tr("task.options_card"))
        column.addWidget(self._options_header)
        self._option_rows: list[FieldRow] = []
        self.build_options(column)
        FieldRow.align(self._option_rows)
        self._cards.addWidget(card)
        self._options_card = card

    def _build_run_card(self) -> None:
        card = Card()
        column = QVBoxLayout(card)
        column.setContentsMargins(_CARD_PADDING, _CARD_PADDING, _CARD_PADDING, _CARD_PADDING)
        column.setSpacing(_CARD_SPACING)

        self._run_button = QPushButton(strings.tr("task.run"), self)
        self._run_button.setCursor(Qt.PointingHandCursor)
        self._run_button.setProperty('primary', True)
        self._run_button.clicked.connect(self.run)

        self._run_header = CardHeader(strings.tr("task.run_card"), trailing=self._run_button)
        column.addWidget(self._run_header)

        self._stage = ValueLabel(strings.tr("task.idle"), muted=True)

        self._progress = QProgressBar(self)
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setTextVisible(False)
        self._progress.setFixedHeight(18)

        self._cancel_button = QPushButton(strings.tr("task.cancel"), self)
        self._cancel_button.setCursor(Qt.PointingHandCursor)
        self._cancel_button.setEnabled(False)
        self._cancel_button.clicked.connect(self._job.cancel)

        controls = QHBoxLayout()
        controls.setSpacing(SPACE["sm"])
        controls.addWidget(self._stage, 1)
        controls.addWidget(self._cancel_button, 0)
        column.addLayout(controls)
        column.addWidget(self._progress)

        self.log = QPlainTextEdit(self)
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(400)
        self.log.setPlaceholderText(strings.tr("task.log_empty"))
        self.log.setVisible(False)
        self._log_toggle = QToolButton(self)
        self._log_toggle.setText(strings.tr('task.logs'))
        self._log_toggle.setCheckable(True)
        from vstudio_ui.motion import reveal
        self._log_toggle.toggled.connect(lambda visible: reveal(self.log, visible))
        controls.insertWidget(1, self._log_toggle)
        column.addWidget(self.log)

        self._cards.addWidget(card, 1)
        self._run_card = card

    def refresh_theme(self) -> None:
        """Restyle, and recompute the log box's minimum height.

        The height is measured from a font, so it goes stale the moment the font
        scale changes: the application font grows but the minimum stays at the old
        value, and the log ends up showing three lines instead of five — the exact
        defect that made the height dynamic in the first place.
        """
        super().refresh_theme()
        self.log.setFixedHeight(min(180, QFontMetrics(app_font()).lineSpacing() * _LOG_MIN_LINES + 10))
        if hasattr(self, '_compact'): self.resizeEvent(None)

    def _build_result_card(self) -> None:
        card = Card()
        column = QVBoxLayout(card)
        column.setContentsMargins(_CARD_PADDING, _CARD_PADDING, _CARD_PADDING, _CARD_PADDING)
        column.setSpacing(_CARD_SPACING)

        self._result_header = CardHeader(strings.tr("task.result_card"))
        column.addWidget(self._result_header)

        self._result_hint = ValueLabel(strings.tr(self.result_hint_key), muted=True)
        column.addWidget(self._result_hint)

        self._result_rows: dict[str, FieldRow] = {}
        self._result_values: dict[str, ValueLabel] = {}
        rows = []
        for row_key, strings_key in self.result_spec:
            value = ValueLabel(_PENDING, muted=True)
            row = FieldRow(strings.tr(strings_key), value)
            self._result_rows[row_key] = row
            self._result_values[row_key] = value
            rows.append(row)
        FieldRow.align(rows)
        for row in rows:
            column.addWidget(row)

        self._cards.addWidget(card)
        self._result_card = card
        self.backend_summary = ValueLabel('', muted=True)
        column.addWidget(self.backend_summary)
        self.transcript = QPlainTextEdit(self)
        self.transcript.setReadOnly(True)
        self.transcript.setMinimumHeight(120)
        self.transcript.setVisible(self.key in ('asr', 'pipeline'))
        column.addWidget(self.transcript)
        self.file_list = QListWidget(self)
        self.file_list.setMinimumHeight(130)
        column.addWidget(self.file_list)
        self.file_list.itemDoubleClicked.connect(lambda item: self._open_selected())
        actions = QHBoxLayout()
        self._file_actions = []
        for key, slot in [('task.open_dir', self._open_directory), ('task.copy_path', self._copy_path), ('task.export', self._export_selected)]:
            button = QPushButton(strings.tr(key), self)
            button.clicked.connect(slot); actions.addWidget(button)
            self._file_actions.append((key, button))
        column.addLayout(actions)
        from vstudio_ui.widgets.audio_preview import AudioPreview
        self.audio_preview = AudioPreview(self)
        self.file_list.currentRowChanged.connect(self._select_audio)
        column.addWidget(self.audio_preview)
        self._preview_input = QPushButton(strings.tr('task.preview_input'), self)
        self._preview_input.clicked.connect(lambda: self.audio_preview.set_path(self.audio_edit.text().strip()))
        column.addWidget(self._preview_input)
        self._last_result = None

    def resizeEvent(self, event):
        if event is not None: super().resizeEvent(event)
        from vstudio_ui.theme import current_font_scale
        compact = self.width() < 1120 or current_font_scale() >= 1.5
        if hasattr(self, '_compact') and compact != self._compact:
            self._compact = compact
            if compact:
                self._right_column.removeWidget(self._result_card)
                self._cards.insertWidget(self._cards.count()-1, self._result_card)
            else:
                self._cards.removeWidget(self._result_card)
                self._right_column.insertWidget(0, self._result_card)
            self._result_scroll.setVisible(not compact)

    def _selected_path(self):
        item = self.file_list.currentItem()
        return item.data(Qt.UserRole) if item is not None else ''

    def _select_audio(self, index): self.audio_preview.set_path(self._selected_path())
    def _open_selected(self):
        path = self._selected_path()
        if path: QDesktopServices.openUrl(QUrl.fromLocalFile(path))
    def _open_directory(self):
        path = self._last_result.output_dir if self._last_result else ''
        if path: QDesktopServices.openUrl(QUrl.fromLocalFile(path))
    def _copy_path(self): QApplication.clipboard().setText(self._selected_path())
    def _export_selected(self):
        path = self._selected_path()
        if not path: return
        target, _ = QFileDialog.getSaveFileName(self, strings.tr('task.export'), Path(path).name)
        if target:
            import shutil
            try: shutil.copy2(path, target)
            except OSError as exc: self._append_log(services.short_error(exc))

    def _sync_task_lock(self):
        tasks = self._shell.tasks
        self._run_button.setEnabled(not tasks.busy and not self._job.running)

    # -- subclass hooks -----------------------------------------------------

    def build_options(self, column: QVBoxLayout) -> None:
        """Add this page's knobs to *column*; register rows in ``_option_rows``."""

    def collect_options(self) -> tuple[tuple[str, object], ...]:
        """Options to apply on top of :attr:`base_options`."""
        return ()

    def render_result(self, result: services.RunResult) -> None:
        """Fill the result card from a finished run."""

    def language(self) -> str:
        """Pipeline language. Pages with a language control override this."""
        return "auto"

    def steps_for_run(self) -> tuple[str, ...]:
        """Steps to run. Pages whose step set depends on a control override this.

        Passing an explicit list to the orchestrator takes priority over its
        ``enable_*`` flags, so a page that exposes a per-step checkbox must compute
        the list here — setting ``enable_diarization`` alone would be ignored.
        """
        return self.steps

    def extra_request_fields(self) -> dict:
        """Anything else on ``RunRequest`` the page wants to set."""
        return {}

    def make_job(self):
        """Return ``(fn, ctx, cfg)`` for one run.

        The default is the pipeline orchestrator, which is what five of the six
        pages need. Training overrides it: it has its own job body and its own
        spec object, and pretending otherwise would mean a pipeline config with a
        ``task`` field bolted on.
        """
        return services.run_pipeline_job, None, services.build_config(self.build_request())

    def add_option_row(self, column: QVBoxLayout, row: FieldRow) -> None:
        """Register an option row so :meth:`retranslate` and alignment see it."""
        self._option_rows.append(row)
        column.addWidget(row)

    # -- public API ---------------------------------------------------------

    def set_audio_path(self, path: str) -> None:
        """Set the input path, and follow it with a default output directory.

        The output field is only overwritten while it still holds a value *we*
        suggested — a directory the user typed is never clobbered.
        """
        path = str(path or "")
        self.audio_edit.setText(path)
        current = self.output_edit.text().strip()
        if current and current != self._auto_output:
            return
        suggestion = self.suggest_output_dir(path)
        self._auto_output = suggestion
        self.output_edit.setText(suggestion)

    def suggest_output_dir(self, input_path: str) -> str:
        """Default output directory for *input_path*. Overridable."""
        return services.suggest_output_dir(input_path, self.key)

    def device_options(self) -> tuple[str, ...]:
        """Device names this page offers."""
        return self.device_choices or ('inherit', *services.DEVICE_CHOICES)

    def set_output_dir(self, path: str) -> None:
        self._auto_output = ""
        self.output_edit.setText(str(path or ""))

    def validate_request(self) -> str:
        """Return a localised problem description, or ``""`` when runnable."""
        audio = self.audio_edit.text().strip()
        if not audio:
            return strings.tr("task.missing_input")
        path = Path(audio)
        if self.input_is_dir:
            if not path.is_dir():
                return strings.tr("task.input_not_dir").format(path=audio)
        elif not path.is_file():
            return strings.tr("task.input_not_found").format(path=audio)
        if not self.output_edit.text().strip():
            return strings.tr("task.missing_output")
        return ""

    def build_request(self) -> services.RunRequest:
        return services.RunRequest(
            audio_path=self.audio_edit.text().strip(),
            output_dir=self.output_edit.text().strip(),
            device=str(self.device_combo.currentData() or "auto"),
            language=self.language(),
            steps=self.steps_for_run(),
            output_formats=self.output_formats,
            options=tuple(self.base_options) + tuple(self.collect_options()),
            **self.extra_request_fields(),
        )

    def run(self) -> None:
        """Validate, build the job and hand it to a worker thread."""
        if self._job.running:
            return
        problem = self.validate_request()
        if problem:
            self._set_status("failed")
            self.set_stage(problem)
            self._append_log(problem)
            return
        try:
            fn, ctx, cfg = self.make_job()
        except Exception as exc:  # noqa: BLE001 - a bad option is a user-visible problem
            self._set_status("failed")
            self.set_stage(services.short_error(exc))
            self._append_log(services.short_error(exc))
            return

        self.log.clear()
        if self._shell is not None and hasattr(self._shell, 'tasks'):
            if not self._shell.tasks.acquire(self.key, self._job):
                self.set_stage(strings.tr('task.busy'))
                return
        self._set_status("running")
        self._set_buttons(running=True)
        self._progress.setValue(0)
        self.set_stage(strings.tr("task.running"))
        if not self._job.start(fn, ctx, cfg):
            if self._shell is not None and hasattr(self._shell, 'tasks'): self._shell.tasks.release()
            self._finish_run()

    def shutdown(self) -> None:
        """Ask a running job to stop and give it a short grace period."""
        if self._job.running:
            self._job.cancel()
        self._job.wait(2000)

    # -- job plumbing -------------------------------------------------------

    def _on_progress(self, fraction: float, stage: str) -> None:
        self._progress.setValue(int(max(0.0, min(1.0, float(fraction))) * 100))
        if stage:
            self.set_stage(stage)
            self._append_log(stage)

    def _on_done(self, result, cancelled: bool) -> None:
        self._finish_run()
        if cancelled:
            self._set_status("idle")
            self.set_stage(strings.tr("task.cancelled"))
            self._append_log(strings.tr("task.cancelled"))
            return
        if not isinstance(result, services.RunResult):
            self._set_status("failed")
            self.set_stage(strings.tr("task.failed"))
            return
        # The run reached the end, so the bar is 100% by definition. Engines report
        # step-local fractions and are not obliged to land exactly on 1.0, so
        # without this the bar sits at whatever the last report was while the label
        # says "done" — a screenshot-level contradiction.
        self._progress.setValue(100)
        self.render_result(result)
        self._last_result = result
        self._result_hint.set_value(strings.tr('task.result_files').format(count=len(result.produced_paths())), muted=True)
        self.backend_summary.set_value(' · '.join(f'{strings.tr(f"step.{step}", step)}: {backend}' for step, backend in result.backends), muted=True)
        self.transcript.setPlainText(result.asr_text)
        self.file_list.clear()
        from PyQt5.QtWidgets import QListWidgetItem
        for label, path in result.produced_paths():
            item = QListWidgetItem(f'{label}  ·  {Path(path).name}')
            item.setData(Qt.UserRole, path); item.setToolTip(path)
            self.file_list.addItem(item)
        if self.file_list.count(): self.file_list.setCurrentRow(0)
        if result.backends: self._append_log('\n'.join(f'{s}: {b}' for s, b in result.backends))
        if result.error:
            # amber, not red: the run reached the end and produced something
            # readable. Training is the realistic case — it reports a non-zero
            # exit code rather than raising, and that is not the same as a crash.
            self._append_log(result.error)
            self.set_stage(result.error)
        else:
            # The stage label is a headline, so it gets the localised summary; the
            # orchestrator's own "[1/6] separation done" lines stay in the log,
            # where mixing raw step ids into a localised UI is appropriate.
            self.set_stage(strings.tr("task.done").format(count=len(result.ran)))
            self._append_log(strings.tr("task.done").format(count=len(result.ran)))
        self._set_status("done" if result.ok else "ready")

    def _on_failed(self, exc) -> None:
        self._finish_run()
        message = services.short_error(exc)
        self._set_status("failed")
        self.set_stage(strings.tr("task.failed"))
        self._append_log(strings.tr("task.failed_detail").format(message=message))

    def _finish_run(self) -> None:
        self._set_buttons(running=False)
        if self._shell is not None and hasattr(self._shell, 'tasks'): self._sync_task_lock()

    # -- rendering ----------------------------------------------------------

    def set_result(self, row_key: str, value: str, *, muted: bool | None = None) -> None:
        label = self._result_values.get(row_key)
        if label is not None:
            label.set_value(value or _PENDING, muted=muted)

    @staticmethod
    def audio_type_label(value: str) -> str:
        """Localised audio-type name; falls back to the raw value if it is new."""
        return strings.tr(f"audio_type.{value}", value) if value else ""

    def clear_results(self) -> None:
        for label in self._result_values.values():
            label.set_value(_PENDING, muted=True)

    def set_stage(self, text: str) -> None:
        self._stage.set_value(text, muted=True)

    def _set_status(self, state: str) -> None:
        self._status.set_state(state)
        self._status.setToolTip(strings.tr(f"status.{state}"))

    def _set_buttons(self, *, running: bool) -> None:
        self._run_button.setEnabled(not running)
        self._cancel_button.setEnabled(running)
        for widget in (self.audio_edit, self.output_edit, self.audio_button, self.output_button):
            widget.setEnabled(not running)

    def _append_log(self, line: str) -> None:
        if line:
            self.log.appendPlainText(line)

    # -- browsing -----------------------------------------------------------

    def browse_audio(self) -> None:
        start = self.audio_edit.text().strip() or str(Path.home())
        if self.input_is_dir:
            chosen = QFileDialog.getExistingDirectory(
                self, strings.tr("task.pick_input"), start
            )
        else:
            chosen, _filter = QFileDialog.getOpenFileName(
                self, strings.tr("task.pick_input"), start, AUDIO_FILTER
            )
        if chosen:
            self.set_audio_path(chosen)

    def browse_output(self) -> None:
        start = self.output_edit.text().strip() or str(Path.home())
        chosen = QFileDialog.getExistingDirectory(self, strings.tr("task.pick_output"), start)
        if chosen:
            self.set_output_dir(chosen)

    def _on_audio_changed(self, text: str) -> None:
        if text.strip() and not self.output_edit.text().strip():
            self.set_audio_path(text)

    # -- i18n ---------------------------------------------------------------

    def retranslate(self) -> None:
        self.set_title(strings.tr(f"{self.key}.title"))
        self.set_caption(strings.tr(f"{self.key}.caption"))
        self._input_header.set_title(strings.tr("task.input_card"))
        self._input_header.set_caption(
            strings.tr(f"{self.key}.input_hint", strings.tr("task.input_hint"))
        )
        self._options_header.set_title(strings.tr("task.options_card"))
        self._run_header.set_title(strings.tr("task.run_card"))
        self._result_header.set_title(strings.tr("task.result_card"))
        self._audio_row.set_label(strings.tr(f"{self.key}.input", strings.tr("task.input")))
        self._output_row.set_label(strings.tr("task.output"))
        self._device_row.set_label(strings.tr("task.device"))
        self.audio_edit.setPlaceholderText(
            strings.tr(f"{self.key}.input_placeholder", strings.tr("task.input_placeholder"))
        )
        self.output_edit.setPlaceholderText(strings.tr("task.output_placeholder"))
        self.audio_button.setText(strings.tr("task.browse"))
        self.output_button.setText(strings.tr("task.browse"))
        self._run_button.setText(strings.tr("task.run"))
        self._cancel_button.setText(strings.tr("task.cancel"))
        self._log_toggle.setText(strings.tr('task.logs'))
        self._preview_input.setText(strings.tr('task.preview_input'))
        for key, button in self._file_actions: button.setText(strings.tr(key))
        self.log.setPlaceholderText(strings.tr("task.log_empty"))
        self._result_hint.set_value(strings.tr(self.result_hint_key), muted=True)
        for index, name in enumerate(self.device_options()):
            self.device_combo.setItemText(index, strings.tr(f"device.{name}", name))
        for row_key, strings_key in self.result_spec:
            row = self._result_rows.get(row_key)
            if row is not None:
                row.set_label(strings.tr(strings_key))
        self.retranslate_extra()
        self._set_status(self._status.state)
        self.refresh_theme()

    def retranslate_extra(self) -> None:
        """Re-label the page's own knobs. Subclasses override."""
