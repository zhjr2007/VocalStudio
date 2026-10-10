"""Overview page — environment, device and model readiness, plus a live self-check.

This is the first page wired to the business layer, and it is where the two rules
that make the rest of the rebuild safe are established:

**Nothing slow runs on the GUI thread.** ``detect_system_hardware`` shells out to
WMI and ``auto_select_device`` imports torch; together they can take seconds on a
cold process. Both run through :class:`~vstudio_ui.jobs.JobHandle`, so the window
paints immediately and fills in when the answer arrives. The page never calls a
probe inline — not even the first one, which is why the initial probe is
``auto_start()`` rather than work done in ``__init__``.

**The page reads the business layer only through** :mod:`vstudio_ui.services`.
``core`` is not imported here; that keeps the frozen contract
(``ui-design/contract.md``) enforced in one place.

The self-check button is not decoration: it walks all six contract surfaces and
prints one line per stage, which is the fastest way to tell a broken install from
a broken page.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFontMetrics
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget, QLabel, QScrollArea, QBoxLayout, QFileDialog, QToolButton,
)

from vstudio_ui import services, strings
from vstudio_ui.jobs import JobHandle
from vstudio_ui.tokens import SPACE
from vstudio_ui.widgets import (
    Card,
    CardHeader,
    FieldRow,
    HudRing,
    PageScaffold,
    StatusDot,
    ValueLabel,
)

#: Placeholder shown in a row that has no value yet. A dash, not "N/A": the point
#: is that the value is *pending*, and the dash reads the same in both languages.
_PENDING = "—"

_CARD_PADDING = 14
_CARD_SPACING = 10

#: How many log lines a full self-check produces: one per contract stage plus the
#: summary. Derived rather than hard-coded so adding a stage to ``CHECK_ORDER``
#: cannot quietly push a line out of sight.
_LOG_LINES = len(services.CHECK_ORDER) + 1


class OverviewPage(PageScaffold):
    def __init__(self, shell=None, parent=None):
        super().__init__(
            strings.tr("overview.title"), parent, caption=strings.tr("overview.caption")
        )
        self._shell = shell

        self._status = StatusDot("idle")
        self.add_trailing_widget(self._status)

        # Two independent jobs: re-probing must not be blocked by a running
        # self-check, and vice versa.
        self._probe = JobHandle(self)
        self._probe.progress.connect(self._on_progress)
        self._probe.done.connect(self._on_probe_done)
        self._probe.failed.connect(self._on_probe_failed)

        self._check = JobHandle(self)
        self._check.progress.connect(self._on_progress)
        self._check.log.connect(self._append_log)
        self._check.done.connect(self._on_check_done)
        self._check.failed.connect(self._on_check_failed)

        self._build_runtime_card()
        self._build_check_card()
        # Keep the probe/check widgets, but compose them as a launcher dashboard.
        runtime = self.content_layout.takeAt(0).widget()
        check = self.content_layout.takeAt(0).widget()
        host = QWidget(); body = QVBoxLayout(host); body.setContentsMargins(0, 0, 6, 0)
        self._hero = QLabel(strings.tr('overview.hero')); self._hero.setWordWrap(True)
        self._hero_caption = QLabel(strings.tr('overview.hero_caption'))
        body.addSpacing(24); body.addWidget(self._hero); body.addWidget(self._hero_caption)
        body.addSpacing(90)
        self._panels = QBoxLayout(QBoxLayout.LeftToRight)
        self._panels.setSpacing(16); self._panels.addWidget(runtime, 1); self._panels.addWidget(check, 1)
        body.addLayout(self._panels); body.addStretch(1)
        self._dashboard_scroll = QScrollArea(); self._dashboard_scroll.setWidgetResizable(True)
        self._dashboard_scroll.setWidget(host); self._dashboard_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.content_layout.addWidget(self._dashboard_scroll, 1)
        footer = QHBoxLayout()
        self._session_label = QPushButton(strings.tr('overview.session_empty'))
        self._session_label.clicked.connect(self._show_last_result)
        footer.addWidget(self._session_label, 1)
        self._import_button = QPushButton(strings.tr('overview.import'))
        self._import_button.setProperty('primary', True); self._import_button.clicked.connect(self._import_audio)
        footer.addWidget(self._import_button)
        self.content_layout.addLayout(footer)
        self._capabilities = JobHandle(self)
        self._capabilities.done.connect(self._on_capabilities)
        if shell is not None and hasattr(shell, 'tasks'):
            shell.tasks.changed.connect(self._update_session)
        self._log.setVisible(False)
        self._check_log_toggle = QToolButton()
        self._check_log_toggle.setText(strings.tr('task.logs')); self._check_log_toggle.setCheckable(True)
        from vstudio_ui.motion import reveal
        self._check_log_toggle.toggled.connect(lambda visible: reveal(self._log, visible))
        check.layout().addWidget(self._check_log_toggle)
        self.refresh_theme()

        self._set_status("idle")
        self._set_stage(strings.tr("overview.not_probed"))

    def refresh_theme(self):
        super().refresh_theme()
        from vstudio_ui.theme import color_hex, current_font_scale
        from vstudio_ui.tokens import TYPE_SCALE
        self._hero.setStyleSheet(f"font-size: {TYPE_SCALE['display']}pt; font-weight: 700; color: {color_hex('foreground')};")
        self._hero_caption.setStyleSheet(f"font-size: {TYPE_SCALE['body']}pt; color: {color_hex('muted_foreground')};")
        self._panels.setDirection(QBoxLayout.TopToBottom if self.width() < 1120 or current_font_scale() >= 1.5 else QBoxLayout.LeftToRight)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, '_panels'): self.refresh_theme()

    def _import_audio(self):
        path, _ = QFileDialog.getOpenFileName(self, strings.tr('overview.import'), '', 'Audio (*.wav *.mp3 *.flac *.m4a *.ogg)')
        if path and self._shell:
            self._shell.page('pipeline').set_audio_path(path)
            self._shell.navigate(self._shell.page_keys().index('pipeline'))

    def _show_last_result(self):
        if self._shell and self._shell.tasks.results:
            key, _ = self._shell.tasks.results[-1]
            self._shell.navigate(self._shell.page_keys().index(key))

    def _update_session(self):
        task = self._shell.tasks
        if task.busy:
            self._session_label.setText(f"{strings.NAV_LABELS.get(task.owner)} · {task.fraction:.0%} · {task.stage}")
        elif task.results:
            key, result = task.results[-1]
            self._session_label.setText(f"{strings.tr('overview.session')}  {len(task.results):02d}  /  {strings.NAV_LABELS.get(key)}  ·  {len(result.produced_paths())} files")
        else: self._session_label.setText(strings.tr('overview.session_empty'))

    def _on_capabilities(self, result, cancelled):
        if not cancelled and self._shell and isinstance(result, tuple):
            self._shell.capability_snapshot = result
            self._shell.capabilities_changed.emit()

    # -- construction -------------------------------------------------------

    def _build_runtime_card(self) -> None:
        card = Card()
        column = QVBoxLayout(card)
        column.setContentsMargins(_CARD_PADDING, _CARD_PADDING, _CARD_PADDING, _CARD_PADDING)
        column.setSpacing(_CARD_SPACING)

        self._refresh_button = QPushButton(strings.tr("overview.refresh"), self)
        self._refresh_button.setCursor(Qt.PointingHandCursor)
        self._refresh_button.clicked.connect(self.refresh_runtime)

        self._runtime_header = CardHeader(
            strings.tr("overview.device_card"),
            caption="device / models",
            trailing=self._refresh_button,
        )
        column.addWidget(self._runtime_header)

        self._cpu_value = ValueLabel()
        self._gpu_value = ValueLabel()
        self._vram_value = ValueLabel()
        self._backend_value = ValueLabel()
        self._device_value = ValueLabel()
        self._models_value = ValueLabel()

        self._cpu_row = FieldRow(strings.tr("overview.cpu"), self._cpu_value)
        self._gpu_row = FieldRow(strings.tr("overview.gpu"), self._gpu_value)
        self._vram_row = FieldRow(strings.tr("overview.vram"), self._vram_value)
        self._backend_row = FieldRow(strings.tr("overview.backend"), self._backend_value)
        self._device_row = FieldRow(strings.tr("overview.device"), self._device_value)
        self._models_row = FieldRow(strings.tr("overview.models"), self._models_value)

        self._rows = [
            self._cpu_row,
            self._gpu_row,
            self._vram_row,
            self._backend_row,
            self._device_row,
            self._models_row,
        ]
        FieldRow.align(self._rows)
        for row in self._rows:
            column.addWidget(row)

        self._values = (
            self._cpu_value,
            self._gpu_value,
            self._vram_value,
            self._backend_value,
            self._device_value,
            self._models_value,
        )

        self.content_layout.addWidget(card)

    def _build_check_card(self) -> None:
        card = Card()
        column = QVBoxLayout(card)
        column.setContentsMargins(_CARD_PADDING, _CARD_PADDING, _CARD_PADDING, _CARD_PADDING)
        column.setSpacing(_CARD_SPACING)

        self._check_header = CardHeader(
            strings.tr("overview.selfcheck_card"), caption=strings.tr("overview.selfcheck_hint")
        )
        column.addWidget(self._check_header)

        row = QHBoxLayout()
        row.setSpacing(SPACE["md"])

        self.ring = HudRing(diameter=64)
        row.addWidget(self.ring, 0, Qt.AlignTop)

        state_column = QVBoxLayout()
        state_column.setSpacing(SPACE["xs"])
        self._stage = ValueLabel(strings.tr("overview.not_probed"), muted=True)
        self._progress = QProgressBar(self)
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setTextVisible(False)
        self._progress.setFixedHeight(18)
        state_column.addWidget(self._stage)
        state_column.addWidget(self._progress)
        row.addLayout(state_column, 1)

        buttons = QVBoxLayout()
        buttons.setSpacing(SPACE["xs"])
        self._run_button = QPushButton(strings.tr("overview.run_check"), self)
        self._run_button.setCursor(Qt.PointingHandCursor)
        self._run_button.clicked.connect(self.run_self_check)
        self._cancel_button = QPushButton(strings.tr("overview.cancel"), self)
        self._cancel_button.setCursor(Qt.PointingHandCursor)
        self._cancel_button.setEnabled(False)
        self._cancel_button.clicked.connect(self._check.cancel)
        buttons.addWidget(self._run_button)
        buttons.addWidget(self._cancel_button)
        row.addLayout(buttons, 0)

        column.addLayout(row)

        self._log = QPlainTextEdit(self)
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(400)
        self._log.setPlaceholderText(strings.tr("overview.log_empty"))
        # Sized from the font, not in pixels: a hard-coded height fits four lines
        # at the system font size on this machine, so a finished self-check would
        # need scrolling to be read. The base font is an app-level setting this
        # page does not own, so ask the metrics.
        self._log.setFixedHeight(
            QFontMetrics(self._log.font()).lineSpacing() * (_LOG_LINES + 1) + 12
        )
        column.addWidget(self._log)

        self.content_layout.addWidget(card, 1)

    # -- public API ---------------------------------------------------------

    def auto_start(self) -> None:
        """Run the first probe. Called by ``app.main()``, deliberately not by ``__init__``.

        Constructing the shell happens in tests, on machines without PowerShell,
        and during screenshots; a WMI round trip per construction would make all
        three slower and less predictable for no benefit.
        """
        self.refresh_runtime()
        self._capabilities.start(services.as_job(services.capabilities))

    def refresh_runtime(self) -> None:
        """Re-read hardware / backend / models on a worker thread."""
        if self._probe.running:
            return
        self._set_status("running")
        self._set_progress(0.0, "hardware")
        for value in self._values:
            value.set_value(_PENDING, muted=True)
        self._probe.start(services.as_job(services.probe_runtime))

    def run_self_check(self) -> None:
        """Walk every contract surface and print the result to the log."""
        if self._check.running:
            return
        self._log.clear()
        self._set_status("running")
        self._set_buttons(running=True)
        self._set_progress(0.0, "hardware")
        self.ring.start()
        if not self._check.start(services.as_job(services.run_self_check)):
            self._finish_check()

    def shutdown(self) -> None:
        """Stop the animation and give running jobs a moment to observe the cancel.

        Cooperative cancellation cannot interrupt a blocking business-layer call
        (the WMI subprocess is the realistic worst case), so this waits briefly and
        then lets the process exit rather than blocking the window close.
        """
        self.ring.stop()
        for handle in (self._probe, self._check):
            if handle.running:
                handle.cancel()
        for handle in (self._probe, self._check):
            handle.wait(2000)

    # -- job plumbing -------------------------------------------------------

    def _on_progress(self, fraction: float, stage: str) -> None:
        self._set_progress(fraction, stage)

    def _set_progress(self, fraction: float, stage: str) -> None:
        self._progress.setValue(int(max(0.0, min(1.0, float(fraction))) * 100))
        if stage:
            self._set_stage(strings.tr(f"check.{stage}"))

    def _set_stage(self, text: str) -> None:
        self._stage.set_value(text, muted=True)

    def _on_probe_done(self, result, cancelled: bool) -> None:
        # The bar belongs to the self-check; a finished probe must not leave it
        # sitting at 100% under a "not probed yet" label.
        if not self._check.running:
            self._progress.setValue(0)
        if cancelled or not isinstance(result, services.RuntimeSnapshot):
            self._set_status("idle")
            return
        self._apply_snapshot(result)
        if result.error:
            self._set_status("failed")
            self._set_stage(
                strings.tr("overview.probe_failed").format(message=result.error)
            )
            return
        # amber = "works, but not fully set up"; green = everything present.
        self._set_status("done" if result.models_ok else "ready")
        self._set_stage(strings.tr("overview.not_probed"))

    def _on_probe_failed(self, exc) -> None:
        if not self._check.running:
            self._progress.setValue(0)
        self._set_status("failed")
        self._set_stage(
            strings.tr("overview.probe_failed").format(message=services.short_error(exc))
        )

    def _on_check_done(self, result, cancelled: bool) -> None:
        self._finish_check()
        if cancelled:
            self._append_log(strings.tr("check.cancelled"))
            self._set_stage(strings.tr("check.cancelled"))
            self._set_status("idle")
            return
        if not isinstance(result, services.SelfCheckReport):
            self._set_stage(strings.tr("check.summary_failed").format(message="?"))
            self._set_status("failed")
            return
        for check in result.checks:
            marker = strings.tr("check.pass") if check.ok else strings.tr("check.fail")
            self._append_log(f"[{marker}] {strings.tr(f'check.{check.key}')}: {check.detail}")
        passed = sum(1 for check in result.checks if check.ok)
        summary = strings.tr("check.summary").format(passed=passed, total=len(result.checks))
        self._append_log(summary)
        # The stage label is a *progress* readout; leaving the last stage name
        # ("settings") under a finished ring reads as if it were still working.
        self._set_stage(summary)
        self._apply_snapshot(result.snapshot)
        self._set_status("done" if result.ok else "ready")

    def _on_check_failed(self, exc) -> None:
        self._finish_check()
        message = services.short_error(exc)
        self._append_log(strings.tr("check.summary_failed").format(message=message))
        self._set_stage(strings.tr("check.summary_failed").format(message=message))
        self._set_status("failed")

    def _finish_check(self) -> None:
        self.ring.stop()
        self._set_buttons(running=False)

    # -- rendering ----------------------------------------------------------

    def _apply_snapshot(self, snapshot: services.RuntimeSnapshot) -> None:
        self._cpu_value.set_value(snapshot.cpu or _PENDING, muted=not snapshot.cpu)

        # Prefer torch's answer over WMI's list: WMI reports every adapter —
        # emulator virtual displays included — in arbitrary order, and its
        # AdapterRAM is a 32-bit field that saturates at 4 GB. The full list is
        # kept as a tooltip so nothing is lost, but the headline is the adapter
        # that will actually do the work.
        if snapshot.gpu_name:
            self._gpu_value.set_value(snapshot.gpu_name)
        elif snapshot.gpu_names:
            self._gpu_value.set_value(" · ".join(snapshot.gpu_names))
        else:
            self._gpu_value.set_value(strings.tr("overview.gpu_none"), muted=True)
        self._gpu_value.setToolTip("\n".join(self._gpu_tooltip(snapshot)))

        if snapshot.has_vram:
            self._vram_value.set_value(
                strings.tr("overview.vram_value").format(
                    free=f"{snapshot.vram_free_gb:.1f}",
                    total=f"{snapshot.vram_total_gb:.1f}",
                )
            )
        else:
            self._vram_value.set_value(_PENDING, muted=True)

        self._backend_value.set_value(snapshot.backend)
        if snapshot.backend_source == "torch":
            self._backend_row.set_hint(
                strings.tr("overview.backend_torch").format(version=snapshot.torch_version or "?")
            )
            self._backend_value.setToolTip("")
        else:
            self._backend_row.set_hint(strings.tr("overview.backend_wmi"))
            self._backend_value.setToolTip(strings.tr("overview.backend_wmi_tip"))

        self._device_value.set_value(snapshot.device or _PENDING, muted=not snapshot.device)
        self._device_value.setToolTip(strings.tr("overview.device_tip"))
        self._device_row.set_hint(
            strings.tr("overview.device_hint").format(value=snapshot.device_setting)
        )

        if snapshot.models_total == 0:
            self._models_value.set_value(_PENDING, muted=True)
        elif snapshot.models_ok:
            self._models_value.set_value(
                strings.tr("overview.models_ready").format(
                    ready=snapshot.models_ready, total=snapshot.models_total
                )
            )
            self._models_value.setToolTip("")
        else:
            self._models_value.set_value(
                strings.tr("overview.models_missing").format(count=len(snapshot.models_missing))
            )
            self._models_value.setToolTip("\n".join(snapshot.models_missing))

    @staticmethod
    def _gpu_tooltip(snapshot: services.RuntimeSnapshot) -> list[str]:
        """Every adapter WMI reported, with the 4 GB caveat spelled out."""
        lines = []
        for name, vram in snapshot.gpus:
            if vram:
                lines.append(f"{name}  (~{vram:.0f} GB reported by WMI, capped at 4 GB)")
            else:
                lines.append(name)
        if snapshot.gpu_name and snapshot.gpu_name not in [n for n, _v in snapshot.gpus]:
            lines.insert(0, snapshot.gpu_name)
        return lines

    def _set_status(self, state: str) -> None:
        self._status.set_state(state)
        self._status.setToolTip(strings.tr(f"status.{state}"))

    def _set_buttons(self, *, running: bool) -> None:
        self._run_button.setEnabled(not running)
        self._cancel_button.setEnabled(running)
        self._refresh_button.setEnabled(not running)

    def _append_log(self, line: str) -> None:
        self._log.appendPlainText(line)

    # -- i18n ---------------------------------------------------------------

    def retranslate(self) -> None:
        self._hero.setText(strings.tr('overview.hero'))
        self._hero_caption.setText(strings.tr('overview.hero_caption'))
        self._import_button.setText(strings.tr('overview.import'))
        self._check_log_toggle.setText(strings.tr('task.logs'))
        if self._shell: self._update_session()
        self.set_title(strings.tr("overview.title"))
        self.set_caption(strings.tr("overview.caption"))
        self._runtime_header.set_title(strings.tr("overview.device_card"))
        self._check_header.set_title(strings.tr("overview.selfcheck_card"))
        self._check_header.set_caption(strings.tr("overview.selfcheck_hint"))
        self._cpu_row.set_label(strings.tr("overview.cpu"))
        self._gpu_row.set_label(strings.tr("overview.gpu"))
        self._vram_row.set_label(strings.tr("overview.vram"))
        self._backend_row.set_label(strings.tr("overview.backend"))
        self._device_row.set_label(strings.tr("overview.device"))
        self._models_row.set_label(strings.tr("overview.models"))
        self._refresh_button.setText(strings.tr("overview.refresh"))
        self._run_button.setText(strings.tr("overview.run_check"))
        self._cancel_button.setText(strings.tr("overview.cancel"))
        self._log.setPlaceholderText(strings.tr("overview.log_empty"))
        self._set_status(self._status.state)
        self.refresh_theme()
