"""Training page.

Structurally different from the other five, and the differences are all real:

* the input is a **dataset directory**, not a file;
* it does not go through the pipeline orchestrator — training has its own job body
  (``engines.training.job_runner.run_training_job``) and its own spec object, so the
  page overrides :meth:`TaskPage.make_job`;
* **Cancel really kills.** Training runs a subprocess and its ``cancel()``
  terminates then kills it, unlike the pipeline's cooperative-only cancellation.
  That is the one place in this UI where Cancel is immediate.
* **A non-zero exit is not an exception.** The engine records the exit code and
  returns normally; :func:`vstudio_ui.services.run_training_job` turns that into
  ``RunResult.error``, which the page then shows in amber rather than green.

The device list is restricted to ``cuda`` / ``cpu`` on purpose: the adapters pin
``CUDA_VISIBLE_DEVICES`` and treat anything that is not CUDA as "train on CPU", so
offering ``rocm`` / ``zluda`` / ``directml`` here would silently mean CPU.
"""

from __future__ import annotations

from PyQt5.QtWidgets import QComboBox, QDoubleSpinBox, QSpinBox, QVBoxLayout, QLabel

from vstudio_ui import services, strings
from vstudio_ui.pages.task_page import TaskPage
from vstudio_ui.widgets import FieldRow

TRAINING_DEVICES = ("cuda", "cpu")


class TrainingPage(TaskPage):
    key = "training"
    input_is_dir = True
    device_choices = TRAINING_DEVICES
    result_spec = (
        ("output", "training.result_output"),
        ("checkpoints", "training.result_checkpoints"),
        ("exit", "training.result_exit"),
    )
    result_hint_key = "training.result_empty"

    def __init__(self, shell=None, parent=None):
        super().__init__(shell, parent)
        self.extension_label = QLabel(self)
        self.extension_label.setWordWrap(True)
        self._options_card.layout().insertWidget(1, self.extension_label)
        self.task_combo.currentIndexChanged.connect(self._update_extension)
        self.device_combo.currentIndexChanged.connect(self._update_extension)
        self._update_extension()

    def _update_extension(self):
        status = services.training_availability(str(self.task_combo.currentData()), str(self.device_combo.currentData()))
        label = strings.tr('training.extension_ready') if status.ready else strings.tr('training.extension_missing')
        self.extension_label.setText(f"{strings.tr('training.extension')} · {label}\n{status.state}: {status.detail}")
        self._run_button.setEnabled(status.ready and not self._job.running and not (self._shell and self._shell.tasks.busy))

    def _sync_task_lock(self):
        super()._sync_task_lock()
        if hasattr(self, 'extension_label'): self._update_extension()

    def validate_request(self):
        status = services.training_availability(str(self.task_combo.currentData()), str(self.device_combo.currentData()))
        if not status.ready: return self.extension_label.text()
        return super().validate_request()

    def build_options(self, column: QVBoxLayout) -> None:
        self.task_combo = QComboBox()
        for name in services.training_tasks():
            self.task_combo.addItem(strings.tr(f"training.task.{name}", name), userData=name)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 10000)
        # Matches the adapters' own default (``params.get("epochs", 100)``), so a
        # page that does not touch the control trains the same as the CLI does.
        self.epochs_spin.setValue(100)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 512)
        self.batch_spin.setValue(4)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(6)
        self.lr_spin.setRange(1e-6, 1.0)
        self.lr_spin.setSingleStep(1e-4)
        self.lr_spin.setValue(5e-4)

        self.task_row = FieldRow(strings.tr("training.task"), self.task_combo)
        self.epochs_row = FieldRow(strings.tr("training.epochs"), self.epochs_spin)
        self.batch_row = FieldRow(strings.tr("training.batch"), self.batch_spin)
        self.lr_row = FieldRow(strings.tr("training.lr"), self.lr_spin)
        for row in (self.task_row, self.epochs_row, self.batch_row, self.lr_row):
            self.add_option_row(column, row)

    def make_job(self):
        return services.run_training_job, None, services.build_training_spec(self.build_training_request())

    def build_training_request(self) -> services.TrainingRequest:
        return services.TrainingRequest(
            task=str(self.task_combo.currentData() or "sensevoice_singing"),
            data_dir=self.audio_edit.text().strip(),
            output_dir=self.output_edit.text().strip(),
            device=str(self.device_combo.currentData() or "cuda"),
            epochs=int(self.epochs_spin.value()),
            batch=int(self.batch_spin.value()),
            lr=float(self.lr_spin.value()),
        )

    def render_result(self, result: services.RunResult) -> None:
        self.set_result("output", result.output_dir)
        self.set_result(
            "checkpoints",
            strings.tr("training.checkpoint_count").format(count=len(result.checkpoints)),
            muted=not result.checkpoints,
        )
        if result.checkpoints:
            label = self._result_values.get("checkpoints")
            if label is not None:
                label.setToolTip("\n".join(result.checkpoints))
        self.set_result(
            "exit",
            "" if result.exit_code is None else str(result.exit_code),
            muted=result.exit_code is None,
        )

    def retranslate_extra(self) -> None:
        if hasattr(self, 'extension_label'): self._update_extension()
        self.task_row.set_label(strings.tr("training.task"))
        self.epochs_row.set_label(strings.tr("training.epochs"))
        self.batch_row.set_label(strings.tr("training.batch"))
        self.lr_row.set_label(strings.tr("training.lr"))
        for index, name in enumerate(services.training_tasks()):
            self.task_combo.setItemText(index, strings.tr(f"training.task.{name}", name))
