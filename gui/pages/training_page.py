"""TrainingPage: fine-tune selected models (wave G-1/G-4 controls).

Task list (>=4): sensevoice_singing / gsv_tts / campplus / game; data-dir
browse; lr (1e-5~1e-3, default 5e-4), batch (2~8, default 4), epochs
(10~200, default 100). Start() runs the adapted training command through
:class:`engines.training.job_runner.run_training_job` on a JobRunner thread;
progress = epoch/total, log lines stream into the log box, cancel terminates
the child process within ~2 s.
"""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from gui.i18n import tr
from gui.widgets.info_card import InfoCard
from gui.widgets.job_runner import JobRunner
from engines.training.adapters import TASK_IDS

USAGE_KEYS = ("training_usage_1", "training_usage_2", "training_usage_3")
MODEL_KEYS = (
    "training_models_1",
    "training_models_2",
    "training_models_3",
)

TASK_LABEL_KEYS = {
    "sensevoice_singing": "training_task_sensevoice",
    "gsv_tts": "training_task_gsv",
    "campplus": "training_task_campplus",
    "game": "training_task_game",
}

LR_MIN, LR_MAX, LR_DEFAULT = 1e-5, 1e-3, 5e-4
BATCH_MIN, BATCH_MAX, BATCH_DEFAULT = 2, 8, 4
EPOCHS_MIN, EPOCHS_MAX, EPOCHS_DEFAULT = 10, 200, 100


class TrainingPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self._title_label = QLabel(self)
        title_font = self._title_label.font()
        title_font.setBold(True)
        self._title_label.setFont(title_font)
        layout.addWidget(self._title_label)

        # -- task selection ---------------------------------------------------
        task_row = QHBoxLayout()
        self._task_label = QLabel(self)
        task_row.addWidget(self._task_label)
        self._task_combo = QComboBox(self)
        for task in TASK_IDS:
            self._task_combo.addItem("", userData=task)
        task_row.addWidget(self._task_combo, 1)
        layout.addLayout(task_row)

        # -- data dir + params -------------------------------------------------
        dir_row = QHBoxLayout()
        self._dir_label = QLabel(self)
        dir_row.addWidget(self._dir_label)
        self._dir_edit = QLineEdit(self)
        dir_row.addWidget(self._dir_edit, 1)
        self._browse_button = QPushButton(self)
        self._browse_button.clicked.connect(self._on_browse)
        dir_row.addWidget(self._browse_button)
        layout.addLayout(dir_row)

        params_row = QHBoxLayout()
        self._lr_label = QLabel(self)
        params_row.addWidget(self._lr_label)
        self._lr_spin = QDoubleSpinBox(self)
        self._lr_spin.setDecimals(5)
        self._lr_spin.setRange(LR_MIN, LR_MAX)
        self._lr_spin.setSingleStep(5e-5)
        self._lr_spin.setValue(LR_DEFAULT)
        params_row.addWidget(self._lr_spin, 1)
        self._batch_label = QLabel(self)
        params_row.addWidget(self._batch_label)
        self._batch_spin = QSpinBox(self)
        self._batch_spin.setRange(BATCH_MIN, BATCH_MAX)
        self._batch_spin.setValue(BATCH_DEFAULT)
        params_row.addWidget(self._batch_spin, 1)
        self._epochs_label = QLabel(self)
        params_row.addWidget(self._epochs_label)
        self._epochs_spin = QSpinBox(self)
        self._epochs_spin.setRange(EPOCHS_MIN, EPOCHS_MAX)
        self._epochs_spin.setValue(EPOCHS_DEFAULT)
        params_row.addWidget(self._epochs_spin, 1)
        layout.addLayout(params_row)

        # -- compute device (default cuda so fine-tuning always hits the CUDA GPU)
        device_row = QHBoxLayout()
        self._device_label = QLabel(self)
        device_row.addWidget(self._device_label)
        self._device_combo = QComboBox(self)
        self._device_combo.addItem("cuda", userData="cuda")
        self._device_combo.addItem("cpu", userData="cpu")
        device_row.addWidget(self._device_combo, 1)
        layout.addLayout(device_row)

        # -- run controls ------------------------------------------------------
        run_row = QHBoxLayout()
        run_row.addStretch(1)
        self._start_button = QPushButton(self)
        self._start_button.clicked.connect(self.start)
        run_row.addWidget(self._start_button)
        self._cancel_button = QPushButton(self)
        self._cancel_button.setEnabled(False)
        self._cancel_button.clicked.connect(self.cancel)
        run_row.addWidget(self._cancel_button)
        layout.addLayout(run_row)

        progress_row = QHBoxLayout()
        self._progress_label = QLabel(self)
        progress_row.addWidget(self._progress_label)
        self._progress_bar = QProgressBar(self)
        self._progress_bar.setRange(0, 100)
        progress_row.addWidget(self._progress_bar, 1)
        self._status_label = QLabel(self)
        progress_row.addWidget(self._status_label)
        layout.addLayout(progress_row)

        self._log = QPlainTextEdit(self)
        self._log.setReadOnly(True)
        layout.addWidget(self._log, 3)

        self._usage_card = InfoCard("info_usage", USAGE_KEYS, self)
        self._models_card = InfoCard("info_models", MODEL_KEYS, self)
        layout.addWidget(self._usage_card, 2)
        layout.addWidget(self._models_card, 3)

        self._job_runner: JobRunner | None = None
        self._last_progress: float = 0.0
        self._last_message: str = ""
        self._stage_raw: str = ""

        self.retranslate()

    # -- public API ----------------------------------------------------------

    def retranslate(self) -> None:
        self._title_label.setText(tr("training"))
        self._task_label.setText(tr("training_task"))
        for index, task in enumerate(TASK_IDS):
            self._task_combo.setItemText(index, tr(TASK_LABEL_KEYS[task]))
        self._dir_label.setText(tr("training_data_dir"))
        self._browse_button.setText(tr("browse"))
        self._lr_label.setText(tr("training_lr"))
        self._batch_label.setText(tr("training_batch"))
        self._epochs_label.setText(tr("training_epochs"))
        self._device_label.setText(tr("training_device"))
        self._start_button.setText(tr("start"))
        self._cancel_button.setText(tr("cancel"))
        self._progress_label.setText(tr("progress"))
        self._usage_card.retranslate()
        self._models_card.retranslate()

    def task(self) -> str:
        return str(self._task_combo.currentData() or "sensevoice_singing")

    def set_task(self, task: str) -> None:
        for i in range(self._task_combo.count()):
            if self._task_combo.itemData(i) == task:
                self._task_combo.setCurrentIndex(i)
                return

    def data_dir(self) -> str:
        return self._dir_edit.text().strip()

    def set_data_dir(self, path: str) -> None:
        self._dir_edit.setText(str(path))

    def lr(self) -> float:
        return float(self._lr_spin.value())

    def set_lr(self, value: float) -> None:
        self._lr_spin.setValue(float(value))

    def batch(self) -> int:
        return int(self._batch_spin.value())

    def set_batch(self, value: int) -> None:
        self._batch_spin.setValue(int(value))

    def epochs(self) -> int:
        return int(self._epochs_spin.value())

    def set_epochs(self, value: int) -> None:
        self._epochs_spin.setValue(int(value))

    def device(self) -> str:
        return str(self._device_combo.currentData() or "cuda")

    def set_device(self, value: str) -> None:
        idx = self._device_combo.findData(str(value))
        if idx >= 0:
            self._device_combo.setCurrentIndex(idx)

    def config(self) -> dict:
        return {
            "task": self.task(),
            "data_dir": self.data_dir(),
            "lr": self.lr(),
            "batch": self.batch(),
            "epochs": self.epochs(),
            "device": self.device(),
        }

    def is_running(self) -> bool:
        runner = self._job_runner
        return bool(runner is not None and runner.isRunning())

    def last_progress(self) -> float:
        return self._last_progress

    def log_text(self) -> str:
        return self._log.toPlainText()

    def start(self) -> JobRunner | None:
        if self.is_running():
            return self._job_runner
        data_dir = self.data_dir()
        if not data_dir or not Path(data_dir).is_dir():
            self._append_log(f"error: {tr('training_data_dir')} {tr('pl_source_missing')}")
            return None

        from core.config import PROJECT_ROOT

        cfg = _TrainingJobCfg(
            task=self.task(),
            data_dir=data_dir,
            params={"lr": self.lr(), "batch": self.batch(), "epochs": self.epochs()},
            output_dir=str(PROJECT_ROOT / "outputs" / "training" / self.task()),
            device=self.device(),
        )
        ctx: dict = {"task": cfg.task, "data_dir": cfg.data_dir,
                     "params": dict(cfg.params), "device": cfg.device}
        runner = JobRunner(run_training_job, ctx=ctx, cfg=cfg, parent=self)
        self._job_runner = runner
        self._last_progress = 0.0
        self._last_message = ""
        self._stage_raw = ""
        self._log.clear()
        runner.progress.connect(self._on_progress)
        runner.log_signal.connect(self._on_log)
        runner.done.connect(self._on_done)
        runner.failed.connect(self._on_failed)
        runner.start_job()
        self._start_button.setEnabled(False)
        self._cancel_button.setEnabled(True)
        return runner

    def cancel(self) -> None:
        if self._job_runner is not None:
            self._job_runner.cancel()

    def _on_browse(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, tr("browse"), self.data_dir() or str(Path.home()))
        if chosen:
            self.set_data_dir(chosen)

    # -- slots ----------------------------------------------------------------

    def _on_progress(self, value: float, message: str) -> None:
        fraction = max(0.0, min(1.0, float(value)))
        self._last_progress = fraction
        self._progress_bar.setValue(int(round(fraction * 100)))
        if message:
            self._last_message = str(message)
            self._status_label.setText(self._last_message)

    def _append_log(self, text: str) -> None:
        self._log.appendPlainText(str(text))

    def _on_log(self, message: str) -> None:
        self._append_log(message)

    def _on_done(self, result, cancelled: bool) -> None:
        self._start_button.setEnabled(True)
        self._cancel_button.setEnabled(False)
        if cancelled:
            self._append_log("cancelled")
            return
        checkpoints = list(((result or {}).get("checkpoints") or []))
        self._append_log(f"done: {len(checkpoints)} checkpoint(s)")
        for path in checkpoints[:5]:
            self._append_log(f"  ckpt: {path}")

    def _on_failed(self, exc) -> None:
        self._start_button.setEnabled(True)
        self._cancel_button.setEnabled(False)
        self._append_log(f"error: {exc}")


class _TrainingJobCfg:
    """Attributes consumed by :func:`run_training_job` (JobRunner wiring)."""

    def __init__(self, task="sensevoice_singing", data_dir="", params=None, output_dir="",
                 device="cuda") -> None:
        self.task = str(task)
        self.data_dir = str(data_dir)
        self.params = dict(params or {})
        self.output_dir = str(output_dir or "")
        self.device = str(device or "cuda")
        self.log_cb = None
        self.cancel_checker = lambda: False
