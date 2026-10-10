"""Subprocess-based training job runner (wave G-2).

A :class:`TrainingJob` maps a spec onto a real training command
(:func:`engines.training.adapters.build_command`), spawns it as a hidden
subprocess, tails stdout/stderr line-by-line and turns epoch lines into
monotone progress. :meth:`TrainingJob.cancel` terminates the process
(polling via :class:`TrainingJobHandle`) and joins with a 2 s deadline.

The GUI page runs :func:`run_training_job` on a :class:`JobRunner` thread so
the signals (progress / log / done / failed) surface in the UI; the CLI tests
use it directly.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from core.config import PROJECT_ROOT
from core.exceptions import CancellationError

from .adapters import build_command, parse_progress

TERMINATE_TIMEOUT_SEC = 2.0


class TrainingJobSpec:
    """User-provided settings for one training job."""

    def __init__(self, task: str = "sensevoice_singing", data_dir: str = "",
                 params: dict | None = None, output_dir: str = "",
                 device: str = "cuda") -> None:
        self.task = str(task)
        self.data_dir = str(data_dir)
        self.params = dict((params or {}))
        self.output_dir = str(output_dir)
        self.device = str(device or "cuda")


class TrainingJobHandle:
    """Live view of a training subprocess."""

    def __init__(self, process: subprocess.Popen, log_path: Path | None = None):
        self.process = process
        self.log_path = log_path

    @property
    def pid(self) -> int:
        return self.process.pid

    def poll(self) -> int | None:
        return self.process.poll()

    def wait(self, timeout: float | None = None) -> int:
        return self.process.wait(timeout=timeout)


class TrainingJob:
    """One training run: spec -> command -> subprocess -> log/progress stream."""

    def __init__(self, spec: TrainingJobSpec, log_cb=None, progress_cb=None,
                 cancel_checker=None):
        self.spec = spec
        self.log_cb = log_cb
        self.progress_cb = progress_cb
        self.cancel_checker = cancel_checker
        self.handle: TrainingJobHandle | None = None
        self.checkpoint_files: list[str] = []
        self.exit_code: int | None = None
        self._cancelled = False

    # -- public -------------------------------------------------------------

    def launch(self) -> JobCommand:
        """Builds the command and starts the process (blocking until it exits)."""
        self.log("launching training job...")
        command = build_command(self.spec.task, self.spec.data_dir, self.spec.params,
                                self.spec.output_dir, device=getattr(self.spec, "device", "cuda"))
        self.log(command.command_line())
        self._check_cancelled()

        out_dir = Path(getattr(self.spec, "output_dir", "") or (PROJECT_ROOT / "outputs" / "training"))
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / f"{self.spec.task}_job.log"
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        process = subprocess.Popen(
            command.argv,
            cwd=command.cwd,
            env=command.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            creationflags=creationflags,
        )
        self.handle = TrainingJobHandle(process, log_path)
        self.log(f"pid={process.pid} log={log_path}")
        return command

    def stream(self, command: JobCommand, total_epochs: int | None = None) -> None:
        """Consumes stdout until EOF, emits progress/log; not cancel-safe by itself.

        :meth:`launch` → :meth:`stream` reproduces the whole blocking flow.
        """
        pattern_count = len(command.parse_patterns) or 0
        total = total_epochs or int(self.spec.params.get("epochs", 100))
        assert self.handle is not None and self.handle.process.stdout is not None
        for line in self.handle.process.stdout:
            self._check_cancelled()
            self.log(line.rstrip("\n"))
            if not (pattern_count or command.parse_patterns):
                continue
            fraction = parse_progress(line, command.parse_patterns, total)
            if fraction is not None and self.progress_cb is not None:
                try:
                    self.progress_cb(fraction, "epoch")
                except Exception:
                    pass
        self.exit_code = self.handle.process.wait()
        self.log(f"training process exited with code {self.exit_code}")
        self.checkpoint_files = [str(p) for p in out_dir_glob(self.spec, "*.pt")]

    def cancel(self) -> None:
        self._cancelled = True
        handle = self.handle
        if handle is None or handle.poll() is not None:
            return
        try:
            handle.process.terminate()
            handle.process.wait(timeout=TERMINATE_TIMEOUT_SEC)
        except subprocess.TimeoutExpired:
            handle.process.kill()
            handle.process.wait(timeout=5.0)
        self.log("cancelled")

    # -- internals ----------------------------------------------------------

    def _check_cancelled(self) -> None:
        cancelled = self._cancelled
        if not cancelled and callable(self.cancel_checker):
            try:
                cancelled = bool(self.cancel_checker())
            except Exception:
                cancelled = False
        if cancelled:
            self.cancel()
            raise CancellationError("training cancelled")

    def log(self, message: str) -> None:
        if self.log_cb is not None:
            try:
                self.log_cb(str(message))
            except Exception:
                pass


def out_dir_glob(spec: TrainingJobSpec, pattern: str) -> list[Path]:
    out_dir = Path(getattr(spec, "output_dir", "") or (PROJECT_ROOT / "outputs" / "training"))
    return list(out_dir.rglob(pattern)) if out_dir.is_dir() else []


def run_training_job(ctx: dict, cfg, progress_cb) -> dict:
    """GUI/CLI job body: build -> launch -> stream -> collect checkpoints.

    ``cfg`` is duck-typed (TrainingJobSpec or page job cfg with the same
    fields plus settable ``log_cb`` / ``cancel_checker``).
    """
    task = str(getattr(cfg, "task", "") or ctx.get("task") or "sensevoice_singing")
    spec = TrainingJobSpec(
        task=task,
        data_dir=str(getattr(cfg, "data_dir", "") or ctx.get("data_dir") or ""),
        params=dict(getattr(cfg, "params", {}) or ctx.get("params") or {}),
        output_dir=str(getattr(cfg, "output_dir", "") or ctx.get("output_dir") or ""),
        device=str(getattr(cfg, "device", "") or ctx.get("device") or "cuda"),
    )
    job = TrainingJob(
        spec,
        log_cb=getattr(cfg, "log_cb", None),
        progress_cb=progress_cb,
        cancel_checker=getattr(cfg, "cancel_checker", None),
    )
    command = job.launch()
    job.stream(command)
    return _finish(ctx, job)


def _finish(ctx: dict, job: TrainingJob) -> dict:
    ctx["exit_code"] = job.exit_code
    ctx["checkpoints"] = list(job.checkpoint_files)
    ctx["pid"] = job.handle.pid if job.handle else None
    return ctx
