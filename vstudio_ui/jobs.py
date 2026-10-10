"""Worker threads: ``JobRunner`` (the frozen contract) and ``JobHandle`` (lifetime).

``JobRunner`` is the **canonical** implementation. It used to live in
``gui/widgets/job_runner.py``; it moved here when the first-generation GUI was
archived, because the new UI may not import ``gui`` but must not reimplement the
signal semantics either. ``gui/widgets/job_runner.py`` is now a re-export shim, so
there is exactly one implementation and the two can never drift.

Semantics (``ui-design/contract.md`` §6 — **do not change**):

* ``fn(ctx, cfg, progress_cb)`` runs on the worker thread.
* Exactly one terminal signal per run: normal -> ``done(result, False)``;
  cancellation -> ``done(None, True)``; anything else -> ``failed(exc)``.
* Cancellation is cooperative: ``cancel()`` only sets a flag, the business layer
  exits at its own checkpoints. Nothing is ever killed.

``JobHandle`` exists because ``JobRunner`` is deliberately one-shot and is a
``QThread``: a page that wants "run again" must build a fresh runner, and must
keep the C++ object alive for the whole run. Every page that runs a job would
otherwise repeat that bookkeeping — and get it wrong once, which shows up as
"the progress bar froze" rather than a traceback.
"""

from __future__ import annotations

from collections.abc import Callable

from PyQt5.QtCore import QObject, QThread, pyqtSignal

from core.exceptions import CancellationError

__all__ = ["CancellationError", "JobHandle", "JobRunner"]


class JobRunner(QThread):
    """Runs ``fn(ctx, cfg, progress_cb)`` on a worker thread.

    Signals (safe to connect from the main thread; Qt queues them across
    threads automatically):
        progress: (float fraction clamped to [0, 1], str stage)
        log_signal: str message - forwarded when fn calls ``cfg.log_cb(...)``
        done: (object result, bool cancelled)
        failed: Exception instance

    Exactly one terminal signal is emitted per run: a normal return emits
    ``done(result, False)``; an engine that observes cancellation and raises
    ``CancellationError`` produces ``done(None, True)`` (never ``failed``); any
    other exception produces ``failed(exc)`` only.

    Cancellation is cooperative: :meth:`cancel` sets the ``cancelled`` flag and,
    when ``cfg`` has a settable ``cancel_checker``, wraps it so that further
    calls return True (an earlier checker is still consulted before cancel).
    Engines polling ``cfg.cancel_checker()`` then raise CancellationError.

    A JobRunner instance may only run once: after it has finished, further
    :meth:`start_job` calls are ignored (create a new instance to rerun).
    """

    progress = pyqtSignal(float, str)
    log_signal = pyqtSignal(str)
    done = pyqtSignal(object, bool)
    failed = pyqtSignal(object)

    def __init__(
        self,
        fn: Callable[[object, object, Callable], object],
        ctx=None,
        cfg=None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._fn = fn
        self._ctx = ctx
        self._cfg = cfg
        self._cancelled = False
        self._started_once = False
        self._base_cancel_checker = (
            getattr(cfg, "cancel_checker", None) if cfg is not None else None
        )

    # -- public surface -----------------------------------------------------

    @property
    def fn(self):
        return self._fn

    @property
    def ctx(self):
        return self._ctx

    @property
    def cfg(self):
        return self._cfg

    @property
    def cancelled(self) -> bool:
        """True once :meth:`cancel` ran or the job reported cancellation."""
        return self._cancelled

    # -- control ------------------------------------------------------------

    def start_job(self) -> None:
        """Starts the worker thread. Each JobRunner runs exactly one job."""
        if self._started_once:
            return
        self._started_once = True
        self.start()

    def cancel(self) -> None:
        """Requests cooperative cancellation of a pending or running job."""
        self._cancelled = True
        cfg = self._cfg
        if cfg is None:
            return
        base = self._base_cancel_checker

        def checker() -> bool:
            if self._cancelled:
                return True
            if callable(base):
                try:
                    return bool(base())
                except Exception:
                    return False
            return False

        try:
            cfg.cancel_checker = checker
        except (AttributeError, TypeError):
            pass  # read-only attribute: engines that poll it keep working

    # -- QThread ------------------------------------------------------------

    def run(self) -> None:
        """Worker-thread entry point: executes fn and emits terminal signals."""
        self._install_log_cb()
        try:
            result = self._fn(self._ctx, self._cfg, self._on_progress)
        except CancellationError:
            self._cancelled = True
            self.done.emit(None, True)
        except Exception as exc:  # noqa: BLE001 - re-emitted via `failed`
            self.failed.emit(exc)
        else:
            self.done.emit(result, False)

    # -- internals ----------------------------------------------------------

    def _install_log_cb(self) -> None:
        cfg = self._cfg
        if cfg is None or not hasattr(cfg, "log_cb"):
            return
        original = getattr(cfg, "log_cb", None)

        def log(message):
            text = message if isinstance(message, str) else str(message)
            self.log_signal.emit(text)
            if callable(original):
                try:
                    original(message)
                except Exception:
                    pass

        try:
            cfg.log_cb = log
        except (AttributeError, TypeError):
            pass

    def _on_progress(self, value: float, stage: str = "") -> None:
        """Progress callback handed to fn; clamps the fraction into [0, 1]."""
        try:
            clamped = max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return
        self.progress.emit(clamped, "" if stage is None else str(stage))


class JobHandle(QObject):
    """Owns one job at a time and re-emits its signals on the main thread.

    Why this wrapper exists:

    * ``JobRunner`` may run **once**. A page with a "run again" button needs a
      fresh runner per run; this class hides that.
    * The runner must outlive its run. :class:`JobHandle` holds the reference
      until ``QThread.finished`` fires, then hands it to ``deleteLater``. Without
      that, a Python-side reference drop while the thread is still winding down
      produces "QThread: Destroyed while thread is still running".
    * Re-emitting instead of forwarding the runner itself means pages connect
      once, in ``__init__``, and never touch a short-lived object.

    Signals mirror :class:`JobRunner` exactly (``log_signal`` is exposed as
    ``log`` to avoid clashing with ``QObject``'s own ``objectName``-era names and
    to read better at the call site).
    """

    progress = pyqtSignal(float, str)
    log = pyqtSignal(str)
    done = pyqtSignal(object, bool)
    failed = pyqtSignal(object)
    settled = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._runner: JobRunner | None = None

    # -- state --------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._runner is not None and self._runner.isRunning()

    @property
    def cancelled(self) -> bool:
        return self._runner is not None and self._runner.cancelled

    @property
    def runner(self) -> JobRunner | None:
        """The live runner, or None. Exposed for tests and for ``cfg`` access."""
        return self._runner

    # -- control ------------------------------------------------------------

    def start(self, fn, ctx=None, cfg=None) -> bool:
        """Run *fn* in the background. Returns False if a job is already running."""
        if self.running:
            return False
        runner = JobRunner(fn, ctx, cfg, self)
        runner.progress.connect(self.progress.emit)
        runner.log_signal.connect(self.log.emit)
        runner.done.connect(self._on_done)
        runner.failed.connect(self._on_failed)
        # Bind the runner explicitly instead of reading self._runner in the slot:
        # a finished thread may still be winding down when the next run starts, and
        # by then self._runner points at the new one — clearing it would orphan a
        # live thread.
        runner.finished.connect(lambda r=runner: self._on_finished(r))
        self._runner = runner
        runner.start_job()
        return True

    def cancel(self) -> None:
        """Ask the running job to stop at its next checkpoint (never kills it)."""
        runner = self._runner
        if runner is not None:
            runner.cancel()

    def wait(self, timeout_ms: int = 2000) -> bool:
        """Block until the job finishes. Returns True if it did.

        Used on shutdown: a cooperative cancel cannot interrupt a blocking
        business-layer call, so the window gives it a short grace period and then
        lets the process exit.
        """
        runner = self._runner
        if runner is None or not runner.isRunning():
            return True
        return bool(runner.wait(timeout_ms))

    # -- internals ----------------------------------------------------------

    def _on_done(self, result, cancelled: bool) -> None:
        self.done.emit(result, cancelled)

    def _on_failed(self, exc) -> None:
        self.failed.emit(exc)

    def _on_finished(self, runner: JobRunner) -> None:
        if runner is self._runner:
            self._runner = None
        runner.deleteLater()
        self.settled.emit()
