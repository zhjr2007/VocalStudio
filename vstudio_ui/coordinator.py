"""Application-wide inference lease. Release only after QThread.finished."""
from PyQt5.QtCore import QObject, pyqtSignal


class TaskCoordinator(QObject):
    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.handle = None
        self.owner = ''
        self.fraction = 0.0
        self.stage = ''
        self.results = []

    @property
    def busy(self): return self.handle is not None

    def acquire(self, owner, handle):
        if self.busy: return False
        self.owner, self.handle = owner, handle
        self.fraction, self.stage = 0.0, ''
        handle.progress.connect(self._progress)
        handle.done.connect(self._result)
        handle.settled.connect(self.release)
        self.changed.emit()
        return True

    def _progress(self, fraction, stage):
        self.fraction, self.stage = fraction, stage
        self.changed.emit()

    def _result(self, result, cancelled):
        if not cancelled and result is not None:
            self.results.append((self.owner, result))
            self.results = self.results[-20:]

    def release(self):
        if self.handle is not None:
            self.handle.progress.disconnect(self._progress)
            self.handle.done.disconnect(self._result)
            self.handle.settled.disconnect(self.release)
        self.handle = None
        self.changed.emit()

    def cancel(self):
        if self.handle: self.handle.cancel()
