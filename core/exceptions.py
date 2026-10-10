"""VocalStudio exception hierarchy."""


class VocalStudioError(Exception):
    def __init__(self, message: str, step: str | None = None):
        super().__init__(message)
        self.step = step


class ModelNotFoundError(VocalStudioError):
    def __init__(self, message: str, step: str | None = None, model: str | None = None):
        super().__init__(message, step)
        self.model = model


class SeparationError(VocalStudioError):
    pass


class ASRError(VocalStudioError):
    pass


class AlignmentError(VocalStudioError):
    pass


class ExportError(VocalStudioError):
    pass


class TrainingError(VocalStudioError):
    pass


class CancellationError(VocalStudioError):
    pass
