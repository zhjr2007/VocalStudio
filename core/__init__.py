"""VocalStudio core layer: config, context, detection, orchestration."""

from .config import (
    QUANTIZATION_MODES,
    SEPARATION_MODELS,
    SLICING_METHODS,
    SUPPORTED_LANGUAGES,
    TTS_VERSIONS,
    AudioType,
    ModelPaths,
    PipelineConfig,
    PipelineStep,
    validate_slice_bounds,
)
from .context import ASRSegment, NoteInfo, PipelineContext
from .text_frontend import LANGS, phonemize
from .orchestrator import STEP_ORDER, VocalStudioOrchestrator
from .exceptions import (
    ASRError,
    AlignmentError,
    CancellationError,
    ExportError,
    ModelNotFoundError,
    SeparationError,
    TrainingError,
    VocalStudioError,
)

__all__ = [
    "ASRSegment",
    "ASRError",
    "AlignmentError",
    "AudioType",
    "CancellationError",
    "ExportError",
    "ModelNotFoundError",
    "ModelPaths",
    "NoteInfo",
    "PipelineContext",
    "PipelineStep",
    "STEP_ORDER",
    "LANGS",
    "phonemize",
    "VocalStudioOrchestrator",
    "QUANTIZATION_MODES",
    "SEPARATION_MODELS",
    "SLICING_METHODS",
    "SUPPORTED_LANGUAGES",
    "TTS_VERSIONS",
    "SeparationError",
    "TrainingError",
    "VocalStudioError",
    "validate_slice_bounds",
]
