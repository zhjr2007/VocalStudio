"""VocalStudio engine registry (wave A skeleton).

`import engines` is side-effect free and loads no heavy dependencies; each
engine subpackage is imported lazily inside get_engine() on first use.
"""

from __future__ import annotations

import importlib

#: Registry name -> the subpackage `engines.<name>` exposing `engine_class`.
ENGINE_NAMES: tuple[str, ...] = (
    "separation",
    "vocal2midi",
    "speaker",
    "speech_asr",
    "tts",
    "training",
)


class UnknownEngineError(ValueError):
    """Raised when get_engine() receives a name outside ENGINE_NAMES."""

    def __init__(self, name: str) -> None:
        valid = ", ".join(ENGINE_NAMES)
        super().__init__(f"Unknown engine {name!r}; expected one of: {valid}")
        self.name = name


def get_engine(name: str):
    """Return the engine class registered under `name` (lazy import).

    The subpackage is imported on first fetch, so torch/funasr/onnxruntime
    stay unloaded until an engine is actually requested.
    """
    if name not in ENGINE_NAMES:
        raise UnknownEngineError(name)
    module = importlib.import_module(f"engines.{name}")
    return module.engine_class


__all__ = ["ENGINE_NAMES", "UnknownEngineError", "get_engine"]
