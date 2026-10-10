"""PipelineContext: carries intermediate results between pipeline steps."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .config import AudioType


@dataclass
class NoteInfo:
    onset: float
    offset: float
    pitch: float
    lyric: str = ""

    @property
    def duration(self) -> float:
        return self.offset - self.onset


@dataclass
class ASRSegment:
    start: float
    end: float
    text: str
    speaker: int = -1
    emotion: str = ""


@dataclass
class PipelineContext:
    audio_path: str = ""
    audio_type: AudioType = AudioType.UNKNOWN
    detected_language: str = ""

    vocals_path: str | None = None
    accompaniment_path: str | None = None
    denoised_vocals_path: str | None = None

    speaker_segments: list = field(default_factory=list)
    speakers: dict = field(default_factory=dict)

    notes: list = field(default_factory=list)
    asr_text: str = ""
    srt_path: str | None = None
    tts_audio_path: str | None = None

    pitch_curve: dict = field(default_factory=dict)
    results: dict = field(default_factory=dict)
    step_status: dict = field(default_factory=dict)

    def set_step_status(self, step: str, status: str, detail: str = "") -> None:
        self.step_status[step] = {"status": status, "detail": detail}

    @property
    def active_vocals_path(self) -> str:
        return self.denoised_vocals_path or self.vocals_path or self.audio_path

    def to_dict(self) -> dict:
        data = asdict(self)
        data["audio_type"] = self.audio_type.value
        return data

    @classmethod
    def from_dict(cls, data: dict) -> PipelineContext:
        data = dict(data)
        if isinstance(data.get("audio_type"), str):
            data["audio_type"] = AudioType(data["audio_type"])
        allowed = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in allowed})
