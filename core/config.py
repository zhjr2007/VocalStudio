"""Central configuration: AudioType, ModelPaths, PipelineConfig, validation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETTINGS_FILE = PROJECT_ROOT / "settings" / "defaults.json"

SLICE_DURATION_MIN_SEC = 0.0
SLICE_DURATION_MAX_SEC = 60.0
DEFAULT_SLICE_MIN_SEC = 5.0
DEFAULT_SLICE_MAX_SEC = 10.0

SUPPORTED_LANGUAGES = ("auto", "zh", "ja", "en")
SEPARATION_MODELS = (
    # preset names resolved by the UVR5 engine (wave B-2/B-6)
    "Karaoke",
    "FullVocal",
    "Band4",
    "Producer5",
    # raw weight basenames (extension optional in the CLI/engine)
    "HP2_all_vocals",
    "HP5_only_main_vocal",
    "VR-DeEchoNormal",
    "VR-DeEchoDeReverb",
    "VR-DeEchoAggressive",
    "model_bs_roformer_ep_317",
    "model_bs_roformer_ep_317_sdr_12.9755",
)
SLICING_METHODS = ("default", "smart", "heuristic", "grid")
QUANTIZATION_MODES = ("simple", "smart", "bayes", "dp")
TTS_VERSIONS = ("v2Pro", "v2ProPlus")
ASR_ENGINES = ("sensevoice", "paraformer")


class AudioType(str, Enum):
    SINGING = "singing"
    SPEECH = "speech"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class PipelineStep(str, Enum):
    SEPARATION = "separation"
    DIARIZATION = "diarization"
    CONTENT_DETECTION = "content_detection"
    VOCAL2MIDI = "vocal2midi"
    SPEECH_ASR = "speech_asr"
    TTS = "tts"


def validate_slice_bounds(slice_min_sec: float, slice_max_sec: float) -> None:
    if not SLICE_DURATION_MIN_SEC <= slice_min_sec <= SLICE_DURATION_MAX_SEC:
        raise ValueError(
            f"slice_min_sec must be within {SLICE_DURATION_MIN_SEC:.0f}-{SLICE_DURATION_MAX_SEC:.0f} seconds"
        )
    if not SLICE_DURATION_MIN_SEC <= slice_max_sec <= SLICE_DURATION_MAX_SEC:
        raise ValueError(
            f"slice_max_sec must be within {SLICE_DURATION_MIN_SEC:.0f}-{SLICE_DURATION_MAX_SEC:.0f} seconds"
        )
    if slice_max_sec <= 0:
        raise ValueError("slice_max_sec must be greater than 0 seconds")
    if slice_min_sec > slice_max_sec:
        raise ValueError("slice_min_sec must be less than or equal to slice_max_sec")


@dataclass
class ModelPaths:
    """Resolved locations of every model VocalStudio can load."""

    game: str = ""
    hubertfa: str = ""
    qwen3_asr: str = ""
    rmvpe: str = ""
    romaji_asr: str = ""
    sensevoice: str = ""
    vad: str = ""
    campplus: str = ""
    ct_punc: str = ""
    paraformer: str = ""
    uvr5_weights: str = ""
    denoise: str = ""
    gsv_root: str = ""
    gsv_v2pro_gpt: str = ""
    gsv_v2pro_sovits: str = ""
    gsv_v2proplus_gpt: str = ""
    gsv_v2proplus_sovits: str = ""
    chinese_hubert: str = ""
    chinese_roberta: str = ""
    drums: str = ""
    bass: str = ""
    piano: str = ""

    @classmethod
    def from_settings(cls, settings_file: Path | str = SETTINGS_FILE) -> ModelPaths:
        path = Path(settings_file)
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        models = data.get("models", {})
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in models.items() if k in fields})

    def resolve(self, name: str) -> Path:
        value = getattr(self, name)
        if not value:
            raise ValueError(f"Model path '{name}' is not configured")
        p = Path(value)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p

    def missing(self) -> list[str]:
        return [name for name in self.__dataclass_fields__ if not getattr(self, name)]

    def to_dict(self) -> dict:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass
class PipelineConfig:
    """Full pipeline configuration (30+ fields, mirrors the GUI settings)."""

    audio_path: str = ""
    output_dir: str = ""
    device: str = "auto"
    language: str = "auto"
    output_formats: list = field(default_factory=lambda: ["mid", "txt"])

    enable_separation: bool = False
    separation_model: str = "HP5_only_main_vocal"
    separation_agg: int = 0
    enable_denoise: bool = False

    enable_diarization: bool = False
    num_speakers: int = 0
    auto_detect_content: bool = True

    enable_vocal2midi: bool = False
    slicing_method: str = "default"
    slice_min_sec: float = DEFAULT_SLICE_MIN_SEC
    slice_max_sec: float = DEFAULT_SLICE_MAX_SEC
    tempo: float = 120.0
    quantization_mode: str = "bayes"
    quantization_step: int = 16
    original_lyrics: str = ""
    lyric_output_mode: str = "auto"
    seg_threshold: float = 0.2
    seg_radius: float = 0.02
    est_threshold: float = 0.2
    batch_size: int = 4
    asr_batch_size: int = 4
    t0: float = 0.0
    nsteps: int = 8
    output_pitch_curve: bool = False

    enable_speech_asr: bool = False
    asr_engine: str = "sensevoice"
    enable_punctuation: bool = True
    output_srt: bool = False
    output_json: bool = False

    enable_tts: bool = False
    tts_version: str = "v2Pro"
    tts_text: str = ""
    tts_lang: str = "zh"
    tts_speed: float = 1.0
    ref_audio: str = ""
    ref_text: str = ""
    ref_lang: str = "zh"

    pipeline_steps: list = field(default_factory=list)
    debug_mode: bool = False
    cancel_checker: object = None
    model_paths: ModelPaths = field(default_factory=ModelPaths)

    def validate(self) -> None:
        from .device_utils import normalize_runtime_device

        if self.audio_path and not Path(self.audio_path).exists():
            raise ValueError(f"Audio file not found: {self.audio_path}")
        normalize_runtime_device(self.device)
        if self.language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language '{self.language}'. Choices: {SUPPORTED_LANGUAGES}")
        if self.separation_model not in SEPARATION_MODELS:
            raise ValueError(f"Unsupported separation model '{self.separation_model}'")
        if self.slicing_method not in SLICING_METHODS:
            raise ValueError(f"Unsupported slicing method '{self.slicing_method}'")
        if self.quantization_mode not in QUANTIZATION_MODES:
            raise ValueError(f"Unsupported quantization mode '{self.quantization_mode}'")
        if self.tts_version not in TTS_VERSIONS:
            raise ValueError(f"Unsupported TTS version '{self.tts_version}'")
        if not 0.5 <= self.tts_speed <= 1.5:
            raise ValueError(f"tts_speed must be within [0.5, 1.5], got {self.tts_speed}")
        if self.asr_engine not in ASR_ENGINES:
            raise ValueError(f"Unsupported ASR engine '{self.asr_engine}'")
        if self.tempo <= 0:
            raise ValueError(f"tempo must be greater than 0, got {self.tempo}")
        if self.batch_size <= 0 or self.asr_batch_size <= 0:
            raise ValueError("batch_size and asr_batch_size must be positive")
        if self.nsteps <= 0:
            raise ValueError("nsteps must be positive")
        validate_slice_bounds(self.slice_min_sec, self.slice_max_sec)

    def to_kwargs(self) -> dict:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in ("cancel_checker", "model_paths")
        }
