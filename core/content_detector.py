"""Content type detection (singing / speech / mixed) via f0 distribution."""

from __future__ import annotations

from dataclasses import dataclass

from .config import AudioType

SINGING_MIN_RANGE_HZ = 300.0
SINGING_MIN_STD_HZ = 40.0
SPEECH_MAX_RANGE_HZ = 150.0
SPEECH_MAX_STD_HZ = 25.0
MIN_VOICED_RATIO = 0.1


@dataclass
class PitchMetrics:
    pitch_range: float
    pitch_std: float
    voiced_ratio: float
    median: float

    def to_dict(self) -> dict:
        return {
            "pitch_range": round(self.pitch_range, 2),
            "pitch_std": round(self.pitch_std, 2),
            "voiced_ratio": round(self.voiced_ratio, 3),
            "median": round(self.median, 2),
        }


class ContentDetector:
    def detect(self, audio_path: str) -> AudioType:
        import librosa

        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        return self.detect_waveform(y, sr)

    def detect_waveform(self, y, sr: int) -> AudioType:
        metrics = self.measure(y, sr)
        if metrics is None or metrics.voiced_ratio < MIN_VOICED_RATIO:
            return AudioType.UNKNOWN
        if metrics.pitch_range > SINGING_MIN_RANGE_HZ and metrics.pitch_std > SINGING_MIN_STD_HZ:
            return AudioType.SINGING
        if metrics.pitch_range < SPEECH_MAX_RANGE_HZ and metrics.pitch_std < SPEECH_MAX_STD_HZ:
            return AudioType.SPEECH
        return AudioType.MIXED

    def measure(self, y, sr: int) -> PitchMetrics | None:
        import librosa
        import numpy as np

        if y is None or len(y) < int(0.5 * sr):
            return None
        f0, voiced_flag, _ = librosa.pyin(y, fmin=60.0, fmax=1200.0, sr=sr)
        voiced_f0 = f0[voiced_flag] if voiced_flag is not None else f0
        if voiced_f0 is None or not len(voiced_f0):
            return None
        return PitchMetrics(
            pitch_range=float(np.max(voiced_f0) - np.min(voiced_f0)),
            pitch_std=float(np.std(voiced_f0)),
            voiced_ratio=float(np.mean(voiced_flag)) if voiced_flag is not None else 0.0,
            median=float(np.median(voiced_f0)),
        )
