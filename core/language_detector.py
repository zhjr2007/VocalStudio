"""Language detection for audio (ASR-based) and text (script heuristics)."""

from __future__ import annotations

import re

SUPPORTED = ("zh", "ja", "en")

_KANA = re.compile(r"[\u3040-\u309F\u30A0-\u30FF]")
_HAN = re.compile(r"[\u4E00-\u9FFF\u3400-\u4DBF]")
_LATIN = re.compile(r"[A-Za-z]")


class LanguageDetector:
    def __init__(self, asr_detector=None):
        self._asr_detector = asr_detector

    def detect(self, audio_path: str) -> str:
        if self._asr_detector is not None:
            try:
                lang = self._asr_detector(audio_path)
                if lang in SUPPORTED:
                    return lang
            except Exception:
                pass
        return "auto"

    def detect_batch(self, audio_paths: list[str]) -> list[str]:
        return [self.detect(p) for p in audio_paths]

    def detect_text(self, text: str) -> str:
        kana = len(_KANA.findall(text or ""))
        han = len(_HAN.findall(text or ""))
        latin = len(_LATIN.findall(text or ""))
        if kana >= 3 and kana > han * 0.3:
            return "ja"
        if han > latin:
            return "zh"
        if latin > 0:
            return "en"
        return "auto"
