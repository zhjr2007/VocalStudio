"""TTS engine package: GPT-SoVITS v2Pro/v2ProPlus voice clone (wave E)."""

from __future__ import annotations


class TTSEngine:
    """Placeholder kept for the engines registry (see voice_clone)."""


def get_engine():
    from engines.tts.voice_clone import VoiceCloneEngine

    return VoiceCloneEngine


engine_class = TTSEngine

__all__ = ["TTSEngine", "get_engine", "VoiceCloneEngine"]
