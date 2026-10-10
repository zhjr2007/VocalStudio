"""Lyric reference/ASR matcher (wave C2-4).

Thin wrapper around the vendored LyricFA matcher
(``Vocal2Midi/inference/API/lfa_api.py`` -> ``LyricFA/tools/lyric_matcher``).
It is used in two modes:

- reference mode: ``reference_lyrics`` is non-empty -> the matcher loads the
  reference lyrics and aligns each chunk's ASR transcript to it;
- auto mode: ``reference_lyrics`` is empty -> no reference text is bound, and
  the ASR transcript is returned as-is (with a ``confidence`` score from the
  matcher's sequence alignment when possible).

:meth:`LyricMatcherEngine.align` returns a per-word timeline plus a confidence
and the matched lyric string, in the shape the pipeline/lower layers expect.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any

from core.config import PROJECT_ROOT

_V2M_REF_DIR = PROJECT_ROOT / "Vocal2Midi"

_SUPPORTED_LANGS = ("zh", "en", "ja")


def _ensure_ref_path() -> None:
    ref = str(_V2M_REF_DIR)
    if _V2M_REF_DIR.is_dir() and ref not in sys.path:
        sys.path.append(ref)


@dataclass
class WordTiming:
    """One aligned word with its time interval and score contribution."""

    text: str
    start: float
    end: float
    score: float = 1.0


@dataclass
class AlignResult:
    """Result of aligning an ASR transcript (and optional reference lyrics)."""

    words: list[WordTiming] = field(default_factory=list)
    matched_lyric: str = ""
    matched_phonetic: str = ""
    confidence: float = 0.0
    reason: str = ""


class LyricMatcherEngine:
    """Loads a LyricMatcher for a language and aligns ASR text to it."""

    def __init__(self, language: str = "zh") -> None:
        _ensure_ref_path()
        from inference.LyricFA.tools.lyric_matcher import LyricMatcher

        self.language = language if language in _SUPPORTED_LANGS else "zh"
        self.matcher = LyricMatcher(self.language)
        self._bound_reference = False

    # -- reference binding ------------------------------------------------

    def bind_reference(self, reference_lyrics: str) -> bool:
        """Load ``reference_lyrics`` for alignment; returns False if unusable."""
        from inference.API.lfa_api import create_lyric_matcher

        matcher = create_lyric_matcher(self.language, reference_lyrics)
        if matcher is None:
            self._bound_reference = False
            self.matcher = type(self.matcher)(self.language)
            return False
        self.matcher = matcher
        self._bound_reference = True
        return True

    # -- alignment --------------------------------------------------------

    def align(self, asr_text: str, reference_lyrics: str = "") -> AlignResult:
        """Align ``asr_text`` (optionally against ``reference_lyrics``).

        Returns an :class:`AlignResult`; when no reference is bound and no ASR
        text is present, the result is empty (never raises).
        """
        text = str(asr_text or "").strip()
        if reference_lyrics and reference_lyrics.strip():
            self.bind_reference(reference_lyrics)

        if not text:
            return AlignResult(reason="empty asr")

        try:
            asr_text_list, asr_phonetic_list = self.matcher.process_asr_content(text)
        except Exception:
            asr_text_list, asr_phonetic_list = [text], [text]

        # reference mode: align ASR phonemes to the bound reference lyrics
        if self._bound_reference:
            lyric_text = list(getattr(self.matcher, "lyric_text_list", []) or [])
            lyric_phonetic = list(getattr(self.matcher, "lyric_phonetic_list", []) or [])
            if asr_phonetic_list and lyric_phonetic and lyric_text:
                try:
                    matched_text, matched_phonetic, reason = self.matcher.align_lyric_with_asr(
                        asr_phonetic=asr_phonetic_list,
                        lyric_text=lyric_text,
                        lyric_phonetic=lyric_phonetic,
                    )
                except Exception:
                    matched_text, matched_phonetic, reason = "", "", "matcher align error"
                confidence = self._confidence(len(matched_text or ""), len(lyric_text))
                return AlignResult(
                    words=self._words_from_phonetic(matched_phonetic, asr_text_list),
                    matched_lyric=str(matched_text or ""),
                    matched_phonetic=str(matched_phonetic or ""),
                    confidence=confidence,
                    reason=str(reason or ""),
                )

        # auto mode: return the ASR text with a length-based confidence
        words = [WordTiming(text=token, start=0.0, end=0.0) for token in asr_text_list if token]
        return AlignResult(
            words=words,
            matched_lyric=text,
            matched_phonetic=" ".join(asr_phonetic_list),
            confidence=min(1.0, len(asr_text_list) / 8.0),
            reason="auto-mode (no reference lyrics bound)",
        )

    @staticmethod
    def _confidence(matched: int, total: int) -> float:
        if total <= 0:
            return 0.0
        return round(min(1.0, matched / total), 4)

    @staticmethod
    def _words_from_phonetic(phonetic: str, fallback: list[str]) -> list[WordTiming]:
        tokens = [t for t in str(phonetic or "").split() if t] or fallback
        return [WordTiming(text=t, start=0.0, end=0.0) for t in tokens]
