"""Wave C2 tests — v2midi lyric flow (lyric area, ASR/HFA engines, matcher)."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

import torch  # noqa: F401 (must load before PyQt5/soundfile on this host)


@pytest.fixture(scope="session")
def app():
    from PyQt5.QtWidgets import QApplication

    return QApplication.instance() or QApplication(sys.argv[:1])


# ---------------------------------------------------------------------------
# C2-1 v2midi page lyric area
# ---------------------------------------------------------------------------

def _make_v2midi_page(app):
    from gui.pages.v2midi_page import V2MidiPage

    page = V2MidiPage()
    page.set_output_dir(str(Path(tempfile.mkdtemp(prefix="v2m_c2_"))))
    return page


def test_v2midi_lyric_area_defaults(app):
    page = _make_v2midi_page(app)
    assert page.lyrics_align_enabled() is True
    assert page.reference_lyrics() == ""
    assert page.pitch_curve_enabled() is False


def test_v2midi_lyric_cfg_roundtrip(app):
    page = _make_v2midi_page(app)
    page.set_lyrics_align(False)
    page.set_reference_lyrics("我 爱 你")
    page.set_pitch_curve(True)
    cfg = page.config()
    assert cfg["output_lyrics"] is False
    assert cfg["reference_lyrics"] == "我 爱 你"
    assert cfg["output_pitch_curve"] is True


# ---------------------------------------------------------------------------
# C2-4 lyric matcher
# ---------------------------------------------------------------------------

def test_lyric_matcher_reference_mode():
    from engines.vocal2midi.lyric_matcher import LyricMatcherEngine

    engine = LyricMatcherEngine("zh")
    bound = engine.bind_reference("你好世界")
    assert bound is True
    result = engine.align("你好 世界", reference_lyrics="你好世界")
    assert result.words, "reference alignment should produce words"
    assert result.matched_lyric
    assert 0.0 <= result.confidence <= 1.0


def test_lyric_matcher_auto_mode_empty():
    from engines.vocal2midi.lyric_matcher import LyricMatcherEngine

    engine = LyricMatcherEngine("zh")
    result = engine.align("", reference_lyrics="")
    assert result.words == []
    assert result.confidence == 0.0
    assert result.reason  # empty-asr reason recorded, never raises


def test_lyric_matcher_one_word_replacement():
    from engines.vocal2midi.lyric_matcher import LyricMatcherEngine

    engine = LyricMatcherEngine("zh")
    result = engine.align("你好 世界", reference_lyrics="你好 世界")
    assert result.words
    assert all(w.text for w in result.words)
    assert result.confidence > 0.0

# ---------------------------------------------------------------------------
# C2-2 / C2-3 engine paths + sanitizers (no heavy weight inference)
# ---------------------------------------------------------------------------

def test_qwen_default_model_path_resolves():
    from engines.vocal2midi.qwen3asr_engine import resolve_asr_model_path

    path = resolve_asr_model_path()
    if not path.exists():
        pytest.skip("Qwen model dir not bundled (weights live outside the repository)")


def test_qwen_clean_text_strips_special_tokens():
    from engines.vocal2midi.qwen3asr_engine import Qwen3AsrEngine

    eng = Qwen3AsrEngine()
    raw = "<|im_start|>你好世界<|im_end|>\n  <|USER|> "
    assert eng._clean_text(raw) == "你好世界"


def test_hfa_default_model_path_resolves():
    from engines.vocal2midi.hfa_engine import resolve_hfa_model_path

    path = resolve_hfa_model_path()
    if not path.exists():
        pytest.skip("HubertFA model dir not bundled (weights live outside the repository)")


def test_hfa_words_from_pred_parses():
    from engines.vocal2midi.hfa_engine import HfaEngine

    class _Word:
        def __init__(self, text, start, end):
            self.text, self.start, self.end = text, start, end

    pred = ("chunk_0.wav", None, [_Word("你", 0.1, 0.3), _Word("好", 0.4, 0.6)])
    words = HfaEngine.words_from_pred(pred)
    assert len(words) == 2
    assert words[0] == {"text": "你", "start": 0.1, "end": 0.3}
    assert HfaEngine.words_from_pred(None) == []
    assert HfaEngine.words_from_pred(("x", None, None)) == []


def test_hfa_compat_installs_textgrid_stub():
    import sys as _sys

    from engines.vocal2midi.hfa_compat import install_textgrid_stub

    assert "textgrid" not in _sys.modules
    install_textgrid_stub()
    assert _sys.modules["textgrid"] is not None
