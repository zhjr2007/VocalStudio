"""Offscreen pytest suite for the TTS page controls and ref-audio rule (wave E)."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

import torch  # noqa: F401 (must load before PyQt5 on this host)

from PyQt5.QtWidgets import QApplication

import soundfile as sf

from gui.pages.tts_page import (
    LANG_CHOICES,
    REF_MAX_SEC,
    REF_MIN_SEC,
    SPEED_DEFAULT,
    SPEED_MAX,
    SPEED_MIN,
    VERSION_CHOICES,
    RefAudioArea,
    TtsPage,
    audio_duration_sec,
)
from gui.widgets.audio_input import REJECTED_GRAY


@pytest.fixture(scope="module")
def app(qapp) -> QApplication:
    return qapp


def _write_wav(path: Path, seconds: float, samplerate: int = 24000) -> str:
    t = np.linspace(0.0, seconds, max(int(seconds * samplerate), 1), endpoint=False)
    signal = (0.3 * np.sin(2.0 * np.pi * 440.0 * t)).astype("float32")
    sf.write(str(path), signal, samplerate)
    return str(path)


# -- E-1: page defaults / ranges -------------------------------------------


def test_page_defaults(app):
    page = TtsPage()
    try:
        assert page.lang() == "zh"
        assert page.version() == "v2Pro"
        assert page.speed() == pytest.approx(SPEED_DEFAULT)
        assert page.target_text() == ""
        assert page.ref_path() == ""
    finally:
        page.deleteLater()


def test_combo_choices(app):
    page = TtsPage()
    try:
        langs = [page._lang_combo.itemData(i) for i in range(page._lang_combo.count())]
        versions = [
            page._version_combo.itemData(i) for i in range(page._version_combo.count())
        ]
        assert langs == [value for value, _label in LANG_CHOICES] == ["zh", "en", "ja"]
        assert versions == [value for value, _label in VERSION_CHOICES] == [
            "v2Pro",
            "v2ProPlus",
        ]
    finally:
        page.deleteLater()


def test_speed_slider_range_and_roundtrip(app):
    page = TtsPage()
    try:
        slider = page._speed_slider
        assert slider.minimum() / 20.0 == pytest.approx(SPEED_MIN)
        assert slider.maximum() / 20.0 == pytest.approx(SPEED_MAX)
        for value in (SPEED_MIN, SPEED_DEFAULT, SPEED_MAX):
            page.set_speed(value)
            assert page.speed() == pytest.approx(value)
    finally:
        page.deleteLater()


def test_setters_roundtrip(app):
    page = TtsPage()
    try:
        page.set_lang("ja")
        assert page.lang() == "ja"
        page.set_version("v2ProPlus")
        assert page.version() == "v2ProPlus"
        page.set_target_text(" 你好，VocalStudio。 ")
        assert page.target_text() == "你好，VocalStudio。"
    finally:
        page.deleteLater()


def test_config_shape(app):
    page = TtsPage()
    try:
        cfg = page.config()
        assert set(cfg) == {
            "tts_text",
            "tts_lang",
            "tts_version",
            "tts_speed",
            "ref_audio",
            "ref_text",
            "ref_lang",
        }
        assert cfg["tts_lang"] == "zh"
        assert cfg["tts_version"] == "v2Pro"
    finally:
        page.deleteLater()


# -- E-1: reference-audio 3~10 s rule ---------------------------------------


def test_ref_area_accepts_3s_rejects_1s(app, tmp_path):
    good = _write_wav(tmp_path / "ref3.wav", 3.0)
    bad = _write_wav(tmp_path / "ref1.wav", 1.0)

    area = RefAudioArea()
    warnings: list[str] = []
    area.warning.connect(lambda msg: warnings.append(msg))
    try:
        assert area.add_files([good]) == 1
        assert warnings == []
        assert area.ref_valid(good) is True

        added = area.add_files([bad])
        assert added == 0
        assert len(warnings) == 1
        assert "ref1.wav" in warnings[0]
        item = area.queue().item(1)
        assert item.foreground().color().name() == REJECTED_GRAY.name()
        assert "3~10" in item.toolTip() or "range" in item.toolTip()
    finally:
        area.deleteLater()


def test_ref_area_boundaries_inclusive(app, tmp_path):
    low = _write_wav(tmp_path / "ref_lo.wav", REF_MIN_SEC)
    high = _write_wav(tmp_path / "ref_hi.wav", REF_MAX_SEC)
    over = _write_wav(tmp_path / "ref_over.wav", 12.0)

    area = RefAudioArea()
    warnings: list[str] = []
    area.warning.connect(lambda msg: warnings.append(msg))
    try:
        assert area.add_files([low, high]) == 2
        assert warnings == []
        assert area.ref_valid(low) is True and area.ref_valid(high) is True
        assert area.add_files([over]) == 0
        assert len(warnings) == 1
    finally:
        area.deleteLater()


def test_ref_area_unknown_extension_and_missing_file(app, tmp_path):
    txt = tmp_path / "notes.txt"
    txt.write_text("not audio", encoding="utf-8")

    area = RefAudioArea()
    warnings: list[str] = []
    area.warning.connect(lambda msg: warnings.append(msg))
    try:
        assert area.add_files([str(txt)]) == 0
        # extension rejection is handled by the base class (no duration warning)
        assert warnings == []
        assert area.ref_valid(str(txt)) is False
        assert area.ref_valid(None) is False
    finally:
        area.deleteLater()


def test_audio_duration_sec(app, tmp_path):
    wav = _write_wav(tmp_path / "probe.wav", 4.0)
    assert audio_duration_sec(wav) == pytest.approx(4.0, abs=0.05)
    assert audio_duration_sec(str(tmp_path / "missing.wav")) is None


# -- E-2: text frontends (zh/ja/en G2P + phonemize) -------------------------

from core import LANGS, phonemize  # noqa: E402  (core export wiring check)

PHONEMIZE_CASES = [
    ("zh", "你好，世界"),
    ("ja", "こんにちは、セカイ"),
    ("en", "hello world"),
    ("zh", "VocalStudio 你好"),   # mixed: latin run + han run (zh frontend)
    ("ja", "こんにちは hello"),   # kana run always ja, latin run en
    ("en", "one two 3"),          # digits spoken by g2p_en normalization
]


@pytest.mark.parametrize(("lang", "text"), PHONEMIZE_CASES)
def test_phonemize_basic(lang, text):
    tokens = phonemize(text, lang)
    assert len(tokens) > 0
    for tok in tokens:
        assert isinstance(tok, str) and tok
        tok.encode("ascii")  # every token must be ASCII-safe


def test_phonemize_zh_tones_and_punctuation():
    tokens = phonemize("你好，世界", "zh")
    assert "," in tokens
    syllables = [t for t in tokens if re.match(r"^[a-z]+\d$", t)]
    assert len(syllables) == 4  # ni3 hao3 shi4 jie4


def test_phonemize_ja_kana_and_han():
    kana = phonemize("こんにちは", "ja")
    assert len(kana) >= 8 and all(tok.isalpha() for tok in kana)
    han = phonemize("セカイ", "ja")  # katakana run -> ja frontend
    assert len(han) == 5  # s e k a i


def test_phonemize_en_arpabet():
    tokens = phonemize("hello world", "en")
    assert len(tokens) >= 6
    assert all(re.fullmatch(r"[A-Z][A-Z0-9]*", tok) for tok in tokens)


def test_phonemize_empty_and_invalid_lang():
    assert phonemize("", "zh") == []
    assert phonemize("   ", "en") == []
    with pytest.raises(ValueError):
        phonemize("x", "fr")


def test_core_exports_text_frontend():
    assert set(LANGS) == {"zh", "ja", "en"}


# -- E-3: engine wrappers + VoiceCloneEngine ----------------------------------

REF_AUDIO = ROOT / "experiments/speaker/iic/speech_campplus_sv_zh-cn_16k-common/examples/speaker1_a_cn_16k.wav"


def test_vocoder_not_applicable_for_v2pro():
    from engines.tts.vocoder import VocoderEngine

    assert VocoderEngine.is_applicable("v2Pro") is False
    assert VocoderEngine.is_applicable("v3") is True


def test_run_tts_rapid_validation_errors():
    """Missing ref/text fail fast, long before any model load."""
    from dataclasses import dataclass

    from core.context import PipelineContext
    from engines.tts.voice_clone import run_tts

    @dataclass
    class _CfgTTSPartial:
        ref_audio: str = ""
        tts_text: str = "你好"

    with pytest.raises(FileNotFoundError):
        run_tts(PipelineContext(), _CfgTTSPartial())

    @dataclass
    class _CfgTTSNoText:
        ref_audio: str = str(REF_AUDIO)
        tts_text: str = ""

    with pytest.raises(ValueError):
        run_tts(PipelineContext(), _CfgTTSNoText())


def test_load_missing_weights_clear_error(tmp_path):
    from core.config import ModelPaths
    from engines.tts.voice_clone import VoiceCloneEngine

    fake = str(tmp_path / "nope")
    paths = ModelPaths(
        gsv_v2pro_gpt=fake,
        gsv_v2pro_sovits=fake,
        chinese_hubert=fake,
        chinese_roberta=fake,
    )
    engine = VoiceCloneEngine()
    with pytest.raises(FileNotFoundError, match="t2s"):
        engine.load("v2Pro", model_paths=paths)


def test_clone_zh_audio_and_speed(tmp_path):
    """Real v2Pro clone: ref 3.72s + short zh text -> wav >0.5 s (E-3 smoke)."""
    from engines.tts.voice_clone import VoiceCloneEngine

    engine = VoiceCloneEngine()
    out1 = tmp_path / "clone1.wav"
    out2 = tmp_path / "clone2.wav"
    stages: list[str] = []
    try:
        info = engine.load("v2Pro", device="auto")
        assert info["version"] == "v2Pro"
        sr, audio = engine.clone(
            "你好，VocalStudio。", "zh",
            ref_audio=str(REF_AUDIO),
            speed=1.0,
            seed=42,
            out_path=str(out1),
            progress_cb=lambda fraction, stage: stages.append(str(stage)),
        )
        assert sr == 32000 and len(audio) / sr > 0.5
        assert {"preprocess", "t2s", "sovits", "save"}.issubset(set(stages))
        assert out1.is_file()

        sr2, audio2 = engine.clone(
            "你好，VocalStudio。", "zh",
            ref_audio=str(REF_AUDIO),
            speed=1.2,
            seed=42,
            out_path=str(out2),
            progress_cb=lambda fraction, stage: None,
        )
        assert out2.is_file()
        # same seed, 1.2x speed -> duration ratio close to 1/1.2 (serial path)
        ratio = len(audio2) / len(audio)
        assert 0.70 <= ratio <= 0.95, f"speed ratio unexpected: {ratio:.3f}"
    finally:
        engine.unload()


# -- E-4: CLI tts subcommand ---------------------------------------------------


def test_cli_tts_parser_has_speed():
    from main import build_parser

    args = build_parser().parse_args(["tts", "--text", "hi", "--ref-audio", "ref.wav", "--speed", "1.3"])
    assert args.speed == pytest.approx(1.3)
    assert args.version == "v2Pro"
    parser_defaults = build_parser().parse_args(["tts", "--text", "hi"])
    assert parser_defaults.speed == pytest.approx(1.0)


def test_cli_tts_requires_text():
    from main import build_parser

    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["tts", "--ref-audio", str(REF_AUDIO)])
    assert exc.value.code in (2, -2)


def test_pipeline_config_speed_bounds():
    from core.config import PipelineConfig

    assert PipelineConfig().tts_speed == 1.0
    assert PipelineConfig(tts_speed=1.4).validate() is None
    with pytest.raises(ValueError):
        PipelineConfig(tts_speed=2.0).validate()


# -- E-5: page-level synthesis flow --------------------------------------------


def test_page_start_job_writes_output(app, qapp, tmp_path):
    """Offscreen E2E: start() -> JobRunner -> done -> waveform + player."""
    import time as _time

    from gui.pages.tts_page import TtsJobCfg, TtsPage

    page = TtsPage()
    ref_area = page.ref_area()
    ref_area.add_files([str(REF_AUDIO)])
    ref_area.select(0)
    page.set_target_text("你好，VocalStudio。")
    assert page.can_start() is True
    runner = page.start()
    assert runner is not None
    deadline = _time.time() + 420.0
    while runner.isRunning() and _time.time() < deadline:
        qapp.processEvents()
        _time.sleep(0.2)
    qapp.processEvents()
    assert not runner.isRunning()
    assert page._stage_raw == "done", f"stage={page._stage_raw!r}"
    out = Path(page.output_path())
    assert out.is_file()
    info = sf.info(str(out))
    assert info.frames / info.samplerate > 0.5
    assert page.waveform_view()._tracks, "waveform must be populated"
    assert page.audio_player().duration_ms > 500
    page._start_button.setEnabled(True)  # reset for fixture teardown
    page.deleteLater()
