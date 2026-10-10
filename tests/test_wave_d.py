"""Offscreen pytest suite for the ASR page controls (wave D).

D-1: engine combo (SenseVoice/Paraformer) + speaker-separation checkbox
with 0~16 spinbox (disabled while separation is off).
D-2: punctuation checkbox + TXT/SRT/JSON output-format row.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

import torch  # noqa: F401 (must load before PyQt5 on this host)
import numpy as np  # noqa: E402

from PyQt5.QtCore import Qt  # noqa: E402


@pytest.fixture(scope="module")
def app(qapp):
    return qapp


# -- D-1: engine combo + speaker separation controls -------------------------


def test_engine_combo_defaults(app):
    from gui.pages.asr_page import AsrPage, ENGINE_CHOICES

    page = AsrPage()
    try:
        data = [page._engine_combo.itemData(i) for i in range(page._engine_combo.count())]
        assert data == ["sensevoice", "paraformer"]
        assert len(ENGINE_CHOICES) == 2
        assert page.engine() == "sensevoice"
        model = page._engine_combo.model()
        tips = [model.data(model.index(i, 0), Qt.ToolTipRole) for i in range(len(data))]
        assert all(tips), "per-item tooltips must be non-empty"
        assert len(set(tips)) == 2, "tooltips should differ per engine"
    finally:
        page.deleteLater()


def test_set_engine_roundtrip(app):
    from gui.pages.asr_page import AsrPage

    page = AsrPage()
    try:
        page.set_engine("paraformer")
        assert page.engine() == "paraformer"
        page.set_engine("sensevoice")
        assert page.engine() == "sensevoice"
    finally:
        page.deleteLater()


def test_speaker_sep_enabled_matrix(app):
    from gui.pages.asr_page import AsrPage

    page = AsrPage()
    try:
        spin = page._spk_num_spin
        assert (spin.minimum(), spin.maximum()) == (0, 16)
        # default state: separation off -> spinbox disabled, value 3
        assert not page.speaker_sep_enabled()
        assert page.spk_num() == 3
        assert not spin.isEnabled()

        page.set_speaker_sep(True)
        assert spin.isEnabled()
        page.set_spk_num(5)
        assert page.spk_num() == 5

        page.set_speaker_sep(False)
        assert not spin.isEnabled()
        # value preserved while disabled
        assert page.spk_num() == 5
    finally:
        page.deleteLater()


def test_asr_page_retranslate_labels(app):
    from gui.i18n import i18n

    from gui.pages.asr_page import AsrPage

    prev = i18n.language
    page = AsrPage()
    try:
        i18n.set_language("zh")
        page.retranslate()
        assert page._engine_label.text() == "引擎"
        assert page._speaker_sep_check.text() == "说话人分离"
        assert page._spk_num_label.text() == "说话人数"
        assert page._punct_check.text() == "标点恢复"
        fmt_texts = [page._format_checks[v].text() for v in ("txt", "srt", "json")]
        assert fmt_texts == ["TXT", "SRT", "JSON"]

        i18n.set_language("en")
        page.retranslate()
        assert page._engine_label.text() == "Engine"
        assert page._speaker_sep_check.text() == "Speaker separation"
        assert page._spk_num_label.text() == "Speakers"
        assert page._punct_check.text() == "Punctuation restore"
        fmt_texts = [page._format_checks[v].text() for v in ("txt", "srt", "json")]
        assert fmt_texts == ["TXT", "SRT", "JSON"]
        model = page._engine_combo.model()
        tips = [model.data(model.index(i, 0), Qt.ToolTipRole) for i in range(2)]
        assert all(tips) and len(set(tips)) == 2
    finally:
        i18n.set_language(prev)
        page.deleteLater()


# -- D-2: punctuation + output formats ---------------------------------------


def test_punctuation_default_on(app):
    from gui.pages.asr_page import AsrPage

    page = AsrPage()
    try:
        assert page.punctuation_enabled()
        page.set_punctuation(False)
        assert not page.punctuation_enabled()
        page.set_punctuation(True)
        assert page.punctuation_enabled()
    finally:
        page.deleteLater()


def test_output_formats_defaults(app):
    from gui.pages.asr_page import AsrPage

    page = AsrPage()
    try:
        # TXT checked by default, SRT/JSON unchecked
        assert page.output_formats() == ("txt",)
        page.set_output_formats(("srt", "json"))
        assert page.output_formats() == ("srt", "json")
        page.set_output_formats("TXT")  # string input, case-insensitive
        assert page.output_formats() == ("txt",)
    finally:
        page.deleteLater()


def test_config_dict(app):
    from gui.pages.asr_page import AsrPage

    page = AsrPage()
    try:
        cfg = page.config()
        assert cfg["asr_engine"] == "sensevoice"
        assert cfg["enable_punctuation"] is True
        assert cfg["output_srt"] is False
        assert cfg["output_json"] is False
        assert cfg["num_speakers"] == 0  # separation off by default

        page.set_engine("paraformer")
        page.set_output_formats(("txt", "srt"))
        page.set_punctuation(False)
        page.set_speaker_sep(True)
        page.set_spk_num(2)
        cfg = page.config()
        assert cfg["asr_engine"] == "paraformer"
        assert cfg["enable_punctuation"] is False
        assert cfg["output_srt"] is True
        assert cfg["num_speakers"] == 2
    finally:
        page.deleteLater()


# -- D-3: CAM++ diarization engine ------------------------------------------


def _campplus_examples_dir() -> Path:
    from core.config import PROJECT_ROOT, ModelPaths

    p = Path(ModelPaths.from_settings().campplus)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p / "examples"


@pytest.fixture(scope="module")
def camp_engine():
    """Load CAM++ + FSMN-VAD once for the whole module (funasr import is slow)."""
    from engines.speaker import CAMPPEngine

    eng = CAMPPEngine()
    eng.load(device="auto")
    yield eng
    eng.unload()


@pytest.fixture(scope="module")
def two_speaker_clip(tmp_path_factory):
    """speaker1_a + 2 s silence + speaker2_a, written as a 16 kHz mono wav.

    The gap is long enough for FSMN-VAD to split the clip into two segments.
    """
    import soundfile as sf

    ex = _campplus_examples_dir()
    s1, sr = sf.read(ex / "speaker1_a_cn_16k.wav")
    s2, _ = sf.read(ex / "speaker2_a_cn_16k.wav")
    gap = np.zeros(2 * sr, dtype=s1.dtype)  # 2 seconds of silence
    clip_path = tmp_path_factory.mktemp("d3") / "two_speaker.wav"
    sf.write(str(clip_path), np.concatenate([s1, gap, s2]), sr)
    return clip_path


def test_cluster_embeddings_modes(app):
    from engines.speaker.campp_engine import cluster_embeddings

    rng = np.random.default_rng(0)
    a1 = rng.normal(size=8)
    b1 = rng.normal(size=8)
    embs = np.vstack([a1, a1 + 0.05 * rng.normal(size=8), b1, b1 + 0.05 * rng.normal(size=8)])

    labels = cluster_embeddings(embs, spk_num=2)
    assert set(labels.tolist()) == {0, 1}
    assert labels[0] == labels[1] and labels[2] == labels[3]

    auto = cluster_embeddings(embs, spk_num=0)
    assert len(set(auto.tolist())) >= 1
    # identical embeddings must collapse to a single speaker in auto mode
    same = np.vstack([a1, a1])
    assert set(cluster_embeddings(same, spk_num=0).tolist()) == {0}


def _naive_agglomerate(sim, target_clusters=None, threshold=None):
    """Reference: original O(n^3) per-pair average-linkage implementation."""
    n = sim.shape[0]
    clusters = [[i] for i in range(n)]
    while len(clusters) > 1:
        if target_clusters is not None and len(clusters) <= target_clusters:
            break
        best_i = best_j = -1
        best_s = -2.0
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                s = float(sim[np.ix_(clusters[i], clusters[j])].mean())
                if s > best_s:
                    best_i, best_j, best_s = i, j, s
        if threshold is not None and best_s < threshold:
            break
        clusters[best_i] = sorted(clusters[best_i] + clusters[best_j])
        del clusters[best_j]
    clusters.sort(key=lambda c: min(c))
    return clusters


def test_agglomerate_matches_naive_reference(app):
    from engines.speaker.campp_engine import _agglomerate, _cosine_similarity_matrix

    rng = np.random.default_rng(7)
    a = rng.normal(size=16)
    b = rng.normal(size=16)
    embs = np.vstack(
        [a, a + 0.05 * rng.normal(size=16), b, b + 0.05 * rng.normal(size=16)]
    )
    sim = _cosine_similarity_matrix(embs)

    fixed = _agglomerate(sim, target_clusters=2)
    assert fixed == _naive_agglomerate(sim, target_clusters=2)
    auto = _agglomerate(sim, threshold=0.65)
    assert auto == _naive_agglomerate(sim, threshold=0.65)
    assert _agglomerate(sim[:1, :1], threshold=0.65) == [[0]]


def test_agglomerate_large_n_is_fast(app):
    import time

    from engines.speaker.campp_engine import _agglomerate, _cosine_similarity_matrix

    rng = np.random.default_rng(11)
    blobs = [rng.normal(size=16) for _ in range(3)]
    embs = np.vstack(
        [blob + 0.05 * rng.normal(size=16) for blob in blobs for _ in range(500)]
    )
    sim = _cosine_similarity_matrix(embs)
    t0 = time.time()
    clusters = _agglomerate(sim, threshold=0.65)
    elapsed = time.time() - t0
    # naive version needs hours here; vectorized must finish in seconds
    assert elapsed < 15
    assert len(clusters) >= 1


def test_diarize_fixed_k_two_speakers(camp_engine, two_speaker_clip):
    import soundfile as sf

    out_dir = two_speaker_clip.parent / "out"
    speakers = camp_engine.diarize(str(two_speaker_clip), spk_num=2, out_dir=out_dir)

    assert len(speakers) == 2
    info = sf.info(str(two_speaker_clip))
    duration_ms = int(round(info.frames * 1000 / info.samplerate))
    all_segs = []
    for spk_id, data in speakers.items():
        segs = data["segments"]
        assert len(segs) >= 1
        # sorted, non-overlapping, within the clip duration
        for (s, e), prev in zip(segs, [None] + list(segs[:-1])):
            assert 0 <= s < e <= duration_ms
            if prev is not None:
                assert s >= prev[1], "segments of one speaker must not overlap"
        all_segs.extend(segs)
        wav = data["audio_path"]
        assert wav and Path(wav).exists()
        merged = sf.info(wav)
        assert merged.samplerate == 16000 and merged.frames > 0

    coverage = sum(e - s for s, e in all_segs) / duration_ms
    print(f"D-3 two-speaker clip: {len(speakers)} speakers, VAD coverage={coverage:.1%}")
    assert coverage >= 0.5


def test_diarize_auto_mode(camp_engine, two_speaker_clip):
    speakers = camp_engine.diarize(str(two_speaker_clip), spk_num=0)
    print(f"D-3 auto mode: {len(speakers)} speaker(s)")
    assert len(speakers) >= 1


# -- D-4: SenseVoice / Paraformer ASR engine ---------------------------------


def _clip_duration_ms(path) -> int:
    import soundfile as sf

    info = sf.info(str(path))
    return int(round(info.frames * 1000 / info.samplerate))


@pytest.fixture(scope="module")
def asr_engine():
    """Load SenseVoice + FSMN-VAD (+ CT punctuation) once for the module."""
    from engines.speech_asr import SpeechASREngine

    eng = SpeechASREngine()
    eng.load(engine="sensevoice", device="auto")
    yield eng
    eng.unload()


@pytest.fixture(scope="module")
def paraformer_engine():
    from engines.speech_asr import SpeechASREngine

    eng = SpeechASREngine()
    eng.load(engine="paraformer", device="auto")
    yield eng
    eng.unload()


def test_transcribe_sensevoice_structure(asr_engine, two_speaker_clip):
    result = asr_engine.transcribe(str(two_speaker_clip), enable_punctuation=True)

    text = result["text"]
    assert text.strip(), "transcribed text must be non-empty"
    assert "<|" not in text and "|>" not in text, f"tag residue in: {text!r}"
    assert result["language"] in {"zh", "en", "ja", "yue", "ko"}

    duration_ms = _clip_duration_ms(two_speaker_clip)
    segments = result["segments"]
    assert len(segments) >= 1
    prev_end = 0
    for start_ms, end_ms, seg_text in segments:
        assert 0 <= start_ms < end_ms <= duration_ms
        assert start_ms >= prev_end, "segments must be disjoint and ordered"
        assert seg_text.strip()
        prev_end = end_ms

    print(
        f"D-4 sensevoice: lang={result['language']} emo={result['emotion']!r} "
        f"event={result['event']!r} segments={len(segments)} text={text[:60]!r}"
    )


def _srt_time_to_ms(tc: str) -> int:
    h, m, rest = tc.split(":")
    s, ms = rest.split(",")
    return ((int(h) * 60 + int(m)) * 60 + int(s)) * 1000 + int(ms)


def test_transcribe_punctuation_and_run_speech_asr(two_speaker_clip):
    import json

    from core.config import PipelineConfig
    from core.context import PipelineContext
    from engines.speech_asr import run_speech_asr

    out_dir = two_speaker_clip.parent / "asr_out"
    cfg = PipelineConfig(
        audio_path=two_speaker_clip,
        output_dir=out_dir,
        asr_engine="sensevoice",
        enable_punctuation=True,
        output_formats=["txt", "srt", "json"],
        device="auto",
    )
    ctx = PipelineContext(audio_path=str(two_speaker_clip))
    run_speech_asr(ctx, cfg)

    text = ctx.asr_text
    assert text and "<|" not in text and "|>" not in text
    cjk_punct = [ch for ch in "。，！？、；：" if ch in text]
    assert cjk_punct, f"expected CJK punctuation in: {text!r}"

    stem = two_speaker_clip.stem
    txt_path = out_dir / "asr" / f"{stem}.txt"
    srt_path = out_dir / "asr" / f"{stem}.srt"
    json_path = out_dir / "asr" / f"{stem}.json"
    for p in (txt_path, srt_path, json_path):
        assert p.exists(), f"missing output file {p}"

    duration_ms = _clip_duration_ms(two_speaker_clip)
    blocks = [b.strip() for b in srt_path.read_text(encoding="utf-8").split("\n\n") if b.strip()]
    assert blocks, "SRT must contain at least one block"
    prev_end_ms = 0
    for i, block in enumerate(blocks, start=1):
        lines = block.splitlines()
        assert len(lines) >= 3, f"SRT block {i} too short: {block!r}"
        assert lines[0] == str(i), "SRT index line must be sequential"
        parts = lines[1].split(" --> ")
        assert len(parts) == 2, f"bad SRT timecode line: {lines[1]!r}"
        s_ms = _srt_time_to_ms(parts[0])
        e_ms = _srt_time_to_ms(parts[1])
        assert 0 <= s_ms < e_ms <= duration_ms
        assert s_ms >= prev_end_ms, "SRT timecodes must be monotonic"
        prev_end_ms = e_ms
        assert any(line.strip() for line in lines[2:]), f"SRT block {i} has no text"

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["text"] == ctx.asr_text
    assert isinstance(data["segments"], list) and data["segments"]
    for seg in data["segments"]:
        assert {"start_ms", "end_ms", "text", "speaker"} <= set(seg)
        assert 0 <= seg["start_ms"] < seg["end_ms"] <= duration_ms
    assert isinstance(data["speakers"], dict)

    res = ctx.results["speech_asr"]
    assert res["engine"] == "sensevoice"
    assert res["num_segments"] == len(data["segments"])
    print(f"D-4 run_speech_asr: {len(blocks)} SRT block(s), text={ctx.asr_text[:60]!r}")


def test_transcribe_paraformer_plain_text(paraformer_engine, two_speaker_clip):
    result = paraformer_engine.transcribe(str(two_speaker_clip), enable_punctuation=False)

    text = result["text"]
    assert text.strip(), "paraformer text must be non-empty"
    assert "<|" not in text and "|>" not in text
    assert len(result["segments"]) >= 1
    print(f"D-4 paraformer: segments={len(result['segments'])} text={text[:60]!r}")
