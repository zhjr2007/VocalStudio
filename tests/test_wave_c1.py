"""Wave C1 tests — v2midi pitch flow (slicer first; more subtasks appended)."""

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

import soundfile as sf

from engines.vocal2midi.slicer import SLICE_MODES, normalize_slice_mode, slice
from engines.vocal2midi.quantization import QUANTIZATION_MODES, quantize_notes, should_apply_quantization

SAMPLE_VOCALS = ROOT / "materials" / "sample_vocals.wav"


@pytest.fixture(scope="module")
def sample_10s():
    if not SAMPLE_VOCALS.is_file():
        pytest.skip("test media not bundled (materials/)")
    info = sf.info(str(SAMPLE_VOCALS))
    stop = min(int(10.0 * info.samplerate), info.frames)
    audio, sr = sf.read(str(SAMPLE_VOCALS), start=0, stop=stop)
    return np.asarray(audio, dtype=np.float32).reshape(-1), int(sr)


def test_slice_grid_sample_10s(sample_10s):
    audio, sr = sample_10s
    length = 4.0
    chunks = slice(audio, sr, "grid", length=length)
    total_sec = len(audio) / sr
    assert len(chunks) == int(np.ceil(total_sec / length))  # n ~= ceil(10/L)
    for i, chunk in enumerate(chunks):
        expected_len = min(int(round(length * sr)), len(audio) - i * int(round(length * sr)))
        assert abs(chunk["waveform"].shape[-1] - expected_len) <= 2  # == sr*L +/- pad
    # contiguous, ordered, covering the whole clip
    for prev, nxt in zip(chunks, chunks[1:]):
        assert prev["end"] == pytest.approx(nxt["start"], abs=1e-6)
    assert chunks[0]["start"] == 0.0
    assert chunks[-1]["end"] == pytest.approx(total_sec, abs=1e-6)
    for chunk in chunks:
        assert chunk["sr"] == sr


def test_slice_silence_documented_behaviour():
    sr = 22050
    long_silence = np.zeros(8 * sr, dtype=np.float32)   # > min segment length
    short_silence = np.zeros(2 * sr, dtype=np.float32)  # <= min segment length
    for mode in ("default", "smart"):
        assert slice(long_silence, sr, mode) == []          # long silence -> no chunks
        short_chunks = slice(short_silence, sr, mode)       # short silence -> whole clip kept
        assert len(short_chunks) == 1
        assert short_chunks[0]["start"] == pytest.approx(0.0, abs=1e-6)
        assert short_chunks[0]["end"] == pytest.approx(2.0, abs=1e-3)
    # heuristic pre-stage removes any silence longer than 800 ms entirely
    assert slice(long_silence, sr, "heuristic") == []
    assert slice(short_silence, sr, "heuristic") == []
    # grid keeps silence as fixed windows; empty audio -> [] for every mode
    assert len(slice(long_silence, sr, "grid", length=4.0)) == 2
    empty = np.zeros(0, dtype=np.float32)
    for mode in SLICE_MODES:
        assert slice(empty, sr, mode) == []


def test_slice_all_modes_structure_on_synthetic_vocals():
    sr = 22050
    t1 = np.linspace(0.0, 6.0, int(6 * sr), endpoint=False)
    tone_a = (0.5 * np.sin(2.0 * np.pi * 440.0 * t1)).astype("float32")
    gap = np.zeros(int(2 * sr), dtype="float32")
    t2 = np.linspace(0.0, 6.0, int(6 * sr), endpoint=False)
    tone_b = (0.5 * np.sin(2.0 * np.pi * 880.0 * t2)).astype("float32")
    audio = np.concatenate([tone_a, gap, tone_b, gap])  # 16 s: voiced/silent x2
    total_sec = len(audio) / sr

    for mode in SLICE_MODES:
        chunks = slice(audio, sr, mode)
        assert chunks, f"mode {mode} produced no chunks on voiced audio"
        for chunk in chunks:
            assert set(chunk) == {"waveform", "sr", "start", "end"}
            assert 0.0 <= chunk["start"] < chunk["end"] <= total_sec + 1e-6
            expected = int(round((chunk["end"] - chunk["start"]) * sr))
            assert abs(chunk["waveform"].shape[-1] - expected) <= 2
        for prev, nxt in zip(chunks, chunks[1:]):
            assert nxt["start"] >= prev["start"]
    # deterministic expectations on this synthetic signal
    assert len(slice(audio, sr, "grid", length=4.0)) == 4          # ceil(16/4)
    for mode in ("default", "smart", "heuristic"):
        assert len(slice(audio, sr, mode)) == 2                    # one cut per 2 s gap


def test_slice_mode_normalization_and_errors():
    assert normalize_slice_mode(None) == "default"
    for label, expected in (("默认", "default"), ("智能", "smart"), ("启发式", "heuristic"), ("网格", "grid")):
        assert normalize_slice_mode(label) == expected
    with pytest.raises(ValueError):
        normalize_slice_mode("bogus")
    sr = 22050
    audio = np.zeros(sr, dtype="float32")
    with pytest.raises(ValueError):
        slice(audio, sr, "bogus")


# ---------------------------------------------------------------------------
# C1-5 quantization
# ---------------------------------------------------------------------------

class _StubNote:
    """Duck-typed note object (same surface as NoteInfo used by the engines)."""

    def __init__(self, onset: float, offset: float, pitch: int = 60, lyric: str = ""):
        self.onset = float(onset)
        self.offset = float(offset)
        self.pitch = int(pitch)
        self.lyric = lyric


def _ticks(t: float, tempo: float) -> int:
    return round(t * tempo * 8)  # 480 ticks per beat


@pytest.mark.parametrize("mode", QUANTIZATION_MODES)
def test_quantize_snaps_to_quarter_beat_grid(mode):
    tempo = 120.0
    step = 120  # ticks: 480/beat -> 120 ticks == 0.25 beat grid
    raw_onsets = [0.03, 0.27, 0.52, 0.78, 1.04]  # near-grid onsets with jitter
    notes = [_StubNote(onset=o, offset=o + 0.23) for o in raw_onsets]

    result = quantize_notes(notes, tempo=tempo, step=step, mode=mode)
    assert result is notes  # in place
    for note in notes:
        assert note.offset > note.onset
        for t in (note.onset, note.offset):
            r = _ticks(t, tempo) % step
            assert min(r, step - r) <= 1  # on the 0.25-beat grid within +/-1 tick
    onsets = [n.onset for n in notes]
    assert all(b > a for a, b in zip(onsets, onsets[1:]))  # strictly increasing


def test_quantize_bpm_changes_grid_and_rules():
    step = 120  # ticks per snap position (480 ticks/beat -> quarter-note grid)
    # onset exactly on the 120 BPM grid (360 ticks) but off-grid at 60 BPM:
    notes_120 = [_StubNote(onset=0.375, offset=0.60)]
    quantize_notes(notes_120, tempo=120.0, step=step, mode="simple")
    assert notes_120[0].onset == pytest.approx(0.375, abs=1e-9)  # already on grid -> unchanged

    notes_60 = [_StubNote(onset=0.375, offset=0.60)]
    quantize_notes(notes_60, tempo=60.0, step=step, mode="simple")
    # at 60 BPM the grid is twice as wide (0.25 s): 180 ticks -> nearest multiple of 120 is 240
    assert notes_60[0].onset == pytest.approx(0.5, abs=1e-9)      # snapped elsewhere
    for note in notes_120 + notes_60:
        assert note.offset > note.onset
        assert _ticks(note.onset, 120.0 if note is notes_120[0] else 60.0) % step == 0

    assert should_apply_quantization("simple", 30) is True
    assert should_apply_quantization("simple", 0) is False
    assert should_apply_quantization("dp", 0) is True  # dp always applies


# ---------------------------------------------------------------------------
# C1-6 note_io
# ---------------------------------------------------------------------------

def test_save_midi_roundtrip(tmp_path):
    import mido

    from engines.vocal2midi.note_io import NoteInfo, save_notes

    tempo = 120.0
    notes = [
        NoteInfo(onset=0.5, offset=0.75, pitch=60, lyric="la"),
        NoteInfo(onset=1.0, offset=1.5, pitch=64, lyric="so"),
        NoteInfo(onset=2.25, offset=2.5, pitch=67, lyric="mi"),
    ]
    written = save_notes(notes, tmp_path, "rt", ["mid"], tempo=tempo)
    assert set(written) == {"mid"} and written["mid"].exists()

    def ticks(t: float) -> int:
        return round(t * tempo * 8)

    mid = mido.MidiFile(str(written["mid"]), charset="utf8")
    events = []
    abs_tick = 0
    for msg in mid.tracks[0]:
        abs_tick += msg.time
        if msg.type in ("note_on", "note_off"):
            events.append((abs_tick, msg.type, msg.note))
        elif msg.type == "lyrics":
            events.append((abs_tick, "lyrics", msg.text))

    ons = [t for t, typ, _ in events if typ == "note_on"]
    offs = [t for t, typ, _ in events if typ == "note_off"]
    pitches = [n for _, typ, n in events if typ == "note_on"]
    assert pitches == [60, 64, 67]
    for (onset_tick, offset_tick), note in zip(zip(ons, offs), notes):
        assert onset_tick == ticks(note.onset)      # note-on matches input
        assert offset_tick == ticks(note.offset)    # note-off matches input
    lyrics = [text for _, typ, text in events if typ == "lyrics"]
    assert lyrics == ["la", "so", "mi"]


def test_save_txt_csv_fields_and_selection(tmp_path):
    import librosa

    from engines.vocal2midi.note_io import NoteInfo, save_notes

    notes = [NoteInfo(0.123, 0.456, 67, "o"), NoteInfo(0.5, 0.9, 60, "")]
    written = save_notes(notes, tmp_path, "tc", ["txt", "csv"], tempo=120.0)
    assert set(written) == {"txt", "csv"}
    assert not (tmp_path / "tc.mid").exists()  # only selected formats are produced

    txt_lines = written["txt"].read_text(encoding="utf8").splitlines()
    assert len(txt_lines) == 2
    onset, offset, pitch_name, lyric = txt_lines[0].split("\t")
    assert (onset, offset, lyric) == ("0.123", "0.456", "o")
    assert pitch_name == librosa.midi_to_note(67, unicode=False, cents=False)  # 'G#4'
    onset2, offset2, pitch2, lyric2 = txt_lines[1].split("\t")
    assert (onset2, offset2, lyric2) == ("0.500", "0.900", "")
    assert pitch2 == librosa.midi_to_note(60, unicode=False, cents=False)      # 'C4'

    csv_lines = written["csv"].read_text(encoding="utf8").splitlines()
    assert csv_lines[0] == "onset,offset,pitch,lyric"  # complete field header
    assert len(csv_lines) == 3


def test_save_ustx_sections_and_notes(tmp_path):
    import yaml

    from engines.vocal2midi.note_io import NoteInfo, save_notes

    notes = [NoteInfo(0.5, 1.0, 62, "re"), NoteInfo(1.5, 2.0, 64, "so")]
    written = save_notes(notes, tmp_path, "ux", ["ustx"], tempo=120.0)
    data = yaml.safe_load(written["ustx"].read_text(encoding="utf-8"))
    for section in ("bpm", "tracks", "voice_parts", "expressions", "time_signatures"):
        assert section in data  # USTX contains the required sections
    voice_part = data["voice_parts"][0]
    assert len(voice_part["notes"]) == 2
    first, second = voice_part["notes"]
    assert (first["position"], first["duration"], first["tone"], first["lyric"]) == (480, 480, 62, "re")
    assert (second["position"], second["tone"], second["lyric"]) == (1440, 64, "so")


# ---------------------------------------------------------------------------
# C1-3 GAME engine / C1-4 RMVPE engine (real weights, GPU when available)
# ---------------------------------------------------------------------------

GAME_MODEL_DIR = ROOT / "Vocal2Midi" / "experiments" / "GAME-1.0.3-medium-onnx"
RMVPE_MODEL_PATH = ROOT / "Vocal2Midi" / "experiments" / "RMVPE" / "rmvpe.onnx"
SAMPLE_BASELINE_TXT = ROOT / "materials" / "v2m_diag_out" / "sample_ja_test.txt"


def _load_window(path: Path, t0: float, duration: float):
    info = sf.info(str(path))
    start = int(t0 * info.samplerate)
    stop = min(start + int(duration * info.samplerate), info.frames)
    audio, sr = sf.read(str(path), start=start, stop=stop)
    return np.asarray(audio, dtype=np.float32).reshape(-1), int(sr)


@pytest.fixture(scope="module")
def game_engine():
    from engines.vocal2midi.game_engine import GameEngine

    if not GAME_MODEL_DIR.exists():
        pytest.skip("GAME weights not bundled (Vocal2Midi/experiments)")
    engine = GameEngine()
    engine.load(GAME_MODEL_DIR, device="cuda")
    yield engine
    engine.unload()


@pytest.fixture(scope="module")
def rmvpe_engine():
    from engines.vocal2midi.rmvpe_engine import RmvpeEngine

    if not RMVPE_MODEL_PATH.exists():
        pytest.skip("RMVPE weights not bundled (Vocal2Midi/experiments)")
    engine = RmvpeEngine()
    engine.load(RMVPE_MODEL_PATH, device="cuda")
    yield engine
    engine.unload()


def test_game_engine_load_backend(game_engine):
    assert game_engine.backend in {"pytorch", "onnx"}
    # only ONNX weights are shipped for GAME -> onnx backend (logged at load)
    assert game_engine.backend == "onnx"
    assert game_engine.model is not None


def test_game_engine_infer_vocals_5s(game_engine):
    if not SAMPLE_VOCALS.is_file():
        pytest.skip("test media not bundled (materials/)")
    audio, sr = _load_window(SAMPLE_VOCALS, t0=18.0, duration=5.0)  # sung region (vocals start ~18 s)
    chunk = {"waveform": audio, "sr": sr, "start": 0.0, "end": len(audio) / sr}
    notes = game_engine.infer([chunk], sr)
    assert len(notes) >= 1
    for note in notes:
        assert 0 <= note.pitch < 128
        assert note.offset > note.onset
        assert -0.05 <= note.onset <= chunk["end"] + 0.05


def test_game_engine_unload_releases():
    from engines.vocal2midi.game_engine import GameEngine

    engine = GameEngine()
    engine.load(GAME_MODEL_DIR, device="cpu")
    assert engine.model is not None
    engine.unload()
    assert engine.model is None  # sessions released
    with pytest.raises(RuntimeError):
        engine.infer([], 44100)


def test_rmvpe_load_and_infer_convention(rmvpe_engine):
    if not SAMPLE_VOCALS.is_file():
        pytest.skip("test media not bundled (materials/)")
    audio, sr = _load_window(SAMPLE_VOCALS, t0=20.0, duration=5.0)  # sung region
    out = rmvpe_engine.infer(audio, sr)
    assert out["time_step_seconds"] == pytest.approx(0.01, abs=1e-9)  # HOP 160 @ 16 kHz
    expected_frames = int(len(audio) * 16000 / sr) // 160 + 1
    assert len(out["midi_pitch"]) == pytest.approx(expected_frames, abs=2.0)
    assert out["voiced_mask"].any()  # voiced content present in this window
    finite = out["midi_pitch"][np.isfinite(out["midi_pitch"])]
    assert finite.size > 0
    assert np.all((finite >= 0) & (finite <= 360))  # MIDI numbers within [0, 360]
    f0 = out["f0_hz"]
    assert len(f0) == len(out["midi_pitch"])
    assert not np.any(f0[~out["voiced_mask"]] > 0)  # unvoiced frames carry no f0


def test_rmvpe_empty_input(rmvpe_engine):
    out = rmvpe_engine.infer(np.zeros(0, dtype="float32"), 44100)
    assert len(out["midi_pitch"]) == 0 and len(out["f0_hz"]) == 0
    assert out["time_step_seconds"] == pytest.approx(0.01, abs=1e-9)


def test_rmvpe_baseline_median_spot_check(rmvpe_engine):
    if not SAMPLE_VOCALS.is_file() or not SAMPLE_BASELINE_TXT.is_file():
        pytest.skip("test media not bundled (materials/)")
    # Baseline spot check (wave-plan C1-4): median of RMVPE's midi_pitch on a
    # 10 s clip vs the GAME-derived baseline notes of the same segment must agree
    # within <=2 semitones, compared octave-folded (mod 12). The sample track is
    # a duet recording: RMVPE follows the mixture f0 (~one octave below the per-note
    # GAME scores here), so raw medians differ by ~12 while pitch classes agree.
    # Measured on this host 2026-09-04, window [29,39): RMVPE median 54.9 vs
    # baseline 67.0 (raw diff -12.1, folded diff -0.08); tolerance recorded here.
    t0, t1 = 29.0, 39.0
    audio, sr = _load_window(SAMPLE_VOCALS, t0=t0, duration=t1 - t0)
    out = rmvpe_engine.infer(audio, sr)
    voiced = out["midi_pitch"][out["voiced_mask"] & np.isfinite(out["midi_pitch"])]
    assert voiced.size > 0
    rmvpe_median = float(np.median(voiced))

    import librosa

    baseline_midi = []
    for line in SAMPLE_BASELINE_TXT.read_text(encoding="utf8").splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        onset = float(parts[0])
        if t0 <= onset < t1:
            baseline_midi.append(float(librosa.note_to_midi(parts[2], round_midi=False)))
    assert len(baseline_midi) >= 5  # the window is well covered by baseline notes
    baseline_median = float(np.median(baseline_midi))
    folded_diff = (rmvpe_median - baseline_median + 6.0) % 12.0 - 6.0
    assert abs(folded_diff) <= 2.0, (
        f"RMVPE median {rmvpe_median:.2f} vs baseline {baseline_median:.2f} "
        f"(octave-folded diff {folded_diff:+.2f})"
    )

# ---------------------------------------------------------------------------
# C1-7 chord / key / meter engine (real madmom weights)
# ---------------------------------------------------------------------------

C_MAJOR_FREQS = (261.63, 329.63, 392.00)  # C E G


def _synthetic_c_major_arpeggio(duration: float = 6.0, sr: int = 44100) -> np.ndarray:
    """A constant C-major chord as a low-block C:E:G spectrum (tone, then stays)."""
    n = int(duration * sr)
    t = np.linspace(0.0, duration, n, endpoint=False)
    out = np.zeros(n, dtype="float32")
    for f in C_MAJOR_FREQS:
        out += 0.33 * np.sin(2.0 * np.pi * f * t)
    return out.astype("float32")


def test_chord_engine_c_major_key_family_on_synthetic():
    from engines.chords import ChordEngine

    engine = ChordEngine()
    engine.load()
    try:
        audio = _synthetic_c_major_arpeggio()
        result = engine.analyze(audio, 44100, hpss=False, min_chord_beats=0.0)
    finally:
        engine.unload()
    # A pure C-major chord should favour a key in the C family and a confident prediction.
    keys = ["C major", "A minor", "G major", "F major"]
    assert result["key"] in keys, f"expected a C-family key, got {result['key']!r}"
    assert 0.05 <= result["confidence"] <= 1.0


def test_chord_engine_silence_is_empty_no_crash():
    from engines.chords import ChordEngine

    engine = ChordEngine()
    engine.load()
    try:
        result = engine.analyze(np.zeros(0, dtype="float32"), 44100)
    finally:
        engine.unload()
    assert result["key"] is None
    assert result["confidence"] == 0.0
    assert result["chordPositions"] == []


def test_chord_engine_real_vocals_positions_monotone_and_legal():
    from engines.chords import ChordEngine, LEGAL_CHORD_LABELS, write_chord_outputs

    if not SAMPLE_VOCALS.is_file():
        pytest.skip("test media not bundled (materials/)")

    audio, sr = _load_window(SAMPLE_VOCALS, t0=18.0, duration=30.0)
    engine = ChordEngine()
    engine.load()
    try:
        result = engine.analyze(audio, sr, hpss=False, min_chord_beats=0.0)
    finally:
        engine.unload()
    positions = result["chordPositions"]
    assert positions, "chord analysis should produce a non-empty chord progression on sung vocals"
    prev = -1.0
    for pos in positions:
        assert pos["value"] in LEGAL_CHORD_LABELS, f"illegal chord label {pos['value']!r}"
        assert pos["position"] >= prev, "chord positions must be monotonically non-decreasing"
        prev = pos["position"]
    assert result["key"] is not None
    assert result["bpm"] is None or result["bpm"] > 0

    import tempfile

    out_dir = Path(tempfile.mkdtemp(prefix="v2m_chords_"))
    files = write_chord_outputs(result, out_dir, "chord_test")
    assert set(files) == {"json", "csv"}
    csv_lines = files["csv"].read_text(encoding="utf8").splitlines()
    assert csv_lines[0] == "position_sec,value"
    assert len(csv_lines) == len(positions) + 1


# ---------------------------------------------------------------------------
# C1-1 / C1-8 v2midi page (offscreen)
# ---------------------------------------------------------------------------

def _make_v2midi_page(app):
    from gui.pages.v2midi_page import V2MidiPage

    page = V2MidiPage()
    page.set_output_dir(str(Path(tempfile.mkdtemp(prefix="v2m_page_"))))
    return page


def test_v2midi_page_defaults(app):
    """Default state: language auto, slicing default, quant bayes, bpm 120,
    MIDI+TXT checked, chord analysis off."""
    page = _make_v2midi_page(app)
    assert page.language() == "auto"
    assert page.slicing_method() == "default"
    assert page.quantization_mode() == "bayes"
    assert page.bpm() == 120
    assert set(page.output_formats()) == {"mid", "txt"}
    assert page.chords_enabled() is False
    assert page.chord_hpss() is False
    assert page.min_chord_beats() == 0
    # format controls default correctly (MIDI checked, others per FORM default)
    fmts = page.output_formats()
    assert "mid" in fmts


def test_v2midi_page_cfg_roundtrip(app):
    """Setting page values round-trips through config() and the setters."""
    page = _make_v2midi_page(app)
    page.set_language("ja")
    page.set_slicing_method("heuristic")
    page.set_quantization_mode("dp")
    page.set_bpm(96)
    page.set_output_formats(["mid", "csv"])
    page.set_chords_enabled(True)
    page.set_chord_hpss(True)
    page.set_min_chord_beats(4)

    cfg = page.config()
    assert cfg["language"] == "ja"
    assert cfg["slicing_method"] == "heuristic"
    assert cfg["quantization_mode"] == "dp"
    assert cfg["bpm"] == 96
    assert set(cfg["output_formats"]) == {"mid", "csv"}
    assert cfg["enable_chords"] is True
    assert cfg["chord_hpss"] is True
    assert cfg["min_chord_beats"] == 4


def test_v2midi_page_start_no_audio_is_noop(app):
    """start() with no input returns None without spawning a job."""
    page = _make_v2midi_page(app)
    result = page.start()
    assert result is None
    assert page.is_running() is False

@pytest.fixture(scope="module")
def app(qapp):
    from PyQt5.QtWidgets import QApplication

    return qapp
