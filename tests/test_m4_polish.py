"""Offscreen pytest suite for the M4 polish features.

Covers:
  1. Live stem mixer (gui/widgets/stem_mixer.py) + WaveformView click-to-seek.
  2. Follow-along "You" recording track + mixdown download.
  3. Result-player transpose (resample-based) + segment loop (audio_player).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

import sounddevice as sd
import soundfile as sf

import torch  # noqa: F401 (must load before PyQt5 on this host)

from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QImage
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication

from gui.widgets.audio_player import AudioPlayerWidget
from gui.widgets.stem_mixer import StemMixerWidget
from gui.widgets.waveform_view import WaveformView


@pytest.fixture(scope="module")
def app(qapp) -> QApplication:
    return qapp


def _has_output_device() -> bool:
    try:
        _, out = sd.default.device
    except Exception:
        return False
    return int(out) >= 0


def _mk_wav(path: Path, seconds: float, samplerate: int = 48000, freq: float = 440.0, amp: float = 0.3) -> str:
    t = np.linspace(0.0, seconds, max(int(seconds * samplerate), 1), endpoint=False)
    signal = (amp * np.sin(2.0 * np.pi * freq * t)).astype("float32")
    sf.write(str(path), signal, samplerate)
    return str(path)


# ---------------------------------------------------------------------------
# Feature 3: AudioPlayerWidget transpose (resample-based) + segment loop
# ---------------------------------------------------------------------------


def test_audio_player_transpose_property_defaults(app, tmp_path):
    wav = _mk_wav(tmp_path / "tmp_m4_p1.wav", 2.0, 48000)
    widget = AudioPlayerWidget()
    try:
        widget.set_file(wav)
        assert widget.transpose == 0
        assert widget._playback_ratio == pytest.approx(1.0)
        widget.set_transpose(7)
        assert widget.transpose == 7
        assert widget._playback_ratio == pytest.approx(2.0 ** (7 / 12.0))
        # clamping
        widget.set_transpose(99)
        assert widget.transpose == 12
        widget.set_transpose(-99)
        assert widget.transpose == -12
        assert widget._playback_ratio == pytest.approx(2.0 ** (-1.0))
    finally:
        widget.stop()
        widget.deleteLater()


def test_audio_player_transpose_callback_interpolates(app, tmp_path):
    wav = _mk_wav(tmp_path / "tmp_m4_p2.wav", 2.0, 48000)
    widget = AudioPlayerWidget()
    try:
        widget.set_file(wav)
        widget.set_position(0)
        widget.set_transpose(12)  # ratio == 2.0 -> consumes 2 input frames per output frame
        frames = 256
        outdata = np.zeros((frames, 2), dtype="float32")
        widget._callback(outdata, frames, None, None)
        # advance is start + frames * ratio
        assert widget._pos_frames == pytest.approx(frames * 2.0, abs=1)
        assert np.any(outdata != 0.0)
        # A raised pitch means the signal differs from the raw first chunk.
        raw, _sr = sf.read(wav, dtype="float32", always_2d=True)
        assert not np.allclose(outdata[: min(frames, raw.shape[0])], raw[:frames] * 0.8)
    finally:
        widget.stop()
        widget.deleteLater()


def test_audio_player_callback_default_ratio_unchanged(app, tmp_path):
    wav = _mk_wav(tmp_path / "tmp_m4_p3.wav", 2.0, 48000)
    widget = AudioPlayerWidget()
    try:
        widget.set_file(wav)
        widget.set_position(0)
        widget.set_transpose(0)
        frames = 256
        outdata = np.zeros((frames, 2), dtype="float32")
        widget._callback(outdata, frames, None, None)
        assert widget._pos_frames == pytest.approx(frames, abs=1)
        raw, _sr = sf.read(wav, dtype="float32", always_2d=True)
        assert np.allclose(outdata, raw[:frames] * 0.8, atol=1e-5)
    finally:
        widget.stop()
        widget.deleteLater()


def test_audio_player_segment_loop_wraps(app, tmp_path):
    wav = _mk_wav(tmp_path / "tmp_m4_p4.wav", 2.0, 48000)
    widget = AudioPlayerWidget()
    try:
        widget.set_file(wav)
        duration = widget.duration_ms
        start_ms = duration // 4
        end_ms = duration // 2
        widget.set_loop(start_ms, end_ms)
        widget.set_loop_enabled(True)
        widget.set_position(start_ms)
        # Use a block large enough to cross the loop boundary.
        frames = int((end_ms - start_ms) * 48000 / 1000) + 2000
        outdata = np.zeros((frames, 2), dtype="float32")
        widget._callback(outdata, frames, None, None)
        pos = widget.position_ms
        assert start_ms <= pos < end_ms
        # loop bounds reported correctly
        assert widget.loop_start_ms == start_ms
        assert widget.loop_end_ms == end_ms
        assert widget.loop_enabled is True
    finally:
        widget.stop()
        widget.deleteLater()


def test_audio_player_loop_bounds_invalid_clears(app, tmp_path):
    wav = _mk_wav(tmp_path / "tmp_m4_p5.wav", 2.0, 48000)
    widget = AudioPlayerWidget()
    try:
        widget.set_file(wav)
        widget.set_loop(1000, 500)  # invalid -> clears
        assert widget.loop_end_ms <= widget.loop_start_ms
        widget.set_loop(200, 1200)
        assert widget.loop_bounds() == (200, 1200)
    finally:
        widget.stop()
        widget.deleteLater()


# ---------------------------------------------------------------------------
# Feature 1: WaveformView N-lane render + click-to-seek
# ---------------------------------------------------------------------------


def test_waveform_multilane_and_seek(app):
    widget = WaveformView()
    try:
        peaks_a = np.linspace(0.1, 1.0, 256, dtype="float32")
        peaks_b = np.linspace(1.0, 0.1, 256, dtype="float32")
        peaks_c = np.linspace(0.5, 0.5, 256, dtype="float32")
        widget.set_waveforms(
            [
                {"label": "A", "peaks": peaks_a, "color": "#4C8DFF"},
                {"label": "B", "peaks": peaks_b, "color": "#3DD68C"},
                {"label": "C", "peaks": peaks_c, "color": "#F7B32B"},
            ]
        )
        widget.set_duration(3000)
        widget.resize(400, 200)

        received: list[int] = []
        widget.seek_requested.connect(received.append)
        QTest.mouseClick(widget, Qt.LeftButton, pos=QPoint(200, 100))
        assert received, "click must emit seek_requested"
        middle = received[-1]
        assert 1100 <= middle <= 1900  # approx half of 3000 ms

        # 3 tracks render distinct lane colors.
        image = widget.grab().toImage().convertToFormat(QImage.Format_RGB32)
        colors: set[tuple[int, int, int]] = set()
        for x in range(0, image.width(), 4):
            for y in range(0, image.height(), 4):
                c = image.pixelColor(x, y)
                colors.add((c.red(), c.green(), c.blue()))
        assert len(colors) >= 3
    finally:
        widget.deleteLater()


# ---------------------------------------------------------------------------
# Feature 1 + 2: StemMixerWidget
# ---------------------------------------------------------------------------


@pytest.fixture()
def stem_pair(tmp_path):
    a = _mk_wav(tmp_path / "stem_vocals.wav", 2.0, 48000, freq=440.0, amp=0.4)
    b = _mk_wav(tmp_path / "stem_accomp.wav", 2.0, 48000, freq=220.0, amp=0.5)
    # load both as stereo so the mixer can normalize channel width
    data_a, sr_a = sf.read(a, dtype="float32", always_2d=True)
    data_b, sr_b = sf.read(b, dtype="float32", always_2d=True)
    return [
        {"name": "vocals", "path": a, "pcm": data_a, "sr": int(sr_a)},
        {"name": "accompaniment", "path": b, "pcm": data_b, "sr": int(sr_b)},
    ]


def test_stem_mixer_set_stems_and_rows(app, stem_pair):
    mixer = StemMixerWidget()
    try:
        mixer.set_stems(stem_pair)
        assert mixer.stem_names == ["vocals", "accompaniment"]
        assert mixer.duration_ms > 0
        view = mixer.waveform_view()
        assert len(view._tracks) == 2
        assert len(mixer.row_controls()) == 2
        assert len(mixer.track_gains()) == 2
    finally:
        mixer.deleteLater()


def test_stem_mixer_mute_solo_gain(app, stem_pair):
    mixer = StemMixerWidget()
    try:
        mixer.set_stems(stem_pair)
        mixer.set_gain(0, 0.5)
        gains = mixer.track_gains()
        assert gains["vocals"] == pytest.approx(0.4)  # 0.5 * volume 0.8
        assert gains["accompaniment"] == pytest.approx(0.8)

        mixer.set_mute(0, True)
        gains = mixer.track_gains()
        assert gains["vocals"] == 0.0
        assert gains["accompaniment"] == pytest.approx(0.8)

        mixer.set_mute(0, False)
        mixer.set_solo(0, True)
        gains = mixer.track_gains()
        assert gains["vocals"] == pytest.approx(0.4)
        assert gains["accompaniment"] == 0.0  # soloed -> other tracks muted

        # mute wins over solo on the soloed track
        mixer.set_mute(0, True)
        gains = mixer.track_gains()
        assert gains["vocals"] == 0.0
    finally:
        mixer.deleteLater()


def test_stem_mixer_mixdown_download(app, stem_pair, tmp_path):
    mixer = StemMixerWidget()
    try:
        mixer.set_stems(stem_pair)
        mixer.set_gain(0, 0.0)  # mute vocals -> output should be accompaniment only
        out = tmp_path / "mix.wav"
        written = mixer.mixdown(str(out))
        assert written == str(out)
        assert out.is_file()
        data, sr = sf.read(str(out), dtype="float32", always_2d=True)
        assert sr == 48000
        assert data.shape[0] > 1000
        # With a sine of amplitude 0.5 and gain 0.8, peak ~0.4.
        assert float(np.max(np.abs(data))) == pytest.approx(0.4, abs=0.05)
    finally:
        mixer.deleteLater()


def test_stem_mixer_you_track_and_mix(app, stem_pair, tmp_path):
    mixer = StemMixerWidget()
    try:
        mixer.set_stems(stem_pair)
        # Simulate a recorded take (mono) aligned to the timeline.
        rec = (0.25 * np.sin(2.0 * np.pi * 660.0 * np.linspace(0.0, 1.0, 48000, endpoint=False))).astype("float32")
        for _ in range(2):
            rec = np.concatenate([rec, rec])
        buf = mixer._recording_to_buffer(rec)
        mixer._add_track("You", buf)
        assert "You" in mixer.stem_names
        assert len(mixer.row_controls()) == 3
        assert len(mixer.waveform_view()._tracks) == 3

        # Mute both original stems so only "You" reaches the mix.
        mixer.set_mute(0, True)
        mixer.set_mute(1, True)
        out = tmp_path / "you_mix.wav"
        mixer.mixdown(str(out))
        data, sr = sf.read(str(out), dtype="float32", always_2d=True)
        # The You track is the only audible source.
        you_seg = data[: len(rec) // 2]
        assert float(np.max(np.abs(you_seg))) == pytest.approx(0.2, abs=0.05)
    finally:
        mixer.deleteLater()


# ---------------------------------------------------------------------------
# Integration: separation page exposes a populated stem mixer
# ---------------------------------------------------------------------------


def test_separation_page_integrates_stem_mixer(app, stem_pair, tmp_path):
    from gui.pages.separation_page import SeparationPage

    page = SeparationPage()
    try:
        mixer = page.stem_mixer()
        assert isinstance(mixer, StemMixerWidget)

        # Drive _on_job_done directly with a fake separation result.
        stems = {
            "vocals": stem_pair[0]["path"],
            "accompaniment": stem_pair[1]["path"],
            "drums": _mk_wav(tmp_path / "stem_drums.wav", 2.0, 48000, freq=110.0, amp=0.3),
        }
        result = {"stems": stems, "duration_ms": 2000}
        page._on_job_done(result, False)

        assert set(mixer.stem_names) >= {"vocals", "accompaniment"}
        # the classic dual-track comparison waveform is preserved
        tracks = page.waveform_view()._tracks
        assert len(tracks) == 2
        player = page.audio_player()
        assert player.duration_ms > 0
        assert player.transpose in range(-12, 13)
    finally:
        page.deleteLater()


# ---------------------------------------------------------------------------
# Integration: v2midi page exposes the result player (transpose + loop)
# ---------------------------------------------------------------------------


def test_v2midi_page_has_result_player(app, tmp_path):
    from gui.pages.v2midi_page import V2MidiPage

    page = V2MidiPage()
    try:
        player = page.audio_player()
        assert isinstance(player, AudioPlayerWidget)

        wav = _mk_wav(tmp_path / "v2m_in.wav", 2.0, 48000)
        result = {"input": wav, "duration_ms": 2000}
        page._on_job_done(result, False)
        assert player.duration_ms > 0

        player.set_transpose(5)
        assert player.transpose == 5
        player.set_loop(200, 1200)
        assert player.loop_bounds() == (200, 1200)
    finally:
        page.deleteLater()
