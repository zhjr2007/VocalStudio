"""Offscreen pytest suite for the separation page audio input area (wave B-1)."""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

import torch  # noqa: F401 (must load before PyQt5 on this host)

from PyQt5.QtGui import QBrush
from PyQt5.QtWidgets import QApplication

import soundfile as sf

from gui.widgets.audio_input import ACCEPTED_EXTENSIONS, REJECTED_GRAY, AudioInputArea


@pytest.fixture(scope="module")
def app(qapp) -> QApplication:
    return qapp


@pytest.fixture(scope="module")
def wav_path(tmp_path_factory) -> str:
    candidates = sorted(glob.glob(str(ROOT / "materials" / "*.wav")))
    if candidates:
        return candidates[0]
    path = tmp_path_factory.mktemp("wave_b_audio") / "sine.wav"
    samplerate = 22050
    t = np.linspace(0.0, 1.0, samplerate, endpoint=False)
    signal = (0.4 * np.sin(2.0 * np.pi * 440.0 * t)).astype("float32")
    sf.write(path, signal, samplerate)
    return str(path)


def test_add_files_rejects_unknown_extension(app, wav_path, tmp_path):
    txt_file = tmp_path / "notes.txt"
    txt_file.write_text("not an audio file", encoding="utf-8")

    area = AudioInputArea()
    try:
        added = area.add_files([wav_path, str(txt_file)])
        assert added == 1
        # queue keeps both rows, only the wav counts as accepted
        assert area.accepted_count() == 1
        rows = area.rows()
        assert len(rows) == 2
        assert rows[0] == (wav_path, True)
        assert rows[1][0] == str(txt_file) and rows[1][1] is False

        list_widget = area.queue()
        accepted_item = list_widget.item(0)
        rejected_item = list_widget.item(1)
        # accepted row keeps the default (NoBrush) foreground, rejected row goes grey
        assert accepted_item.foreground().style() == QBrush().style()
        assert rejected_item.foreground().style() != QBrush().style()
        assert rejected_item.foreground().color().name() == REJECTED_GRAY.name()
    finally:
        area.deleteLater()


def test_select_index_highlight_state(app, wav_path):
    area = AudioInputArea()
    try:
        assert area.selected_index() == -1
        assert area.queue().currentRow() == -1

        area.add_files([wav_path])
        area.select(0)
        assert area.selected_index() == 0
        assert area.queue().currentRow() == 0
        selected = area.queue().selectionModel().selectedIndexes()
        assert [index.row() for index in selected] == [0]
        assert area.queue().item(0).isSelected()

        # out-of-range selection is a no-op, state unchanged
        area.select(-1)
        area.select(99)
        assert area.selected_index() == 0
    finally:
        area.deleteLater()


def test_whitelist_all_extensions_accepted(app, tmp_path):
    assert tuple(sorted(ACCEPTED_EXTENSIONS)) == ("flac", "m4a", "mp3", "ogg", "wav")

    paths = []
    for ext in ACCEPTED_EXTENSIONS:
        path = tmp_path / f"clip.{ext}"
        path.touch()
        paths.append(str(path))

    area = AudioInputArea()
    try:
        added = area.add_files(paths)
        assert added == 5
        assert area.accepted_count() == 5
        rows = area.rows()
        assert len(rows) == 5
        for (path, ok), expected in zip(rows, paths):
            assert path == expected and ok is True
    finally:
        area.deleteLater()


def test_separation_page_wiring_and_retranslate(app, wav_path, tmp_path, monkeypatch):
    from gui.i18n import en_us, i18n, zh_cn

    import gui.pages.separation_page as separation_page_mod
    from gui.pages.separation_page import SeparationPage

    from PyQt5.QtWidgets import QWidget

    # the fluent CardWidget global (qconfig) is unusable once wave A has run in-process;
    # stub the cards so this test focuses on the input-area wiring only
    class _StubCard(QWidget):
        def __init__(self, title_key: str = "", line_keys=(), parent=None, **kwargs):
            super().__init__(parent)

        def retranslate(self) -> None:
            pass

    monkeypatch.setattr(separation_page_mod, "InfoCard", _StubCard)

    txt_file = tmp_path / "readme.txt"
    txt_file.write_text("notes", encoding="utf-8")
    prev_language = i18n.language
    page = SeparationPage()
    try:
        added = page.add_files([wav_path, str(txt_file)])
        assert added == 1
        assert page.accepted_count() == 1
        assert len(page.rows()) == 2
        assert page.selected_index() == -1

        page.select(0)
        assert page.selected_index() == 0

        area = page.input_area()
        page.retranslate()
        assert area._title_label.text() == zh_cn.TEXTS["audio_input"]
        zh_hint = area._hint_label.text()
        assert zh_hint == zh_cn.TEXTS["input_drop_hint"]

        i18n.set_language("en")
        page.retranslate()
        assert area._title_label.text() == en_us.TEXTS["audio_input"]
        assert area._hint_label.text() != zh_hint, "english hint identical to chinese"
    finally:
        i18n.set_language(prev_language)
        page.deleteLater()


def _make_page(monkeypatch):
    """Build a real SeparationPage with the fluent InfoCards stubbed out."""

    from PyQt5.QtWidgets import QWidget

    import gui.pages.separation_page as separation_page_mod

    class _StubCard(QWidget):
        def __init__(self, title_key: str = "", line_keys=(), parent=None, **kwargs):
            super().__init__(parent)

        def retranslate(self) -> None:
            pass

    monkeypatch.setattr(separation_page_mod, "InfoCard", _StubCard)
    return separation_page_mod.SeparationPage()


def _stub_uv5_weights(monkeypatch, tmp_path, present_files):
    from core import config as config_mod

    weights = tmp_path / "uvr5"
    weights.mkdir()
    for name in present_files:
        (weights / name).touch()
    fake_paths = config_mod.ModelPaths(uvr5_weights=str(weights))
    monkeypatch.setattr(config_mod.ModelPaths, "from_settings", staticmethod(lambda: fake_paths))


def test_preset_grid_default_and_weight_gating(app, tmp_path, monkeypatch):
    _stub_uv5_weights(
        monkeypatch,
        tmp_path,
        ("HP5_only_main_vocal.pth", "HP2_all_vocals.pth"),
    )
    page = _make_page(monkeypatch)
    try:
        grid = page.preset_grid()
        assert grid.card_count() == 4
        assert tuple(grid.preset_names()) == ("Karaoke", "FullVocal", "Band4", "Producer5")
        # Karaoke is the default selection
        assert page.selected_preset() == "Karaoke"
        # HP5/HP2 weights present -> usable; drums/bass/piano missing -> greyed out
        assert grid.is_enabled("Karaoke") is True
        assert grid.is_enabled("FullVocal") is True
        assert grid.is_enabled("Band4") is False
        assert grid.is_enabled("Producer5") is False
    finally:
        page.deleteLater()


def test_preset_enable_after_weights_present(app, tmp_path, monkeypatch):
    _stub_uv5_weights(
        monkeypatch,
        tmp_path,
        (
            "HP5_only_main_vocal.pth",
            "HP2_all_vocals.pth",
            "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
            "drums_bs_roformer.ckpt",
            "bass_bs_roformer.ckpt",
            "piano_bs_roformer.ckpt",
        ),
    )
    page = _make_page(monkeypatch)
    try:
        grid = page.preset_grid()
        assert all(grid.is_enabled(name) for name in grid.preset_names())
        grid.select("Band4")
        assert page.selected_preset() == "Band4"
        grid.select("Producer5")
        assert page.selected_preset() == "Producer5"
    finally:
        page.deleteLater()


def test_denoise_and_stage_labels_retranslate(app, monkeypatch):
    from gui.i18n import en_us, i18n, zh_cn

    prev_language = i18n.language
    page = _make_page(monkeypatch)
    try:
        assert page.denoise_enabled() is False
        page.set_denoise(True)
        assert page.denoise_enabled() is True
        page.set_denoise(False)

        zh_stages = [page._stage_combo.itemText(index) for index in range(3)]
        assert zh_stages == [zh_cn.TEXTS["tier_standard"], zh_cn.TEXTS["tier_hq"], zh_cn.TEXTS["tier_ultra"]]

        desc_karaoke_zh = page.preset_grid().card("Karaoke")._desc_label.text()
        i18n.set_language("en")
        page.retranslate()
        assert [page._stage_combo.itemText(index) for index in range(3)][0] == en_us.TEXTS["tier_standard"]
        desc_karaoke_en = page.preset_grid().card("Karaoke")._desc_label.text()
        assert desc_karaoke_en != desc_karaoke_zh, "english preset description identical to chinese"
    finally:
        i18n.set_language(prev_language)
        page.deleteLater()


# ---------------------------------------------------------------- UVR5Engine (B-5)

from core.config import ModelPaths  # noqa: E402

from engines.separation.uvr5_engine import PRESET_WEIGHTS, UVR5Engine  # noqa: E402


def _engine_from_real_weights():
    return UVR5Engine(ModelPaths.from_settings())


def test_uvr5_load_errors_and_registry(app):
    from engines import get_engine

    assert get_engine("separation") is UVR5Engine
    assert set(PRESET_WEIGHTS) == {"Karaoke", "FullVocal", "Band4", "Producer5"}

    # running before load raises a clear error
    engine = _engine_from_real_weights()
    import tempfile

    with tempfile.TemporaryDirectory() as out_dir:
        try:
            engine.run("missing.wav", out_dir)
            assert False, "expected RuntimeError before load"
        except RuntimeError as exc:
            assert "not loaded" in str(exc)

    # a missing model file reports the basename in the message
    with tempfile.TemporaryDirectory() as empty_weights:
        engine2 = UVR5Engine(ModelPaths(uvr5_weights=empty_weights))
        try:
            engine2.load("HP5_only_main_vocal.pth", device="cpu")
            assert False, "expected ValueError"
        except ValueError as exc:
            assert "HP5_only_main_vocal.pth" in str(exc)

    # unknown weight family is rejected
    with tempfile.TemporaryDirectory() as weights_dir:
        (Path(weights_dir) / "model.onnx").touch()
        engine3 = UVR5Engine(ModelPaths(uvr5_weights=weights_dir))
        try:
            engine3.load("model.onnx", device="cpu")
            assert False, "expected ValueError for unknown family"
        except ValueError as exc:
            assert "Unsupported weight family" in str(exc)


def test_uvr5_intensity_boundaries_and_unload(app):
    import soundfile as sf
    import tempfile

    samplerate = 44100
    n_samples = int(0.2 * samplerate)
    t = np.linspace(0.0, 0.2, n_samples, endpoint=False)
    mono = (0.5 * np.sin(2.0 * np.pi * 440.0 * t)).astype("float32")
    clip = np.stack([mono, mono])  # stereo float32 @44.1k

    engine = _engine_from_real_weights()
    with tempfile.TemporaryDirectory(prefix="wave_b_uvr_") as out_dir:
        engine.load("Karaoke", device="cpu")
        assert engine.is_loaded()

        # intensity boundaries 0 and 20 must both run without crashing
        for intensity in (0.0, 20):
            paths = engine.run(clip, Path(out_dir) / f"i{intensity}")
            assert set(paths) == {"vocals", "accompaniment"}
            vocals_arr, _sr = sf.read(paths["vocals"], dtype="float32")
            accompaniment_arr, _sr2 = sf.read(paths["accompaniment"], dtype="float32")
            assert _sr == samplerate and _sr2 == samplerate
            for arr in (vocals_arr, accompaniment_arr):
                duration = len(arr) / samplerate
                assert abs(duration - 0.2) < 0.05

        # out-of-range intensity is rejected with a clear message
        try:
            engine.run(clip, Path(out_dir) / "bad", intensity=21)
            assert False, "expected ValueError"
        except ValueError as exc:
            assert "[0, 20]" in str(exc)

        # unload releases the session; run afterwards must fail cleanly
        engine.unload()
        assert not engine.is_loaded() and engine._loaded == {}
        try:
            engine.run(clip, Path(out_dir) / "after")
            assert False, "expected RuntimeError after unload"
        except RuntimeError as exc:
            assert "not loaded" in str(exc)

        # the engine can be re-loaded for further runs
        engine.load("FullVocal", device="cpu")
        paths = engine.run(clip, Path(out_dir) / "reload", intensity=7)
        assert set(paths) == {"vocals", "accompaniment"}
        engine.unload()


def test_uvr5_bs_roformer_family_runs(app):
    import soundfile as sf
    import tempfile

    samplerate = 44100
    n_samples = int(0.2 * samplerate)
    t = np.linspace(0.0, 0.2, n_samples, endpoint=False)
    mono = (0.5 * np.sin(2.0 * np.pi * 330.0 * t)).astype("float32")
    clip = np.stack([mono, mono])

    engine = _engine_from_real_weights()
    with tempfile.TemporaryDirectory(prefix="wave_b_uvr_bs_") as out_dir:
        # raw ckpt basename spec (BS-RoFormer family), not a preset name
        engine.load("model_bs_roformer_ep_317_sdr_12.9755.ckpt", device="cpu")
        assert engine.is_loaded()
        paths = engine.run(clip, Path(out_dir) / "bs", intensity=0)
        # single BS model trained on the vocal target -> vocals + residual
        assert set(paths) == {"vocals", "accompaniment"}
        for name in ("vocals", "accompaniment"):
            arr, sr = sf.read(paths[name], dtype="float32")
            assert sr == samplerate and abs(len(arr) / samplerate - 0.2) < 0.05
        engine.unload()


def test_intensity_slider_defaults_and_value_display(app, monkeypatch):
    from PyQt5.QtCore import Qt

    page = _make_page(monkeypatch)
    try:
        slider = page._intensity_slider
        assert (slider.minimum(), slider.maximum()) == (0, 20)
        assert slider.value() == 0
        assert slider.orientation() == Qt.Horizontal
        # numeric display synced from the start and on valueChanged
        assert page._value_label.text() == "0"
        slider.setValue(7)
        assert page._value_label.text() == "7"
        slider.setValue(20)
        assert page._value_label.text() == "20"
        slider.setValue(0)

        # retranslate keeps the value display in sync without a reset
        page.set_intensity(13)
        page.retranslate()
        assert page._value_label.text() == "13"
    finally:
        page.deleteLater()


def test_output_dir_start_button_and_config(app, tmp_path, monkeypatch):
    from gui.i18n import en_us, i18n, zh_cn

    import gui.pages.separation_page as separation_page_mod

    prev_language = i18n.language
    out_dir = tmp_path / "wave_b_out"
    page = _make_page(monkeypatch)
    try:
        assert page.output_dir() == ""
        assert page._start_button.isEnabled() is False

        page.set_output_dir(str(out_dir))
        assert page.output_dir() == str(out_dir)
        start_button = page._start_button
        assert start_button.isEnabled() is True

        cfg = page.config()
        assert set(cfg.keys()) >= {"preset", "intensity", "denoise", "output_dir"}
        assert cfg["output_dir"] == str(out_dir)
        assert cfg["preset"] == page.selected_preset()
        assert cfg["intensity"] == 0

        # browse round-trip: offscreen dialog patched to return a fresh dir
        browsed_dir = tmp_path / "wave_b_browse"
        monkeypatch.setattr(
            separation_page_mod.QFileDialog,
            "getExistingDirectory",
            staticmethod(lambda *args, **kwargs: str(browsed_dir)),
        )
        page._browse_button.click()
        assert page.output_dir() == str(browsed_dir)
        assert start_button.isEnabled() is True

        # retranslate covers the new labels / buttons and value display
        zh_value = page._value_label.text()
        i18n.set_language("en")
        page.retranslate()
        assert page._intensity_label.text() == en_us.TEXTS["intensity"]
        assert page._dir_label.text() == en_us.TEXTS["output_dir"]
        assert page._browse_button.text() == en_us.TEXTS["browse"]
        assert start_button.text() == en_us.TEXTS["start"]
        assert page._value_label.text() == zh_value
        i18n.set_language(prev_language)
        page.retranslate()
        assert start_button.text() == zh_cn.TEXTS["start"]
    finally:
        i18n.set_language(prev_language)
        page.deleteLater()


def test_empty_output_dir_disables_start(app, tmp_path, monkeypatch):
    out_dir = str(tmp_path / "wave_b_out")
    page = _make_page(monkeypatch)
    try:
        assert page._start_button.isEnabled() is False  # fresh page, empty dir

        page.set_output_dir(out_dir)
        assert page._start_button.isEnabled() is True

        page.set_output_dir("")
        assert page.output_dir() == ""
        assert page._start_button.isEnabled() is False

        # typing into the line edit also re-evaluates the start state
        page._dir_edit.setText("  ")
        assert page._start_button.isEnabled() is False
    finally:
        page.deleteLater()


# ---------------------------------------------------------------- CLI (B-6)


def test_cli_separate_missing_audio_reports_not_found():
    import contextlib
    import io

    from main import main as cli_main

    stderr_buf = io.StringIO()
    with contextlib.redirect_stderr(stderr_buf):
        code = cli_main(["separate", "--audio", "no_such.wav"])
    assert code != 0
    assert "not found" in stderr_buf.getvalue().lower()


def test_cli_separate_synthetic_wav_two_stems_on_cpu(tmp_path):
    from main import main as cli_main

    samplerate = 44100
    n_samples = int(0.3 * samplerate)
    t = np.linspace(0.0, 0.3, n_samples, endpoint=False)
    mono = (0.5 * np.sin(2.0 * np.pi * 440.0 * t)).astype("float32")
    wav_path = tmp_path / "clip.wav"
    sf.write(str(wav_path), np.column_stack((mono, mono)), samplerate)

    out_dir = tmp_path / "b6_cli_out"
    code = cli_main(
        ["separate", "--audio", str(wav_path), "--model", "Karaoke", "--out", str(out_dir), "--device", "cpu"]
    )
    assert code == 0
    wavs = sorted(p.name for p in out_dir.glob("*.wav"))
    assert len(wavs) == 2, f"expected two stems, got {wavs}"
    assert any(name.endswith("_vocals.wav") for name in wavs)
    assert any(name.endswith("_accompaniment.wav") for name in wavs)


def test_separation_page_eta_follows_stage(app, monkeypatch):
    from gui.i18n import en_us, i18n, zh_cn

    prev_language = i18n.language
    page = _make_page(monkeypatch)
    try:
        eta_zh = [zh_cn.TEXTS[key] for key in ("tier_eta_standard", "tier_eta_hq", "tier_eta_ultra")]
        assert page._eta_label.text() == eta_zh[0]  # default stage is Standard

        for index, expected in enumerate(eta_zh):
            page.set_stage_index(index)
            assert page._eta_label.text() == expected

        i18n.set_language("en")
        page.retranslate()
        en_etas = [en_us.TEXTS[key] for key in ("tier_eta_standard", "tier_eta_hq", "tier_eta_ultra")]
        page.set_stage_index(2)
        assert page._eta_label.text() == en_etas[2]
    finally:
        i18n.set_language(prev_language)
        page.deleteLater()


# ------------------------------------------------------- page-level job flow (B-7)

import time as _time  # noqa: E402


def _pump_until(app, predicate, seconds=2.0):
    end = _time.monotonic() + seconds
    while not predicate():
        if _time.monotonic() >= end:
            return False
        app.processEvents()
        _time.sleep(0.02)
    return True


def _pump(app, seconds):
    end = _time.monotonic() + seconds
    while _time.monotonic() < end:
        app.processEvents()
        _time.sleep(0.02)


def _has_output_device():
    try:
        import sounddevice as sd

        _, out = sd.default.device
    except Exception:
        return False
    return int(out) >= 0


def _make_44k_stereo_wav(tmp_path, name="b7_clip.wav", duration=2.0, frequency=392.0):
    samplerate = 44100
    n_frames = int(duration * samplerate)
    t = np.linspace(0.0, duration, n_frames, endpoint=False)
    mono = (0.5 * np.sin(2.0 * np.pi * frequency * t)).astype("float32")
    wav_path = tmp_path / name
    sf.write(str(wav_path), np.column_stack((mono, mono)), samplerate)
    return str(wav_path), n_frames


def test_run_separation_job_stage_sequence_and_ctx(tmp_path, monkeypatch):
    from engines.separation import uvr5_engine as uvr_mod

    from gui.pages.separation_page import SeparationJobCfg, run_separation_job

    wav_path, n_frames = _make_44k_stereo_wav(tmp_path, duration=1.0)
    out_dir = tmp_path / "b7_job_out"
    calls: dict[str, tuple] = {}

    class FakeEngine:
        def load(self, spec, device="cpu"):
            calls["load"] = (spec, device)

        def run(self, audio, out_dir_arg, intensity=0.0):
            calls["run"] = (str(audio), str(out_dir_arg), float(intensity))
            target = Path(str(out_dir_arg))
            target.mkdir(parents=True, exist_ok=True)
            base = Path(str(audio)).stem
            frames, sr = sf.read(str(audio), dtype="float32", always_2d=True)  # (samples, ch)
            paths: dict[str, str] = {}
            for stem in ("vocals", "accompaniment"):
                path = target / f"{base}_{stem}.wav"
                sf.write(str(path), frames * 0.5, int(sr))
                paths[stem] = str(path)
            return paths

        def unload(self):
            calls["unloaded"] = True

    monkeypatch.setattr(uvr_mod, "UVR5Engine", FakeEngine)

    events: list[tuple[float, str]] = []
    logs: list[str] = []
    cfg = SeparationJobCfg(spec="Karaoke", device="cpu", intensity=3, denoise=True, output_dir=str(out_dir))
    cfg.log_cb = logs.append
    ctx = {"input": wav_path}
    result = run_separation_job(ctx, cfg, lambda value, stage: events.append((float(value), str(stage))))

    values = [value for value, _stage in events]
    assert values and all(a <= b for a, b in zip(values, values[1:])), "progress must be monotone"
    assert values[0] == pytest.approx(0.0)
    assert values[-1] == 1.0

    stages_first_seen: list[str] = []
    for _value, stage in events:
        if stage and stage not in stages_first_seen:
            stages_first_seen.append(stage)
    assert stages_first_seen == ["fetch", "analyze", "separate"]

    assert calls["load"] == ("Karaoke", "cpu")
    assert calls["run"][0] == wav_path and calls["run"][1] == str(out_dir)
    assert calls["run"][2] == pytest.approx(3.0)
    assert calls.get("unloaded") is True

    assert result is ctx
    assert set(result["stems"]) == {"vocals", "accompaniment"}
    for path in result["stems"].values():
        assert Path(path).is_file()
    assert 950 <= int(result["duration_ms"]) <= 1050

    first_log = logs[0]
    assert "preset=Karaoke" in first_log and "denoise=on" in first_log


def test_run_separation_job_missing_input_and_cancel(tmp_path, monkeypatch):
    from core.exceptions import CancellationError

    from engines.separation import uvr5_engine as uvr_mod

    from gui.pages.separation_page import SeparationJobCfg, run_separation_job

    wav_path, _frames = _make_44k_stereo_wav(tmp_path)

    class FakeEngine:
        def load(self, spec, device="cpu"):
            pass

        def run(self, audio, out_dir, intensity=0.0):
            return {}

        def unload(self):
            pass

    monkeypatch.setattr(uvr_mod, "UVR5Engine", FakeEngine)

    for missing in ("", str(tmp_path / "nope.wav")):
        try:
            run_separation_job({"input": missing}, SeparationJobCfg(spec="Karaoke"), lambda v, s="": None)
            assert False, "expected FileNotFoundError"
        except FileNotFoundError as exc:
            assert "not found" in str(exc).lower()

    cfg = SeparationJobCfg(spec="Karaoke", device="cpu", output_dir=str(tmp_path / "out"))
    cfg.cancel_checker = lambda: True  # cancelled right after the fetch stage
    try:
        run_separation_job({"input": wav_path}, cfg, lambda value, stage: None)
        assert False, "expected CancellationError"
    except CancellationError:
        pass


def test_i18n_b7_keys_appended_both_languages():
    from gui.i18n import en_us, zh_cn

    b7_keys = {
        "stage_fetch",
        "stage_analyze",
        "stage_separate",
        "stage_idle",
        "progress",
        "result_title",
        "track_vocals",
        "track_accomp",
        "separation_done",
        "separation_failed",
    }
    # wave D added 6 asr-page keys per language (asr_engine_*, asr_output_formats, fmt_txt/srt/json)
    # wave E added 4 tts-page keys per language (tts_target_text/tts_version/tts_ref_hint/tts_ref_bad_duration)
    # wave gui-usability + m4-polish added 22 keys, training_device added 1 (6 gui_ + 16 m4_ + 1)
    # rhine theme added 4 keys per language (theme.auto/light/dark/rhine)
    assert len(zh_cn.TEXTS) == 229 and len(en_us.TEXTS) == 229
    assert set(zh_cn.TEXTS) == set(en_us.TEXTS), "zh/en key sets diverged"
    assert b7_keys <= set(zh_cn.TEXTS) and b7_keys <= set(en_us.TEXTS)


def test_separation_page_start_full_flow(app, tmp_path, monkeypatch):
    from gui.i18n import i18n, tr

    prev_language = i18n.language
    wav_path, _frames = _make_44k_stereo_wav(tmp_path)
    out_dir = tmp_path / "b7_page_out"

    page = _make_page(monkeypatch)  # InfoCards stubbed like the other page tests here
    try:
        assert page.selected_preset() == "Karaoke"  # real weights present on this host
        assert page.add_files([wav_path]) == 1
        assert page.accepted_count() == 1
        page.select(0)
        page.set_output_dir(str(out_dir))
        assert page._start_button.isEnabled()

        done_events: list[tuple[object, bool]] = []
        failed_events: list[Exception] = []
        page._start_button.click()  # start button -> JobRunner(UVR5Engine)
        runner = page._job_runner
        assert runner is not None and page.is_running()
        runner.done.connect(lambda result, cancelled: done_events.append((result, cancelled)))
        runner.failed.connect(failed_events.append)

        finished = _pump_until(
            app, lambda: len(done_events) == 1 or len(failed_events) == 1, seconds=240
        )
        assert finished and not failed_events, f"job did not finish cleanly: {failed_events}"

        events = page.progress_events()
        values = [value for value, _stage in events]
        stages_first_seen: list[str] = []
        for _value, stage in events:
            if stage and stage not in stages_first_seen:
                stages_first_seen.append(stage)
        assert values and all(a <= b for a, b in zip(values, values[1:])), "progress must be monotone"
        assert values[0] == pytest.approx(0.0)
        assert page.last_progress() == 1.0
        assert stages_first_seen == ["fetch", "analyze", "separate"]

        result, cancelled = done_events[0]
        assert cancelled is False and result is runner.ctx
        stems = page.result_paths()
        assert set(stems) >= {"vocals", "accompaniment"}
        for path in stems.values():
            stem_path = Path(path)
            assert stem_path.is_file(), f"missing output file: {path}"
            assert stem_path.parent == out_dir

        view = page.waveform_view()
        tracks = view._tracks
        assert len(tracks) == 2, "dual-track waveform expected after done"
        for track in tracks:
            peaks = track["peaks"]
            assert isinstance(peaks, np.ndarray) and peaks.size > 0
            assert float(np.max(peaks)) >= 0.9
        assert tracks[0]["color"] != tracks[1]["color"], "vocals/accompaniment lanes use distinct colors"
        assert view.duration_ms() > 1900

        combo = page.track_combo()
        assert combo.count() == 2 and combo.currentIndex() == 0  # default lane: vocals
        assert combo.itemText(0) == tr("track_vocals")
        assert combo.itemData(0) == stems["vocals"]

        player = page.audio_player()
        assert not player.is_playing
        assert player.duration_ms > 1900
        if _has_output_device():
            player.volume = 0.3
            player.play()
            try:
                assert player.is_playing
                _pump(app, 1.0)
                assert player.position_ms > 0
            finally:
                player.stop()
    finally:
        runner = page._job_runner
        if runner is not None and runner.isRunning():
            runner.wait(5000)
        i18n.set_language(prev_language)
        page.deleteLater()


# ------------------------------------------------------------ abnormal paths (B-9)


def test_uvr5_empty_audio_rejected_before_outputs(app):
    import tempfile

    samplerate = 44100
    root = Path(tempfile.mkdtemp(prefix="wave_b_b9_"))
    empty = root / "empty.wav"
    sf.write(str(empty), np.zeros((0, 2), dtype="float32"), samplerate)

    engine = _engine_from_real_weights()
    out_dir = root / "out"
    try:
        engine.load("Karaoke", device="cpu")
        try:
            engine.run(str(empty), str(out_dir))
            assert False, "expected ValueError for empty audio"
        except ValueError as exc:
            assert "no samples" in str(exc)
        # the guard must fire before any output directory is created
        assert not out_dir.exists()
    finally:
        engine.unload()


def test_uvr5_bs_drums_checkpoint_dim384(app):
    """The public drums checkpoint (ep_937) was trained with dim=384, so the
    engine's architecture fallback must select it (strict load enforces shapes)."""
    import tempfile

    samplerate = 44100
    n_samples = int(0.2 * samplerate)
    t = np.linspace(0.0, 0.2, n_samples, endpoint=False)
    mono = (0.5 * np.sin(2.0 * np.pi * 330.0 * t)).astype("float32")
    clip = np.stack([mono, mono])

    engine = _engine_from_real_weights()
    with tempfile.TemporaryDirectory(prefix="wave_b_uvr_drums_") as out_dir:
        engine.load("drums_bs_roformer.ckpt", device="cpu")
        assert "drums" in engine._loaded and engine._loaded["drums"]["kind"] == "bs"
        paths = engine.run(clip, Path(out_dir) / "drums", intensity=0)
        assert set(paths) == {"drums", "other"}
        for name in ("drums", "other"):
            arr, sr = sf.read(paths[name], dtype="float32")
            assert sr == samplerate and abs(len(arr) / samplerate - 0.2) < 0.05
        engine.unload()


def test_cli_separate_abnormal_args(tmp_path):
    import contextlib
    import io

    from main import main as cli_main

    samplerate = 44100
    n_frames = int(0.2 * samplerate)
    t = np.linspace(0.0, 0.2, n_frames, endpoint=False)
    mono = (0.5 * np.sin(2.0 * np.pi * 392.0 * t)).astype("float32")
    wav_path = tmp_path / "b9 clip é.wav"  # unicode + spaces in the filename
    sf.write(str(wav_path), np.column_stack((mono, mono)), samplerate)

    # invalid --device: argparse exits 2 and lists the choices
    stderr_buf = io.StringIO()
    with contextlib.redirect_stderr(stderr_buf):
        try:
            code = cli_main(
                ["separate", "--audio", str(wav_path), "--model", "Karaoke", "--device", "foo"]
            )
        except SystemExit as exc:
            code = int(exc.code or 0)
    assert code == 2 and "invalid choice" in stderr_buf.getvalue()

    # output dir that is an existing regular file: clean error, no traceback
    blocker_file = tmp_path / "blocker.txt"
    blocker_file.write_text("not a directory", encoding="utf-8")
    stderr_buf = io.StringIO()
    with contextlib.redirect_stderr(stderr_buf):
        code = cli_main(
            [
                "separate",
                "--audio",
                str(wav_path),
                "--model",
                "Karaoke",
                "--out",
                str(blocker_file),
                "--device",
                "cpu",
            ]
        )
    assert code == 1
    err = stderr_buf.getvalue()
    assert err.lstrip().startswith("error:") and "traceback" not in err.lower()

    # the unicode+spaces filename round-trips end-to-end with output dir creation
    out_dir = tmp_path / "b9 uni out"
    code = cli_main(
        [
            "separate",
            "--audio",
            str(wav_path),
            "--model",
            "Karaoke",
            "--out",
            str(out_dir),
            "--device",
            "cpu",
        ]
    )
    assert code == 0
    produced = sorted(p.name for p in out_dir.glob("*.wav"))
    assert len(produced) == 2 and all(name.startswith("b9 clip é_") for name in produced)


def test_page_empty_queue_start_and_duplicate_rows(app, tmp_path, monkeypatch):
    from gui.i18n import i18n, tr

    prev_language = i18n.language
    # page 1: start with an empty queue -> start() returns None, UI stays usable
    page = _make_page(monkeypatch)
    try:
        page.set_output_dir(str(tmp_path / "b9_empty_out"))
        runner = page.start()
        assert runner is None and not page.is_running()
        # no failure flashed; status stays idle, start button usable
        assert page._status_stage_label.text() == tr("stage_idle")
        assert page._start_button.isEnabled()
    finally:
        runner = page._job_runner
        if runner is not None and runner.isRunning():
            runner.wait(5000)
        page.deleteLater()

    # page 2: duplicates are kept (one row per add), a dragged directory is rejected
    wav, _frames = _make_44k_stereo_wav(tmp_path, duration=0.2)
    page2 = _make_page(monkeypatch)
    try:
        area = page2.input_area()
        assert area.add_files([wav]) == 1
        assert area.add_files([wav]) == 1
        assert area.accepted_count() == 2 and len(area.rows()) == 2

        added = area.add_files([str(tmp_path)])
        rows = area.rows()
        assert added == 0
        assert rows[-1] == (str(tmp_path), False)
    finally:
        page2.deleteLater()
        i18n.set_language(prev_language)
