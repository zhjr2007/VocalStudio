"""Offscreen pytest suite for AudioPlayerWidget (wave A)."""

from __future__ import annotations

import collections
import glob
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

import torch  # noqa: F401 (must load before PyQt5 on this host)

from PyQt5.QtWidgets import QApplication

import sounddevice as sd
import soundfile as sf

from PyQt5.QtGui import QImage

from core.device_utils import get_gpu_info
from core.exceptions import CancellationError

from gui.widgets.audio_player import AudioPlayerWidget
from gui.widgets.job_runner import JobRunner
from gui.widgets.model_status import ModelStatusWidget
from gui.widgets.waveform_view import WaveformView, compute_peaks


@pytest.fixture(scope="module")
def app(qapp) -> QApplication:
    return qapp


@pytest.fixture(scope="module")
def wav_path(tmp_path_factory) -> str:
    candidates = sorted(glob.glob(str(ROOT / "materials" / "*.wav")))
    if candidates:
        return candidates[0]
    path = tmp_path_factory.mktemp("wave_a_audio") / "sine.wav"
    samplerate = 22050
    t = np.linspace(0.0, 2.0, samplerate * 2, endpoint=False)
    signal = (0.4 * np.sin(2.0 * np.pi * 440.0 * t)).astype("float32")
    sf.write(path, signal, samplerate)
    return str(path)


def _has_output_device() -> bool:
    try:
        _, out = sd.default.device
    except Exception:
        return False
    return int(out) >= 0


def _pump(app: QApplication, seconds: float) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.02)


def test_set_file_sets_duration(app, wav_path):
    widget = AudioPlayerWidget()
    try:
        widget.set_file(wav_path)
        assert widget.duration_ms > 0
        assert widget.position_ms == 0
        assert not widget.is_playing
        assert widget.state == "stopped"
    finally:
        widget.stop()


def test_play_advances_position_and_emits(app, wav_path):
    if not _has_output_device():
        pytest.skip("no default output audio device")
    widget = AudioPlayerWidget()
    emitted: list[int] = []
    try:
        widget.set_file(wav_path)
        widget.volume = 0.2
        widget.position_changed.connect(lambda ms: emitted.append(ms))
        widget.play()
        assert widget.is_playing
        _pump(app, 1.0)
        assert widget.is_playing
        assert widget.position_ms > 0
        assert len(emitted) >= 1
        assert emitted[-1] > 0
    finally:
        widget.stop()


def test_stop_resets_then_play_again(app, wav_path):
    if not _has_output_device():
        pytest.skip("no default output audio device")
    widget = AudioPlayerWidget()
    try:
        widget.set_file(wav_path)
        widget.volume = 0.2

        widget.play()
        _pump(app, 0.5)
        assert widget.is_playing
        assert widget.position_ms > 0

        widget.stop()
        assert widget.state == "stopped"
        assert not widget.is_playing
        assert widget.position_ms == 0

        widget.play()
        assert widget.is_playing
        _pump(app, 0.3)
        assert widget.is_playing
        assert widget.position_ms > 0
    finally:
        widget.stop()


def test_volume_default_and_clamp(app):
    widget = AudioPlayerWidget()
    try:
        assert widget.volume == pytest.approx(0.8)
        widget.volume = 1.5
        assert widget.volume == pytest.approx(1.0)
        widget.volume = -0.3
        assert widget.volume == pytest.approx(0.0)
        widget.volume = 0.45
        assert widget.volume == pytest.approx(0.45)
    finally:
        widget.stop()


def _render_image(widget: WaveformView) -> QImage:
    widget.resize(480, 160)
    return widget.grab().toImage().convertToFormat(QImage.Format_RGB32)


def _distinct_colors(image: QImage, step_x: int = 4, step_y: int = 4) -> int:
    seen: set[tuple[int, int, int]] = set()
    for x in range(0, image.width(), step_x):
        for y in range(0, image.height(), step_y):
            color = image.pixelColor(x, y)
            seen.add((color.red(), color.green(), color.blue()))
    return len(seen)


def _dominant_row(image: QImage, row: int) -> tuple[int, int, int]:
    counter = collections.Counter()
    for x in range(0, image.width(), 2):
        color = image.pixelColor(x, row)
        counter[(color.red(), color.green(), color.blue())] += 1
    return counter.most_common(1)[0][0]


def _load_channels(wav_path: str) -> tuple[list[np.ndarray], int]:
    data, samplerate = sf.read(str(wav_path), dtype="float32", always_2d=True)
    return [data[:, i] for i in range(data.shape[1])], int(samplerate)


def test_waveform_peaks(wav_path):
    channels, sr = _load_channels(wav_path)
    peaks = compute_peaks(channels[0], sr, 512)
    assert isinstance(peaks, np.ndarray)
    assert peaks.shape == (512,)
    assert peaks.dtype.kind in "f"
    assert float(peaks.min()) >= 0.0
    assert float(peaks.max()) <= 1.0
    assert float(peaks.max()) == pytest.approx(1.0, abs=1e-3)

    silence = compute_peaks(np.zeros(4 * sr, dtype="float32"), sr, 8)
    assert silence.shape == (8,)
    assert not np.any(silence)
    empty = compute_peaks(np.array([], dtype="float32"), sr, 5)
    assert empty.shape == (5,)
    assert not np.any(empty)

    quarters = np.concatenate(
        [np.full(100, value, dtype="float64") for value in (0.25, -0.5, 0.75, 1.0)]
    )
    exact = compute_peaks(quarters, sr, 4)
    assert np.allclose(exact, [0.25, 0.5, 0.75, 1.0])


def test_waveform_single_dual_render(app, wav_path):
    channels, sr = _load_channels(wav_path)
    frames = len(channels[0])
    duration_ms = int(frames * 1000 / sr)
    peaks_a = compute_peaks(channels[0], sr, 256)
    peaks_b = compute_peaks(
        channels[-1] if len(channels) > 1 else channels[0], sr, 256
    )

    blue = (0x4C, 0x8D, 0xFF)
    green = (0x3D, 0xD6, 0x8C)
    widget = WaveformView()
    try:
        widget.set_waveforms([{"label": "A", "peaks": peaks_a, "color": "#4C8DFF"}])
        widget.set_duration(duration_ms)
        single_image = _render_image(widget)
        assert _distinct_colors(single_image) >= 2
        assert _dominant_row(single_image, int(single_image.height() * 0.25)) == blue

        widget.set_waveforms(
            [
                {"label": "A", "peaks": peaks_a, "color": "#4C8DFF"},
                {"label": "B", "peaks": peaks_b, "color": "#3DD68C"},
            ]
        )
        dual_image = _render_image(widget)
        assert _distinct_colors(dual_image) >= 3
        top_color = _dominant_row(dual_image, int(dual_image.height() * 0.25))
        bottom_color = _dominant_row(dual_image, int(dual_image.height() * 0.75))
        assert top_color == blue
        assert bottom_color == green
        assert top_color != bottom_color
    finally:
        widget.deleteLater()


def test_waveform_marks(app, wav_path):
    channels, sr = _load_channels(wav_path)
    duration_ms = int(len(channels[0]) * 1000 / sr)
    peaks = compute_peaks(channels[0], sr, 256)

    widget = WaveformView()
    try:
        widget.set_waveforms([{"label": "A", "peaks": peaks, "color": "#4C8DFF"}])
        widget.set_duration(duration_ms)
        before = _distinct_colors(_render_image(widget))

        marks = [
            {"position_ms": max(1, duration_ms // 4), "label": "M1"},
            {"position_ms": max(2, duration_ms // 2), "label": "M2", "duration_ms": 50},
            {
                "position_ms": max(3, (duration_ms * 3) // 4),
                "label": "M3",
                "duration_ms": 120,
            },
        ]
        widget.set_marks(marks)
        assert len(widget._marks) == 3

        image = _render_image(widget)
        after = _distinct_colors(image)
        assert after >= before and after >= 3
    finally:
        widget.deleteLater()


def test_modelstatus_summary_counts(app):
    widget = ModelStatusWidget()
    try:
        summary = widget.summary()
        assert isinstance(summary, dict)
        assert summary["total"] == 22
        assert summary["ready"] >= 1
        assert isinstance(summary["missing"], list)
        green_rows = [name for name, ready in widget.rows() if ready]
        red_rows = [name for name, ready in widget.rows() if not ready]
        assert len(green_rows) + len(red_rows) == summary["total"]
        assert summary["ready"] == len(green_rows)
        assert sorted(summary["missing"]) == sorted(red_rows)
    finally:
        widget.deleteLater()


def test_modelstatus_gpu_name(app):
    info = get_gpu_info()
    if info.get("name") is None:
        pytest.skip("no GPU detected (get_gpu_info().name is None)")
    widget = ModelStatusWidget()
    try:
        gpu_name = widget.summary()["gpu_name"]
        assert isinstance(gpu_name, str)
        # vendor-neutral: widget must report the same name get_gpu_info() sees
        # (works on AMD/ROCm machines and NVIDIA alike)
        assert info["name"] in gpu_name or gpu_name in info["name"]
    finally:
        widget.deleteLater()


def test_modelstatus_refresh_renders(app):
    widget = ModelStatusWidget()
    try:
        widget.refresh()
        widget.resize(384, 420)
        image = widget.grab().toImage().convertToFormat(QImage.Format_RGB32)
        assert image.width() > 0 and image.height() > 0
        assert _distinct_colors(image) >= 2
        assert f"{widget.summary()['ready']}/22" in widget._summary_label.text()
    finally:
        widget.deleteLater()


def _pump_until(app: QApplication, predicate, seconds: float = 2.0) -> bool:
    """Pumps the event loop until predicate() is true or the budget runs out."""
    end = time.monotonic() + seconds
    while not predicate():
        if time.monotonic() >= end:
            return False
        app.processEvents()
        time.sleep(0.02)
    return True


def test_jobrunner_progress_monotone_ends_at_one(app):
    def fn(ctx, cfg, progress_cb):
        for value, stage in ((0.3, "a"), (0.7, "b"), (1.0, "export")):
            time.sleep(0.02)
            progress_cb(value, stage)

    runner = JobRunner(fn)
    events: list[tuple[float, str]] = []
    done_events: list[tuple[object, bool]] = []
    failed_events: list[Exception] = []
    runner.progress.connect(lambda v, s: events.append((v, s)))
    runner.done.connect(lambda r, c: done_events.append((r, c)))
    runner.failed.connect(failed_events.append)
    try:
        assert not runner.cancelled
        runner.start_job()
        assert _pump_until(app, lambda: len(done_events) == 1), "done never emitted"
    finally:
        runner.wait(2000)

    values = [v for v, _ in events]
    stages = [s for _, s in events]
    assert values and all(0.0 <= v <= 1.0 for v in values)
    assert all(b >= a for a, b in zip(values, values[1:]))
    assert events[-1][0] == pytest.approx(1.0)
    assert stages == ["a", "b", "export"]
    assert done_events == [(None, False)]
    assert not failed_events


def test_jobrunner_done_once_result_passthrough(app):
    sentinel = {"name": "payload"}

    def fn(ctx, cfg, progress_cb):
        progress_cb(0.5)
        return sentinel

    runner = JobRunner(fn, ctx="ctx-obj", cfg=None)
    done_events: list[tuple[object, bool]] = []
    failed_events: list[Exception] = []
    runner.done.connect(lambda r, c: done_events.append((r, c)))
    runner.failed.connect(failed_events.append)
    try:
        runner.start_job()
        assert _pump_until(app, lambda: len(done_events) == 1), "done never emitted"
        _pump(app, 0.2)
    finally:
        runner.wait(2000)

    assert len(done_events) == 1
    result, cancelled = done_events[0]
    assert result is sentinel
    assert cancelled is False
    assert not failed_events


def test_jobrunner_failed_on_error(app):
    def fn(ctx, cfg, progress_cb):
        progress_cb(0.2)
        raise ValueError("boom")

    runner = JobRunner(fn)
    done_events: list[tuple[object, bool]] = []
    failed_events: list[Exception] = []
    runner.done.connect(lambda r, c: done_events.append((r, c)))
    runner.failed.connect(failed_events.append)
    try:
        runner.start_job()
        assert _pump_until(app, lambda: len(failed_events) == 1), "failed never emitted"
        _pump(app, 0.2)
    finally:
        runner.wait(2000)

    assert not done_events
    assert isinstance(failed_events[0], ValueError)
    assert str(failed_events[0]) == "boom"


def test_jobrunner_cancel_done(app):
    state = {"runner": None}

    def fn(ctx, cfg, progress_cb):
        i = 0
        runner = state["runner"]
        while not (runner is None or runner.cancelled):
            time.sleep(0.05)
            i += 1
            progress_cb(min(i * 0.2, 0.9), "work")
        raise CancellationError("cancelled")

    runner = JobRunner(fn)
    state["runner"] = runner
    progress_events: list[tuple[float, str]] = []
    done_events: list[tuple[object, bool]] = []
    failed_events: list[Exception] = []
    runner.progress.connect(lambda v, s: progress_events.append((v, s)))
    runner.done.connect(lambda r, c: done_events.append((r, c)))
    runner.failed.connect(failed_events.append)
    try:
        runner.start_job()
        time.sleep(0.15)
        app.processEvents()
        assert len(progress_events) >= 1
        runner.cancel()
        assert _pump_until(app, lambda: len(done_events) == 1), "done never emitted"
    finally:
        runner.wait(2000)

    assert done_events[0][0] is None
    assert done_events[0][1] is True
    assert runner.cancelled
    assert not failed_events


def test_i18n_parity() -> None:
    from gui.i18n import en_us, zh_cn

    assert len(zh_cn.TEXTS) >= 50
    assert len(en_us.TEXTS) >= 50
    missing_zh = set(en_us.TEXTS) - set(zh_cn.TEXTS)
    missing_en = set(zh_cn.TEXTS) - set(en_us.TEXTS)
    assert not missing_zh, f"keys missing in zh: {sorted(missing_zh)}"
    assert not missing_en, f"keys missing in en: {sorted(missing_en)}"


def test_i18n_live_switch(app) -> None:
    from PyQt5.QtWidgets import QWidget

    from gui.fluent_main import NAV_TITLE_KEYS, MainWindow
    from gui.i18n import en_us, zh_cn

    window = MainWindow()

    def nav_titles() -> list[str]:
        trees = [c for c in window.navigationInterface.findChildren(QWidget)
                 if type(c).__name__ == "NavigationTreeWidget"]
        return [t.text() for t in trees]

    try:
        zh_expected = [zh_cn.TEXTS[k] for k in NAV_TITLE_KEYS][: len(nav_titles())]
        assert nav_titles()[:3] == zh_expected[:3], f"initial titles wrong: {nav_titles()}"

        window.switch_language("en")
        for _ in range(6):
            app.processEvents()
        en_expected = [en_us.TEXTS[k] for k in NAV_TITLE_KEYS][: len(nav_titles())]
        assert nav_titles()[:3] == en_expected[:3], f"english titles wrong: {nav_titles()}"
        assert window.windowTitle() == en_us.TEXTS["app"]

        window.switch_language("zh")
        for _ in range(6):
            app.processEvents()
        assert nav_titles()[:3] == zh_expected[:3], f"back-to-zh titles wrong: {nav_titles()}"
    finally:
        window.deleteLater()


def test_appsettings_save_load_roundtrip(tmp_path) -> None:
    from core.app_settings import AppSettings, get_settings

    path = tmp_path / "user.json"
    settings = AppSettings(device="cuda", theme="dark", language="en")
    settings.validate()
    assert (settings.device, settings.theme, settings.language) == ("cuda", "dark", "en")
    settings.save(path)

    loaded = AppSettings.load(path)
    assert (loaded.device, loaded.theme, loaded.language) == ("cuda", "dark", "en")
    # in-process config cache sees the saved values immediately (round-trip)
    cached = get_settings(path)
    assert cached.device == "cuda"
    assert cached is settings


def test_appsettings_alias_and_validation_errors(tmp_path) -> None:
    import pytest as _pytest

    from core.app_settings import AppSettings, USER_SETTINGS_FILE, invalidate_settings

    aliased = AppSettings(device="dml")
    aliased.validate()
    assert aliased.device == "directml"

    direct = AppSettings(device="directml")
    direct.validate()
    assert direct.device == "directml"

    bad_theme = AppSettings(theme="sepia")
    with _pytest.raises(ValueError):
        bad_theme.validate()
    bad_device = AppSettings(device="metal")
    with _pytest.raises(ValueError):
        bad_device.validate()

    # loading a missing file yields defaults without touching the real user file
    fallback = AppSettings.load(tmp_path / "absent.json")
    assert (fallback.device, fallback.theme, fallback.language) == ("auto", "auto", "zh")
    invalidate_settings(USER_SETTINGS_FILE)


def test_settings_page_device_roundtrip(app, tmp_path) -> None:
    from core.app_settings import AppSettings, get_settings
    from core.device_utils import default_backend
    from gui.pages.settings_page import DEVICE_CHOICES, SettingsPage

    path = tmp_path / "user.json"
    page = SettingsPage(settings_file=path)
    try:
        assert page.settings_file == path
        combo_texts = [page._device_combo.itemText(i) for i in range(page._device_combo.count())]
        assert combo_texts == list(DEVICE_CHOICES)
        # fresh file -> concrete backend derived from the detected hardware, theme follows system
        assert (page.device, page.theme, page.language) == (default_backend(), "auto", "zh")

        # selecting cuda persists to the core config cache; other pages read it back
        page.set_device("cuda")
        saved = page.save_settings()
        assert saved.device == "cuda"
        assert AppSettings.load(page.settings_file).device == "cuda"
        assert get_settings(path).device == "cuda"

        # explicit save button takes the same path
        page.set_theme("dark")
        page._save_button.click()
        assert get_settings(path).theme == "dark"
    finally:
        page.deleteLater()


def test_settings_page_language_roundtrip(app, tmp_path) -> None:
    from core.app_settings import AppSettings, get_settings
    from gui.i18n import en_us, i18n, zh_cn
    from gui.pages.settings_page import SettingsPage

    path = tmp_path / "user.json"
    page = SettingsPage(settings_file=path)
    prev_language = i18n.language
    try:
        page.set_language("en")
        assert page.language == "en"
        saved = page.save_settings()
        assert saved.language == "en"
        assert get_settings(path).language == "en"

        # retranslating in english updates the language combo labels
        i18n.set_language("en")
        page.retranslate()
        item_texts = [page._language_combo.itemText(i) for i in range(page._language_combo.count())]
        assert en_us.TEXTS["lang.en"] in item_texts
        assert zh_cn.TEXTS["lang.en"] not in item_texts

        # switching back to zh restores the original labels and language value
        i18n.set_language("zh")
        page.retranslate()
        item_texts = [page._language_combo.itemText(i) for i in range(page._language_combo.count())]
        assert item_texts == [zh_cn.TEXTS["lang.zh"], zh_cn.TEXTS["lang.en"]]
        page.set_language("zh")
        page.save_settings()
    finally:
        i18n.set_language(prev_language)
        page.deleteLater()


def test_settings_page_theme_switch_and_modelstatus_refresh(app, tmp_path) -> None:
    from qfluentwidgets import Theme

    try:
        from qfluentwidgets import isDarkTheme, setTheme as apply_fluent_theme
    except ModuleNotFoundError:  # renamed distribution
        from qfluent_widgets import isDarkTheme, setTheme as apply_fluent_theme

    from gui.pages.settings_page import SettingsPage

    path = tmp_path / "user.json"
    page = SettingsPage(settings_file=path)
    try:
        page.set_theme("dark")  # auto-save + theme applied, no exception
        assert isDarkTheme() is True
        page.refresh_model_status()

        summary = page.model_status.summary()
        assert summary["total"] == 22
        assert summary["ready"] >= 1

        # switching back to light also applies cleanly and ModelStatus re-renders
        page.set_theme("light")
        assert not isDarkTheme()
    finally:
        apply_fluent_theme(Theme.LIGHT)
        page.deleteLater()


def test_settings_page_renders(app, tmp_path) -> None:
    from gui.pages.settings_page import SettingsPage

    page = SettingsPage(settings_file=tmp_path / "user.json")
    try:
        page.resize(520, 760)
        image = page.grab().toImage().convertToFormat(QImage.Format_RGB32)
        assert image.width() > 0 and image.height() > 0
        assert _distinct_colors(image) >= 3
        assert f"{page.model_status.summary()['ready']}/22" in page.model_status._summary_label.text()
    finally:
        page.deleteLater()


def test_engines_registry_enumerates_six():
    from engines import ENGINE_NAMES

    assert len(ENGINE_NAMES) >= 6
    for name in ("separation", "vocal2midi", "speaker", "speech_asr", "tts", "training"):
        assert name in ENGINE_NAMES


def test_get_engine_tts_lazy_import():
    from engines import get_engine

    # importing the registry must not have pulled in any engine subpackage yet
    assert "engines.tts" not in sys.modules
    engine_cls = get_engine("tts")
    assert "engines.tts" in sys.modules
    assert isinstance(engine_cls, type), f"expected a class object, got {engine_cls!r}"
    # second fetch reuses the same module-level placeholder
    assert get_engine("tts") is engine_cls


def test_get_engine_unknown_name_lists_valid():
    from engines import ENGINE_NAMES, UnknownEngineError, get_engine

    with pytest.raises(UnknownEngineError) as excinfo:
        get_engine("vocals")
    message = str(excinfo.value)
    for name in ENGINE_NAMES:
        assert name in message
    assert isinstance(excinfo.value, ValueError)


def test_engines_import_fast_in_fresh_interpreter():
    import subprocess

    script = (
        "import sys, time\n"
        "t0 = time.perf_counter()\n"
        "import engines\n"
        "dt = time.perf_counter() - t0\n"
        "assert dt < 1.0, f'engines import took {dt:.3f}s'\n"
        "for heavy in ('torch', 'funasr'):\n"
        "    assert heavy not in sys.modules, heavy\n"
        "print('ok')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        timeout=30,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"


def test_settings_page_language_signal(app, tmp_path) -> None:
    from gui.pages.settings_page import SettingsPage

    page = SettingsPage(settings_file=str(tmp_path / "user.json"))
    emitted: list[str] = []
    page.language_changed.connect(emitted.append)
    try:
        assert page.language == "zh"
        page.set_language("en")
        for _ in range(3):
            app.processEvents()
        assert emitted == ["en"]
        page.set_language("zh")
        for _ in range(3):
            app.processEvents()
        assert emitted == ["en", "zh"]
    finally:
        page.deleteLater()


def test_detect_system_hardware_reports_gpu_and_cpu():
    from core.device_utils import default_backend, detect_system_hardware

    hw = detect_system_hardware()
    assert hw["cpu"], "CPU name should not be empty"
    assert len(hw["gpus"]) >= 1, f"expected at least one GPU entry, got {hw['gpus']}"
    names_lower = [name.lower() for name, _vram in hw["gpus"]]
    # Target backend order is rocm/zluda (AMD) > cuda (NVIDIA) > cpu.
    if any(("radeon" in n or "amd" in n) for n in names_lower):
        assert default_backend() == "rocm"
    elif any(("nvidia" in n or "rtx" in n) for n in names_lower):
        assert default_backend() == "cuda"
    else:
        assert default_backend() == "cpu"
    # cached on subsequent calls (same object identity)
    assert detect_system_hardware() is hw


def test_settings_page_theme_auto_and_hardware_labels(app, tmp_path):
    from core.device_utils import detect_system_hardware
    from gui.pages.settings_page import THEME_ITEMS, SettingsPage

    page = SettingsPage(settings_file=tmp_path / "user.json")
    try:
        # display text is localised; the stored value rides in userData
        theme_values = [page._theme_combo.itemData(i) for i in range(page._theme_combo.count())]
        assert theme_values == [value for value, _t, _k in THEME_ITEMS]
        assert "rhine" in theme_values
        theme_texts = [page._theme_combo.itemText(i) for i in range(page._theme_combo.count())]
        assert all(text.strip() for text in theme_texts)
        assert page.theme == "auto"

        # the rhine theme round-trips through the combo and validates
        page.set_theme("rhine")
        assert page.theme == "rhine"
        assert page.save_settings().theme == "rhine"
        page.set_theme("auto")

        first_gpu = detect_system_hardware()["gpus"][0][0] if detect_system_hardware()["gpus"] else ""
        gpu_text = page._hw_gpu_label.text()
        cpu_text = page._hw_cpu_label.text()
        assert len(gpu_text) > 0 and len(cpu_text) > 0
        if first_gpu:
            assert first_gpu in gpu_text

        # auto theme saves/validates cleanly and applies without exception
        saved = page.save_settings()
        assert saved.theme == "auto"
    finally:
        page.deleteLater()


def test_rhine_theme_is_additive(app):
    """The rhine theme must be purely additive: light/dark keep an empty sheet."""
    from PyQt5.QtWidgets import QApplication, QWidget

    from core.app_settings import THEME_CHOICES
    from gui.design_tokens import RADIUS, RHINE, build_qss
    from gui.theme import apply_app_qss, is_rhine_theme, set_app_theme, set_qss_root

    assert "rhine" in THEME_CHOICES
    assert build_qss("light") == "" and build_qss("dark") == ""
    assert build_qss("rhine").strip()

    # tokens that reach the stylesheet must be concrete hex colours
    for key in ("background", "background_alt", "card", "foreground", "border",
                "border_strong", "primary", "secondary"):
        value = RHINE[key]
        assert value.startswith("#") and len(value) == 7, f"{key}={value!r}"

    # the reference deck is square-cornered and shadowless — keep it that way
    assert RADIUS == 0
    assert "box-shadow" not in build_qss("rhine")

    # the sheet rides on the root window, never on the application (qfluentwidgets
    # owns the application sheet; overwriting it crashed the event loop)
    root = QWidget()
    try:
        set_qss_root(root)
        set_app_theme("rhine")
        assert is_rhine_theme()
        assert root.styleSheet().strip()
        assert QApplication.instance().styleSheet() == ""

        set_app_theme("light")
        assert not is_rhine_theme()
        apply_app_qss("light")
        assert root.styleSheet() == ""
    finally:
        set_qss_root(None)
        root.deleteLater()


def test_token_naming_is_aligned_and_complete():
    """S2: token names follow the shadcn vocabulary, and the rename is total."""
    from gui import design_tokens as dt

    # every legacy name has a documented successor, and the successor exists
    for old, new in dt.LEGACY_TOKEN_NAMES.items():
        assert new in dt.RHINE, f"{old} -> {new} is missing from RHINE"
        assert old not in dt.RHINE, f"legacy name {old} still present in RHINE"

    # the whole table is concrete hex, and `muted` mirrors `card_alt` by design
    for key, value in dt.RHINE.items():
        assert value.startswith("#") and len(value) == 7, f"{key}={value!r}"
    assert dt.RHINE["muted"] == dt.RHINE["card_alt"]

    # motion / focus tokens exist with the shapes the ui_kit will consume
    assert dt.MOTION["enter_ms"] > 0 and dt.MOTION["exit_ms"] > 0
    for name in ("enter", "exit"):
        assert len(dt.MOTION[f"{name}_curve"]) == 4
    assert dt.FOCUS["ring_width"] >= 1 and dt.FOCUS["ring_offset"] >= 0

    # the registry is the single source of theme facts
    assert set(dt.THEMES) == set(dt.THEME_ORDER)
    assert dt.THEMES["rhine"].has_custom_qss and not dt.THEMES["light"].has_custom_qss
    for name, spec in dt.THEMES.items():
        assert spec.name == name
        assert spec.fluent_base in ("auto", "light", "dark")


def test_modelstatus_dark_theme_styling(app):
    from qfluentwidgets import Theme, setTheme as apply_fluent_theme

    from gui.widgets.model_status import ModelStatusWidget

    widget = ModelStatusWidget()
    try:
        apply_fluent_theme(Theme.DARK)
        widget.retranslate()
        dark_sheet = widget._list.styleSheet()
        assert "#323232" in dark_sheet

        apply_fluent_theme(Theme.LIGHT)
        widget.retranslate()
        light_sheet = widget._list.styleSheet()
        assert "#F2F2F2" in light_sheet
    finally:
        apply_fluent_theme(Theme.LIGHT)
        widget.deleteLater()


def test_module_pages_show_usage_and_models(app):
    from gui.i18n import i18n

    from gui.pages.asr_page import AsrPage
    from gui.pages.pipeline_page import PipelinePage
    from gui.pages.separation_page import SeparationPage
    from gui.pages.training_page import TrainingPage
    from gui.pages.tts_page import TtsPage
    from gui.pages.v2midi_page import V2MidiPage

    prev_language = i18n.language
    pages = [SeparationPage(), V2MidiPage(), AsrPage(), TtsPage(), PipelinePage(), TrainingPage()]
    try:
        zh_usage_by_id = {}
        for page in pages:
            zh_texts = page._usage_card.line_texts()
            assert all(text for text in zh_texts), f"{type(page).__name__} usage lines empty"
            assert all(text for text in page._models_card.line_texts())
            zh_usage_by_id[id(page)] = zh_texts

        i18n.set_language("en")
        for page in pages:
            page.retranslate()
            en_usage = page._usage_card.line_texts()
            assert all(text for text in en_usage)
            assert en_usage != zh_usage_by_id[id(page)], (
                f"{type(page).__name__} english usage identical to chinese"
            )
    finally:
        i18n.set_language(prev_language)
        for page in pages:
            page.deleteLater()


def test_ort_bootstrap_and_provider_selection():
    from core.device_utils import (
        bootstrap_onnxruntime,
        detect_rocm,
        detect_zluda,
        ort_execution_providers,
    )

    ort = bootstrap_onnxruntime()
    assert ort is bootstrap_onnxruntime(), "bootstrap must cache the module"
    available = set(ort.get_available_providers())
    has_cuda = "CUDAExecutionProvider" in available
    has_dml = "DMLExecutionProvider" in available

    cpu_providers, cpu_note = ort_execution_providers("cpu")
    assert cpu_providers == ["CPUExecutionProvider"] and cpu_note == ""

    for device in ("auto", "cuda", "rocm", "directml", "cpu"):
        providers, note = ort_execution_providers(device)
        assert isinstance(providers, list) and len(providers) >= 1
        assert providers[-1] == "CPUExecutionProvider"

    if has_cuda:
        cuda_p, _note = ort_execution_providers("cuda")
        assert cuda_p[0] == "CUDAExecutionProvider"
        auto_p, _n = ort_execution_providers("auto")
        if detect_rocm():
            # native ROCm torch: auto must not initialize the NVIDIA EP
            assert auto_p[0] != "CUDAExecutionProvider"
        else:
            assert auto_p[0] == "CUDAExecutionProvider"
        rocm_p, rocm_note = ort_execution_providers("rocm")
        if detect_zluda():
            assert rocm_p[0] == "CUDAExecutionProvider" and "ZLUDA" in rocm_note
        else:
            # NV isolation: without ZLUDA a rocm request never uses the CUDA EP
            assert rocm_p[0] == "CPUExecutionProvider"

    if has_dml:
        dml_p, _n = ort_execution_providers("directml")
        assert dml_p[0] == "DMLExecutionProvider"
    else:
        dml_p, dml_note = ort_execution_providers("directml")
        assert dml_p == ["CPUExecutionProvider"] and len(dml_note) > 0


def test_resolve_pytorch_device_all_backends():
    from core.device_utils import (
        cuda_available,
        detect_rocm,
        detect_zluda,
        resolve_pytorch_device,
    )

    assert "cpu" in str(resolve_pytorch_device("cpu"))
    # DirectML is an ORT-only backend; torch engines fall back to CPU.
    assert "cuda" not in str(resolve_pytorch_device("directml")).lower()
    if cuda_available():
        assert "cuda" in str(resolve_pytorch_device("auto"))
        if detect_rocm() or detect_zluda():
            # hip torch or ZLUDA-emulated CUDA -> rocm lands on the AMD device
            assert "cuda" in str(resolve_pytorch_device("rocm"))
        else:
            # plain NVIDIA CUDA torch: rocm refuses instead of leaking to NV
            with pytest.raises(RuntimeError):
                resolve_pytorch_device("rocm")
