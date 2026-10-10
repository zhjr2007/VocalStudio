"""Tests for the new UI (``vstudio_ui``).

These lock the properties that make the rebuild worth doing:

* it does **not** import the archived ``gui`` package,
* both themes are fully specified and share one token vocabulary,
* the shell can build and navigate every page without crashing,
* the business layer is reached only through ``vstudio_ui.services``, and every
  probe runs off the GUI thread through ``JobHandle``.

The navigation test is a regression guard: an earlier version of
``refresh_theme_tree`` called itself through ``AppShell.refresh_theme``, and the
resulting infinite recursion killed the process with a *silent native crash* —
no traceback, no error, just a dead window on the first page switch.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1] / "vstudio_ui"


def _pump(app, predicate, timeout_ms: int = 5000) -> bool:
    """Spin the event loop until *predicate* holds or the deadline passes.

    Needed because ``JobHandle`` reports through queued signals: a worker thread
    that has finished still needs the main thread to process its events.
    """
    from PyQt5.QtCore import QElapsedTimer, QThread
    from PyQt5.QtWidgets import QApplication

    timer = QElapsedTimer()
    timer.start()
    while not predicate() and timer.elapsed() < timeout_ms:
        QApplication.processEvents()
        QThread.msleep(5)
    QApplication.processEvents()
    return bool(predicate())


# ---------------------------------------------------------------------------
# isolation from the archived UI
# ---------------------------------------------------------------------------


def test_new_ui_never_imports_the_archived_gui():
    """``vstudio_ui`` must be self-contained; only ``core``/``engines`` are shared."""
    import_stmt = re.compile(r"^\s*(from|import)\s+gui\b")
    offenders = []
    for path in sorted(PKG.rglob("*.py")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if import_stmt.match(line):
                offenders.append(f"{path.relative_to(PKG)}:{lineno}: {line.strip()}")
    assert not offenders, "vstudio_ui must not import the archived gui package:\n" + "\n".join(offenders)


def test_new_ui_does_not_use_third_party_widget_libraries():
    """The kit is PyQt5-only, so no widget library can move our pixels."""
    banned = ("qfluentwidgets", "qfluent_widgets", "PySide")
    offenders = []
    for path in sorted(PKG.rglob("*.py")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if line.strip().startswith("#"):
                continue
            if re.match(r"^\s*(from|import)\s+\S*", line) and any(name in line for name in banned):
                offenders.append(f"{path.relative_to(PKG)}:{lineno}: {line.strip()}")
    assert not offenders, "vstudio_ui must stay PyQt5-only:\n" + "\n".join(offenders)


# ---------------------------------------------------------------------------
# tokens
# ---------------------------------------------------------------------------


def test_both_themes_define_the_same_tokens():
    from vstudio_ui.tokens import DARK, LIGHT, THEME_ORDER

    assert set(LIGHT) == set(DARK), "light/dark token sets diverged"
    assert set(THEME_ORDER) == {"light", "dark"}
    for name, value in {**LIGHT, **DARK}.items():
        assert re.fullmatch(r"#[0-9A-Fa-f]{6}", value), f"{name}={value!r}"


def test_scale_tables_are_ordered_and_positive():
    from vstudio_ui.tokens import MOTION, SPACE, TYPE_SCALE

    assert TYPE_SCALE["display"] > TYPE_SCALE["h1"] > TYPE_SCALE["h2"] > TYPE_SCALE["h3"]
    assert TYPE_SCALE["h3"] > TYPE_SCALE["body"] > TYPE_SCALE["caption"]
    assert all(value > 0 for value in SPACE.values())
    assert MOTION["enter_ms"] > 0 and MOTION["exit_ms"] > 0
    assert len(MOTION["enter_curve"]) == 4 and len(MOTION["exit_curve"]) == 4


# ---------------------------------------------------------------------------
# strings
# ---------------------------------------------------------------------------


def test_string_tables_have_key_parity():
    from vstudio_ui import strings

    zh = set(strings._LANGUAGES["zh"])
    en = set(strings._LANGUAGES["en"])
    assert zh == en, f"zh-only: {sorted(zh - en)}; en-only: {sorted(en - zh)}"


def test_every_nav_key_has_a_label():
    from vstudio_ui import strings
    from vstudio_ui.app import PAGE_SPECS

    for key, _factory in PAGE_SPECS:
        label = strings.NAV_LABELS.get(key)
        assert label and label != f"nav.{key}", f"missing nav label for {key!r}"


# ---------------------------------------------------------------------------
# theme
# ---------------------------------------------------------------------------


def test_theme_switching_updates_palette_and_sheet(app):
    from vstudio_ui.theme import apply_theme, build_qss, color, color_hex, current_theme

    for name in ("light", "dark"):
        apply_theme(name)
        assert current_theme() == name
        sheet = build_qss(name)
        assert sheet.strip()
        assert color("card").isValid()
        assert color_hex("primary").startswith("#")


def test_unknown_token_raises(app):
    from vstudio_ui.theme import color

    with pytest.raises(KeyError):
        color("not_a_token")


def test_qss_carries_no_font_family(app):
    """The family must be set at application level to keep its fallback chain."""
    from vstudio_ui.theme import build_qss

    # strip /* ... */ comments so the explanatory note does not count as a hit
    sheet = re.sub(r"/\*.*?\*/", "", build_qss("light"), flags=re.S)
    assert "font-family" not in sheet


def test_application_font_family_is_applied(app):
    from vstudio_ui.theme import apply_theme
    from vstudio_ui.tokens import FONT_FAMILY

    apply_theme("light")
    assert app.font().family() == FONT_FAMILY[0]


def test_easing_helpers(app):
    from vstudio_ui.theme import easing_curve, motion_duration, prefers_reduced_motion

    assert easing_curve("enter").type() is not None
    assert motion_duration("enter") > 0
    assert prefers_reduced_motion() is False


# ---------------------------------------------------------------------------
# widgets
# ---------------------------------------------------------------------------


def test_widgets_construct_and_restyle(app):
    from vstudio_ui.theme import apply_theme
    from vstudio_ui.widgets import (
        Card,
        CardHeader,
        EmptyState,
        FieldRow,
        Hairline,
        HudRing,
        NavRail,
        PageScaffold,
        StatusDot,
    )

    widgets = [
        Card(),
        Card(alt=True),
        Hairline(),
        Hairline(vertical=True, strong=True),
        CardHeader("T", caption="C", trailing=StatusDot("ready")),
        PageScaffold("P", caption="c"),
        FieldRow("Label", StatusDot("idle"), hint="hint"),
        EmptyState("nothing", "hint"),
        NavRail(),
        HudRing(diameter=64),
    ]
    try:
        for theme in ("light", "dark"):
            apply_theme(theme)
            for widget in widgets:
                widget.refresh_theme()
    finally:
        for widget in widgets:
            widget.deleteLater()


def test_status_dot_states(app):
    from vstudio_ui.widgets import STATE_TOKENS, StatusDot

    for state in STATE_TOKENS:
        assert StatusDot(state).state == state
    assert StatusDot("nonsense").state == "idle"


def test_hud_ring_start_stop_and_reduced_motion(app, monkeypatch):
    from vstudio_ui.widgets import HudRing

    ring = HudRing(diameter=64)
    try:
        ring.start()
        assert ring.is_running() is True
        ring.stop()
        assert ring.is_running() is False
    finally:
        ring.deleteLater()

    monkeypatch.setenv("VOCALSTUDIO_REDUCED_MOTION", "1")
    ring = HudRing(diameter=64)
    try:
        ring.start()
        assert ring.is_running() is False, "reduced motion must not spin"
    finally:
        ring.deleteLater()


def test_field_row_align_uses_the_widest_label(app):
    from vstudio_ui.widgets import FieldRow

    rows = [FieldRow("Short"), FieldRow("A much longer label")]
    try:
        width = FieldRow.align(rows)
        assert width >= rows[1].label.sizeHint().width()
        assert rows[0].label.minimumWidth() == rows[1].label.minimumWidth() == width
    finally:
        for row in rows:
            row.deleteLater()


def test_refresh_theme_tree_does_not_recurse(app):
    """Guards the silent native crash: the root's own hook must not be re-entered."""
    from vstudio_ui.widgets import Card, PageScaffold, refresh_theme_tree

    page = PageScaffold("P")
    card = Card()
    page.content_layout.addWidget(card)
    try:
        refresh_theme_tree(page)                      # includes root — fine here
        refresh_theme_tree(page, include_root=False)  # what AppShell does
    finally:
        page.deleteLater()


# ---------------------------------------------------------------------------
# shell
# ---------------------------------------------------------------------------


def test_shell_builds_every_page(app):
    from vstudio_ui.app import PAGE_SPECS, build_shell

    shell = build_shell(theme="light")
    try:
        assert shell.page_keys() == [key for key, _f in PAGE_SPECS]
        assert len(PAGE_SPECS) == 8
        for key, _factory in PAGE_SPECS:
            assert shell.page(key) is not None
    finally:
        shell.deleteLater()


def test_shell_navigates_every_page_without_crashing(app):
    """Regression guard for the infinite-recursion crash on the first page switch."""
    from vstudio_ui.app import build_shell

    shell = build_shell(theme="light")
    try:
        for index in range(shell._nav.count()):
            shell.navigate(index)
            assert shell.current_index() == index
            assert shell._nav.current_index() == index
    finally:
        shell.deleteLater()


def test_shell_theme_toggle_round_trips(app):
    from vstudio_ui.app import build_shell
    from vstudio_ui.theme import current_theme

    shell = build_shell(theme="light")
    try:
        assert current_theme() == "light"
        shell.toggle_theme()
        assert current_theme() == "dark"
        shell.toggle_theme()
        assert current_theme() == "light"
    finally:
        shell.deleteLater()


def test_settings_page_theme_combo_drives_the_shell(app):
    from vstudio_ui.app import build_shell
    from vstudio_ui.theme import current_theme

    shell = build_shell(theme="light")
    try:
        settings = shell.page("settings")
        assert settings is not None
        settings.theme_combo.setCurrentIndex(1)  # dark
        assert current_theme() == "dark"
        settings.sync_theme("light")
        assert settings.theme_combo.currentData() == "light"
    finally:
        shell.deleteLater()


def test_shell_retranslate_switches_language(app):
    from vstudio_ui import strings
    from vstudio_ui.app import build_shell

    shell = build_shell(theme="light")
    try:
        strings.set_language("en")
        shell.retranslate()
        assert shell._nav._items[0].label == strings.tr("nav.overview")
    finally:
        strings.set_language("zh")
        shell.retranslate()
        shell.deleteLater()


# ---------------------------------------------------------------------------
# jobs — the frozen terminal-signal semantics (contract.md §6)
# ---------------------------------------------------------------------------


def test_job_runner_has_exactly_one_implementation():
    """The archived GUI re-exports the canonical runner; two copies would drift."""
    from gui.widgets.job_runner import JobRunner as archived
    from vstudio_ui.jobs import JobRunner

    assert archived is JobRunner


def test_job_handle_emits_exactly_one_terminal_signal(app):
    from vstudio_ui.jobs import JobHandle

    events: list = []
    handle = JobHandle()
    handle.done.connect(lambda result, cancelled: events.append(("done", result, cancelled)))
    handle.failed.connect(lambda exc: events.append(("failed", exc)))

    assert handle.start(lambda ctx, cfg, cb: 42) is True
    assert _pump(app, lambda: bool(events) and not handle.running)
    assert events == [("done", 42, False)]


def test_job_handle_reports_cancellation_as_done_not_failed(app):
    from core.exceptions import CancellationError
    from vstudio_ui.jobs import JobHandle

    class Cfg:
        cancel_checker = None

    cfg = Cfg()
    events: list = []
    entered = []

    def job(_ctx, cfg, _cb):
        entered.append(True)
        while not (cfg.cancel_checker and cfg.cancel_checker()):
            time.sleep(0.005)
        raise CancellationError("cancelled")

    handle = JobHandle()
    handle.done.connect(lambda result, cancelled: events.append((result, cancelled)))
    handle.failed.connect(lambda exc: events.append(("failed", exc)))

    assert handle.start(job, None, cfg) is True
    assert _pump(app, lambda: bool(entered), timeout_ms=3000)
    handle.cancel()
    assert _pump(app, lambda: bool(events))
    assert events == [(None, True)]


def test_job_handle_can_run_again_after_finishing(app):
    """JobRunner is one-shot; JobHandle is what makes a "run again" button work."""
    from vstudio_ui.jobs import JobHandle

    results: list = []
    handle = JobHandle()
    handle.done.connect(lambda result, _cancelled: results.append(result))

    for index in range(3):
        assert handle.start(lambda _ctx, _cfg, _cb, i=index: i) is True
        assert _pump(app, lambda: len(results) == index + 1 and not handle.running)
    assert results == [0, 1, 2]


def test_job_handle_refuses_a_second_concurrent_start(app):
    import threading

    from vstudio_ui.jobs import JobHandle

    release = threading.Event()
    handle = JobHandle()

    assert handle.start(lambda _ctx, _cfg, _cb: release.wait(5)) is True
    assert handle.start(lambda _ctx, _cfg, _cb: None) is False
    release.set()
    assert _pump(app, lambda: not handle.running)


def test_as_job_adapts_the_frozen_signature():
    from vstudio_ui import services

    seen: list = []
    job = services.as_job(lambda cb: (cb(0.5, "half"), "result")[1])
    assert job(None, None, lambda fraction, stage: seen.append((fraction, stage))) == "result"
    assert seen == [(0.5, "half")]


# ---------------------------------------------------------------------------
# services — the single contract surface
# ---------------------------------------------------------------------------


def test_probe_runtime_reads_the_business_layer(monkeypatch):
    from core import device_utils
    from vstudio_ui import services

    monkeypatch.setattr(
        device_utils,
        "detect_system_hardware",
        lambda: {"cpu": "Test CPU", "gpus": [("Test GPU", 8.0)]},
    )
    monkeypatch.setattr(
        device_utils,
        "get_gpu_info",
        lambda: {
            "backend": "cuda",
            "name": "Test GPU",
            "total_vram_gb": 16.0,
            "free_vram_gb": 14.0,
            "torch_version": "2.9.1+cu128",
        },
    )
    monkeypatch.setattr(device_utils, "default_backend", lambda: "rocm")
    monkeypatch.setattr(device_utils, "auto_select_device", lambda: "cuda")
    monkeypatch.setattr(services.ModelPaths, "missing", lambda self: ["vad", "drums"])

    seen: list = []
    snapshot = services.probe_runtime(lambda fraction, stage: seen.append((fraction, stage)))

    assert snapshot.cpu == "Test CPU"
    assert snapshot.gpu_names == ("Test GPU",)
    assert snapshot.gpu_name == "Test GPU"
    assert snapshot.backend == "cuda", "the torch answer must beat the WMI guess"
    assert snapshot.backend_source == "torch"
    assert snapshot.vram_total_gb == 16.0 and snapshot.has_vram is True
    assert snapshot.device == "cuda"
    assert snapshot.models_missing == ("vad", "drums")
    assert snapshot.models_ready == snapshot.models_total - 2
    assert snapshot.models_ok is False
    assert snapshot.error == ""
    assert seen[0][0] == 0.0 and seen[-1][0] == 1.0
    assert [stage for _f, stage in seen if stage] == ["hardware", "backend", "device", "models"]


def test_probe_runtime_falls_back_to_the_wmi_backend_guess(monkeypatch):
    """Without torch the WMI heuristic is all there is; it must still be labelled."""
    from core import device_utils
    from vstudio_ui import services

    monkeypatch.setattr(
        device_utils,
        "detect_system_hardware",
        lambda: {"cpu": "Test CPU", "gpus": [("AMD Radeon 780M", 0.0)]},
    )
    monkeypatch.setattr(device_utils, "get_gpu_info", lambda: {"backend": "none"})
    monkeypatch.setattr(device_utils, "default_backend", lambda: "rocm")
    monkeypatch.setattr(device_utils, "auto_select_device", lambda: "cpu")
    monkeypatch.setattr(services.ModelPaths, "missing", lambda self: [])

    snapshot = services.probe_runtime()
    assert snapshot.backend == "rocm"
    assert snapshot.backend_source == "wmi"
    assert snapshot.gpu_name == ""
    assert snapshot.has_vram is False


def test_torch_backend_wins_over_the_wmi_guess(monkeypatch):
    """Regression guard for what the first Overview screenshot exposed.

    This machine has an AMD iGPU *and* an NVIDIA card, so ``default_backend()``
    answers ``rocm`` while torch computes on ``cuda``. Showing the guess as the
    headline made the primary diagnostic page lie.
    """
    from core import device_utils
    from vstudio_ui import services

    monkeypatch.setattr(
        device_utils,
        "detect_system_hardware",
        lambda: {
            "cpu": "AMD Ryzen 7 H 255",
            "gpus": [("MuMu Virtual Display Adapter", 0.0), ("AMD Radeon 780M", 0.0)],
        },
    )
    monkeypatch.setattr(
        device_utils,
        "get_gpu_info",
        lambda: {
            "backend": "cuda",
            "name": "NVIDIA GeForce RTX 5060 Ti",
            "total_vram_gb": 16.0,
            "free_vram_gb": 14.8,
            "torch_version": "2.9.1+cu128",
        },
    )
    monkeypatch.setattr(device_utils, "default_backend", lambda: "rocm")
    monkeypatch.setattr(device_utils, "auto_select_device", lambda: "cuda")
    monkeypatch.setattr(services.ModelPaths, "missing", lambda self: [])

    snapshot = services.probe_runtime()
    assert snapshot.backend == "cuda"
    assert snapshot.backend_source == "torch"
    assert snapshot.gpu_name == "NVIDIA GeForce RTX 5060 Ti"
    assert snapshot.device == "cuda"


def test_probe_runtime_reports_breakage_instead_of_raising(monkeypatch):
    """A diagnostic page that dies when the thing it diagnoses is broken is useless."""
    from core import device_utils
    from vstudio_ui import services

    def boom(*_args, **_kwargs):
        raise RuntimeError("wmi exploded")

    monkeypatch.setattr(device_utils, "detect_system_hardware", boom)
    monkeypatch.setattr(device_utils, "get_gpu_info", boom)
    monkeypatch.setattr(device_utils, "auto_select_device", boom)

    snapshot = services.probe_runtime()
    assert "wmi exploded" in snapshot.error
    assert snapshot.cpu == ""
    assert snapshot.device == ""


def test_probe_runtime_survives_a_broken_progress_callback(monkeypatch):
    from core import device_utils
    from vstudio_ui import services

    monkeypatch.setattr(device_utils, "detect_system_hardware", lambda: {"cpu": "C", "gpus": []})
    monkeypatch.setattr(device_utils, "get_gpu_info", lambda: {"backend": "cpu"})
    monkeypatch.setattr(device_utils, "default_backend", lambda: "cpu")
    monkeypatch.setattr(device_utils, "auto_select_device", lambda: "cpu")

    def bad_callback(_fraction, _stage):
        raise RuntimeError("callback exploded")

    assert services.probe_runtime(bad_callback).cpu == "C"


def test_self_check_covers_every_contract_surface(monkeypatch):
    from core import app_settings, device_utils
    from vstudio_ui import services

    monkeypatch.setattr(
        device_utils, "detect_system_hardware", lambda: {"cpu": "Test CPU", "gpus": []}
    )
    monkeypatch.setattr(device_utils, "get_gpu_info", lambda: {"backend": "cuda", "name": "Test GPU"})
    monkeypatch.setattr(device_utils, "default_backend", lambda: "cuda")
    monkeypatch.setattr(device_utils, "auto_select_device", lambda: "cuda")
    monkeypatch.setattr(services.ModelPaths, "missing", lambda self: [])
    monkeypatch.setattr(app_settings, "get_settings", lambda *a, **k: app_settings.AppSettings())

    report = services.run_self_check()
    assert [check.key for check in report.checks] == list(services.CHECK_ORDER)
    assert report.ok is True
    assert report.snapshot.gpu_name == "Test GPU"


def test_self_check_reports_failures_as_data(monkeypatch):
    from core import app_settings, device_utils
    from vstudio_ui import services

    monkeypatch.setattr(
        device_utils, "detect_system_hardware", lambda: {"cpu": "Test CPU", "gpus": []}
    )
    monkeypatch.setattr(device_utils, "get_gpu_info", lambda: {"backend": "cpu"})
    monkeypatch.setattr(device_utils, "default_backend", lambda: "cpu")
    monkeypatch.setattr(device_utils, "auto_select_device", lambda: "cpu")
    monkeypatch.setattr(services.ModelPaths, "missing", lambda self: ["vad"])
    monkeypatch.setattr(app_settings, "get_settings", lambda *a, **k: app_settings.AppSettings())

    report = services.run_self_check()
    models = next(check for check in report.checks if check.key == "models")
    assert models.ok is False
    assert report.ok is False


def test_load_preferences_maps_first_generation_choices(monkeypatch):
    from core import app_settings
    from vstudio_ui import services

    monkeypatch.setattr(
        app_settings,
        "get_settings",
        lambda *a, **k: app_settings.AppSettings(
            theme="rhine", language="en", device="cuda", font_scale=1.25
        ),
    )
    preferences = services.load_preferences()
    assert preferences.theme == "light", "rhine has no equivalent in a two-theme UI"
    assert preferences.language == "en"
    assert preferences.device == "cuda"
    assert preferences.font_scale == 1.25


def test_load_preferences_never_raises(monkeypatch):
    from core import app_settings
    from vstudio_ui import services

    def boom(*_args, **_kwargs):
        raise RuntimeError("corrupt user.json")

    monkeypatch.setattr(app_settings, "get_settings", boom)
    assert services.load_preferences() == services.Preferences()


# ---------------------------------------------------------------------------
# ValueLabel
# ---------------------------------------------------------------------------


def test_value_label_elides_but_keeps_the_full_text(app):
    from vstudio_ui.widgets import ValueLabel

    label = ValueLabel()
    try:
        label.resize(60, 20)
        long_text = "NVIDIA GeForce RTX 4090 Laptop GPU with a very long marketing name"
        label.set_value(long_text)
        assert label.full_text == long_text
        assert label.toolTip() == long_text
        assert label.text() != long_text
        assert "\u2026" in label.text()
    finally:
        label.deleteLater()


def test_value_label_infers_muted_from_an_empty_value(app):
    from vstudio_ui.widgets import ValueLabel

    label = ValueLabel()
    try:
        label.set_value("")
        assert label.muted is True
        label.set_value("cuda")
        assert label.muted is False
        label.set_value("", muted=False)
        assert label.muted is False
    finally:
        label.deleteLater()


# ---------------------------------------------------------------------------
# overview — business layer wired in
# ---------------------------------------------------------------------------


def test_building_the_shell_does_not_probe(app):
    """Construction must stay cheap: tests, screenshots and cold starts build shells."""
    from vstudio_ui.app import build_shell

    shell = build_shell(theme="light")
    try:
        page = shell.page("overview")
        assert page._probe.running is False
        assert page._probe.runner is None
        assert page._check.runner is None
    finally:
        shell.deleteLater()


def test_overview_renders_a_probed_snapshot(app, monkeypatch):
    from vstudio_ui import services, strings
    from vstudio_ui.app import build_shell

    snapshot = services.RuntimeSnapshot(
        cpu="Test CPU",
        gpus=(("Test GPU", 0.0),),
        gpu_name="Test GPU",
        vram_total_gb=16.0,
        vram_free_gb=14.0,
        backend="cuda",
        backend_source="torch",
        torch_version="2.9.1+cu128",
        device="cuda",
        device_setting="auto",
        models_total=22,
        models_missing=(),
    )

    def fake_probe(progress_cb=None):
        if progress_cb is not None:
            progress_cb(1.0, "models")
        return snapshot

    monkeypatch.setattr(services, "probe_runtime", fake_probe)

    shell = build_shell(theme="light")
    page = shell.page("overview")
    try:
        page.refresh_runtime()
        assert _pump(app, lambda: page._cpu_value.full_text == "Test CPU" and not page._probe.running)
        assert page._gpu_value.full_text == "Test GPU"
        assert page._vram_value.full_text == strings.tr("overview.vram_value").format(
            free="14.0", total="16.0"
        )
        assert page._backend_value.full_text == "cuda"
        assert page._backend_row.hint.text() == strings.tr("overview.backend_torch").format(
            version="2.9.1+cu128"
        )
        assert page._device_value.full_text == "cuda"
        assert page._models_value.full_text == strings.tr("overview.models_ready").format(
            ready=22, total=22
        )
        assert page._status.state == "done"
        assert page._progress.value() == 0, "the bar belongs to the self-check, not the probe"
        assert page._stage.full_text == strings.tr("overview.not_probed")
    finally:
        page.shutdown()
        shell.deleteLater()


def test_overview_labels_a_guessed_backend_as_a_guess(app, monkeypatch):
    """The WMI heuristic must never look like a fact on the page."""
    from vstudio_ui import services, strings
    from vstudio_ui.app import build_shell

    snapshot = services.RuntimeSnapshot(
        cpu="Test CPU",
        gpus=(("AMD Radeon 780M", 0.0),),
        backend="rocm",
        backend_source="wmi",
        device="cpu",
        models_total=22,
    )
    monkeypatch.setattr(services, "probe_runtime", lambda progress_cb=None: snapshot)

    shell = build_shell(theme="light")
    page = shell.page("overview")
    try:
        page.refresh_runtime()
        assert _pump(app, lambda: page._backend_value.full_text == "rocm" and not page._probe.running)
        assert page._backend_row.hint.text() == strings.tr("overview.backend_wmi")
        assert page._backend_value.toolTip() == strings.tr("overview.backend_wmi_tip")
        assert page._vram_value.muted is True
        assert page._gpu_value.full_text == "AMD Radeon 780M"
    finally:
        page.shutdown()
        shell.deleteLater()


def test_overview_marks_incomplete_models_as_amber(app, monkeypatch):
    """Amber means "works, but not fully set up" — not an error and not green."""
    from vstudio_ui import services
    from vstudio_ui.app import build_shell

    snapshot = services.RuntimeSnapshot(
        cpu="Test CPU",
        backend="cuda",
        backend_source="torch",
        device="cuda",
        models_total=22,
        models_missing=("vad", "drums"),
    )
    monkeypatch.setattr(services, "probe_runtime", lambda progress_cb=None: snapshot)

    shell = build_shell(theme="light")
    page = shell.page("overview")
    try:
        page.refresh_runtime()
        assert _pump(app, lambda: page._cpu_value.full_text == "Test CPU")
        assert page._status.state == "ready"
        assert "vad" in page._models_value.toolTip()
    finally:
        page.shutdown()
        shell.deleteLater()


def test_overview_probe_failure_is_shown_not_swallowed(app, monkeypatch):
    from vstudio_ui import services
    from vstudio_ui.app import build_shell

    def boom(progress_cb=None):
        raise RuntimeError("probe exploded")

    monkeypatch.setattr(services, "probe_runtime", boom)

    shell = build_shell(theme="light")
    page = shell.page("overview")
    try:
        page.refresh_runtime()
        assert _pump(app, lambda: page._status.state == "failed" and not page._probe.running)
        assert "probe exploded" in page._stage.full_text
        assert page._probe.runner is None
    finally:
        page.shutdown()
        shell.deleteLater()


def test_overview_self_check_logs_every_stage(app, monkeypatch):
    from vstudio_ui import services, strings
    from vstudio_ui.app import build_shell

    report = services.SelfCheckReport(
        checks=(
            services.Check("hardware", True, "Test CPU"),
            services.Check("models", False, "19/21"),
        ),
        snapshot=services.RuntimeSnapshot(cpu="Test CPU", models_total=21, models_missing=("vad",)),
    )
    monkeypatch.setattr(services, "run_self_check", lambda progress_cb=None: report)

    shell = build_shell(theme="light")
    page = shell.page("overview")
    try:
        page.run_self_check()
        assert _pump(app, lambda: bool(page._log.toPlainText().strip()) and not page._check.running)
        text = page._log.toPlainText()
        assert strings.tr("check.hardware") in text
        assert strings.tr("check.models") in text
        assert strings.tr("check.fail") in text
        assert strings.tr("check.summary").format(passed=1, total=2) in text
        assert page._status.state == "ready"
        assert page._run_button.isEnabled() is True
        assert page._cancel_button.isEnabled() is False
        assert page.ring.is_running() is False
        assert page._probe.runner is None and page._check.runner is None
    finally:
        page.shutdown()
        shell.deleteLater()


def test_overview_log_box_fits_a_whole_self_check(app):
    """A finished self-check has to be readable without scrolling.

    The height was a hard-coded 150 px, which fits four lines at this machine's
    font size while a self-check prints one line per stage plus a summary.
    """
    from PyQt5.QtGui import QFontMetrics

    from vstudio_ui import services
    from vstudio_ui.app import build_shell
    from vstudio_ui.pages.overview import _LOG_LINES

    shell = build_shell(theme="light")
    page = shell.page("overview")
    try:
        line = QFontMetrics(page._log.font()).lineSpacing()
        assert _LOG_LINES == len(services.CHECK_ORDER) + 1
        assert page._log.minimumHeight() >= line * _LOG_LINES
    finally:
        shell.deleteLater()


def test_overview_self_check_replaces_the_stage_with_the_summary(app, monkeypatch):
    """A finished check must not leave a stage name ("settings") under the ring.

    The stage label is a progress readout. Leaving the last stage name there after
    the run ends reads as if the page were still working on it.
    """
    from vstudio_ui import services, strings
    from vstudio_ui.app import build_shell

    report = services.SelfCheckReport(
        checks=(
            services.Check("hardware", True, "Test CPU"),
            services.Check("settings", True, "settings/user.json"),
        ),
        snapshot=services.RuntimeSnapshot(cpu="Test CPU"),
    )

    def fake(progress_cb=None):
        if progress_cb is not None:
            progress_cb(1.0, "settings")
        return report

    monkeypatch.setattr(services, "run_self_check", fake)

    shell = build_shell(theme="light")
    page = shell.page("overview")
    try:
        page.run_self_check()
        assert _pump(
            app, lambda: not page._check.running and page._status.state == "done"
        )
        assert page._stage.full_text == strings.tr("check.summary").format(passed=2, total=2)
        assert page._stage.full_text != strings.tr("check.settings")
    finally:
        page.shutdown()
        shell.deleteLater()


def test_overview_self_check_failure_stops_the_ring(app, monkeypatch):
    from vstudio_ui import services, strings
    from vstudio_ui.app import build_shell

    def boom(progress_cb=None):
        raise RuntimeError("check exploded")

    monkeypatch.setattr(services, "run_self_check", boom)

    shell = build_shell(theme="light")
    page = shell.page("overview")
    try:
        page.run_self_check()
        assert _pump(app, lambda: page._status.state == "failed" and not page._check.running)
        assert page.ring.is_running() is False
        assert strings.tr("check.summary_failed").format(message="check exploded") in (
            page._log.toPlainText()
        )
        assert page._run_button.isEnabled() is True
    finally:
        page.shutdown()
        shell.deleteLater()


def test_overview_retranslate_covers_the_new_labels(app):
    from vstudio_ui import strings
    from vstudio_ui.app import build_shell

    shell = build_shell(theme="light")
    page = shell.page("overview")
    try:
        strings.set_language("en")
        page.retranslate()
        assert page._cpu_row.label.text() == strings.tr("overview.cpu")
        assert page._run_button.text() == strings.tr("overview.run_check")
        assert page._cancel_button.text() == strings.tr("overview.cancel")
        assert page._refresh_button.text() == strings.tr("overview.refresh")
        assert page._status.toolTip() == strings.tr("status.idle")
    finally:
        strings.set_language("zh")
        page.retranslate()
        shell.deleteLater()
