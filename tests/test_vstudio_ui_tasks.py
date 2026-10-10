"""Tests for the task pages and the run surface of ``vstudio_ui.services``.

Separate module from ``test_vstudio_ui.py`` because that one locks the *structure*
(isolation, tokens, shell) while this one locks *behaviour*: what a page does when
you press Start, what it does when the job fails, and — the part that is easy to get
wrong and invisible in a screenshot — what happens to ``cfg`` before the job runs.

The regression guards worth naming:

* ``test_build_config_populates_model_paths`` — an empty ``ModelPaths()`` is
  **truthy**, so the engines' ``cfg.model_paths or ModelPaths.from_settings()``
  fallback never fires. Every step except separation would run with blank paths.
* ``test_training_exit_code_decides_success`` — the training engine records a
  non-zero exit and returns *normally*, so without checking the code a crashed run
  looks exactly like a successful one.
* ``test_strings_used_in_the_source_all_exist`` — ``strings.tr`` falls back to the
  key itself, so a typo renders the raw key instead of raising.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import os
import re
import tempfile
from pathlib import Path

import pytest

#: Every test here that builds a page takes the session-scoped ``app`` fixture
#: explicitly. That is not just convention: a widget created with no
#: ``QApplication`` does not raise, it **kills the process** with no traceback and
#: no output at all — a genuinely confusing thing to debug.
#:
#: Temporary directories come from :func:`_scratch`, not from a fixture, for the
#: reason spelled out in its docstring.


def _scratch() -> Path:
    """A fresh empty directory for one test.

    Deliberately **not** a fixture. In this pytest version, any extra function-scoped
    fixture used together with the ones these tests already take trips
    ``assert not self._finalizers`` in ``_pytest/fixtures.py`` at setup — including
    a fixture of my own that merely wraps ``tmp_path_factory``. A plain call keeps
    the fixture graph to exactly what each test asks for.

    Nothing deletes these; they hold a couple of empty files each and live in the
    system temp.
    """
    return Path(tempfile.mkdtemp(prefix="vstudio-ui-task-"))


def _pump(app, predicate, timeout_ms: int = 5000) -> bool:
    """Spin the event loop until *predicate* holds or the deadline passes."""
    from PyQt5.QtCore import QElapsedTimer, QThread
    from PyQt5.QtWidgets import QApplication

    timer = QElapsedTimer()
    timer.start()
    while not predicate() and timer.elapsed() < timeout_ms:
        QApplication.processEvents()
        QThread.msleep(5)
    QApplication.processEvents()
    return bool(predicate())


class _FakeCfg:
    """Stands in for a real config in page tests: no hardware, no torch.

    It deliberately carries ``log_cb`` / ``cancel_checker``, because that is what
    ``JobRunner`` looks for before the job starts.
    """

    def __init__(self):
        self.log_cb = None
        self.cancel_checker = None


def _stub_pipeline(monkeypatch, result, *, progress=(), log=()):
    """Make ``TaskPage.run`` touch nothing real. Returns what it recorded."""
    from vstudio_ui import services

    calls: dict = {"cfg": None, "request": None}

    def build_config(request):
        calls["request"] = request
        return _FakeCfg()

    def job(_ctx=None, _cfg=None, progress_cb=None):
        calls["cfg"] = _cfg
        for fraction, stage in progress:
            if progress_cb is not None:
                progress_cb(fraction, stage)
        callback = getattr(_cfg, "log_cb", None)
        if callable(callback):
            for line in log:
                callback(line)
        return result

    monkeypatch.setattr(services, "build_config", build_config)
    monkeypatch.setattr(services, "run_pipeline_job", job)
    return calls


# ---------------------------------------------------------------------------
# services: vocabularies and config building
# ---------------------------------------------------------------------------


def test_pipeline_steps_come_from_the_business_layer():
    """A new orchestrator step must show up here instead of having no page."""
    from core.orchestrator import STEP_ORDER

    from vstudio_ui import services

    assert services.PIPELINE_STEPS == tuple(step.value for step in STEP_ORDER)


def test_option_vocabularies_come_from_the_business_layer():
    """A hard-coded list in a page is how the UI offers a choice the config rejects."""
    from core import config as config_module
    from core import device_utils

    from vstudio_ui import services

    assert services.DEVICE_CHOICES == tuple(device_utils.RUNTIME_DEVICE_CHOICES)
    assert services.LANGUAGE_CHOICES == tuple(config_module.SUPPORTED_LANGUAGES)
    assert services.SEPARATION_MODELS == tuple(config_module.SEPARATION_MODELS)
    assert services.TTS_VERSIONS == tuple(config_module.TTS_VERSIONS)
    assert services.ASR_ENGINES == tuple(config_module.ASR_ENGINES)


def test_build_config_populates_model_paths():
    """Regression guard for a silent-misconfiguration trap.

    Engines read ``getattr(cfg, "model_paths", None) or ModelPaths.from_settings()``
    — and an empty ``ModelPaths()`` is **truthy**, so the ``or`` short-circuits and
    the engine receives blank paths. Only the separation engine has its own
    fallback, so every other step would fail with "model path not configured".
    ``main.py cmd_tts`` works around this by hand, which is the confirmation.
    """
    from core.config import ModelPaths, PipelineConfig

    from vstudio_ui import services

    assert bool(ModelPaths()) is True, "the premise: an empty ModelPaths is truthy"
    assert bool(PipelineConfig().model_paths) is True, "the premise: the default is truthy"

    cfg = services.build_config(services.RunRequest(audio_path="a.wav"))
    assert cfg.model_paths.sensevoice, "model_paths was left blank"


def test_build_config_creates_log_cb_before_the_job_starts():
    """``JobRunner`` only wraps a ``log_cb`` that already exists.

    It installs its forwarder before calling the job body, so a body that adds the
    attribute itself is too late and the log pane stays empty forever.
    """
    from vstudio_ui import services

    cfg = services.build_config(services.RunRequest())
    assert hasattr(cfg, "log_cb")
    assert cfg.log_cb is None


def test_build_config_rejects_unknown_and_runner_owned_options():
    from vstudio_ui import services

    with pytest.raises(ValueError, match="unknown pipeline option"):
        services.build_config(services.RunRequest(options=(("nope", 1),)))
    for owned in ("cancel_checker", "log_cb", "model_paths", "pipeline_steps"):
        with pytest.raises(ValueError, match="owned by the runner"):
            services.build_config(services.RunRequest(options=((owned, 1),)))


def test_build_config_carries_the_request_through():
    from vstudio_ui import services

    request = services.RunRequest(
        audio_path="song.wav",
        output_dir="out",
        device="cpu",
        language="ja",
        steps=("separation", "vocal2midi"),
        output_formats=("mid",),
        options=(("separation_agg", 3),),
    )
    cfg = services.build_config(request)
    assert cfg.audio_path == "song.wav"
    assert cfg.output_dir == "out"
    assert cfg.device == "cpu"
    assert cfg.language == "ja"
    assert cfg.pipeline_steps == ["separation", "vocal2midi"]
    assert cfg.output_formats == ["mid"]
    assert cfg.separation_agg == 3


def test_suggest_output_dir_follows_the_cli_convention():
    """``main.py separate`` documents ``<audio parent>/outputs/separation``."""
    from vstudio_ui import services

    suggested = services.suggest_output_dir("E:/a/b/song.mp3", "separation")
    assert suggested.endswith(os.path.join("outputs", "separation"))
    assert services.suggest_output_dir("", "separation") == ""


# ---------------------------------------------------------------------------
# services: the pipeline job body
# ---------------------------------------------------------------------------


def test_run_pipeline_job_flattens_the_context(monkeypatch):
    from vstudio_ui import services

    class FakeOrchestrator:
        def __init__(self, cfg):
            self.cfg = cfg

        def with_progress(self, callback):
            self.callback = callback
            return self

        def execute(self, context=None):
            from core.config import AudioType
            from core.context import PipelineContext

            ctx = PipelineContext(audio_path="a.wav", audio_type=AudioType.SINGING)
            ctx.vocals_path = "v.wav"
            ctx.accompaniment_path = "acc.wav"
            ctx.detected_language = "zh"
            ctx.asr_text = "hello"
            ctx.speakers = {"s0": {}}
            ctx.notes = [1, 2, 3]
            ctx.set_step_status("separation", "done")
            ctx.set_step_status("vocal2midi", "skipped", "content detected as speech")
            return ctx

    monkeypatch.setattr(services, "VocalStudioOrchestrator", FakeOrchestrator)
    monkeypatch.setattr(services, "preflight", lambda cfg: None)
    cfg = _FakeCfg()
    cfg.output_dir = ""
    result = services.run_pipeline_job(None, cfg, None)

    assert isinstance(result, services.RunResult)
    assert result.vocals_path == "v.wav"
    assert result.accompaniment_path == "acc.wav"
    assert result.audio_type == "singing"
    assert result.detected_language == "zh"
    assert result.asr_text == "hello"
    assert result.notes_count == 3
    assert result.speakers == ("s0",)
    assert result.ok is True
    assert [outcome.step for outcome in result.skipped] == ["vocal2midi"]
    assert [outcome.step for outcome in result.ran] == ["separation"]


def test_run_pipeline_job_lets_cancellation_escape(monkeypatch):
    """Cancellation must reach ``JobRunner``, which maps it to ``done(None, True)``.

    Swallowing it here would turn "the user pressed Cancel" into "the run finished",
    and the page would report success. Note the exception is a ``VocalStudioError``
    subclass, *not* a ``ValueError`` — see contract.md §1.1.
    """
    scratch = _scratch()
    from core.exceptions import CancellationError, VocalStudioError

    from vstudio_ui import services

    assert issubclass(CancellationError, VocalStudioError)

    class FakeOrchestrator:
        def __init__(self, cfg):
            pass

        def with_progress(self, callback):
            return self

        def execute(self, context=None):
            raise CancellationError("cancelled")

    monkeypatch.setattr(services, "VocalStudioOrchestrator", FakeOrchestrator)
    monkeypatch.setattr(services, "preflight", lambda cfg: None)
    cfg = _FakeCfg()
    cfg.output_dir = str(scratch / "out")
    with pytest.raises(CancellationError):
        services.run_pipeline_job(None, cfg, None)


def test_run_pipeline_job_creates_the_output_directory(monkeypatch):
    """Not every engine mkdirs its output; failing after a model load wastes it."""
    scratch = _scratch()
    from vstudio_ui import services

    class FakeOrchestrator:
        def __init__(self, cfg):
            pass

        def with_progress(self, callback):
            return self

        def execute(self, context=None):
            from core.context import PipelineContext

            return PipelineContext()

    monkeypatch.setattr(services, "VocalStudioOrchestrator", FakeOrchestrator)
    monkeypatch.setattr(services, "preflight", lambda cfg: None)
    target = scratch / "made" / "by" / "the" / "job"
    cfg = _FakeCfg()
    cfg.output_dir = str(target)
    services.run_pipeline_job(None, cfg, None)
    assert target.is_dir()


def test_run_pipeline_job_needs_a_config():
    from vstudio_ui import services

    with pytest.raises(ValueError, match="needs a config"):
        services.run_pipeline_job(None, None, None)


# ---------------------------------------------------------------------------
# services: training
# ---------------------------------------------------------------------------


def test_training_tasks_come_from_the_adapters():
    from engines.training.adapters import TASK_IDS

    from vstudio_ui import services

    assert services.training_tasks() == tuple(TASK_IDS)


def test_build_training_spec_carries_the_runner_hooks():
    """``log_cb`` / ``cancel_checker`` must exist before the job starts."""
    from vstudio_ui import services

    spec = services.build_training_spec(
        services.TrainingRequest(task="gsv_tts", data_dir="d", output_dir="o", epochs=2, batch=8)
    )
    assert spec.task == "gsv_tts"
    assert spec.params == {"epochs": 2, "batch": 8, "lr": 5e-4}
    assert hasattr(spec, "log_cb") and hasattr(spec, "cancel_checker")
    assert spec.device == "cuda", "training pins CUDA; 'auto' would mean CPU here"


@pytest.mark.parametrize("exit_code,expected_ok", [(0, True), (1, False), (3, False)])
def test_training_exit_code_decides_success(monkeypatch, exit_code, expected_ok):
    """The engine records a non-zero exit and returns normally — it does not raise.

    Without checking the code a crashed training run looks exactly like a
    successful one, and the page shows green.
    """
    import engines.training.job_runner as engine

    from vstudio_ui import services

    def fake(ctx, cfg, progress_cb):
        ctx["exit_code"] = exit_code
        ctx["checkpoints"] = ["ckpt.pt"]
        return ctx

    monkeypatch.setattr(engine, "run_training_job", fake)
    spec = services.build_training_spec(services.TrainingRequest(output_dir="out"))
    result = services.run_training_job(None, spec, None)

    assert result.ok is expected_ok
    assert result.exit_code == exit_code
    assert result.checkpoints == ("ckpt.pt",)
    if not expected_ok:
        assert "exited with code" in result.error


def test_run_training_job_needs_a_spec():
    from vstudio_ui import services

    with pytest.raises(ValueError, match="needs a spec"):
        services.run_training_job(None, None, None)


# ---------------------------------------------------------------------------
# task pages
# ---------------------------------------------------------------------------

TASK_PAGE_KEYS = ("separation", "midi", "asr", "tts", "pipeline", "training")


def test_task_page_keys_match_the_shell():
    """``TaskPage.key`` drives the strings and the output folder, so it must match."""
    from vstudio_ui.app import PAGE_SPECS

    shell_keys = [key for key, _factory in PAGE_SPECS]
    assert set(TASK_PAGE_KEYS) <= set(shell_keys)


def test_every_task_page_key_is_a_task_page(app):
    from vstudio_ui.app import build_shell
    from vstudio_ui.pages.task_page import TaskPage

    shell = build_shell(theme="light")
    try:
        for key in TASK_PAGE_KEYS:
            page = shell.page(key)
            assert isinstance(page, TaskPage), key
            assert page.key == key, f"{key} page reports key={page.key!r}"
    finally:
        shell.deleteLater()


def test_building_the_shell_starts_no_job(app):
    """Same rule as the Overview page: constructing a page must be cheap."""
    from vstudio_ui.app import build_shell

    shell = build_shell(theme="light")
    try:
        for key in TASK_PAGE_KEYS:
            page = shell.page(key)
            assert page._job.running is False, f"{key} started a job while building"
            assert page._status.state == "idle"
    finally:
        shell.deleteLater()


def test_task_page_refuses_to_run_without_input(app):
    from vstudio_ui import strings
    from vstudio_ui.app import build_shell

    shell = build_shell(theme="light")
    page = shell.page("separation")
    try:
        page.run()
        assert page._job.running is False
        assert page._status.state == "failed"
        assert page._stage.full_text == strings.tr("task.missing_input")
    finally:
        shell.deleteLater()


def test_task_page_reports_a_missing_file(app):
    scratch = _scratch()
    from vstudio_ui.app import build_shell

    shell = build_shell(theme="light")
    page = shell.page("separation")
    try:
        page.set_audio_path(str(scratch / "nope.wav"))
        page.run()
        assert page._status.state == "failed"
        assert "nope.wav" in page._stage.full_text
    finally:
        shell.deleteLater()


def test_task_page_suggests_an_output_dir_but_never_clobbers(app):
    scratch = _scratch()
    from vstudio_ui import services
    from vstudio_ui.app import build_shell

    audio = scratch / "song.wav"
    audio.write_bytes(b"")
    shell = build_shell(theme="light")
    page = shell.page("separation")
    try:
        page.set_audio_path(str(audio))
        assert page.output_edit.text() == services.suggest_output_dir(str(audio), "separation")

        # A directory the user typed must survive a change of input file.
        page.set_output_dir(str(scratch / "mine"))
        page.set_audio_path(str(scratch / "other.wav"))
        assert page.output_edit.text() == str(scratch / "mine")

        # A value *we* suggested is allowed to follow the input.
        page.set_audio_path(str(audio))
        page.set_output_dir(services.suggest_output_dir(str(audio), "separation"))
        page.set_audio_path(str(scratch / "third.wav"))
        assert page.output_edit.text() == services.suggest_output_dir(
            str(scratch / "third.wav"), "separation"
        )
    finally:
        shell.deleteLater()


def test_task_page_run_reports_progress_and_result(app, monkeypatch):
    scratch = _scratch()
    from vstudio_ui import services
    from vstudio_ui.app import build_shell

    audio = scratch / "song.wav"
    audio.write_bytes(b"")
    result = services.RunResult(
        steps=(services.StepOutcome("separation", "done"),),
        vocals_path="vocals.wav",
        accompaniment_path="acc.wav",
    )
    calls = _stub_pipeline(
        monkeypatch, result, progress=((0.5, "separation: analyze"),), log=("engine says hi",)
    )

    shell = build_shell(theme="light")
    page = shell.page("separation")
    try:
        page.set_audio_path(str(audio))
        page.run()
        assert page._status.state == "running"
        assert _pump(app, lambda: not page._job.running and page._status.state == "done")
        assert page._progress.value() == 100
        assert page._result_values["vocals"].full_text == "vocals.wav"
        assert page._result_values["accompaniment"].full_text == "acc.wav"
        log = page.log.toPlainText()
        assert "engine says hi" in log, "cfg.log_cb did not reach the log pane"
        assert "separation: analyze" in log
        assert calls["request"].steps == ("separation",)
        assert page._run_button.isEnabled() is True
        assert page._cancel_button.isEnabled() is False
    finally:
        page.shutdown()
        shell.deleteLater()


def test_task_page_cancel_is_reported_as_cancelled_not_failed(app, monkeypatch):
    scratch = _scratch()
    from vstudio_ui import services, strings
    from vstudio_ui.app import build_shell

    audio = scratch / "song.wav"
    audio.write_bytes(b"")
    _stub_pipeline(monkeypatch, services.RunResult())

    shell = build_shell(theme="light")
    page = shell.page("separation")
    try:
        page.set_audio_path(str(audio))
        page.run()
        # Cancellation arrives as done(None, True) — contract.md §6.
        page._on_done(None, True)
        assert page._status.state == "idle"
        assert page._stage.full_text == strings.tr("task.cancelled")
        assert page._cancel_button.isEnabled() is False
    finally:
        page.shutdown()
        shell.deleteLater()


def test_task_page_failure_is_shown_not_swallowed(app, monkeypatch):
    scratch = _scratch()
    from vstudio_ui import strings
    from vstudio_ui.app import build_shell

    audio = scratch / "song.wav"
    audio.write_bytes(b"")
    _stub_pipeline(monkeypatch, None)

    shell = build_shell(theme="light")
    page = shell.page("separation")
    try:
        page.set_audio_path(str(audio))
        page.run()
        assert _pump(app, lambda: not page._job.running)
        page._on_failed(RuntimeError("model missing for step: separation"))
        assert page._status.state == "failed"
        assert page._stage.full_text == strings.tr("task.failed")
        assert "model missing" in page.log.toPlainText()
    finally:
        page.shutdown()
        shell.deleteLater()


def test_task_page_rejects_a_bad_option(app, monkeypatch):
    """A page that sets a knob the config does not have must say so, not no-op."""
    scratch = _scratch()
    from vstudio_ui import services
    from vstudio_ui.app import build_shell

    audio = scratch / "song.wav"
    audio.write_bytes(b"")

    def explode(request):
        raise ValueError("unknown pipeline option 'nope'")

    monkeypatch.setattr(services, "build_config", explode)

    shell = build_shell(theme="light")
    page = shell.page("separation")
    try:
        page.set_audio_path(str(audio))
        page.run()
        assert page._job.running is False
        assert page._status.state == "failed"
        assert "unknown pipeline option" in page._stage.full_text
    finally:
        shell.deleteLater()


def test_asr_diarization_checkbox_changes_the_step_list(app):
    """An explicit step list wins over enable_* flags, so it has to be computed."""
    from vstudio_ui.app import build_shell

    shell = build_shell(theme="light")
    page = shell.page("asr")
    try:
        assert page.steps_for_run() == ("speech_asr",)
        page.diarize_check.setChecked(True)
        assert page.steps_for_run() == ("diarization", "speech_asr")
        request = page.build_request()
        assert request.steps == ("diarization", "speech_asr")
        assert dict(request.options)["enable_diarization"] is True
    finally:
        shell.deleteLater()


def test_pipeline_page_refuses_an_empty_step_selection(app):
    scratch = _scratch()
    from vstudio_ui import strings
    from vstudio_ui.app import build_shell

    audio = scratch / "song.wav"
    audio.write_bytes(b"")
    shell = build_shell(theme="light")
    page = shell.page("pipeline")
    try:
        page.set_audio_path(str(audio))
        for check in page.step_checks.values():
            check.setChecked(False)
        page.run()
        assert page._job.running is False
        assert page._status.state == "failed"
        assert page._stage.full_text == strings.tr("pipeline.no_steps")
    finally:
        shell.deleteLater()


def test_pipeline_page_keeps_steps_in_canonical_order(app):
    from vstudio_ui import services
    from vstudio_ui.app import build_shell

    shell = build_shell(theme="light")
    page = shell.page("pipeline")
    try:
        for check in page.step_checks.values():
            check.setChecked(True)
        assert page.steps_for_run() == services.PIPELINE_STEPS
    finally:
        shell.deleteLater()


def test_tts_page_requires_target_text(app):
    """The orchestrator *skips* TTS without tts_text, so the run would "succeed"."""
    scratch = _scratch()
    from vstudio_ui import strings
    from vstudio_ui.app import build_shell

    audio = scratch / "ref.wav"
    audio.write_bytes(b"")
    shell = build_shell(theme="light")
    page = shell.page("tts")
    try:
        page.set_audio_path(str(audio))
        page.run()
        assert page._job.running is False
        assert page._stage.full_text == strings.tr("tts.missing_text")

        page.text_edit.setPlainText("hello")
        request = page.build_request()
        options = dict(request.options)
        assert options["tts_text"] == "hello"
        assert options["ref_audio"] == str(audio), "without ref_audio the step is skipped"
        assert request.language == "zh"
    finally:
        shell.deleteLater()


def test_training_page_takes_a_directory_and_offers_only_real_devices(app):
    """Offering rocm/zluda/directml here would silently mean "train on CPU"."""
    scratch = _scratch()
    from vstudio_ui import strings
    from vstudio_ui.app import build_shell

    shell = build_shell(theme="light")
    page = shell.page("training")
    try:
        assert page.input_is_dir is True
        assert page.device_options() == ("cuda", "cpu")

        file_path = scratch / "not_a_dir.wav"
        file_path.write_bytes(b"")
        page.set_audio_path(str(file_path))
        page.run()
        assert page._job.running is False
        assert page._stage.full_text == strings.tr("task.input_not_dir").format(path=file_path)
    finally:
        shell.deleteLater()


def test_training_page_uses_its_own_job_body(app, monkeypatch):
    scratch = _scratch()
    from vstudio_ui import services
    from vstudio_ui.app import build_shell

    dataset = scratch / "data"
    dataset.mkdir()
    seen: dict = {}

    def build_spec(request):
        seen["request"] = request
        return _FakeCfg()

    def job(_ctx=None, _cfg=None, progress_cb=None):
        seen["ran"] = True
        return services.RunResult(exit_code=0, output_dir=str(scratch / "out"))

    monkeypatch.setattr(services, "build_training_spec", build_spec)
    monkeypatch.setattr(services, "run_training_job", job)
    # If the page used the pipeline path instead, this would raise.
    monkeypatch.setattr(
        services, "run_pipeline_job", lambda *a, **k: pytest.fail("training used the pipeline")
    )

    shell = build_shell(theme="light")
    page = shell.page("training")
    try:
        page.set_audio_path(str(dataset))
        page.run()
        assert _pump(app, lambda: not page._job.running and seen.get("ran"))
        assert page._status.state == "done"
        assert seen["request"].data_dir == str(dataset)
        assert seen["request"].task in services.training_tasks()
    finally:
        page.shutdown()
        shell.deleteLater()


def test_training_page_reports_a_non_zero_exit_as_amber(app, monkeypatch):
    """A failed training run must not look green."""
    scratch = _scratch()
    from vstudio_ui import services
    from vstudio_ui.app import build_shell

    dataset = scratch / "data"
    dataset.mkdir()
    monkeypatch.setattr(services, "build_training_spec", lambda request: _FakeCfg())
    monkeypatch.setattr(
        services,
        "run_training_job",
        lambda _ctx=None, _cfg=None, progress_cb=None: services.RunResult(
            steps=(services.StepOutcome("training", "failed", "exit 2"),),
            output_dir=str(scratch / "out"),
            exit_code=2,
            error="training exited with code 2",
        ),
    )

    shell = build_shell(theme="light")
    page = shell.page("training")
    try:
        page.set_audio_path(str(dataset))
        page.run()
        assert _pump(app, lambda: not page._job.running)
        assert page._status.state == "ready", "amber, not green and not red"
        assert page._result_values["exit"].full_text == "2"
        assert "exited with code 2" in page.log.toPlainText()
    finally:
        page.shutdown()
        shell.deleteLater()


def test_every_task_page_retranslates(app):
    from vstudio_ui import strings
    from vstudio_ui.app import build_shell

    shell = build_shell(theme="light")
    try:
        for language in ("en", "zh"):
            strings.set_language(language)
            for key in TASK_PAGE_KEYS:
                page = shell.page(key)
                page.retranslate()
                assert page._title.text() == strings.tr(f"{key}.title")
                assert page._run_button.text() == strings.tr("task.run")
    finally:
        strings.set_language("zh")
        shell.deleteLater()


def test_strings_used_in_the_source_all_exist():
    """Every literal ``tr("...")`` key must resolve.

    Added after a page shipped with a placeholder key spelled differently from the
    table: ``strings.tr`` falls back to the key itself, so the UI renders the raw
    key instead of raising. The key-parity test cannot catch that — a key missing
    from *both* tables is perfectly balanced.
    """
    import vstudio_ui
    from vstudio_ui import strings

    pattern = re.compile(r'tr\(\s*f?"([A-Za-z0-9_.]+)"')
    root = Path(vstudio_ui.__file__).resolve().parent
    have = strings.keys()
    missing: list[str] = []
    for path in sorted(root.rglob("*.py")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for match in pattern.finditer(line):
                key = match.group(1)
                if key not in have:
                    missing.append(f"{path.relative_to(root)}:{lineno}: {key}")
    assert not missing, "strings referenced in source but absent from the tables:\n" + "\n".join(
        missing
    )


def test_every_dynamic_string_family_is_populated():
    """The ``f"{family}.{value}"`` keys, expanded over the real value lists.

    The scanner above cannot see these, so they get their own guard: adding a device
    or a pipeline step without a label would otherwise render ``device.zluda``.
    """
    from vstudio_ui import services, strings

    have = strings.keys()
    families = {
        "device": services.DEVICE_CHOICES,
        "language": services.LANGUAGE_CHOICES,
        "slicing": services.SLICING_METHODS,
        "quantization": services.QUANTIZATION_MODES,
        "step": services.PIPELINE_STEPS,
        "training.task": services.training_tasks(),
        "audio_type": ("singing", "speech", "mixed", "unknown"),
        "status": ("idle", "ready", "running", "done", "failed"),
        "check": services.CHECK_ORDER + ("pass", "fail", "summary", "summary_failed", "cancelled"),
        "nav": ("overview", *TASK_PAGE_KEYS, "settings"),
        "theme": ("light", "dark"),
    }
    missing = [
        f"{family}.{value}"
        for family, values in families.items()
        for value in values
        if f"{family}.{value}" not in have
    ]
    assert not missing, f"labels missing for: {sorted(missing)}"


def test_result_outcome_helpers(app):
    """``did_run`` is what stops a page reporting a number nobody computed."""
    from vstudio_ui import services

    result = services.RunResult(
        steps=(
            services.StepOutcome("separation", "done"),
            services.StepOutcome("vocal2midi", "skipped", "content detected as speech"),
        ),
        notes_count=0,
    )
    assert result.did_run("separation") is True
    assert result.did_run("vocal2midi") is False
    assert result.did_run("speech_asr") is False, "absent step is not 'ran'"
    assert result.outcome("speech_asr") is None
    assert result.outcome("vocal2midi").detail == "content detected as speech"


def test_pipeline_page_omits_a_note_count_it_did_not_compute(app, monkeypatch):
    """A run with only content_detection must not claim "0 notes"."""
    from vstudio_ui import services, strings
    from vstudio_ui.app import build_shell

    scratch = _scratch()
    audio = scratch / "song.wav"
    audio.write_bytes(b"")
    _stub_pipeline(
        monkeypatch,
        services.RunResult(
            steps=(services.StepOutcome("content_detection", "done"),),
            audio_type="speech",
            notes_count=0,
        ),
    )

    shell = build_shell(theme="light")
    page = shell.page("pipeline")
    try:
        page.set_audio_path(str(audio))
        page.run()
        assert _pump(app, lambda: not page._job.running and page._status.state == "done")
        assert page._result_values["audio_type"].full_text == strings.tr("audio_type.speech")
        assert page._result_values["notes"].full_text == "—", "reported a note count it never made"
        assert page._progress.value() == 100
    finally:
        page.shutdown()
        shell.deleteLater()


def test_task_page_stage_is_localised_on_completion(app, monkeypatch):
    """The headline is localised; the orchestrator's raw lines stay in the log."""
    from vstudio_ui import services, strings
    from vstudio_ui.app import build_shell

    scratch = _scratch()
    audio = scratch / "song.wav"
    audio.write_bytes(b"")
    _stub_pipeline(
        monkeypatch,
        services.RunResult(steps=(services.StepOutcome("separation", "done"),)),
        progress=((1.0, "[1/1] separation done"),),
    )

    shell = build_shell(theme="light")
    page = shell.page("separation")
    try:
        page.set_audio_path(str(audio))
        page.run()
        assert _pump(app, lambda: not page._job.running and page._status.state == "done")
        assert page._stage.full_text == strings.tr("task.done").format(count=1)
        assert "[1/1] separation done" in page.log.toPlainText()
    finally:
        page.shutdown()
        shell.deleteLater()
