"""Acceptance tests for the launcher integration, beyond widget construction."""
import json
import tempfile
import threading
from pathlib import Path
import pytest
from vstudio_ui import services


def scratch(): return Path(tempfile.mkdtemp(prefix='launcher-test-'))


def pump(app, predicate):
    from PyQt5.QtCore import QElapsedTimer, QThread
    timer = QElapsedTimer(); timer.start()
    while not predicate() and timer.elapsed() < 5000:
        app.processEvents(); QThread.msleep(5)
    app.processEvents(); assert predicate()


def test_result_keeps_directory_and_midi_exports():
    from core.context import PipelineContext
    ctx = PipelineContext()
    ctx.results['vocal2midi'] = {'files': {'mid':'out/notes.mid', 'txt':'out/notes.txt'}, 'chord_files':{'chords':'out/chords.json'}, 'backend':'onnx-cpu'}
    result = services._result_from_context(ctx, 'out')
    assert result.output_dir == 'out'
    assert {p for _,p in result.produced_paths()} == {'out/notes.mid', 'out/notes.txt', 'out/chords.json'}
    assert result.backends == (('vocal2midi', 'onnx-cpu'),)


def test_raw_separation_stems_feed_following_steps_and_remain_exportable():
    from core.context import PipelineContext
    from core.config import PipelineConfig
    ctx = PipelineContext(audio_path='mix.wav')
    ctx.results['separation_stems'] = {'hp5_only_main_vocal':'v.wav', 'other':'acc.wav', 'drums':'drums.wav'}
    services._normalize_separation(ctx, PipelineConfig(separation_model='HP5_only_main_vocal'))
    assert ctx.active_vocals_path == 'v.wav' and ctx.accompaniment_path == 'acc.wav'
    assert {p for _,p in services._result_from_context(ctx, 'out').produced_paths()} == {'v.wav','acc.wav','drums.wav'}


def test_pipeline_uses_shared_parameters_and_checks_tts(app):
    from vstudio_ui.app import build_shell
    from vstudio_ui.parameters import TtsParameters, MidiParameters
    shell = build_shell(theme='dark', font_scale=1)
    try:
        page = shell.page('pipeline')
        sample = scratch()/'a.wav'; sample.write_bytes(b'input')
        page.set_audio_path(str(sample))
        for name, check in page.step_checks.items(): check.setChecked(name in ('vocal2midi','tts'))
        assert isinstance(page.step_editors['tts'], TtsParameters)
        assert isinstance(shell.page('midi')._parameters, MidiParameters)
        assert page.validate_request() == __import__('vstudio_ui.strings', fromlist=['tr']).tr('tts.missing_text')
        page.step_editors['tts'].text_edit.setPlainText('Hello')
        assert page.validate_request()
        page.ref_audio_edit.setText(str(sample))
        page.step_editors['midi' if 'midi' in page.step_editors else 'vocal2midi'].tempo_spin.setValue(97)
        cfg = services.build_config(page.build_request())
        assert not page.validate_request()
        assert cfg.tts_text == 'Hello' and cfg.ref_audio == str(sample) and cfg.tempo == 97
    finally: shell.close(); shell.deleteLater()


def test_task_coordinator_holds_lease_until_thread_finishes(app):
    from vstudio_ui.coordinator import TaskCoordinator
    from vstudio_ui.jobs import JobHandle
    gate = threading.Event(); first = JobHandle(); other = JobHandle(); task = TaskCoordinator()
    def work(*args):
        gate.wait(3)
        return services.RunResult()
    assert task.acquire('asr', first)
    assert first.start(work)
    assert not task.acquire('midi', other)
    try:
        first.cancel()
        assert task.busy  # cancel is cooperative, never frees memory immediately
    finally: gate.set()
    pump(app, lambda:not task.busy)
    assert task.acquire('midi', other)
    task.release()


def test_close_waits_asynchronously_for_worker(app):
    from vstudio_ui.app import build_shell
    shell = build_shell(theme='dark', font_scale=1); page = shell.page('asr')
    gate = threading.Event()
    def work(*args): gate.wait(3); return services.RunResult()
    shell.show(); page._job.start(work)
    shell.close()
    try:
        assert shell._closing and shell.isVisible() and page._job.cancelled
    finally: gate.set()
    pump(app, lambda:not shell.isVisible())
    shell.deleteLater()


def test_long_transcript_and_artifact_paths_are_available(app):
    from vstudio_ui.pages import AsrPage
    page = AsrPage(); text = 'A complete transcript. '*500
    result = services.RunResult(asr_text=text, output_dir='out', files=(('txt','out/transcript.txt'),))
    page._on_done(result, False)
    assert page.transcript.toPlainText() == text
    assert page._selected_path() == 'out/transcript.txt'
    assert page._last_result.output_dir == 'out'
    page.deleteLater()


def test_extension_contract_versions_backend_tasks_and_paths():
    root = scratch(); ext = root/'extensions'/'training'; ext.mkdir(parents=True)
    assert services.training_extension_status(root=root).state == 'extension_missing'
    (ext/'python.exe').write_bytes(b'fake'); (ext/'run.py').write_text('')
    manifest = dict(schema_version=1, api_version='1', version='0.1', backends=['cuda'], tasks=['gsv_tts'], python='python.exe', entrypoint='run.py')
    file = ext/'manifest.json'
    def write(): file.write_text(json.dumps(manifest), encoding='utf-8')
    write()
    assert services.training_extension_status('gsv_tts','cuda',root).ready
    assert services.training_extension_status('gsv_tts','rocm',root).state == 'device_unsupported'
    assert services.training_extension_status('other','cuda',root).state == 'task_unsupported'
    manifest['api_version'] = '2'; write()
    assert services.training_extension_status(root=root).state == 'version_mismatch'
    manifest['api_version'] = '1'; manifest['python'] = '../../escape.exe'; write()
    assert services.training_extension_status(root=root).state == 'invalid_manifest'


def test_capabilities_distinguish_dependencies_from_models(monkeypatch):
    from core.config import PipelineConfig, ModelPaths
    monkeypatch.setattr(services.importlib.util, 'find_spec', lambda name: None if name == 'funasr' else object())
    statuses = {s.step:s for s in services.capabilities(PipelineConfig(model_paths=ModelPaths()))}
    assert statuses['speech_asr'].state == 'dependency_missing'
    assert statuses['separation'].state == 'model_missing'
    assert statuses['content_detection'].state == 'ready'


def test_preflight_rejects_invalid_io_and_missing_tts_before_execution(monkeypatch):
    from core.config import PipelineConfig
    root = scratch(); sample = root/'a.wav'; sample.write_bytes(b'input')
    cfg = PipelineConfig(audio_path=str(root), output_dir=str(root/'out'), device='cpu')
    with pytest.raises(ValueError, match='audio file'): services.preflight(cfg)
    cfg.audio_path = str(sample); cfg.pipeline_steps = ['tts']
    monkeypatch.setattr(services, 'capabilities', lambda cfg: ())
    with pytest.raises(ValueError, match='target text'): services.preflight(cfg)
    cfg.tts_text = 'Hello'
    with pytest.raises(ValueError, match='reference audio'): services.preflight(cfg)


def test_selecting_audio_results_does_not_start_native_audio_backend(app, monkeypatch):
    from PyQt5 import QtMultimedia
    from vstudio_ui.widgets.audio_preview import AudioPreview
    def unexpected_backend(*args, **kwargs):
        raise AssertionError('Displaying an input/result must not open audio hardware')
    monkeypatch.setattr(QtMultimedia, 'QMediaPlayer', unexpected_backend)
    preview=AudioPreview()
    preview.set_path('first.wav'); preview.set_path('second.wav')
    assert preview.play.isEnabled() and not preview.stop.isEnabled()
    preview.deleteLater()
