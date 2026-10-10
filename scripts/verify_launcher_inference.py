"""Real UI acceptance: configured pipeline jobs, route persistence and artifacts."""
import argparse
import json
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch  # native-library load order: torch before Qt
from vstudio_ui.app import build_shell
from PyQt5.QtWidgets import QApplication


def verify(out):
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    shell = build_shell(theme='dark', font_scale=1); shell.show()
    page = shell.page('pipeline'); samples = ROOT/'temp/rocm10-inference/samples'
    report = []
    for kind, steps in [('singing', ('separation','content_detection','vocal2midi','speech_asr')), ('speech', ('diarization','content_detection','speech_asr','tts'))]:
        page.set_audio_path(str(samples/(kind+'.wav'))); page.set_output_dir(str(out/kind))
        page.device_combo.setCurrentIndex(page.device_combo.findData('rocm'))
        for key, check in page.step_checks.items(): check.setChecked(key in steps)
        asr = page.step_editors['speech_asr']; asr.lang_combo.setCurrentIndex(asr.lang_combo.findData('ja' if kind == 'singing' else 'en'))
        asr.srt_check.setChecked(True); asr.json_check.setChecked(True)
        if kind == 'speech':
            page.speakers_spin.setValue(1); page.ref_audio_edit.setText(str(samples/'reference.wav'))
            tts = page.step_editors['tts']; tts.lang_combo.setCurrentIndex(tts.lang_combo.findData('en'))
            tts.text_edit.setPlainText('Hello from Vocal Studio. Every voice has more possibilities.')
        terminals = []; progress = []
        def done(result, cancelled): terminals.append(('done', result, cancelled))
        def failed(error): terminals.append(('failed', str(error)))
        def changed(value, stage): progress.append(value)
        page._job.done.connect(done); page._job.failed.connect(failed); page._job.progress.connect(changed)
        shell.navigate(shell.page_keys().index('pipeline')); page.run()
        assert shell.tasks.busy, page._stage.text()
        shell.navigate(shell.page_keys().index('overview'))
        assert shell._header._global_cancel.isEnabled()
        start = time.monotonic()
        while shell.tasks.busy and time.monotonic()-start < 600:
            app.processEvents(); time.sleep(.01)
        app.processEvents()
        assert not shell.tasks.busy, 'UI job timed out'
        assert len(terminals) == 1 and terminals[0][0] == 'done' and not terminals[0][2], str(terminals)
        result = terminals[0][1]
        assert result.output_dir == str(out/kind) and result.ok and max(progress) == 1
        assert result.produced_paths() and all(Path(path).is_file() for _,path in result.produced_paths())
        if kind == 'singing': assert result.vocals_path and result.notes_count and any('DmlExecutionProvider' in backend for step,backend in result.backends if step == 'vocal2midi')
        else: assert result.asr_text and result.tts_audio_path and page.transcript.toPlainText() == result.asr_text
        shell.navigate(shell.page_keys().index('pipeline')); app.processEvents()
        assert page.audio_edit.text() == str(samples/(kind+'.wav'))
        from PyQt5.QtTest import QTest
        QTest.qWait(360)  # Capture the settled result, after the route transition.
        shell.grab().save(str(out/(kind+'-result.png')))
        item = dict(case=kind, seconds=round(time.monotonic()-start,2), terminal_count=len(terminals), progress=max(progress), output_dir=result.output_dir, notes=result.notes_count, text=result.asr_text, backends=result.backends, files=result.produced_paths())
        report.append(item); print(json.dumps(item, ensure_ascii=False), flush=True)
        page._job.done.disconnect(done); page._job.failed.disconnect(failed); page._job.progress.disconnect(changed)
    shell.close(); app.processEvents()
    (out/'ui-acceptance.json').write_text(json.dumps(dict(status='PASSED', cases=report),ensure_ascii=False,indent=2),encoding='utf-8')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--out',required=True)
    verify(parser.parse_args().out)
