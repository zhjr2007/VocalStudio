"""Real AMD UI failure, cooperative cancellation and asynchronous-close checks."""
import json
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from vstudio_ui.app import build_shell
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication


def main():
    app = QApplication.instance() or QApplication([])
    shell = build_shell(theme='dark',font_scale=1); shell.show()
    page = shell.page('midi'); sample = ROOT/'temp/rocm10-inference/samples/singing.wav'
    page.set_audio_path(str(sample)); report = []
    for name, device in [('incompatible_gpu_failure','cuda'),('cancel_real_midi','rocm'),('close_real_midi','rocm')]:
        page.set_output_dir(str(ROOT/'temp/launcher-lifecycle'/name))
        page.device_combo.setCurrentIndex(page.device_combo.findData(device))
        terminals = []
        def done(result, cancelled): terminals.append(('done',cancelled))
        def failed(exc): terminals.append(('failed',str(exc)))
        page._job.done.connect(done); page._job.failed.connect(failed)
        page.run(); assert shell.tasks.busy
        if name == 'cancel_real_midi': QTimer.singleShot(200,page._job.cancel)
        if name == 'close_real_midi': QTimer.singleShot(200,shell.close)
        start=time.monotonic()
        while (shell.tasks.busy or (name == 'close_real_midi' and shell.isVisible())) and time.monotonic()-start < 120:
            app.processEvents(); time.sleep(.01)
        app.processEvents()
        assert not shell.tasks.busy and len(terminals)==1, str(terminals)
        if name == 'incompatible_gpu_failure': assert terminals[0][0]=='failed' and 'ROCm' in terminals[0][1]
        else: assert terminals[0]==('done',True), str(terminals)
        if name == 'close_real_midi': assert not shell.isVisible()
        report.append(dict(case=name,terminal=terminals[0],seconds=round(time.monotonic()-start,2)))
        page._job.done.disconnect(done); page._job.failed.disconnect(failed)
    out=ROOT/'temp/launcher-lifecycle'; out.mkdir(parents=True,exist_ok=True)
    (out/'acceptance.json').write_text(json.dumps(dict(status='PASSED',cases=report),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))


if __name__ == '__main__': main()
