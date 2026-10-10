"""Capture real Qt transitions and compose a before/after presentation sheet."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    # Preserve the same native-library loading order as the desktop entry point.
    import torch  # noqa: F401
    from PyQt5.QtCore import Qt, QRectF, QTimer
    from PyQt5.QtGui import QImage, QPainter, QColor, QFont, QPen
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication
    from PIL import Image
    from vstudio_ui.app import build_shell
    from vstudio_ui import strings

    app = QApplication.instance() or QApplication([])
    strings.set_language('zh')
    shell = build_shell(theme='dark', font_scale=1.0)
    shell.resize(1440, 900)
    shell.show()
    QTest.qWait(500)
    out = ROOT / 'ui-design' / 'launcher-20261010'
    out.mkdir(parents=True, exist_ok=True)
    frames, times = [], []
    start = time.perf_counter()

    def capture():
        times.append(time.perf_counter() - start)
        image = shell.grab().toImage().scaled(960, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        image = image.convertToFormat(QImage.Format_RGBA8888)
        data = image.bits(); data.setsize(image.byteCount())
        frames.append(Image.frombytes('RGBA', (image.width(), image.height()), bytes(data)).convert('RGB'))

    timer = QTimer(); timer.setInterval(32); timer.timeout.connect(capture); timer.start()
    QTimer.singleShot(180, lambda: QTest.mouseClick(shell._nav._items[1], Qt.LeftButton))
    QTimer.singleShot(720, lambda: shell.page('separation').audio_edit.setFocus())
    QTimer.singleShot(1150, lambda: QTest.mouseClick(shell._nav._items[5], Qt.LeftButton))
    QTimer.singleShot(1700, lambda: QTest.mouseClick(shell.page('pipeline')._step_toggles['vocal2midi'], Qt.LeftButton))
    QTimer.singleShot(2200, lambda: QTest.mouseClick(shell.page('pipeline')._step_toggles['vocal2midi'], Qt.LeftButton))
    QTimer.singleShot(2700, lambda: QTest.mouseClick(shell._nav._items[0], Qt.LeftButton))
    QTimer.singleShot(3300, app.quit)
    app.exec_()
    timer.stop()
    durations = [max(20, round((b-a)*1000)) for a, b in zip(times, times[1:])] + [100]
    frames[0].save(out/'motion-preview.gif', save_all=True, append_images=frames[1:], duration=durations, loop=0)
    gaps = [b-a for a, b in zip(times, times[1:])]
    (out/'motion-capture.json').write_text(json.dumps({'frames':len(frames), 'seconds':times[-1], 'largest_capture_interval_ms':round(max(gaps)*1000, 1), 'page_after':shell.current_index(), 'duration_ms':320, 'reduced_motion_env':'VOCALSTUDIO_REDUCED_MOTION=1'}, indent=2), encoding='utf-8')

    sheet = QImage(1920, 900, QImage.Format_RGB32)
    sheet.fill(QColor('#0a101c'))
    painter = QPainter(sheet)
    painter.setRenderHint(QPainter.SmoothPixmapTransform)
    for x, title, detail, path in [
        (32, '改前', '原有界面 · 总览页', ROOT/'temp/connection-audit-screens/dark_0_overview.png'),
        (984, '改后', '启动器布局 · 声学视觉 · 交互动效', out/'dark_0_overview.png'),
    ]:
        painter.setPen(QColor('#d6fc64' if x > 32 else '#f0f4fa'))
        painter.setFont(QFont('Microsoft YaHei UI', 24, QFont.Bold))
        painter.drawText(x, 52, title)
        painter.setPen(QColor('#93a3bc')); painter.setFont(QFont('Microsoft YaHei UI', 11))
        painter.drawText(x, 82, detail)
        image = QImage(str(path))
        if image.isNull(): raise FileNotFoundError(path)
        bounds = QRectF(x, 106, 904, 742)
        painter.setPen(QPen(QColor('#263349'), 1)); painter.setBrush(QColor('#101827'))
        painter.drawRoundedRect(bounds, 12, 12)
        scaled = image.scaled(888, 726, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        painter.drawImage(QRectF(bounds.center().x()-scaled.width()/2, bounds.center().y()-scaled.height()/2, scaled.width(), scaled.height()), scaled)
    painter.end()
    sheet.save(str(out/'before-after.png'))
    shell.close()
    print('MOTION_PREVIEW_OK', len(frames), flush=True)


if __name__ == '__main__':
    main()
