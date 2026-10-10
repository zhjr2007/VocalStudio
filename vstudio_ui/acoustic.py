"""Original static acoustic artwork. No external graphics or fonts required."""
import math
from PyQt5.QtCore import QPointF
from PyQt5.QtGui import QPainter, QPen, QPainterPath, QLinearGradient, QColor, QPixmap
from PyQt5.QtWidgets import QWidget
from vstudio_ui.theme import color, current_theme


class AcousticCanvas(QWidget):
    def paintEvent(self, event):
        # Translucent cards and motion overlays repaint their ancestors. Cache
        # the static artwork at native DPI instead of rebuilding 9,774 points
        # on every animation frame. Theme, resize and DPI changes invalidate it.
        dpr = self.devicePixelRatioF()
        key = (self.width(), self.height(), dpr, current_theme())
        if getattr(self, '_art_key', None) != key:
            self._art = QPixmap(round(self.width()*dpr), round(self.height()*dpr))
            self._art.setDevicePixelRatio(dpr)
            painter = QPainter(self._art)
            self._paint_art(painter)
            painter.end()
            self._art_key = key
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self._art)

    def _paint_art(self, p):
        p.setRenderHint(QPainter.Antialiasing); p.fillRect(self.rect(), color('background'))
        w, h = self.width(), self.height(); p.translate(w*.64, h*.37)
        s = min(w/1000, h/650); p.scale(s, s); dark = current_theme() == 'dark'
        glow = QLinearGradient(-420, -250, 420, 270)
        glow.setColorAt(0, QColor('#16466B' if dark else '#C6D9E5'))
        glow.setColorAt(1, QColor('#132B39' if dark else '#E3EBD4'))
        p.setPen(QPen(glow, 1.2))
        for j in range(54):
            path = QPainterPath()
            for i in range(181):
                x = -540 + i*6
                y = (j-27)*8 + math.sin(x/83 + j*.15)*math.exp(-(x/275)**2)*(75+j*1.8)
                if i == 0: path.moveTo(x, y)
                else: path.lineTo(x, y)
            p.drawPath(path)
        accent = color('primary'); accent.setAlpha(100 if dark else 120); p.setPen(QPen(accent, 2))
        for i in range(63):
            x = (i-31)*10; height = 8 + 100*math.exp(-(x/120)**2)*abs(math.sin(i*.71))
            p.drawLine(QPointF(x, 120-height), QPointF(x, 120+height))
    def refresh_theme(self): self.update()

