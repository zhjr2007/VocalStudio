"""Frameless launcher shell with native Windows resizing, persistent pages and asynchronous shutdown."""

from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QRectF, QEvent
from PyQt5.QtGui import QFontMetrics, QPainter, QPen, QPainterPath, QRegion, QCursor
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from vstudio_ui import services, strings
from vstudio_ui.acoustic import AcousticCanvas
from vstudio_ui.coordinator import TaskCoordinator
from vstudio_ui.jobs import JobHandle
from vstudio_ui.motion import PageMotion, InteractionMotion
from vstudio_ui.theme import (
    apply_font_scale,
    app_font,
    apply_theme,
    color,
    color_hex,
    current_font_scale,
    current_theme,
    set_qss_root,
)
from vstudio_ui.tokens import (
    BORDER_WIDTH,
    HEADER_HEIGHT,
    SPACE,
    TYPE_SCALE,
)
from vstudio_ui.widgets import NavRail, refresh_theme_tree


class WindowButton(QPushButton):
    def __init__(self, action, parent=None):
        super().__init__(parent)
        self.action = action
        self.setFixedSize(34, 34)
        self.setCursor(Qt.PointingHandCursor)

    def paintEvent(self, event):
        painter = QPainter(self); painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(color('border'), 1))
        painter.setBrush(color('card_alt') if self.underMouse() else color('card'))
        painter.drawEllipse(QRectF(1, 1, 32, 32))
        painter.setPen(QPen(color('foreground'), 1.4))
        if self.action == 'minimize': painter.drawLine(11, 17, 23, 17)
        elif self.action == 'maximize': painter.drawRect(12, 12, 10, 10)
        else:
            painter.drawLine(12, 12, 22, 22); painter.drawLine(22, 12, 12, 22)


class HeaderBar(QWidget):
    """App name plus the current page name on the left, actions on the right.

    The page's own heading lives in its ``PageScaffold``; the header only carries
    *context* (muted, small), so the two never repeat each other at display size.
    """

    theme_toggled = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(HEADER_HEIGHT)

        self._app_label = QLabel(strings.APP_NAME, self)
        self._page_label = QLabel("", self)
        self._separator = QLabel("/", self)
        self._theme_button = QPushButton("", self)
        self._theme_button.setCursor(Qt.PointingHandCursor)
        self._theme_button.clicked.connect(self.theme_toggled.emit)

        row = QHBoxLayout(self)
        row.setContentsMargins(SPACE["lg"], 0, SPACE["lg"], 0)
        row.setSpacing(SPACE["sm"])
        row.addWidget(self._app_label, 0)
        row.addWidget(self._separator, 0)
        row.addWidget(self._page_label, 1)
        row.addWidget(self._theme_button, 0)
        self._task_label = QPushButton('', self)
        self._task_label.setMaximumWidth(360)
        self._task_label.clicked.connect(self._show_task)
        row.insertWidget(3, self._task_label)
        self._global_cancel = QPushButton('×', self)
        self._global_cancel.clicked.connect(lambda: self.window().tasks.cancel())
        row.insertWidget(4, self._global_cancel)
        self._window_buttons = []
        for name, action in [('minimize', lambda: self.window().showMinimized()),
                             ('maximize', self._maximize), ('close', lambda: self.window().close())]:
            button = WindowButton(name, self)
            button.clicked.connect(action)
            row.addWidget(button)
            self._window_buttons.append(button)

        self.refresh_theme()

    def _maximize(self):
        window = self.window()
        window.showNormal() if window.isMaximized() else window.showMaximized()

    def _show_task(self):
        window = self.window()
        if window.tasks.owner in window.page_keys():
            window.navigate(window.page_keys().index(window.tasks.owner))

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton: self._maximize()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.window().windowHandle():
            self.window().windowHandle().startSystemMove()

    def update_task(self):
        task = self.window().tasks
        self._task_label.setVisible(task.busy)
        self._global_cancel.setVisible(task.busy)
        self._global_cancel.setToolTip(strings.tr('task.cancel'))
        self._task_label.setText(f"{strings.NAV_LABELS.get(task.owner, task.owner)} · {task.fraction:.0%}  {task.stage}")

    def set_page_title(self, text: str) -> None:
        self._page_label.setText(text)

    def refresh_theme(self) -> None:
        self._app_label.setStyleSheet(
            f"color: {color_hex('foreground')};"
            f"font-size: {TYPE_SCALE['h3']}pt; font-weight: 700;"
        )
        self._separator.setStyleSheet(
            f"color: {color_hex('border')}; font-size: {TYPE_SCALE['h3']}pt;"
        )
        self._page_label.setStyleSheet(
            f"color: {color_hex('muted_foreground')}; font-size: {TYPE_SCALE['body']}pt;"
        )
        self._theme_button.setText(strings.THEME_LABELS[current_theme()])
        for button in self._window_buttons:
            label = strings.tr(f'window.{button.action}')
            button.setAccessibleName(label); button.setToolTip(label); button.update()
        # HEADER_HEIGHT is a floor, not a fixed value: the app name is set in h3,
        # which is 32pt at a 2x scale and would otherwise be clipped by the bar.
        metrics = QFontMetrics(app_font(TYPE_SCALE["h3"]))
        self.setFixedHeight(max(HEADER_HEIGHT, metrics.height() + SPACE["md"]))
        self.update()

    def paintEvent(self, event):  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.end()


class AppShell(QMainWindow):
    """The application window. Add pages with :meth:`add_page`."""

    #: Emitted with a short description when a preference could not be written.
    #: The settings page listens and shows it; nothing else has to care.
    preference_failed = pyqtSignal(str)
    capabilities_changed = pyqtSignal()

    def __init__(self, parent=None, *, theme: str | None = None, font_scale: float | None = None):
        super().__init__(parent)
        self.setWindowTitle(strings.APP_NAME)
        self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)
        self.setMinimumSize(1080, 720)
        self.tasks = TaskCoordinator(self)
        self.capability_snapshot = ()
        self._closing = False
        self._close_timer = QTimer(self)
        self._close_timer.setInterval(100)
        self._close_timer.timeout.connect(self._finish_close)

        set_qss_root(self)
        apply_theme(theme or current_theme())
        # Applied here rather than by the caller so that *every* shell — including
        # the ones the tests build — starts from a known scale instead of whatever
        # a previous shell left in the module-level token table.
        apply_font_scale(font_scale if font_scale is not None else current_font_scale())

        self._nav = NavRail(self)
        self._header = HeaderBar(self)
        self._stack = QStackedWidget(self)
        self._page_motion = PageMotion(self._stack)
        self._interaction_motion = InteractionMotion(self)
        self._page_keys: list[str] = []

        self._nav.navigate.connect(self._on_navigate)
        self._header.theme_toggled.connect(self.toggle_theme)

        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(0)
        right.addWidget(self._header)
        right.addWidget(self._stack, 1)

        central = AcousticCanvas(self)
        row = QHBoxLayout(central)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        row.addWidget(self._nav)
        row.addLayout(right, 1)
        self.setCentralWidget(central)

        self.tasks.changed.connect(self._header.update_task)
        self._header.update_task()
        screen = self.screen().availableGeometry()
        self.resize(min(1440, screen.width()), min(900, screen.height()))

    def showEvent(self, event):
        super().showEvent(event)
        import sys
        if sys.platform == 'win32' and not getattr(self, '_native_frame', False):
            import ctypes
            from ctypes import wintypes
            user = ctypes.windll.user32
            user.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
            user.SetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_long]
            user.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint]
            hwnd = int(self.winId())
            self._native_frame = True
            # Windows snap and system resizing require WS_THICKFRAME. Keep the
            # native non-client frame hidden through WM_NCCALCSIZE below.
            user.SetWindowLongW(hwnd, -16, user.GetWindowLongW(hwnd, -16) | 0x40000)
            user.SetWindowPos(hwnd, None, 0, 0, 0, 0, 0x37)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_window_mask()

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.WindowStateChange: self._update_window_mask()

    def _update_window_mask(self):
        if self.isMaximized(): self.clearMask()
        else:
            path = QPainterPath(); path.addRoundedRect(QRectF(self.rect()), 16, 16)
            self.setMask(QRegion(path.toFillPolygon().toPolygon()))

    def nativeEvent(self, event_type, message):
        # Native hit testing gives Windows resize cursors, Aero snap and DPI-aware
        # edge sizing, rather than maintaining a second geometry implementation.
        import sys
        if sys.platform == 'win32':
            import ctypes
            from ctypes import wintypes
            msg = wintypes.MSG.from_address(int(message))
            if msg.message == 0x83: return True, 0  # WM_NCCALCSIZE: custom chrome
            if msg.message == 0x24 and getattr(self, '_native_frame', False):
                class MinMaxInfo(ctypes.Structure):
                    _fields_ = [(name, wintypes.POINT) for name in ('reserved','max_size','max_position','min_track','max_track')]
                class MonitorInfo(ctypes.Structure):
                    _fields_ = [('size', wintypes.DWORD), ('monitor', wintypes.RECT), ('work', wintypes.RECT), ('flags', wintypes.DWORD)]
                user = ctypes.windll.user32
                user.MonitorFromWindow.restype = wintypes.HANDLE
                user.MonitorFromWindow.argtypes = [wintypes.HWND, wintypes.DWORD]
                user.GetMonitorInfoW.argtypes = [wintypes.HANDLE, ctypes.POINTER(MonitorInfo)]
                info = MonitorInfo(); info.size = ctypes.sizeof(info)
                if user.GetMonitorInfoW(user.MonitorFromWindow(int(self.winId()), 2), ctypes.byref(info)):
                    limits = MinMaxInfo.from_address(msg.lParam)
                    limits.max_position.x, limits.max_position.y = info.work.left-info.monitor.left, info.work.top-info.monitor.top
                    limits.max_size.x, limits.max_size.y = info.work.right-info.work.left, info.work.bottom-info.work.top
                    limits.min_track.x = round(self.minimumWidth()*self.devicePixelRatioF())
                    limits.min_track.y = round(self.minimumHeight()*self.devicePixelRatioF())
                    return True, 0
            if msg.message == 0x84 and not self.isMaximized():  # WM_NCHITTEST
                rect = wintypes.RECT()
                ctypes.windll.user32.GetWindowRect(int(self.winId()), ctypes.byref(rect))
                point = wintypes.POINT(ctypes.c_short(msg.lParam & 0xffff).value, ctypes.c_short((msg.lParam >> 16) & 0xffff).value)
                edge = round(8*self.devicePixelRatioF())
                left, right = point.x < rect.left+edge, point.x >= rect.right-edge
                top, bottom = point.y < rect.top+edge, point.y >= rect.bottom-edge
                hit = 13 if top and left else 14 if top and right else 16 if bottom and left else 17 if bottom and right else 10 if left else 11 if right else 12 if top else 15 if bottom else 0
                if hit: return True, hit
        return super().nativeEvent(event_type, message)

    def closeEvent(self, event):
        handles = self.findChildren(JobHandle)
        running = [handle for handle in handles if handle.running]
        if running:
            event.ignore(); self._closing = True
            for handle in running: handle.cancel()
            self._header._page_label.setText(strings.tr('task.closing'))
            self.setEnabled(False); self._close_timer.start()
        else:
            self._close_timer.stop(); event.accept()

    def _finish_close(self):
        if not any(handle.running for handle in self.findChildren(JobHandle)):
            self._close_timer.stop(); self.close()

    # -- pages ------------------------------------------------------------

    def add_page(self, key: str, label: str, widget: QWidget) -> int:
        """Append a page. Returns its index (also its nav position)."""
        index = self._stack.addWidget(widget)
        self._nav.add_item(label)
        self._page_keys.append(key)
        if index == 0:
            self._header.set_page_title(label)
        return index

    def page_keys(self) -> list[str]:
        return list(self._page_keys)

    def page(self, key: str) -> QWidget | None:
        if key not in self._page_keys:
            return None
        return self._stack.widget(self._page_keys.index(key))

    def current_index(self) -> int:
        return self._stack.currentIndex()

    def navigate(self, index: int) -> None:
        self._nav.set_current(index)

    def _on_navigate(self, index: int) -> None:
        self._page_motion.switch_to(index)
        self._header.set_page_title(self._nav._items[index].label)
        self.refresh_theme()

    # -- theme ------------------------------------------------------------

    def toggle_theme(self) -> None:
        self.set_theme("dark" if current_theme() == "light" else "light")

    def set_theme(self, name: str) -> None:
        apply_theme(name)
        self.refresh_theme()
        self.persist_preference(theme=current_theme())

    def refresh_theme(self) -> None:
        # include_root=False: this method *is* the tree walk, so including the
        # shell would recurse forever (and die as a native stack overflow, not a
        # RecursionError — see refresh_theme_tree's docstring).
        refresh_theme_tree(self, include_root=False)

    # -- font scale -------------------------------------------------------

    def set_font_scale(self, scale: float) -> float:
        """Rescale the whole UI. Returns the scale actually applied.

        ``refresh_theme`` is the whole of the "relayout" step: every control in the
        kit reads ``TYPE_SCALE`` inside its own ``refresh_theme``, and the three
        pieces of chrome with fixed geometry (header, nav rail, run log) recompute
        their size there too.
        """
        applied = apply_font_scale(scale)
        self.refresh_theme()
        self.persist_preference(font_scale=applied)
        return applied

    # -- i18n -------------------------------------------------------------

    def set_language(self, language: str) -> None:
        """Switch language, re-label every page and remember the choice."""
        strings.set_language(language)
        self.retranslate()
        self.persist_preference(language=strings.current_language())

    def retranslate(self) -> None:
        self.setWindowTitle(strings.APP_NAME)
        for index, key in enumerate(self._page_keys):
            label = strings.NAV_LABELS.get(key, key)
            self._nav.set_label(index, label)
            page = self._stack.widget(index)
            hook = getattr(page, "retranslate", None)
            if callable(hook):
                hook()
        self._header.set_page_title(strings.NAV_LABELS.get(self._page_keys[self.current_index()], ""))
        self._header.refresh_theme()

    # -- persistence ------------------------------------------------------

    def persist_preference(self, **changes) -> None:
        """Write preference changes, reporting a failure instead of raising.

        Public because a preference that is stored but never *applied* — the
        device is the one — still has to go through the single writer. A
        read-only ``settings/`` directory must not take the window down, but it
        must not be silent either: the failure is emitted as
        :attr:`preference_failed`.
        """
        try:
            services.save_preferences(**changes)
        except Exception as exc:  # noqa: BLE001 - an unwritable settings file is survivable
            self.preference_failed.emit(services.short_error(exc))
