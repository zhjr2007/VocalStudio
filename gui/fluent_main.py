"""VocalStudio main window (qfluentwidgets / PyQt-Fluent-Widgets)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch  # noqa: F401 (load native libs before Qt on Windows)
except Exception:  # keep GUI bootable when the torch install is broken
    pass

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFrame, QScrollArea, QWidget

try:
    from qfluentwidgets import FluentIcon, FluentWindow, Theme, setTheme as apply_fluent_theme
except ModuleNotFoundError:  # renamed distribution
    from qfluent_widgets import FluentIcon, FluentWindow, Theme, setTheme as apply_fluent_theme

NAV_TITLE_KEYS = ("overview", "separation", "midi", "asr", "tts", "pipeline", "training", "settings")

from gui.i18n import tr
from gui.theme import apply_font_scale, set_app_theme, set_qss_root
from gui.pages.asr_page import AsrPage
from gui.pages.overview_page import OverviewPage
from gui.pages.pipeline_page import PipelinePage
from gui.pages.separation_page import SeparationPage
from gui.pages.settings_page import SettingsPage
from gui.pages.training_page import TrainingPage
from gui.pages.tts_page import TtsPage
from gui.pages.v2midi_page import V2MidiPage


def _icon(name: str):
    return getattr(FluentIcon, name)


def _make_scroll(page: QWidget) -> QScrollArea:
    """Wrap a page in a scroll area so it can shrink below its content's
    minimum size hint. This keeps the main window freely resizable on all four
    edges: at small window sizes the page scrolls instead of forcing the window
    to a large minimum (which would clip/overlap content)."""
    scroll = QScrollArea()
    scroll.setObjectName(page.objectName())
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    scroll.setWidget(page)
    return scroll


class MainWindow(FluentWindow):
    def __init__(self):
        super().__init__()
        # Register this window as the rhine stylesheet root *before* the saved
        # theme is applied, so the first set_app_theme() call has somewhere to
        # put the sheet. See gui/theme.set_qss_root for why it is not the
        # application stylesheet.
        set_qss_root(self)
        self._apply_saved_settings()
        self.resize(1200, 900)
        self.setWindowTitle(tr("app"))

        self.overview_page = OverviewPage(self)
        self.separation_page = SeparationPage(self)
        self.midi_page = V2MidiPage(self)
        self.asr_page = AsrPage(self)
        self.tts_page = TtsPage(self)
        self.pipeline_page = PipelinePage(self)
        self.training_page = TrainingPage(self)
        self.settings_page = SettingsPage(self)

        self.overview_page.setObjectName("overview_page")
        self.separation_page.setObjectName("separation_page")
        self.midi_page.setObjectName("midi_page")
        self.asr_page.setObjectName("asr_page")
        self.tts_page.setObjectName("tts_page")
        self.pipeline_page.setObjectName("pipeline_page")
        self.training_page.setObjectName("training_page")
        self.settings_page.setObjectName("settings_page")

        self.overview_scroll = _make_scroll(self.overview_page)
        self.separation_scroll = _make_scroll(self.separation_page)
        self.midi_scroll = _make_scroll(self.midi_page)
        self.asr_scroll = _make_scroll(self.asr_page)
        self.tts_scroll = _make_scroll(self.tts_page)
        self.pipeline_scroll = _make_scroll(self.pipeline_page)
        self.training_scroll = _make_scroll(self.training_page)
        self.settings_scroll = _make_scroll(self.settings_page)

        self.addSubInterface(self.overview_scroll, _icon("HOME"), tr("overview"))
        self.addSubInterface(self.separation_scroll, _icon("MUSIC"), tr("separation"))
        self.addSubInterface(self.midi_scroll, _icon("EDIT"), tr("midi"))
        self.addSubInterface(self.asr_scroll, _icon("DOCUMENT"), tr("asr"))
        self.addSubInterface(self.tts_scroll, _icon("MICROPHONE"), tr("tts"))
        self.addSubInterface(self.pipeline_scroll, _icon("PLAY"), tr("pipeline"))
        self.addSubInterface(self.training_scroll, _icon("TRAIN"), tr("training"))
        self.addSubInterface(self.settings_scroll, _icon("SETTING"), tr("settings"))

        self._apply_saved_font_scale()
        self._enable_switch_fade()

        language_changed = getattr(self.settings_page, "language_changed", None)
        if hasattr(language_changed, "connect"):
            self.settings_page.language_changed.connect(self.switch_language)

    def _apply_saved_settings(self) -> None:
        from core.app_settings import get_settings
        from gui.i18n import i18n

        try:
            settings = get_settings()
        except Exception:
            return
        if settings.language in ("zh", "en"):
            i18n.set_language(settings.language)
        set_app_theme(settings.theme)
# palette + fluent theme handled by set_app_theme above

    def _apply_saved_font_scale(self) -> None:
        from core.app_settings import get_settings

        try:
            settings = get_settings()
        except Exception:
            return
        apply_font_scale(settings.font_scale)

    def _enable_switch_fade(self) -> None:
        from PyQt5.QtCore import QEasingCurve, QPropertyAnimation
        from PyQt5.QtWidgets import QGraphicsOpacityEffect

        self._fade_anims = {}
        for i in range(self.stackedWidget.count()):
            widget = self.stackedWidget.widget(i)
            effect = QGraphicsOpacityEffect(widget)
            effect.setOpacity(1.0)
            widget.setGraphicsEffect(effect)
            anim = QPropertyAnimation(effect, b"opacity")
            anim.setDuration(240)
            anim.setStartValue(0.0)
            anim.setEndValue(1.0)
            anim.setEasingCurve(QEasingCurve.OutCubic)
            self._fade_anims[i] = (effect, anim)

        def on_switch(index: int) -> None:
            effect, anim = self._fade_anims[index]
            if anim.state() == 2:  # Running
                return
            effect.setOpacity(0.0)
            anim.start()

        self.stackedWidget.currentChanged.connect(on_switch)

    def switch_language(self, language: str) -> None:
        from gui.i18n import i18n

        nav = self.navigationInterface
        trees = [c for c in nav.findChildren(QWidget) if type(c).__name__ == "NavigationTreeWidget"]
        i18n.set_language(language)
        for tree, key in zip(trees, NAV_TITLE_KEYS):
            title = tr(key)
            tree.setText(title)
            for item in (c for c in tree.children() if type(c).__name__ == "NavigationTreeItem"):
                item.setText(title)
        self.setWindowTitle(tr("app"))
        pages = (
            self.overview_page, self.separation_page, self.midi_page, self.asr_page,
            self.tts_page, self.pipeline_page, self.training_page, self.settings_page,
        )
        for page in pages:
            retranslate = getattr(page, "retranslate", None)
            if callable(retranslate):
                retranslate()
            for child in page.findChildren(QWidget):
                child_retranslate = getattr(child, "retranslate", None)
                if callable(child_retranslate):
                    child_retranslate()


def main() -> int:
    from PyQt5.QtCore import QTimer
    from PyQt5.QtWidgets import QApplication

    smoke = os.environ.get("VOCALSTUDIO_SMOKE") == "1"
    if smoke:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    app = QApplication(sys.argv)
    from core.app_settings import get_settings as _get_settings
    from gui.theme import apply_app_palette
    try:
        apply_app_palette(_get_settings().theme)
    except Exception:
        pass
    window = MainWindow()
    window.show()

    def _smoke_quit() -> None:
        try:
            import torch as _torch

            print(f"SMOKE_TORCH {_torch.__version__} cuda={_torch.cuda.is_available()}")
        except Exception:  # slim frontend exe does not bundle torch; the backend exe reports it
            pass
        print("SMOKE_FROZEN_PASS")
        app.quit()

    if smoke:
        QTimer.singleShot(1200, _smoke_quit)
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
