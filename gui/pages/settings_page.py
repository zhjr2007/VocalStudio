"""SettingsPage: detected hardware, device / theme / language / font-size selection + model status."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

try:
    from qfluentwidgets import (
        BodyLabel as FluentLabel,
        CardWidget,
        CaptionLabel,
        ComboBox as FluentComboBox,
        PushButton as FluentPushButton,
        Theme,
        setTheme as apply_fluent_theme,
    )
except ModuleNotFoundError:  # renamed distribution
    from qfluent_widgets import (
        BodyLabel as FluentLabel,
        CardWidget,
        CaptionLabel,
        ComboBox as FluentComboBox,
        PushButton as FluentPushButton,
        Theme,
        setTheme as apply_fluent_theme,
    )

from core.app_settings import AppSettings, USER_SETTINGS_FILE
from core.device_utils import default_backend, detect_system_hardware
from gui.i18n import tr
from gui.theme import apply_font_scale, set_app_theme
from gui.widgets.model_status import ModelStatusWidget

DEVICE_CHOICES = ("cuda", "rocm", "directml", "cpu")
# (stored value, qfluentwidgets base theme, i18n key). `rhine` is the warm-white
# theme built from the RHINE LAB reference deck — see ui-design/design.md.
# It is based on Fluent LIGHT because its canvas is warm off-white; the near-black
# "ink" is carried by the palette + QSS rather than by Fluent's dark mode.
THEME_ITEMS = (
    ("auto", Theme.AUTO, "theme.auto"),
    ("light", Theme.LIGHT, "theme.light"),
    ("dark", Theme.DARK, "theme.dark"),
    ("rhine", Theme.LIGHT, "theme.rhine"),
)
THEME_VALUES = tuple(value for value, _theme, _key in THEME_ITEMS)
LANGUAGE_KEYS = {"zh": "lang.zh", "en": "lang.en"}

# (i18n key, English fallback text, multiplier applied to the base font)
FONT_SCALE_ITEMS = (
    ("gui_font_small", "Small", 0.85),
    ("gui_font_normal", "Normal", 1.0),
    ("gui_font_large", "Large", 1.2),
    ("gui_font_xlarge", "Extra large", 1.5),
    ("gui_font_xxlarge", "Huge", 2.0),
)


class SettingsPage(QWidget):
    language_changed = pyqtSignal(str)

    def __init__(self, parent=None, settings_file: Path | str | None = None):
        super().__init__(parent)
        self._settings_file = Path(settings_file) if settings_file else USER_SETTINGS_FILE

        layout = QVBoxLayout(self)
        self._title_label = QLabel(self)
        layout.addWidget(self._title_label)

        card = CardWidget()
        card_layout = QVBoxLayout(card)
        form_column = QVBoxLayout()
        form_column.setContentsMargins(14, 10, 14, 10)
        card_layout.addLayout(form_column)

        self._hardware_title_label = FluentLabel()
        self._hw_gpu_label = CaptionLabel("")
        self._hw_cpu_label = CaptionLabel("")
        form_column.addWidget(self._hardware_title_label)
        form_column.addWidget(self._hw_gpu_label)
        form_column.addWidget(self._hw_cpu_label)
        form_column.addSpacing(8)

        self._device_label = FluentLabel()
        self._theme_label = FluentLabel()
        self._language_label = FluentLabel()
        self._font_size_label = FluentLabel()
        self._save_button = FluentPushButton()
        self.save_button = self._save_button

        self._device_combo = FluentComboBox()
        for value in DEVICE_CHOICES:
            self._device_combo.addItem(value)
        self._theme_combo = FluentComboBox()
        for value, _theme, _key in THEME_ITEMS:
            # display text is set in retranslate(); the stored value rides in
            # userData so a translated label can never break the lookup.
            self._theme_combo.addItem(value, userData=value)
        self._language_combo = FluentComboBox()
        self._font_size_combo = FluentComboBox()
        for _key, _text, scale in FONT_SCALE_ITEMS:
            self._font_size_combo.addItem("", userData=scale)

        device_row = QHBoxLayout()
        device_row.addWidget(self._device_label)
        device_row.addWidget(self._device_combo, 1)
        theme_row = QHBoxLayout()
        theme_row.addWidget(self._theme_label)
        theme_row.addWidget(self._theme_combo, 1)
        language_row = QHBoxLayout()
        language_row.addWidget(self._language_label)
        language_row.addWidget(self._language_combo, 1)
        font_size_row = QHBoxLayout()
        font_size_row.addWidget(self._font_size_label)
        font_size_row.addWidget(self._font_size_combo, 1)
        button_row = QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(self._save_button)

        form_column.addLayout(device_row)
        form_column.addLayout(theme_row)
        form_column.addLayout(language_row)
        form_column.addLayout(font_size_row)
        form_column.addSpacing(6)
        form_column.addLayout(button_row)
        layout.addWidget(card)

        self._status_title_label = FluentLabel()
        layout.addWidget(self._status_title_label)
        self.model_status = ModelStatusWidget()
        layout.addWidget(self.model_status, 1)
        layout.addStretch(0)

        self._load_settings_into_ui()
        self.retranslate()

        self._save_button.clicked.connect(self.save_settings)
        self._device_combo.currentIndexChanged.connect(self._on_choice_changed)
        self._theme_combo.currentIndexChanged.connect(self._on_choice_changed)
        self._language_combo.currentIndexChanged.connect(self._on_choice_changed)
        self._language_combo.currentIndexChanged.connect(self._emit_language)
        self._font_size_combo.currentIndexChanged.connect(self._on_font_size_changed)

    # -- public API -------------------------------------------------------

    @property
    def settings_file(self) -> Path:
        return self._settings_file

    @property
    def device(self) -> str:
        return str(self._device_combo.currentText())

    def set_device(self, value: str) -> None:
        index = self._device_combo.findText(value)
        if index >= 0:
            self._device_combo.setCurrentIndex(index)

    @property
    def theme(self) -> str:
        return str(self._theme_combo.currentData() or "auto")

    def set_theme(self, value: str) -> None:
        for index in range(self._theme_combo.count()):
            if self._theme_combo.itemData(index) == value:
                self._theme_combo.setCurrentIndex(index)
                return

    @property
    def language(self) -> str:
        return str(self._language_combo.currentData() or "zh")

    def set_language(self, value: str) -> None:
        for index in range(self._language_combo.count()):
            if self._language_combo.itemData(index) == value:
                self._language_combo.setCurrentIndex(index)
                return

    @property
    def font_scale(self) -> float:
        data = self._font_size_combo.currentData()
        if data is None:
            return 1.0
        try:
            return float(data)
        except (TypeError, ValueError):
            return 1.0

    def set_font_scale(self, value: float) -> None:
        """Select the combo entry whose scale is closest to *value*."""
        try:
            target = float(value)
        except (TypeError, ValueError):
            target = 1.0
        best_index = 0
        best_delta = float("inf")
        for index in range(self._font_size_combo.count()):
            data = self._font_size_combo.itemData(index)
            try:
                delta = abs(float(data) - target)
            except (TypeError, ValueError):
                continue
            if delta < best_delta:
                best_delta = delta
                best_index = index
        self._font_size_combo.setCurrentIndex(best_index)

    def save_settings(self) -> AppSettings:
        settings = AppSettings(
            device=self.device,
            theme=self.theme,
            language=self.language,
            font_scale=self.font_scale,
        )
        settings.validate()
        settings.save(self._settings_file)
        self._apply_ui_settings(settings)
        return settings

    def refresh_model_status(self) -> None:
        self.model_status.refresh()

    # -- internals --------------------------------------------------------

    def _on_choice_changed(self, *_args) -> None:
        self.save_settings()
        self.refresh_model_status()

    def _on_font_size_changed(self, *_args) -> None:
        # Live-apply the new font scale and persist it; hardware detection is
        # expensive and does not depend on the font, so skip the refresh here.
        self.save_settings()

    def _emit_language(self, *_args) -> None:
        self.language_changed.emit(str(self.language))

    def _load_settings_into_ui(self) -> None:
        stored = AppSettings.load(self._settings_file)
        for value, key in LANGUAGE_KEYS.items():
            self._language_combo.addItem(tr(key), userData=value)
        if stored.device in DEVICE_CHOICES:
            self.set_device(stored.device)
        else:  # legacy "auto" / unknown -> derive from detected hardware
            self.set_device(default_backend())
        if stored.theme not in THEME_VALUES:
            self.set_theme("auto")
        self.set_language(stored.language if stored.language in LANGUAGE_KEYS else "zh")
        self.set_font_scale(stored.font_scale)

    def _update_hardware(self) -> None:
        hw = detect_system_hardware()
        gpus = [name for name, _vram in hw.get("gpus", [])]
        gpu_text = tr("hw_gpu") + ": " + ("; ".join(gpus) if gpus else tr("hw_none"))
        self._hw_gpu_label.setText(gpu_text)
        cpu_name = hw.get("cpu") or ""
        self._hw_cpu_label.setText(tr("hw_cpu") + (f": {cpu_name}" if cpu_name else ""))

    def _apply_ui_settings(self, settings: AppSettings) -> None:
        set_app_theme(settings.theme)
        apply_font_scale(settings.font_scale)

    def retranslate(self) -> None:
        self._title_label.setText(tr("settings"))
        self._device_label.setText(tr("device"))
        self._theme_label.setText(tr("theme"))
        self._language_label.setText(tr("language"))
        self._font_size_label.setText(tr("gui_font_size", "Font size"))
        self._save_button.setText(tr("save"))
        self._hardware_title_label.setText(tr("hardware_title"))
        self._status_title_label.setText(tr("models_title"))
        for index, (key, default, _scale) in enumerate(FONT_SCALE_ITEMS):
            self._font_size_combo.setItemText(index, tr(key, default))
        for index, (value, _theme, key) in enumerate(THEME_ITEMS):
            self._theme_combo.setItemText(index, tr(key, value))
        for index in range(self._language_combo.count()):
            value = self._language_combo.itemData(index)
            if value in LANGUAGE_KEYS:
                self._language_combo.setItemText(
                    index, tr(LANGUAGE_KEYS[value])
                )
        self._update_hardware()
        self.model_status.retranslate()
