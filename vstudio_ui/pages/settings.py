"""Settings page — appearance, interface and runtime preferences.

Four cards, one per decision the user actually makes:

* **appearance** — light / dark
* **interface** — UI language, font scale
* **runtime** — the device preference stored in ``settings/user.json``
* **about** — what this window is

Every control applies **immediately and persists**. There is no Save button
because a settings page that silently forgets your choice on restart is worse
than one that writes as you go, and the write is a single small JSON file. The
write goes through ``AppShell``, which is the only writer — so the header's theme
button and the combo on this page cannot disagree about what is on disk.

The theme combo also lives in the header. That is not duplication: a settings page
is where a user looks for it, and a header toggle is where they reach for it. Both
read and write the same shell state, and ``sync_theme`` reflects a change made
elsewhere without re-emitting the signal.
"""

from __future__ import annotations

from PyQt5.QtWidgets import QComboBox, QVBoxLayout, QScrollArea, QWidget

from vstudio_ui import services, strings, tokens
from vstudio_ui.theme import apply_font_scale, apply_theme, current_font_scale, current_theme
from vstudio_ui.tokens import THEME_ORDER
from vstudio_ui.widgets import (
    Card,
    CardHeader,
    FieldRow,
    PageScaffold,
    ValueLabel,
    refresh_theme_tree,
)

_CARD_PADDING = 14
_CARD_SPACING = 10


class SettingsPage(PageScaffold):
    def __init__(self, shell=None, parent=None):
        super().__init__(strings.tr("settings.title"), parent, caption=strings.tr("settings.caption"))
        self._shell = shell

        self._failure = ValueLabel("", muted=True)
        self._failure.setVisible(False)
        self.add_trailing_widget(self._failure)

        self._build_appearance_card()
        self._build_interface_card()
        self._build_runtime_card()
        self._build_about_card()
        self.content_layout.addStretch(1)
        host = QWidget(); column = QVBoxLayout(host); column.setContentsMargins(0, 0, 6, 0)
        while self.content_layout.count(): column.addItem(self.content_layout.takeAt(0))
        self._scroll = QScrollArea(); self._scroll.setWidgetResizable(True); self._scroll.setWidget(host)
        self.content_layout.addWidget(self._scroll)

        if shell is not None:
            hook = getattr(shell, "preference_failed", None)
            if hook is not None:
                hook.connect(self._on_write_failed)

    # -- cards --------------------------------------------------------------

    def _card(self, title_key: str, caption_key: str = "") -> tuple[Card, QVBoxLayout]:
        card = Card()
        column = QVBoxLayout(card)
        column.setContentsMargins(_CARD_PADDING, _CARD_PADDING, _CARD_PADDING, _CARD_PADDING)
        column.setSpacing(_CARD_SPACING)
        caption = strings.tr(caption_key) if caption_key else ""
        column.addWidget(CardHeader(strings.tr(title_key), caption=caption))
        return card, column

    def _build_appearance_card(self) -> None:
        card, column = self._card("settings.appearance", "settings.theme_hint")

        self.theme_combo = QComboBox()
        for name in THEME_ORDER:
            self.theme_combo.addItem(strings.THEME_LABELS[name], userData=name)
        self.theme_combo.setCurrentIndex(max(0, THEME_ORDER.index(current_theme())))
        self.theme_combo.currentIndexChanged.connect(self._on_theme_changed)

        self._theme_row = FieldRow(strings.tr("settings.theme"), self.theme_combo)
        column.addWidget(self._theme_row)
        self.content_layout.addWidget(card)

    def _build_interface_card(self) -> None:
        card, column = self._card("settings.interface", "settings.interface_hint")

        # The UI-language list is deliberately not the pipeline's language list:
        # ``language.*`` is reused for the labels because for zh/en the two mean
        # exactly the same thing, but the *choices* come from the business layer's
        # interface languages.
        self.language_combo = QComboBox()
        for code in services.UI_LANGUAGES:
            self.language_combo.addItem(strings.tr(f"language.{code}", code), userData=code)
        self._sync_combo(self.language_combo, strings.current_language())
        self.language_combo.currentIndexChanged.connect(self._on_language_changed)

        self.font_combo = QComboBox()
        for step in tokens.FONT_SCALE_STEPS:
            self.font_combo.addItem(self._font_label(step), userData=float(step))
        self.select_font_scale(current_font_scale())
        self.font_combo.currentIndexChanged.connect(self._on_font_scale_changed)

        self._language_row = FieldRow(strings.tr("settings.language"), self.language_combo)
        self._font_row = FieldRow(strings.tr("settings.font_scale"), self.font_combo)
        FieldRow.align([self._language_row, self._font_row])
        column.addWidget(self._language_row)
        column.addWidget(self._font_row)
        self.content_layout.addWidget(card)

    def _build_runtime_card(self) -> None:
        card, column = self._card("settings.runtime", "settings.runtime_hint")

        self.device_combo = QComboBox()
        for name in services.DEVICE_CHOICES:
            self.device_combo.addItem(strings.tr(f"device.{name}", name), userData=name)
        self._sync_combo(self.device_combo, services.load_preferences().device)
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)

        self._device_row = FieldRow(strings.tr("settings.device"), self.device_combo)
        column.addWidget(self._device_row)
        self.content_layout.addWidget(card)

    def _build_about_card(self) -> None:
        card, column = self._card("settings.about", "settings.about_body")
        self.content_layout.addWidget(card)

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _font_label(step: float) -> str:
        return strings.tr("settings.font_scale_value").format(percent=round(float(step) * 100))

    @staticmethod
    def _sync_combo(combo: QComboBox, value) -> None:
        """Select the entry for *value* without emitting a change signal.

        Blocking the signal matters: without it, reflecting a value that came from
        elsewhere (startup, the header button) would be re-broadcast as a fresh
        user edit, and every reflection would write the settings file again.
        """
        index = combo.findData(value)
        if index >= 0 and index != combo.currentIndex():
            combo.blockSignals(True)
            combo.setCurrentIndex(index)
            combo.blockSignals(False)

    def select_font_scale(self, value: float) -> None:
        """Select *value*, adding it to the list when it is not one of the steps.

        ``settings/user.json`` is a text file a user can edit, so the stored scale
        can be any number in range. Snapping it to the nearest step would make the
        combo *claim* a setting the application is not using — the label would read
        125% while the type scale is 1.2.
        """
        combo = self.font_combo
        try:
            target = float(value)
        except (TypeError, ValueError):
            target = tokens.DEFAULT_FONT_SCALE
        if combo.findData(target) < 0:
            position = 0
            for index in range(combo.count()):
                if float(combo.itemData(index)) < target:
                    position = index + 1
            combo.blockSignals(True)
            combo.insertItem(position, self._font_label(target), userData=target)
            combo.setCurrentIndex(position)
            combo.blockSignals(False)
            return
        self._sync_combo(combo, target)

    # -- API --------------------------------------------------------------

    def sync_theme(self, name: str) -> None:
        """Reflect an externally changed theme without re-emitting the signal."""
        self._sync_combo(self.theme_combo, name)

    def refresh_theme(self) -> None:
        super().refresh_theme()
        # The header's toggle changes the theme behind this page's back; without
        # this line the combo still shows the theme from when the page was built.
        self.sync_theme(current_theme())

    def _on_theme_changed(self, _index: int) -> None:
        name = self.theme_combo.currentData()
        if name:
            self._apply(theme=str(name))

    def _on_language_changed(self, _index: int) -> None:
        code = self.language_combo.currentData()
        if code:
            self._apply(language=str(code))

    def _on_font_scale_changed(self, _index: int) -> None:
        data = self.font_combo.currentData()
        if data is None:
            return
        applied = self._apply(font_scale=float(data))
        if applied is not None:
            self.select_font_scale(float(applied))

    def _on_device_changed(self, _index: int) -> None:
        name = self.device_combo.currentData()
        if name:
            self._apply(device=str(name))

    def _apply(self, **change):
        """Apply one preference and persist it. Returns the applied value, if any.

        With a shell the work is the shell's, which is what keeps the header and
        this page on one code path. Without one — a page built on its own, as the
        tests and the screenshot harness do — it is applied locally and not
        written, because there is nothing here that should be writing settings.
        """
        if self._shell is not None:
            if "theme" in change:
                self._shell.set_theme(change["theme"])
            if "language" in change:
                self._shell.set_language(change["language"])
            if "font_scale" in change:
                return self._shell.set_font_scale(change["font_scale"])
            if "device" in change:
                # The device is a stored preference, not a live setting: the
                # shell has no set_device because there is nothing to apply it to
                # (see new-ui.md N16). It still goes through the shell so that
                # this page is not a second writer of settings/user.json.
                self._shell.persist_preference(device=str(change["device"]))
            return None

        if "theme" in change:
            apply_theme(change["theme"])
        if "language" in change:
            strings.set_language(change["language"])
        if "font_scale" in change:
            apply_font_scale(change["font_scale"])
        refresh_theme_tree(self)
        self.retranslate()
        return change.get("font_scale")

    def _on_write_failed(self, message: str) -> None:
        self._failure.set_value(
            strings.tr("settings.write_failed").format(message=message), muted=True
        )
        self._failure.setVisible(True)

    # -- i18n ---------------------------------------------------------------

    def retranslate(self) -> None:
        self.set_title(strings.tr("settings.title"))
        self.set_caption(strings.tr("settings.caption"))
        self._theme_row.set_label(strings.tr("settings.theme"))
        self._language_row.set_label(strings.tr("settings.language"))
        self._font_row.set_label(strings.tr("settings.font_scale"))
        self._device_row.set_label(strings.tr("settings.device"))
        for index, name in enumerate(THEME_ORDER):
            self.theme_combo.setItemText(index, strings.THEME_LABELS[name])
        for index, code in enumerate(services.UI_LANGUAGES):
            self.language_combo.setItemText(index, strings.tr(f"language.{code}", code))
        # Iterate the combo, not the step table: select_font_scale may have added a
        # stored, non-standard scale, and re-labelling by table position would
        # shift every label after it by one.
        for index in range(self.font_combo.count()):
            data = self.font_combo.itemData(index)
            self.font_combo.setItemText(index, self._font_label(float(data)))
        for index, name in enumerate(services.DEVICE_CHOICES):
            self.device_combo.setItemText(index, strings.tr(f"device.{name}", name))
        # Keep the combos truthful after a language switch: the labels changed, but
        # so may have the values (the language combo's own selection).
        self._sync_combo(self.language_combo, strings.current_language())
        self.select_font_scale(current_font_scale())
        self._sync_combo(self.theme_combo, current_theme())
        self.refresh_theme()
