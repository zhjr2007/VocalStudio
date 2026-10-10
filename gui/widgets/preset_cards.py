"""Preset cards for the separation page (wave B-2). Pure PyQt5 + QSS."""

from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QGridLayout, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from gui.i18n import tr

PRESET_ORDER = ("Karaoke", "FullVocal", "Band4", "Producer5")
TIER_BY_PRESET = {
    "Karaoke": "tier_standard",
    "FullVocal": "tier_standard",
    "Band4": "tier_hq",
    "Producer5": "tier_hq",
}
NAME_KEYS = {
    "Karaoke": "preset_karaoke",
    "FullVocal": "preset_fullvocal",
    "Band4": "preset_band4",
    "Producer5": "preset_producer5",
}
DESC_KEYS = {
    "Karaoke": "preset_karaoke_desc",
    "FullVocal": "preset_fullvocal_desc",
    "Band4": "preset_band4_desc",
    "Producer5": "preset_producer5_desc",
}

_BASE_QSS = """
#PresetCard { border: 1px solid #B9C0CC; border-radius: 8px; background: transparent; }
#PresetName { font-weight: bold; }
"""
_SELECTED_QSS = _BASE_QSS + "\n#PresetCard { border: 2px solid #4A7DBD; }"
_DISABLED_QSS = (
    "#PresetCard { border: 1px dashed #B9C0CC; background: rgba(150, 150, 150, 36); }\n"
    "QLabel { color: #9A9A9A; }"
)


class PresetCard(QWidget):
    """One clickable preset card with name / description / tier label."""

    clicked = pyqtSignal(str)

    def __init__(self, preset_name: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("PresetCard")
        self._preset = preset_name
        self._selected = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)

        self._name_label = QLabel(self)
        self._name_label.setObjectName("PresetName")
        layout.addWidget(self._name_label)

        self._desc_label = QLabel(self)
        self._desc_label.setWordWrap(True)
        layout.addWidget(self._desc_label)

        bottom_row = QHBoxLayout()
        self._tier_label = QLabel(self)
        bottom_row.addWidget(self._tier_label)
        bottom_row.addStretch(1)
        layout.addLayout(bottom_row)

        for label in (self._name_label, self._desc_label, self._tier_label):
            label.setAttribute(Qt.WA_TransparentForMouseEvents)

        self._apply_style()

    @property
    def preset_name(self) -> str:
        return self._preset

    @property
    def is_selected(self) -> bool:
        return self._selected

    def _apply_style(self) -> None:
        if not self.isEnabled():
            self.setStyleSheet(_DISABLED_QSS)
        elif self._selected:
            self.setStyleSheet(_SELECTED_QSS)
        else:
            self.setStyleSheet(_BASE_QSS)

    def setSelected(self, selected: bool) -> None:  # noqa: N802 (qt naming)
        if self._selected != selected and self.isEnabled():
            self._selected = selected
            self._apply_style()

    def setEnabled(self, enabled: bool) -> None:  # noqa: A003 (qt override)
        super().setEnabled(enabled)
        if not enabled:
            self._selected = False
        self._apply_style()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 (qt naming)
        if self.isEnabled() and self.rect().contains(event.pos()):
            self.clicked.emit(self._preset)
        super().mouseReleaseEvent(event)

    def retranslate(self) -> None:
        self._name_label.setText(tr(NAME_KEYS[self._preset], self._preset))
        self._desc_label.setText(tr(DESC_KEYS[self._preset]))
        self._tier_label.setText(tr(TIER_BY_PRESET[self._preset]))


class PresetGrid(QWidget):
    """2x2 grid of preset cards with single selection + readiness gating."""

    currentChanged = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._cards: dict[str, PresetCard] = {}

        layout = QGridLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(10)

        for row, name in enumerate(PRESET_ORDER):
            card = PresetCard(name, self)
            card.clicked.connect(lambda _name=None, n=name: self._on_pressed(n))
            self._cards[name] = card
            layout.addWidget(card, row // 2, row % 2)

        if "Karaoke" in self._cards:
            self.select("Karaoke")
        self.retranslate()

    def preset_names(self) -> tuple[str, ...]:
        return PRESET_ORDER

    def card_count(self) -> int:
        return len(self._cards)

    def card(self, name: str) -> PresetCard | None:
        return self._cards.get(name)

    def is_enabled(self, name: str) -> bool:
        card = self._cards.get(name)
        return bool(card and card.isEnabled())

    def set_enabled(self, name: str, enabled: bool) -> None:
        card = self._cards.get(name)
        if card is not None:
            card.setEnabled(enabled)

    def select(self, name: str) -> None:
        for preset_name, card in self._cards.items():
            want = preset_name == name and card.isEnabled()
            card.setSelected(want)

    def selected_preset(self) -> str | None:
        for name in PRESET_ORDER:
            if self._cards[name].is_selected:
                return name
        return None

    def _on_pressed(self, name: str) -> None:
        card = self._cards.get(name)
        if card is None or not card.isEnabled():
            return
        self.select(name)
        self.currentChanged.emit(name)

    def retranslate(self) -> None:
        for card in self._cards.values():
            card.retranslate()
