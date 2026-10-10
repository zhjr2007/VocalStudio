"""ModelStatusWidget: model readiness list (green/red dots) plus GPU summary."""

from __future__ import annotations

from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QVBoxLayout, QWidget

try:
    from qfluentwidgets import BodyLabel as FluentLabel, CardWidget, CaptionLabel
except ModuleNotFoundError:  # renamed distribution
    from qfluent_widgets import BodyLabel as FluentLabel, CardWidget, CaptionLabel

from core.config import ModelPaths, PROJECT_ROOT
from core.device_utils import detect_system_hardware, get_gpu_info
from gui.i18n import tr

READY_COLOR = "#3DD68C"
MISSING_COLOR = "#F75C49"


def _dot_label(color: str) -> QLabel:
    label = QLabel()
    label.setFixedSize(10, 10)
    label.setStyleSheet(f"background-color:{color}; border-radius:5px;")
    return label


class ModelStatusWidget(QWidget):

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._rows_data: list[tuple[str, bool]] = []
        self._ready_count: int = 0
        self._total: int = 0
        self._gpu: dict = {}
        self.setMinimumSize(320, 360)

        card = CardWidget(self)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(14, 10, 14, 10)

        self._summary_label = FluentLabel("", card)
        self._list = QListWidget(card)
        self._gpu_label = CaptionLabel("", card)
        card_layout.addWidget(self._summary_label)
        card_layout.addWidget(self._list, 1)
        card_layout.addWidget(self._gpu_label)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(card)

        self.refresh()

    def refresh(self) -> None:
        mp = ModelPaths.from_settings()
        gpu = get_gpu_info()
        rows: list[tuple[str, bool]] = []
        for name in mp.__dataclass_fields__:
            value = getattr(mp, name)
            ready = False
            if value:
                try:
                    ready = mp.resolve(name).exists()
                except ValueError:
                    ready = False
            rows.append((name, bool(ready)))

        self._rows_data = rows
        self._gpu = gpu or {}
        self._total = len(rows)
        self._ready_count = sum(1 for _name, ok in rows if ok)

        self._list.clear()
        for name, ready in rows:
            summary_text = self._row_summary(mp, name)
            row = QWidget()
            h_layout = QHBoxLayout(row)
            h_layout.setContentsMargins(6, 2, 6, 2)
            h_layout.addWidget(_dot_label(READY_COLOR if ready else MISSING_COLOR))
            h_layout.addSpacing(8)
            name_label = FluentLabel(name, row)
            name_label.setMinimumWidth(150)
            h_layout.addWidget(name_label)
            path_label = CaptionLabel(summary_text, row)
            h_layout.addWidget(path_label, 1)

            item = QListWidgetItem()
            item.setSizeHint(QSize(-1, 32))
            self._list.addItem(item)
            self._list.setItemWidget(item, row)

        ready_word = tr("modelstatus.ready", "就绪")
        self._summary_label.setText(f"{self._ready_count}/{self._total} {ready_word}")
        self._gpu_label.setText(self._format_gpu_line())
        self._apply_theme_styles()

    def summary(self) -> dict:
        gpu_name = self._gpu.get("name") if isinstance(self._gpu, dict) else None
        try:
            free_vram_gb = float((self._gpu or {}).get("free_vram_gb") or 0.0)
        except (TypeError, ValueError):
            free_vram_gb = 0.0
        return {
            "ready": self._ready_count,
            "total": self._total,
            "missing": [name for name, ok in self._rows_data if not ok],
            "gpu_name": gpu_name,
            "free_vram_gb": free_vram_gb,
        }

    def rows(self) -> list[tuple[str, bool]]:
        return [(name, ready) for name, ready in self._rows_data]

    def retranslate(self) -> None:
        self.refresh()

    def _apply_theme_styles(self) -> None:
        # A widget-level stylesheet wins over the application sheet, so the row
        # colours have to be picked here rather than inherited from QSS.
        rhine = False
        try:
            from gui.theme import is_rhine_theme

            rhine = bool(is_rhine_theme())
        except Exception:
            pass
        if rhine:
            from gui.design_tokens import RADIUS, RHINE

            item_bg = RHINE["card_alt"]
            handle = RHINE["border"]
            radius = RADIUS
        else:
            dark = False
            try:
                from qfluentwidgets import isDarkTheme

                dark = bool(isDarkTheme())
            except Exception:  # renamed distribution or missing helper -> keep light look
                pass
            item_bg = "#323232" if dark else "#F2F2F2"
            handle = "#5C5C5C" if dark else "#C4C4C4"
            radius = 5
        self._list.setStyleSheet(
            "QListWidget{border:none; background:transparent; outline:none;}"
            f"QListWidget::item{{margin:1px 8px; padding:2px 6px; border-radius:{radius}px; background-color:{item_bg};}}"
            "QScrollBar:vertical{width:6px; margin:0; background:transparent;}"
            f"QScrollBar::handle:vertical{{background:{handle}; border-radius:3px;}}"
        )

    @staticmethod
    def _display_path(mp: ModelPaths, name: str) -> str:
        try:
            path = mp.resolve(name)
        except ValueError:
            return ""
        try:
            return path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            return path.as_posix()

    def _row_summary(self, mp: ModelPaths, name: str) -> str:
        value = getattr(mp, name, "")
        if not value:
            return tr("modelstatus.not_set", "未配置")
        text = self._display_path(mp, name) or tr("modelstatus.not_set", "未配置")
        if len(text) > 58:
            text = "…" + text[-57:]
        return text

    def _format_gpu_line(self) -> str:
        gpu = self._gpu or {}
        name = gpu.get("name")
        if not name:
            names = [entry[0] for entry in detect_system_hardware()["gpus"]]
            if names:
                sys_word = tr("modelstatus.sys_gpus", "系统显卡")
                return f"{sys_word}: {', '.join(names)}"
            return tr("modelstatus.gpu_none", "未检测到 GPU")
        cuda_available = bool(gpu.get("torch_cuda_available"))
        try:
            free_gb = float(gpu.get("free_vram_gb") or 0.0)
            total_gb = float(gpu.get("total_vram_gb") or 0.0)
        except (TypeError, ValueError):
            free_gb = 0.0
            total_gb = 0.0
        vram_text = tr("modelstatus.vram", "显存 {free}/{total} GB").format(
            free=f"{free_gb:.1f}", total=f"{total_gb:.1f}"
        )
        return f"{name} · CUDA {'True' if cuda_available else 'False'} · {vram_text}"
