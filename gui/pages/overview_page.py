"""Overview / home page: environment summary, model readiness, quick navigation."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from PyQt5.QtCore import QPropertyAnimation, Qt, QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QGridLayout,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

try:
    from qfluentwidgets import BodyLabel as FluentLabel, CardWidget, CaptionLabel
except ModuleNotFoundError:  # renamed distribution
    from qfluent_widgets import BodyLabel as FluentLabel, CardWidget, CaptionLabel

from gui.i18n import tr


def _base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parents[2]


class OverviewPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 20)
        root.setSpacing(14)

        header = QHBoxLayout()
        logo = QLabel("\U0001F399")
        f = QFont()
        f.setPointSize(34)
        logo.setFont(f)
        self._logo_effect = QGraphicsOpacityEffect(logo)
        logo.setGraphicsEffect(self._logo_effect)
        pulse = QPropertyAnimation(self._logo_effect, b"opacity")
        pulse.setDuration(900)
        pulse.setStartValue(1.0)
        pulse.setEndValue(0.55)
        pulse.setLoopCount(-1)
        pulse.setDirection(1)  # QAbstractAnimation.Alternate
        self._pulse = pulse
        header.addWidget(logo)

        title_box = QVBoxLayout()
        self._title_label = QLabel(tr("app"))
        tf = QFont()
        tf.setPointSize(28)
        tf.setBold(True)
        self._title_label.setFont(tf)

        subtitle = CaptionLabel(tr("overview_subtitle"))
        self._subtitle_label = subtitle
        title_box.addWidget(self._title_label)
        title_box.addWidget(subtitle)
        header.addLayout(title_box)
        root.addLayout(header)

        cards_row = QHBoxLayout()
        cards_row.setSpacing(14)
        self._models_ready, self._models_total = self._read_models_status()
        self._head_labels: list[tuple[QLabel, str]] = []
        self._nav_btns: list[tuple[QPushButton, str]] = []
        cards_row.addWidget(self._env_card(), 3)
        cards_row.addWidget(self._models_card(), 2)
        cards_row.addWidget(self._quicknav_card(), 3)
        root.addLayout(cards_row, 1)

    # ---- data helpers -------------------------------------------------
    def _env_lines(self) -> list[str]:
        from gui.i18n import tr

        lines = [f"Python {platform.python_version()}"]
        try:
            from core.device_utils import check_python_environment, get_gpu_info

            env = check_python_environment()
            gpu = get_gpu_info()
            lines.append(f"torch {env['torch']} (cuda={env['torch_cuda']})")
            if gpu["torch_cuda_available"]:
                cc = str(gpu["compute_capability"]).replace(".", "")
                lines.append(
                    f"{gpu['name']} | sm_{cc} | "
                    f"{gpu['total_vram_gb']:.0f}GB / {gpu['free_vram_gb']:.1f}GB free"
                )
            else:
                lines.append(tr("env_no_cuda"))
        except Exception:
            try:
                import torch

                if torch.cuda.is_available():
                    name = torch.cuda.get_device_name(0)
                    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    lines.append(f"torch {torch.__version__} | cuda ok")
                    lines.append(f"{name} | {vram:.0f}GB")
                else:
                    lines.append(f"torch {torch.__version__}")
            except Exception:
                lines.append(tr("env_torch_missing"))
            nv = self._nvidia_smi()
            if nv:
                lines.append(nv)
        return lines

    @staticmethod
    def _nvidia_smi() -> str | None:
        exe = shutil.which("nvidia-smi")
        if not exe:
            return None
        try:
            out = subprocess.run(
                [exe, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
            if out:
                return f"nvidia-smi: {out.splitlines()[0]}"
        except Exception:
            pass
        return None

    def _read_models_status(self) -> tuple[int, int]:
        settings = _base_dir() / "settings" / "defaults.json"
        try:
            data = json.loads(settings.read_text(encoding="utf-8"))
        except Exception:
            return 0, 0
        models = data.get("models", {})
        total = sum(1 for v in models.values() if v)

        def _exists(value: str) -> bool:
            p = Path(value)
            return p.exists() if p.is_absolute() else (_base_dir() / p).exists()

        ready = sum(1 for v in models.values() if v and _exists(v))
        return ready, total

    # ---- widgets ------------------------------------------------------
    def _card(self, title_key: str) -> tuple[CardWidget, QVBoxLayout]:
        card = CardWidget()
        layout = QVBoxLayout(card)
        head = QLabel(tr(title_key))
        hf = QFont(head.font())
        hf.setBold(True)
        head.setFont(hf)
        self._head_labels.append((head, title_key))
        layout.addWidget(head)
        return card, layout

    def _env_card(self) -> QWidget:
        from gui.i18n import tr

        card, layout = self._card("env_title")
        for line in self._env_lines():
            label = CaptionLabel(line)
            label.setWordWrap(True)
            label.setMinimumWidth(80)
            layout.addWidget(label)
        layout.addStretch(1)
        return card

    def _models_card(self) -> QWidget:
        from gui.i18n import tr

        card, layout = self._card("models_title")
        big = QLabel(f"0 / {self._models_total}")
        bf = QFont()
        bf.setPointSize(30)
        bf.setBold(True)
        big.setFont(bf)
        big.setAlignment(Qt.AlignCenter)
        self._count_label = big
        layout.addWidget(big)
        layout.addWidget(CaptionLabel("settings/defaults.json"))
        layout.addStretch(1)
        target, total = self._models_ready, self._models_total
        if target > 0:
            step = max(1, round(target / 24))

            def _tick() -> None:
                cur = int(self._count_label.text().split(" ")[0].split("/")[0]) + step
                if cur >= target:
                    self._count_timer.stop()
                    self._count_label.setText(f"{target} / {total}")
                else:
                    self._count_label.setText(f"{cur} / {total}")

            self._count_timer = QTimer(self)
            self._count_timer.setInterval(30)
            self._count_timer.timeout.connect(_tick)
            QTimer.singleShot(250, self._count_timer.start)
        else:
            big.setText(f"{target} / {total}")
        return card

    def _quicknav_card(self) -> QWidget:
        from gui.i18n import tr

        targets = [
            ("separation_page", "separation"),
            ("midi_page", "midi"),
            ("asr_page", "asr"),
            ("tts_page", "tts"),
            ("pipeline_page", "pipeline"),
            ("training_page", "training"),
        ]
        card, layout = self._card("quicknav_title")
        grid = QGridLayout()
        grid.setSpacing(8)
        for i, (attr, key) in enumerate(targets):
            btn = QPushButton(tr(key))
            btn.clicked.connect(lambda _=False, a=attr: self._switch_to(a))
            self._nav_btns.append((btn, key))
            grid.addWidget(btn, divmod(i, 2)[0], divmod(i, 2)[1])
        layout.addLayout(grid)
        layout.addStretch(1)
        return card

    def retranslate(self) -> None:
        self._title_label.setText(tr("app"))
        self._subtitle_label.setText(tr("overview_subtitle"))
        for head, key in self._head_labels:
            head.setText(tr(key))
        for btn, key in self._nav_btns:
            btn.setText(tr(key))

    def _switch_to(self, attr: str) -> None:
        window = self.window()
        widget = getattr(window, attr, None)
        if widget is not None:
            idx = window.stackedWidget.indexOf(widget)
            if idx >= 0:
                window.stackedWidget.setCurrentIndex(idx)
