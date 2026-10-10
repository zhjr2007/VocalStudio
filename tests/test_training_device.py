"""Training compute-device targeting: the page must default to cuda (the CUDA GPU)
and carry it through config()/start() so fine-tuning never lands on rocm/cpu by
accident after the backend auto-detect order change."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt5.QtWidgets import QApplication

try:
    _app = QApplication.instance() or QApplication([])
except Exception:
    _app = None


def _page():
    from gui.pages.training_page import TrainingPage
    return TrainingPage()


@pytest.fixture
def app():
    from PyQt5.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_training_page_device_default_is_cuda(app):
    page = _page()
    try:
        assert page.device() == "cuda"
        cfg = page.config()
        assert cfg["device"] == "cuda"
        # device survives round-trip
        page.set_device("cpu")
        assert page.device() == "cpu"
        assert page.config()["device"] == "cpu"
        page.set_device("cuda")
        assert page.config()["device"] == "cuda"
    finally:
        page.deleteLater()


def test_sensevoice_command_env_pins_cuda():
    from engines.training.adapters import build_command
    import tempfile, os
    from pathlib import Path

    data = Path(tempfile.mkdtemp())
    (data / "train.jsonl").write_text("", encoding="utf-8")
    # model dir must be present; reuse the real configured sensevoice path if available
    from core.config import ModelPaths, PROJECT_ROOT
    mp = ModelPaths.from_settings()
    sv = getattr(mp, "sensevoice", "") or ""
    sv_path = Path(sv)
    if not sv_path.is_absolute():
        sv_path = PROJECT_ROOT / sv_path
    if not (sv_path / "model.pt").is_file():
        pytest.skip("sensevoice model dir not ready")
    out = Path(tempfile.mkdtemp())
    cmd = build_command("sensevoice_singing", str(data), {"epochs": 1}, str(out))
    assert cmd.env.get("CUDA_VISIBLE_DEVICES") == "0"
    cpu = build_command("sensevoice_singing", str(data), {"epochs": 1}, str(out), device="cpu")
    assert cpu.env.get("CUDA_VISIBLE_DEVICES") == "-1"
