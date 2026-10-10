"""Fine-tune training-source tests for the CAM++ / GAME adapters (wave G-3).

The upstream training repos (3D-Speaker for CAM++, openvpi/GAME for GAME) are
cloned into ``vendored/`` which is kept OUTSIDE git (see .gitignore).  These
tests assert that ``build_command`` maps each task onto the real vendored
training entry point and that missing data / a missing vendored checkout fail
clearly (``TrainingError``) before any process is spawned.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import PROJECT_ROOT
from core.exceptions import TrainingError
from engines.training.adapters import JobCommand, build_command

VENDORED = PROJECT_ROOT / "vendored"
CAMPPUS_REPO = VENDORED / "3D-Speaker"
GAME_REPO = VENDORED / "GAME"


def _head_sha(repo: Path) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True, text=True, timeout=30,
    )
    return out.stdout.strip() if out.returncode == 0 else ""


@pytest.fixture
def campplus_data(tmp_path):
    d = tmp_path / "campplus_data"
    d.mkdir()
    with open(d / "train.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ID", "path", "spk"])
        w.writerows([
            ["a", "wav/a.wav", "spk1"],
            ["b", "wav/b.wav", "spk2"],
            ["c", "wav/c.wav", "spk2"],
        ])
    return d


@pytest.fixture
def game_data(tmp_path):
    d = tmp_path / "game_data"
    d.mkdir()
    for name in (
        "train.info.npz", "train.data.hdf5",
        "valid.info.npz", "valid.data.hdf5",
        "feature.yaml", "lang_map.json",
    ):
        (d / name).write_text("", encoding="utf-8")
    return d


def test_vendored_sources_cloned_and_committed():
    """The training entry points must exist in the gitignored vendored checkouts."""
    if not CAMPPUS_REPO.is_dir() or not GAME_REPO.is_dir():
        pytest.skip("vendored training sources not cloned in this workspace")
    assert (CAMPPUS_REPO / "speakerlab" / "bin" / "train.py").is_file()
    assert (CAMPPUS_REPO / "egs" / "3dspeaker" / "sv-cam++" / "conf" / "cam++.yaml").is_file()
    assert (GAME_REPO / "train.py").is_file()
    assert (GAME_REPO / "configs" / "midi.yaml").is_file()
    # Record/reproducibility: these are real git checkouts with a pinned HEAD.
    assert _head_sha(CAMPPUS_REPO), "3D-Speaker is not a git checkout"
    assert _head_sha(GAME_REPO), "GAME is not a git checkout"


def test_campplus_build_command(tmp_path, campplus_data):
    if not (CAMPPUS_REPO / "speakerlab" / "bin" / "train.py").is_file():
        pytest.skip("vendored/3D-Speaker not present")
    out = tmp_path / "out"
    cmd = build_command("campplus", str(campplus_data), {"epochs": 1, "lr": 2e-4, "batch": 4}, str(out))
    assert isinstance(cmd, JobCommand)
    cmd.run_dry()  # argv[0] is a real python; data/pretrained pre-validated
    text = " ".join(cmd.argv)
    assert "torch.distributed.run" in text
    assert "speakerlab/bin/train.py" in text
    cfg_path = Path(cmd.argv[cmd.argv.index("--config") + 1])
    assert cfg_path.is_file()
    cfg_text = cfg_path.read_text(encoding="utf-8")
    assert "num_epoch: 1" in cfg_text
    assert "batch_size: 4" in cfg_text
    assert "lr: 0.0002" in cfg_text
    assert "speed_pertub: false" in cfg_text
    assert "aug_prob: 0.0" in cfg_text
    assert str(out / "models").replace("\\", "/") in cfg_text
    assert cmd.cwd == str(CAMPPUS_REPO)


def test_game_build_command(tmp_path, game_data):
    if not (GAME_REPO / "train.py").is_file():
        pytest.skip("vendored/GAME not present")
    out = tmp_path / "out"
    cmd = build_command("game", str(game_data), {"epochs": 1, "lr": 2e-4, "batch": 4}, str(out))
    assert isinstance(cmd, JobCommand)
    cmd.run_dry()
    text = " ".join(cmd.argv)
    assert str(GAME_REPO / "train.py").replace("\\", "/") in text
    assert "--override training.trainer.max_epochs=1" in text
    assert "--override training.dataloader.max_batch_size=4" in text
    assert "--override training.optimizer.kwargs.lr=0.0002" in text
    assert "--override binarizer.data_dir=" in text
    assert cmd.cwd == str(GAME_REPO)


def test_campplus_missing_data_raises_early(tmp_path):
    if not (CAMPPUS_REPO / "speakerlab" / "bin" / "train.py").is_file():
        pytest.skip("vendored/3D-Speaker not present")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(TrainingError) as exc:
        build_command("campplus", str(empty), {"epochs": 1}, str(tmp_path / "out"))
    assert ("CSV" in str(exc.value)) or ("speaker" in str(exc.value))


def test_game_missing_data_raises_early(tmp_path):
    if not (GAME_REPO / "train.py").is_file():
        pytest.skip("vendored/GAME not present")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(TrainingError) as exc:
        build_command("game", str(empty), {"epochs": 1}, str(tmp_path / "out"))
    assert "binarized" in str(exc.value)


# -- CUDA device targeting (training must land on the CUDA GPU) ---------------


def test_device_env_and_is_cuda():
    from engines.training.adapters import _device_env, _is_cuda

    assert _is_cuda("cuda") and _is_cuda("rocm") and _is_cuda("zluda")
    assert not _is_cuda("cpu") and not _is_cuda("directml")
    assert _device_env("cuda") == {"CUDA_VISIBLE_DEVICES": "0"}
    assert _device_env("rocm") == {"CUDA_VISIBLE_DEVICES": "0"}
    assert _device_env("zluda") == {"CUDA_VISIBLE_DEVICES": "0"}
    assert _device_env("cpu") == {"CUDA_VISIBLE_DEVICES": "-1"}
    assert _device_env("directml") == {"CUDA_VISIBLE_DEVICES": "-1"}


def test_campplus_command_targets_cuda_by_default(tmp_path, campplus_data):
    if not (CAMPPUS_REPO / "speakerlab" / "bin" / "train.py").is_file():
        pytest.skip("vendored/3D-Speaker not present")
    out = tmp_path / "out"
    cmd = build_command("campplus", str(campplus_data), {"epochs": 1}, str(out))
    assert cmd.env.get("CUDA_VISIBLE_DEVICES") == "0"
    assert cmd.argv[cmd.argv.index("--gpu") + 1] == "0"
    cpu = build_command("campplus", str(campplus_data), {"epochs": 1}, str(out), device="cpu")
    assert cpu.env.get("CUDA_VISIBLE_DEVICES") == "-1"
    assert cpu.argv[cpu.argv.index("--gpu") + 1] == "-1"


def test_game_command_targets_cuda_by_default(tmp_path, game_data):
    if not (GAME_REPO / "train.py").is_file():
        pytest.skip("vendored/GAME not present")
    out = tmp_path / "out"
    cmd = build_command("game", str(game_data), {"epochs": 1}, str(out))
    assert cmd.env.get("CUDA_VISIBLE_DEVICES") == "0"
    assert "--override training.trainer.accelerator=gpu" in " ".join(cmd.argv)
    assert "--override training.trainer.devices=1" in " ".join(cmd.argv)
    cpu = build_command("game", str(game_data), {"epochs": 1}, str(out), device="cpu")
    assert cpu.env.get("CUDA_VISIBLE_DEVICES") == "-1"
    assert "--override training.trainer.accelerator=cpu" in " ".join(cpu.argv)
