"""Offscreen/unit + real finetune pytest suite for the training page (wave G)."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

import torch  # noqa: F401 (must load before PyQt5 on this host)

from PyQt5.QtWidgets import QApplication

from gui.pages.training_page import (
    BATCH_DEFAULT,
    BATCH_MAX,
    BATCH_MIN,
    EPOCHS_DEFAULT,
    EPOCHS_MAX,
    EPOCHS_MIN,
    LR_DEFAULT,
    LR_MAX,
    LR_MIN,
    TrainingPage,
)


@pytest.fixture(scope="module")
def app(qapp) -> QApplication:
    return qapp


def _dummy_script(tmp_path: Path, *, intervals: int = 3, sleep: float = 0.3, checkpoints: bool = True) -> Path:
    script = tmp_path / "dummy_train.py"
    body = "import os, sys, time\n"
    body += "out = sys.argv[1] if len(sys.argv) > 1 else '.'\n"
    body += f"for i in range(1, {intervals + 1}):\n"
    epoch_line = "    print('epoch: {{}}/{{}}'.format(i, {n}), flush=True)\n"
    body += epoch_line.format(n=intervals)
    body += f"    time.sleep({sleep})\n"
    body += '    if i == 1:\n'
    body += '        print("train loss 0.5", file=sys.stderr, flush=True)\n'
    if checkpoints:
        body += 'open(os.path.join(out, "model.pt.ep1"), "w").write("ckpt")\n'
    script.write_text(body, encoding="utf-8")
    return script


# -- G-1: page controls -------------------------------------------------------


def test_page_defaults(app):
    page = TrainingPage()
    try:
        assert page.task() == "sensevoice_singing"
        assert page.lr() == pytest.approx(LR_DEFAULT)
        assert page.batch() == BATCH_DEFAULT
        assert page.epochs() == EPOCHS_DEFAULT
        assert page.data_dir() == ""
        assert page._task_combo.count() == 4
    finally:
        page.deleteLater()


def test_task_combo_labels(app):
    from gui.i18n import zh_cn

    from engines.training.adapters import TASK_IDS
    from gui.pages.training_page import TASK_LABEL_KEYS

    page = TrainingPage()
    try:
        order = [page._task_combo.itemData(i) for i in range(page._task_combo.count())]
        assert order == list(TASK_IDS)
        for index, task in enumerate(TASK_IDS):
            assert page._task_combo.itemText(index) == zh_cn.TEXTS[TASK_LABEL_KEYS[task]]
    finally:
        page.deleteLater()


def test_param_ranges_and_roundtrip(app):
    page = TrainingPage()
    try:
        assert page._lr_spin.minimum() == pytest.approx(LR_MIN)
        assert page._lr_spin.maximum() == pytest.approx(LR_MAX)
        page.set_lr(LR_MIN)
        assert page.lr() == pytest.approx(LR_MIN)
        page.set_lr(LR_MAX)
        assert page.lr() == pytest.approx(LR_MAX)
        assert page._batch_spin.minimum() == BATCH_MIN and page._batch_spin.maximum() == BATCH_MAX
        page.set_batch(BATCH_MAX)
        assert page.batch() == BATCH_MAX
        assert page._epochs_spin.minimum() == EPOCHS_MIN and page._epochs_spin.maximum() == EPOCHS_MAX
        page.set_epochs(10)
        assert page.epochs() == 10
        page.set_task("game")
        assert page.task() == "game"
    finally:
        page.deleteLater()


def test_config_shape(app, tmp_path):
    page = TrainingPage()
    try:
        page.set_data_dir(str(tmp_path))
        cfg = page.config()
        assert set(cfg) == {"task", "data_dir", "lr", "batch", "epochs", "device"}
        assert cfg["data_dir"] == str(tmp_path)
    finally:
        page.deleteLater()


def test_start_blocked_without_data_dir(app):
    page = TrainingPage()
    try:
        assert page.start() is None
        assert "error" in page.log_text()
    finally:
        page.deleteLater()


# -- G-2: job runner (launch / tail / progress / cancel ----------------------


def test_job_launch_tail_and_progress(tmp_path, monkeypatch):
    import engines.training.job_runner as jr
    from engines.training.adapters import JobCommand
    from engines.training.job_runner import TrainingJob, TrainingJobSpec

    script = _dummy_script(tmp_path)
    out = tmp_path / "out"
    out.mkdir()

    def fake_command(task, data_dir, params=None, output_dir="", device="cuda"):
        return JobCommand(
            task="dummy",
            argv=[sys.executable, str(script), str(out)],
            cwd=str(tmp_path),
            parse_patterns=(r"epoch:\s*(\d+)/(\d+)",),
        )

    monkeypatch.setattr(jr, "build_command", fake_command)
    progress: list[float] = []
    logs: list[str] = []
    job = TrainingJob(
        TrainingJobSpec(task="dummy", data_dir=str(tmp_path), output_dir=str(out)),
        log_cb=logs.append,
        progress_cb=lambda f, stage: progress.append(f),
    )
    command = job.launch()
    job.stream(command)
    assert job.exit_code == 0
    assert progress == [1 / 3, 2 / 3, 1.0], progress
    assert any("train loss" in line for line in logs)
    assert any("pid=" in line for line in logs)


def test_job_cancel_terminates_below_deadline(tmp_path, monkeypatch):
    import engines.training.job_runner as jr
    from engines.training.adapters import JobCommand
    from engines.training.job_runner import TrainingJob, TrainingJobSpec

    script = _dummy_script(tmp_path, intervals=30, sleep=1.0)
    out = tmp_path / "out_cancel"
    out.mkdir()

    def fake_command(task, data_dir, params=None, output_dir="", device="cuda"):
        return JobCommand(
            task="dummy",
            argv=[sys.executable, str(script), str(out)],
            cwd=str(tmp_path),
            parse_patterns=("epoch",),
        )

    monkeypatch.setattr(jr, "build_command", fake_command)
    job = TrainingJob(TrainingJobSpec(task="dummy", data_dir=str(tmp_path), output_dir=str(out)))
    command = job.launch()
    assert job.handle is not None and job.handle.poll() is None
    t0 = time.time()
    job.cancel()
    elapsed = time.time() - t0
    assert elapsed <= jr.TERMINATE_TIMEOUT_SEC + 1.0
    assert job.handle.poll() is not None  # no zombie
    assert job.handle.wait(timeout=5.0) is not None


# -- G-3: adapters -------------------------------------------------------------


def test_adapter_sensevoice_dry_run(tmp_path):
    from engines.training.adapters import build_command

    data = tmp_path / "svdata"
    data.mkdir()
    (data / "train.jsonl").write_text("{}\n", encoding="utf-8")
    (data / "val.jsonl").write_text("{}\n", encoding="utf-8")
    command = build_command("sensevoice_singing", str(data), {"epochs": 1, "lr": 2e-4, "batch": 4}, str(tmp_path / "out"))
    text = " ".join(command.argv)
    assert "funasr.bin.train_ds" in text
    assert "++train_conf.max_epoch=1" in text
    assert "++optim_conf.lr=0.0002" in text
    assert "++dataset_conf.batch_type=example" in text
    assert "++output_dir=" in text


def test_adapter_gsv_tts_dry_run(tmp_path):
    from engines.training.adapters import build_command

    data = tmp_path / "gsvdata"
    data.mkdir()
    (data / "6-name2semantic.tsv").write_text("a\t1 2 3\n", encoding="utf-8")
    (data / "2-name2text.txt").write_text("a\tni3 hao3\t1 1\t你好\n", encoding="utf-8")
    out_dir = tmp_path / "out"
    command = build_command("gsv_tts", str(data), {"epochs": 1, "lr": 1e-4, "batch": 4}, str(out_dir))
    text = " ".join(command.argv)
    assert "s1_train.py" in text
    assert "--config_file" in text
    yaml_file = command.argv[-1]
    yaml_text = Path(yaml_file).read_text(encoding="utf-8")
    assert "epochs: 1" in yaml_text
    assert "batch_size: 4" in yaml_text
    assert "lr: 0.0001" in yaml_text
    assert str(out_dir / "logs_s1_v2Pro").replace("\\", "/") in yaml_text


@pytest.mark.parametrize("task", ["campplus", "game"])
def test_adapter_native_training_sources_dry_run(tmp_path, task):
    """CAM++/GAME now build a real training command from the vendored source.

    Previously these raised ``TrainingError("... not available")`` because the
    upstream training repo was not vendored; now the adapters produce a
    ``JobCommand`` running the vendored entry point (3D-Speaker
    ``speakerlab/bin/train.py`` / openvpi/GAME ``train.py``).  A missing
    vendored checkout still fails with a clear ``TrainingError`` before any
    process is spawned (covered by ``test_training_sources.py``).
    """
    from engines.training.adapters import JobCommand, build_command

    data = tmp_path / "data"
    data.mkdir()
    if task == "campplus":
        if not (ROOT / "vendored" / "3D-Speaker" / "speakerlab" / "bin" / "train.py").is_file():
            pytest.skip("vendored/3D-Speaker not present")
        (data / "train.csv").write_text(
            "ID,path,spk\na,test/a.wav,spk1\nb,test/b.wav,spk2\nc,test/c.wav,spk2\n",
            encoding="utf-8",
        )
    else:
        if not (ROOT / "vendored" / "GAME" / "train.py").is_file():
            pytest.skip("vendored/GAME not present")
        for name in (
            "train.info.npz", "train.data.hdf5",
            "valid.info.npz", "valid.data.hdf5",
            "feature.yaml", "lang_map.json",
        ):
            (data / name).write_text("", encoding="utf-8")
    out = tmp_path / "out"
    command = build_command(task, str(data), {"epochs": 1, "lr": 2e-4, "batch": 4}, str(out))
    assert isinstance(command, JobCommand)
    command.run_dry()  # validates argv[0] is a real program
    text = " ".join(command.argv)
    if task == "campplus":
        assert "torch.distributed.run" in text
        assert "speakerlab/bin/train.py" in text
    else:
        assert str(ROOT / "vendored" / "GAME" / "train.py").replace("\\", "/") in text


# -- G-4: real sensevoice epochs=1 finetune ------------------------------------


SENSEVOICE_DIR = ROOT / "experiments/funasr/iic/SenseVoiceSmall"


def test_g4_sensevoice_epochs1_real(tmp_path):
    if not (SENSEVOICE_DIR / "model.pt").is_file():
        pytest.skip("sensevoice model.pt not present")
    import subprocess

    data = tmp_path / "sv_g4"
    prep = subprocess.run(
        [sys.executable, str(ROOT / "scripts/prepare_sensevoice_finetune.py"), "--out", str(data)],
        capture_output=True, text=True, timeout=900, cwd=str(ROOT),
    )
    assert prep.returncode == 0, prep.stderr[-1500:]
    train_jsonl = (data / "train.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(train_jsonl) >= 3
    val_jsonl = (data / "val.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(val_jsonl) >= 1

    from engines.training.adapters import build_command, parse_progress

    out_dir = tmp_path / "g4_out"
    command = build_command("sensevoice_singing", str(data), {"epochs": 1, "lr": 2e-4, "batch": 4}, str(out_dir))
    import subprocess as sp

    progress: list[float] = []
    with sp.Popen(
        command.argv, cwd=command.cwd, env=command.env,
        stdout=sp.PIPE, stderr=sp.STDOUT, text=True, bufsize=1,
        creationflags=getattr(sp, "CREATE_NO_WINDOW", 0),
    ) as process:
        assert process.stdout is not None
        for line in process.stdout:
            fraction = parse_progress(line, command.parse_patterns, 1)
            if fraction is not None:
                progress.append(fraction)
            if "error" in line.lower() or "traceback" in line.lower():
                print(line.rstrip())
        exit_code = process.wait()
    assert exit_code == 0, f"training exited {exit_code}"
    ckpts = list(out_dir.rglob("model.pt*"))
    assert ckpts, "checkpoint must be written after epochs=1"
    # FunASR logs both "epoch: 1/1" (training) and a trailing "epoch: 0/1"
    # line; the max fraction must reach 1.0.
    assert progress and max(progress) >= 1.0, progress
    print(f"G-4: ckpt={ckpts[0].name} progress_end={progress[-1]:.2f}")
