"""Fine-tune task adapters (wave G-3): map JobSpec onto real training commands.

Four tasks are exposed by the UI:

- ``sensevoice_singing`` — FunASR ``funasr.bin.train_ds`` (SenseVoice CTC),
  real epochs=1 runs (best supported; G-4).
- ``gsv_tts`` — GPT-SoVITS v2Pro ``s1_train.py`` (a tmp config yaml is written
  into the output dir with the requested lr/batch/epochs); data prep (the
  prepare-datasets chain described in the upstream README) is NOT automated,
  so a real run also needs a prepared data dir.
- ``campplus`` — 3D-Speaker CAM++ recipe (``speakerlab/bin/train.py`` via
  ``torch.distributed.run``) with a resolved tmp yaml config; needs a speaker
  CSV (ID,path,spk) plus the pretrained CAM++ model directory for sanity.
- ``game`` — openvpi/GAME ``train.py``; needs a binarized GAME dataset
  (``train/valid.info.npz`` + ``train/valid.data.hdf5``) produced by the
  upstream ``binarize.py``.

All adapters pre-validate the data dir/available sources and fail *before* any
process is spawned.
"""

from __future__ import annotations

import csv
import os
import shlex
import subprocess
import sys
from pathlib import Path

from core.config import PROJECT_ROOT
from core.exceptions import TrainingError

TASK_IDS = ("sensevoice_singing", "gsv_tts", "campplus", "game")

_PARSE_PATTERNS: dict[str, tuple[str, ...]] = {
    "sensevoice_singing": (
        r"epoch:\s*(\d+)/(\d+)",
        r"epoch:\s*(\d+)",
    ),
    "gsv_tts": (
        r"epoch\s*(\d+)/(\d+)",
        r"epoch:\s*(\d+)",
    ),
    "campplus": (
        r"Epoch:\s*\[\s*(\d+)",
        r"epoch:\s*(\d+)/(\d+)",
    ),
    "game": (
        r"Epoch\s+(\d+)",
        r"epoch:\s*(\d+)/(\d+)",
    ),
}


class JobCommand:
    """A ready-to-run training command (argv + cwd + env + parse patterns)."""

    def __init__(self, task: str, argv: list[str], cwd: str, env: dict | None = None,
                 parse_patterns: tuple[str, ...] = ()) -> None:
        self.task = task
        self.argv = list(argv)
        self.cwd = str(cwd)
        self.env = dict(os.environ) | dict(env or {})
        self.parse_patterns = tuple(parse_patterns)

    @property
    def pid_source(self) -> str:
        return Path(self.argv[0]).name

    def command_line(self) -> str:
        return " ".join(shlex.quote(a) for a in self.argv)

    def run_dry(self) -> None:
        """Validate what we can without spawning (used by tests/UI checks)."""
        executable = Path(self.argv[0])
        if not executable.is_file() and shlex.quote(self.argv[0]) not in ("python", "python.exe"):
            # module-style entries have argv[0] == python; data dirs checked
            # separately by the adapters.
            raise TrainingError(f"training executable not found: {self.argv[0]}")


def _venv_python() -> str:
    return str(Path(sys.executable).resolve())


def _require_dir(path: str | Path, label: str) -> Path:
    value = Path(path)
    if not value.is_dir():
        raise TrainingError(f"{label} directory not found: {value}")
    return value


def _is_cuda(device: str | None = "cuda") -> bool:
    return str(device or "cuda").strip().lower() not in ("cpu", "directml")


def _device_env(device: str | None = "cuda") -> dict[str, str]:
    """Pin the training subprocess to a physical GPU.

    The app's auto backend can prefer rocm/zluda (AMD), but fine-tuning must
    land on the CUDA NVIDIA GPU.  ``CUDA_VISIBLE_DEVICES=0`` makes torch/funasr/
    lightning use the CUDA GPU at index 0; ``-1`` forces CPU.  ``rocm`` and
    ``zluda`` are CUDA-visible too, so they map to index 0 as well.
    """
    dev = (device or "cuda").strip().lower()
    if dev in ("cpu", "directml"):
        return {"CUDA_VISIBLE_DEVICES": "-1"}
    return {"CUDA_VISIBLE_DEVICES": "0"}


def build_command(task: str, data_dir: str, params: dict | None = None,
                    output_dir: str = "", device: str = "cuda") -> JobCommand:
    """Build the command for *task*; raises TrainingError for unavailable ones.

    ``device`` defaults to ``cuda`` so fine-tuning always targets the CUDA GPU
    (pinned via ``CUDA_VISIBLE_DEVICES`` and per-entry GPU args), regardless of
    the app's auto backend preference (which may pick rocm/zluda on AMD boxes).
    """
    params = dict(params or {})
    epochs = int(params.get("epochs", 100))
    batch_size = int(params.get("batch", 4))
    lr = float(params.get("lr", 5e-4))
    out = Path(output_dir).resolve() if output_dir else Path.cwd() / "outputs" / "training"
    data = _require_dir(data_dir, "training data")
    out.mkdir(parents=True, exist_ok=True)

    if task == "sensevoice_singing":
        return _sensevoice(data, params, epochs, batch_size, lr, out, device)
    if task == "gsv_tts":
        return _gsv_tts(data, params, epochs, batch_size, lr, out, device)
    if task == "campplus":
        return _campplus(data, params, epochs, batch_size, lr, out, device)
    if task == "game":
        return _game(data, params, epochs, batch_size, lr, out, device)
    raise TrainingError(f"unknown training task: {task!r}")


def _sensevoice(data: Path, params, epochs: int, batch_size: int, lr: float, out: Path,
                  device: str = "cuda") -> JobCommand:
    train_list = data / "train.jsonl"
    valid_list = data / "val.jsonl"
    if not train_list.is_file():
        raise TrainingError(f"sensevoice training data missing: {train_list}")
    args = [
        _venv_python(), "-m", "funasr.bin.train_ds",
        "++model=" + str(_sensevoice_model_dir()),
        "++train_data_set_list=" + str(train_list),
        f"++valid_data_set_list={valid_list}" if valid_list.is_file() else "++valid_data_set_list=",
        "++dataset_conf.data_split_num=1",
        "++dataset_conf.batch_sampler=BatchSampler",
        "++dataset_conf.batch_type=example",
        "++dataset_conf.batch_size=3000",
        "++dataset_conf.num_workers=0",
        f"++train_conf.max_epoch={epochs}",
        f"++device={'cuda:0' if _is_cuda(device) else 'cpu'}",
        "++train_conf.resume=false",
        "++train_conf.log_interval=1",
        "++train_conf.save_checkpoint_interval=99999",
        "++train_conf.validate_interval=99999",
        f"++optim_conf.lr={lr:g}",
        "++scheduler_conf.warmup_steps=10",
        "++output_dir=" + str(out),
    ]
    args = [a for a in args if a]  # drop empty valid arg
    return JobCommand(task="sensevoice_singing", argv=args, cwd=str(PROJECT_ROOT),
                      env=_device_env(device),
                      parse_patterns=_PARSE_PATTERNS["sensevoice_singing"])


def _sensevoice_model_dir() -> Path:
    from core.config import ModelPaths

    mp = ModelPaths.from_settings()
    value = getattr(mp, "sensevoice", "") or ""
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_dir() or not (path / "model.pt").is_file() or not (path / "tokens.json").is_file():
        raise TrainingError(f"sensevoice model dir not ready: {path}")
    return path


def _gsv_tts(data: Path, params, epochs: int, batch_size: int, lr: float, out: Path,
                device: str = "cuda") -> JobCommand:
    """s1_train.py with a generated tmp yaml (webui-style injection)."""
    from core.config import ModelPaths

    gsv_names = ("GPT-SoVITS-v2pro-20250604-nvidia50", "GPT_SoVITS")
    gsv_root = PROJECT_ROOT / gsv_names[0] / gsv_names[1]
    pretrained_path = PROJECT_ROOT / gsv_names[0] / gsv_names[1] / "pretrained_models" / \
        "gsv-v2final-pretrained" / "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
    berts = PROJECT_ROOT / gsv_names[0] / gsv_names[1] / "pretrained_models" / "chinese-roberta-wwm-ext-large"

    semantic_path = data / "6-name2semantic.tsv"
    phoneme_path = data / "2-name2text.txt"
    if not semantic_path.is_file() or not phoneme_path.is_file():
        raise TrainingError(
            f"gsv_tts data not prepared: expected {semantic_path} and {phoneme_path} "
            "(run the GSMV prepare_datasets chain first)"
        )
    if not pretrained_path.is_file():
        raise TrainingError(f"gsv pretrained s1 checkpoint not found: {pretrained_path}")

    tmp_config = out / "gsv_s1_train.yaml"
    tmp_config.write_text(
        "\n".join([
            "pretrained_s1: %s" % str(pretrained_path).replace("\\", "/"),
            "train_semantic_path: %s" % str(semantic_path).replace("\\", "/"),
            "train_phoneme_path: %s" % str(phoneme_path).replace("\\", "/"),
            "bert_base_path: %s" % str(berts).replace("\\", "/"),
            "output_dir: %s" % str(out / "logs_s1_v2Pro").replace("\\", "/"),
            "train: {",
            "  batch_size: %d" % batch_size,
            "  epochs: %d" % epochs,
            "  precision: 16-mixed",
            "  if_save_latest: false",
            "  exp_name: gsv_ft",
            "  half_weights_save_dir: %s" % str(out / "s1_half").replace("\\", "/"),
            "}",
            "optimizer: {",
            "  lr: %g" % lr,
            "  lr_init: 0.00001",
            "  lr_end: 0.0001",
            "  warmup_steps: 10",
            "  decay_steps: 2000",
            "}",
            "data: { max_sec: 54, num_workers: 0 }",
            "model: { n_layer: 24 }",
        ]),
        encoding="utf-8",
    )
    runtime = gsv_root.parent / "runtime" / "python.exe"
    python = str(runtime if runtime.is_file() else _venv_python())
    argv = [
        python, "-s", str(gsv_root / "s1_train.py"),
        "--config_file", str(tmp_config),
    ]
    env = {"hz": "25hz"} | _device_env(device)
    return JobCommand(task="gsv_tts", argv=argv, cwd=str(gsv_root.parent), env=env,
                      parse_patterns=_PARSE_PATTERNS["gsv_tts"])



# ---------------------------------------------------------------------------
# CAM++ (3D-Speaker) and GAME (openvpi/GAME) training adapters
# ---------------------------------------------------------------------------

_VENDORED = PROJECT_ROOT / "vendored"


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _count_campplus_speakers(csv_path: Path) -> int:
    """Read a 3D-Speaker speaker CSV (ID,path,spk) and count unique speakers."""
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, skipinitialspace=True)
            fields = {str(x).strip() for x in (reader.fieldnames or [])}
            if "ID" not in fields:
                raise TrainingError(
                    f"campplus training CSV must have an 'ID' column: {csv_path}"
                )
            if "path" not in fields or "spk" not in fields:
                raise TrainingError(
                    f"campplus training CSV needs 'path' and 'spk' columns: {csv_path}"
                )
            spks = {row.get("spk") for row in reader if row.get("spk")}
    except TrainingError:
        raise
    except Exception as exc:  # noqa: BLE001 - surface a clean TrainingError
        raise TrainingError(f"campplus training CSV unreadable: {csv_path}: {exc}") from exc
    if not spks:
        raise TrainingError(f"campplus training CSV has no 'spk' rows: {csv_path}")
    return len(spks)


def _find_campplus_csv(data: Path) -> Path:
    """Locate the training speaker CSV under the data dir (or the file itself)."""
    if data.is_file() and data.suffix.lower() == ".csv":
        return data
    if not data.is_dir():
        raise TrainingError(f"campplus training data directory not found: {data}")
    csvs = sorted(p for p in data.glob("*.csv"))
    if not csvs:
        raise TrainingError(
            f"campplus training data missing: expected a speaker CSV "
            f"(columns ID,path,spk) under {data}"
        )
    if "train.csv" in {c.name for c in csvs}:
        return data / "train.csv"
    if len(csvs) == 1:
        return csvs[0]
    raise TrainingError(
        f"campplus training data ambiguous: multiple CSVs under {data}; "
        f"put the training set in train.csv"
    )


def _yaml_scalar_str(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _resolve_campplus_placeholder(value, mapping: dict) -> object:
    """Recursively substitute ``<name>`` placeholders with typed values.

    Exact ``<name>`` matches become typed Python values so the dumped yaml
    keeps int/float/bool semantics; embedded occurrences inside a longer
    string (e.g. ``<exp_dir>/models``) are replaced textually.
    """
    if isinstance(value, dict):
        return {k: _resolve_campplus_placeholder(v, mapping) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_campplus_placeholder(v, mapping) for v in value]
    if isinstance(value, str):
        if value.startswith("<") and value.endswith(">"):
            key = value[1:-1]
            if key in mapping:
                return mapping[key]
        if value in mapping:
            return mapping[value]
        if "<" in value:
            out = value
            for key, val in mapping.items():
                if "<%s>" % key in out:
                    out = out.replace("<%s>" % key, _yaml_scalar_str(val))
            return out
    return value


def _write_campplus_config(recipe_yaml: Path, mapping: dict, out: Path) -> Path:
    import yaml

    with open(recipe_yaml, "r", encoding="utf-8") as f:
        conf = yaml.safe_load(f)
    # Override top-level keys directly (e.g. num_epoch, lr, batch_size, data,
    # exp_dir are concrete scalars in the recipe, not '<...>' placeholders).
    for key in (
        "data", "noise", "reverb", "exp_dir", "wav_len", "sample_rate", "aug_prob",
        "speed_pertub", "lr", "min_lr", "num_epoch", "batch_size", "num_workers",
        "fbank_dim", "embedding_size", "num_classes", "save_epoch_freq", "log_batch_freq",
    ):
        if key in mapping:
            conf[key] = mapping[key]
    resolved = _resolve_campplus_placeholder(conf, mapping)
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(resolved, f, sort_keys=False)
    return out


def _campplus(data: Path, params, epochs: int, batch_size: int, lr: float, out: Path,
                 device: str = "cuda") -> JobCommand:
    """3D-Speaker CAM++ recipe via ``speakerlab/bin/train.py``.

    The recipe trains a speaker-embedding model from a speaker CSV
    (``ID,path,spk``).  The upstream ``conf/cam++.yaml`` is read and its
    ``<...>`` placeholders are resolved into a tmp yaml in the output dir so
    the requested ``epochs``/``batch``/``lr`` are actually honoured.  A
    real run needs a prepared speaker corpus plus CUDA (``torch.distributed.run``
    spawns a NCCL worker).
    """
    from core.config import ModelPaths

    repo = _VENDORED / "3D-Speaker"
    train_script = repo / "speakerlab" / "bin" / "train.py"
    recipe_yaml = repo / "egs" / "3dspeaker" / "sv-cam++" / "conf" / "cam++.yaml"
    if not train_script.is_file() or not recipe_yaml.is_file():
        raise TrainingError(
            "CAM++ training source (vendored/3D-Speaker) is not present: "
            "expected speakerlab/bin/train.py and egs/3dspeaker/sv-cam++/conf/cam++.yaml"
        )

    csv_path = _find_campplus_csv(data)
    num_spk = _count_campplus_speakers(csv_path)

    # sanity-check the pretrained CAM++ model dir so a missing download fails
    # before any process is spawned (the recipe itself trains from scratch).
    mp = ModelPaths.from_settings()
    campplus_dir = _resolve_project_path(getattr(mp, "campplus", "") or "")
    if not campplus_dir.is_dir() or not (campplus_dir / "campplus_cn_common.bin").is_file():
        raise TrainingError(f"campplus pretrained model not ready: {campplus_dir}")

    save_freq = max(1, int(epochs))
    mapping = {
        "data": str(csv_path).replace("\\", "/"),
        "noise": "",
        "reverb": "",
        "exp_dir": str(out).replace("\\", "/"),
        "wav_len": 3.0,
        "sample_rate": 16000,
        "aug_prob": 0.0,
        "speed_pertub": False,
        "lr": float(lr),
        "min_lr": 1e-4,
        "num_epoch": int(epochs),
        "batch_size": int(batch_size),
        "num_workers": 0,
        "fbank_dim": 80,
        "embedding_size": 512,
        "num_classes": int(num_spk),
        "save_epoch_freq": save_freq,
        "log_batch_freq": 1,
    }
    tmp_config = out / "campplus_train.yaml"
    _write_campplus_config(recipe_yaml, mapping, tmp_config)

    python = _venv_python()
    master_port = str(int.from_bytes(os.urandom(2), "little") + 20000)
    argv = [
        python, "-m", "torch.distributed.run",
        "--nproc_per_node=1", "--master_port", master_port,
        str(train_script).replace("\\", "/"),
        "--config", str(tmp_config).replace("\\", "/"),
        "--gpu", "0" if _is_cuda(device) else "-1",
    ]
    env = {"PYTHONPATH": str(repo)} | _device_env(device)
    return JobCommand(task="campplus", argv=argv, cwd=str(repo), env=env,
                      parse_patterns=_PARSE_PATTERNS["campplus"])


def _game(data: Path, params, epochs: int, batch_size: int, lr: float, out: Path,
              device: str = "cuda") -> JobCommand:
    """openvpi/GAME training via the top-level ``train.py`` CLI.

    GAME is step-based (``training.trainer.unit=step``) and trains from a
    binarized dataset (the upstream ``binarize.py`` output).  ``epochs``/
    ``batch``/``lr`` are mapped onto the config through ``--override``
    dotlist options.  A real run therefore needs a binarized GAME dataset
    plus a Lightning/PyTorch runtime matching the upstream `requirements.txt`.
    """
    from core.config import ModelPaths

    repo = _VENDORED / "GAME"
    train_script = repo / "train.py"
    cfg = repo / "configs" / "midi.yaml"
    if not train_script.is_file() or not cfg.is_file():
        raise TrainingError(
            "GAME training source (vendored/GAME) is not present: "
            "expected train.py and configs/midi.yaml"
        )

    required = (
        "train.info.npz", "train.data.hdf5",
        "valid.info.npz", "valid.data.hdf5",
        "feature.yaml", "lang_map.json",
    )
    missing = [name for name in required if not (data / name).is_file()]
    if missing:
        raise TrainingError(
            f"GAME training data is not binarized: missing {', '.join(missing)} "
            f"under {data}; run upstream binarize.py with a DiffSinger dataset first"
        )

    # sanity-check the GAME model suite (ONNX weights) so a missing download
    # fails before any process is spawned.
    mp = ModelPaths.from_settings()
    game_dir = _resolve_project_path(getattr(mp, "game", "") or "")
    if not game_dir.is_dir():
        raise TrainingError(f"GAME pretrained model dir not ready: {game_dir}")

    exp_name = "game_ft"
    work_dir = out
    argv = [
        _venv_python(), str(train_script).replace("\\", "/"),
        "--config", str(cfg).replace("\\", "/"),
        "--exp-name", exp_name,
        "--work-dir", str(work_dir).replace("\\", "/"),
        "--override", f"binarizer.data_dir={str(data.resolve()).replace(chr(92), '/')}",
        "--override", f"training.trainer.max_epochs={int(epochs)}",
        "--override", f"training.trainer.max_steps={max(1, int(epochs) * 1000)}",
        "--override", f"training.dataloader.max_batch_size={int(batch_size)}",
        "--override", f"training.optimizer.kwargs.lr={float(lr):g}",
        "--override", f"training.trainer.accelerator={'gpu' if _is_cuda(device) else 'cpu'}",
        "--override", "training.trainer.devices=1",
    ]
    env = {"PYTHONPATH": str(repo)} | _device_env(device)
    return JobCommand(task="game", argv=argv, cwd=str(repo), env=env,
                      parse_patterns=_PARSE_PATTERNS["game"])


# ---------------------------------------------------------------------------
# progress parsing helpers
# ---------------------------------------------------------------------------


def parse_progress(line: str, patterns: tuple[str, ...], total_epochs: int) -> float | None:
    """Extract an epoch -> fraction in [0, 1] from a log line (or None)."""
    import re

    for pattern in patterns:
        match = re.search(pattern, line)
        if not match:
            continue
        try:
            epoch = int(match.group(1))
            total = int(match.group(2)) if match.lastindex and match.lastindex >= 2 else max(total_epochs, 1)
            return min(1.0, epoch / max(total, 1))
        except (IndexError, TypeError, ValueError):
            return None
    return None


__all__ = [
    "TASK_IDS",
    "JobCommand",
    "build_command",
    "parse_progress",
]
