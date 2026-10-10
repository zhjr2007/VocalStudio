"""Training engine package (wave G): subprocess fine-tune jobs + adapters.

- :class:`engines.training.job_runner.TrainingJob` — spec -> command ->
  subprocess -> tailed log/progress -> cancel (terminate within 2 s).
- :func:`engines.training.adapters.build_command` — maps task + params onto
  the real vendor training commands (FunASR SenseVoice / GPT-SoVITS s1;
  CAM++ via 3D-Speaker ``speakerlab/bin/train.py``, GAME via openvpi/GAME
  ``train.py``).
"""

from __future__ import annotations

from .adapters import TASK_IDS, build_command
from .job_runner import TrainingJob, TrainingJobSpec, run_training_job


class TrainingEngine:
    """Resolver for the training package (see job_runner.TrainingJob)."""


def get_engine():
    from engines.training.job_runner import TrainingJob

    return TrainingJob


engine_class = TrainingEngine

__all__ = [
    "TrainingEngine",
    "get_engine",
    "TrainingJob",
    "TrainingJobSpec",
    "run_training_job",
    "build_command",
    "TASK_IDS",
]
