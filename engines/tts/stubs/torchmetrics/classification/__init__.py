"""Minimal torchmetrics stub for the GSV TTS reference code.

Only ``torchmetrics.classification.MulticlassAccuracy`` is imported (by
``AR/models/t2s_model.py``) and its statistics are only used during training;
inference never calls compute().
"""

from __future__ import annotations

import torch


class MulticlassAccuracy(torch.nn.Module):
    def __init__(self, num_classes: int | None = None, *args, **kwargs):
        super().__init__()

    def update(self, *args, **kwargs):
        pass

    def compute(self, *args, **kwargs):
        return torch.tensor(0.0)
