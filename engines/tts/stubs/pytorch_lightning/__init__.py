"""Minimal pytorch_lightning stub for the GSV TTS reference code.

Only ``LightningModule`` is used by ``AR/models/t2s_lightning_module.py``
(and only for ``is_train=False`` inference); the real Lightning package is a
heavy training dependency that we intentionally do not install. Placed on
``sys.path`` ahead of the reference repo by ``engines.tts.voice_clone``.
"""

from __future__ import annotations

import torch


class LightningModule(torch.nn.Module):
    def __init__(self):
        super().__init__()


def seed_everything(seed: int, workers: bool = False):
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    return seed, seed
