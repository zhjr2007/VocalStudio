"""CNHuBERT content encoder (wave E-3, thin wrapper over the reference model).

The reference chain turns the 16 kHz reference audio into HuBERT features
(``last_hidden_state`` transposed) and then into VQ codebook indices via
``vits.extract_latent``; this wrapper only owns the encoder part.
"""

from __future__ import annotations

import torch


class HubertEncoder:
    def __init__(self) -> None:
        self.model = None
        self.device = torch.device("cpu")
        self.is_half = False

    def load(self, base_path, device="cpu", is_half=False) -> None:
        """Load CNHuBERT from a huggingface-style dir (pytorch_model.bin)."""
        from engines.tts._ref import ensure_reference_imported

        ensure_reference_imported()
        from feature_extractor.cnhubert import CNHubert

        self.device = torch.device(str(device))
        self.is_half = bool(is_half)
        self.model = CNHubert(str(base_path)).eval().to(self.device)
        if self.is_half and str(self.device) != "cpu":
            self.model = self.model.half()

    def extract(self, wav16k: torch.Tensor) -> torch.Tensor:
        """Features for a 16 kHz mono tensor shaped ``[1, N]``.

        Returns ``[1, hidden, T]`` (transposed last_hidden_state), matching
        what ``SovitsModel.extract_latent`` digests.
        """
        if self.model is None:
            raise RuntimeError("HubertEncoder.load() must be called before extract().")
        if wav16k.dim() == 1:
            wav16k = wav16k.unsqueeze(0)
        if self.is_half:
            wav16k = wav16k.half()
        with torch.no_grad():
            return self.model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)

    def unload(self) -> None:
        self.model = None
