"""Text2Semantic (GPT) decoder wrapper (wave E-3).

Owns the AR model that predicts semantic tokens from phonemes + BERT features
(+ optional prompt semantic). The heavy reference imports are fully deferred to
the first :meth:`load`.
"""

from __future__ import annotations

import torch


class T2SModel:
    def __init__(self) -> None:
        self.model = None
        self.device = torch.device("cpu")
        self.is_half = False
        self.hz = 50
        self.max_sec = 0.0
        self.mute_token = 486

    def load(self, ckpt_path, device="cpu", is_half=False) -> dict:
        """Load the T2S checkpoint (``s1bert25hz-...-step=... .ckpt``).

        Returns e.g. ``{"hz": 50, "max_sec": <config value>}``.
        """
        from engines.tts._ref import ensure_reference_imported

        ensure_reference_imported()
        from AR.models.t2s_lightning_module import Text2SemanticLightningModule

        self.device = torch.device(str(device))
        self.is_half = bool(is_half)
        state = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)
        config = state["config"]
        self.max_sec = float(config["data"]["max_sec"])
        self.model = Text2SemanticLightningModule(config, "****", is_train=False)
        self.model.load_state_dict(state["weight"])
        self.model = self.model.eval().to(self.device)
        if self.is_half and str(self.device) != "cpu":
            self.model = self.model.half()
        self.hz = 50
        return {"hz": self.hz, "max_sec": self.max_sec}

    def infer(self, all_phoneme_ids, all_phoneme_lens, prompt, all_bert_features,
              *, top_k=15, top_p=1.0, temperature=1.0, repetition_penalty=1.35):
        """Run the (parallel) panel inference; returns ``(semantic_list, idx_list)``.

        All tensors must already live on the right device/dtype (the caller
        prepares them, like the reference ``TTS.run`` does via ``to_batch``).
        ``prompt`` may be None (no reference text supplied).
        """
        if self.model is None:
            raise RuntimeError("T2SModel.load() must be called before infer().")
        return self.model.model.infer_panel_batch_infer(
            all_phoneme_ids,
            all_phoneme_lens,
            prompt,
            all_bert_features,
            top_k=int(top_k),
            top_p=float(top_p),
            temperature=float(temperature),
            early_stop_num=float(self.hz * self.max_sec),
            max_len=all_bert_features.shape[-1] if hasattr(all_bert_features, "shape") else 256,
            repetition_penalty=float(repetition_penalty),
        )

    def unload(self) -> None:
        self.model = None
