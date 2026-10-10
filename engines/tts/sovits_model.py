"""VITS decoder wrapper (wave E-3): SynthesizerTrn for v1/v2/v2Pro/v2ProPlus.

The reference loader (``process_ckpt.get_sovits_version_from_path_fast`` +
``load_sovits_new``) adapts the checkpoint-config to whatever model family the
file is; this wrapper reuses it, then exposes the two ops the chain needs:
``extract_latent`` (prompt semantic codes) and ``decode`` (speech synthesis).
"""

from __future__ import annotations

import gc
import math

import torch


class SovitsModel:
    def __init__(self) -> None:
        self.model = None
        self.device = torch.device("cpu")
        self.is_half = False
        self.model_version = ""
        self.sampling_rate = 32000
        self.use_vocoder = False

    def load(self, ckpt_path, device="cpu", is_half=False) -> dict:
        """Load a SoVITS checkpoint; version is auto-detected from the file.

        Side effects mirror the reference loader: the v2Pro/v2ProPlus SV
        embedding model is initialized (needed for ``decode`` with ``sv_emb``).
        """
        from engines.tts._ref import ensure_reference_imported

        ensure_reference_imported()
        from module.models import SynthesizerTrn
        from process_ckpt import get_sovits_version_from_path_fast, load_sovits_new

        self.device = torch.device(str(device))
        self.is_half = bool(is_half)
        version, model_version, _lora = get_sovits_version_from_path_fast(str(ckpt_path))
        self.model_version = model_version
        state = load_sovits_new(str(ckpt_path))
        hps = state["config"]
        hps["model"]["semantic_frame_rate"] = "25hz"
        if "enc_p.text_embedding.weight" not in state["weight"]:
            hps["model"]["version"] = "v2"
        elif state["weight"]["enc_p.text_embedding.weight"].shape[0] == 322:
            hps["model"]["version"] = "v1"
        else:
            hps["model"]["version"] = "v2"
        if model_version not in {"v3", "v4"}:
            hps["model"]["version"] = model_version if "Pro" in model_version else hps["model"]["version"]

        self.sampling_rate = int(hps["data"]["sampling_rate"])
        filter_length = int(hps["data"]["filter_length"])
        segment_size = int(hps["train"]["segment_size"])
        hop_length = int(hps["data"]["hop_length"])
        n_speakers = int(hps["data"]["n_speakers"])

        self.model = SynthesizerTrn(
            filter_length // 2 + 1,
            segment_size // hop_length,
            n_speakers=n_speakers,
            **hps["model"],
        )
        self.model.load_state_dict(state["weight"], strict=False)
        self.model = self.model.eval().to(self.device)
        if self.is_half and str(self.device) != "cpu":
            self.model = self.model.half()
        self.use_vocoder = model_version in {"v3", "v4"}
        return {
            "version": model_version,
            "sampling_rate": self.sampling_rate,
            "use_vocoder": self.use_vocoder,
        }

    def extract_latent(self, hubert_feature: torch.Tensor) -> torch.Tensor:
        """VQ semantic codes for HuBERT features (prompt side)."""
        if self.model is None:
            raise RuntimeError("SovitsModel.load() must be called before extract_latent().")
        with torch.no_grad():
            codes = self.model.extract_latent(hubert_feature)
        return codes[0, 0]

    def decode(self, pred_semantic: list[torch.Tensor], batch_phones: list[torch.Tensor],
               refer_audio_spec: list, *,
               speed: float = 1.0, sv_emb=None) -> list[torch.Tensor]:
        """Decode one batch of semantic tokens into waveform fragments.

        Mirrors the reference parallel path (``TTS.run`` -> ``vits_model.decode``
        with all fragments concatenated and then split back per idx).
        """
        if self.model is None:
            raise RuntimeError("SovitsModel.load() must be called before decode().")
        upsample_rate = math.prod(self.model.upsample_rates)
        frag_lens = [p.shape[0] * 2 * upsample_rate for p in pred_semantic]
        end_idx = [sum(frag_lens[: i + 1]) for i in range(len(frag_lens))]
        all_pred = torch.cat(pred_semantic).unsqueeze(0).unsqueeze(0).to(self.device)
        phones = torch.cat(batch_phones).unsqueeze(0).to(self.device)
        with torch.no_grad():
            audio = self.model.decode(all_pred, phones, refer_audio_spec, speed=speed, sv_emb=sv_emb).detach()[0, 0, :]
        end_idx.insert(0, 0)
        return [audio[end_idx[i - 1]:end_idx[i]] for i in range(1, len(end_idx))]

    def decode_serial(self, pred_semantic: torch.Tensor, phones: torch.Tensor,
                      refer_audio_spec: list, *, speed: float = 1.0, sv_emb=None) -> torch.Tensor:
        """Decode a single fragment (reference serial path, speed != 1.0)."""
        if self.model is None:
            raise RuntimeError("SovitsModel.load() must be called before decode_serial().")
        pred = pred_semantic.unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            return self.model.decode(pred, phones.unsqueeze(0), refer_audio_spec, speed=speed, sv_emb=sv_emb).detach()[0, 0, :]

    def unload(self) -> None:
        self.model = None
        gc.collect()
