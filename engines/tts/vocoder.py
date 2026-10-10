"""Vocoder wrappers (wave E-3): BigVGAN (v3) / Generator (v4).

``v2Pro`` and ``v2ProPlus`` do NOT use a vocoder (``use_vocoder=False``);
:meth:`VocoderEngine.load` returns ``None`` for them, so the engine chain
never instantiates it. Only the vocoder versions that need the post-coding
stage are supported here.
"""

from __future__ import annotations

import torch


class VocoderEngine:
    def __init__(self) -> None:
        self.model = None
        self.version = ""
        self.sr = 0

    @staticmethod
    def is_applicable(version: str) -> bool:
        return version in ("v3", "v4")

    def load(self, version: str, device="cpu", is_half=False) -> dict | None:
        """Instantiates and loads the matching vocoder; ``None`` when unused."""
        from engines.tts._ref import ensure_reference_imported

        ensure_reference_imported()
        self.version = version
        model_device = torch.device(str(device))
        model = None
        if version == "v3":
            from BigVGAN.bigvgan import BigVGAN

            model = BigVGAN.from_pretrained(str(_v3_weights()), use_cuda_kernel=False)
            model.remove_weight_norm()
            self.sr = 24000
        elif version == "v4":
            from module.models import Generator

            model = Generator(
                initial_channel=100,
                resblock="1",
                resblock_kernel_sizes=[3, 7, 11],
                resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                upsample_rates=[10, 6, 2, 2, 2],
                upsample_initial_channel=512,
                upsample_kernel_sizes=[20, 12, 4, 4, 4],
                gin_channels=0,
                is_bias=True,
            )
            model.remove_weight_norm()
            state = torch.load(str(_v4_weights()), map_location="cpu", weights_only=False)
            model.load_state_dict(state)
            self.sr = 48000
        else:
            return None
        self.model = model.eval()
        if self.model is not None:
            self.model = self.model.half().to(model_device) if (is_half and str(model_device) != "cpu") else self.model.to(model_device)
        return {"version": version, "sr": self.sr}

    def synthesize(self, mel: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("VocoderEngine.load() must be called before synthesize().")
        with torch.inference_mode():
            return self.model(mel)[0][0]

    def unload(self) -> None:
        self.model = None


def _v3_weights():
    from core.config import PROJECT_ROOT

    return PROJECT_ROOT / "GPT-SoVITS-v2pro-20250604-nvidia50" / "GPT_SoVITS" / "pretrained_models" / \
        "models--nvidia--bigvgan_v2_24khz_100band_256x"


def _v4_weights():
    from core.config import PROJECT_ROOT

    return PROJECT_ROOT / "GPT-SoVITS-v2pro-20250604-nvidia50" / "GPT_SoVITS" / "pretrained_models" / "gsv-v4-pretrained" / "vocoder.pth"
