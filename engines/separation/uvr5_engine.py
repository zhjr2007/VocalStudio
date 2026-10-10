"""UVR5 separation engine (wave B-5): one class for both MDX-Net (.pth) and
BS-RoFormer (.ckpt) weight families, following the inference data flow of
``GPT-SoVITS-v2pro-20250604-nvidia50/tools/uvr5`` (vr.py / bsroformer.py).

Stem semantics
--------------
MDX family: HP .pth models are trained with the accompaniment as target, so
  estimate == accompaniment and vocals = mix - estimate.
  Karaoke uses HP5_only_main_vocal -> the vocal track keeps only the lead,
  backing vocals/harmonies remain in the accompaniment (去主唱留和声).
BS family: the ep_317 ckpt is trained with ``target_instrument="vocals"`` and
the drums/bass/piano ckpts each isolate their own instrument, so for a single
BS model estimate == its stem; the residual mix - estimate is "other"/"accompaniment".

Intensity is the UVR "agg" value in [0, 20] (default 0): forwarded as
aggressiveness = agg / 100 to MDX inference. BS-RoFormer inference has no
intensity knob; the value is validated and ignored there.
"""

from __future__ import annotations

import gc
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from core.config import ModelPaths, PROJECT_ROOT

_UVR5_REF_DIR = PROJECT_ROOT / "GPT-SoVITS-v2pro-20250604-nvidia50" / "tools" / "uvr5"

#: preset name -> ordered (stem_name, weight basename) pairs.
PRESET_WEIGHTS: dict[str, tuple[tuple[str, str], ...]] = {
    "Karaoke": (("vocals", "HP5_only_main_vocal.pth"),),
    "FullVocal": (("vocals", "HP2_all_vocals.pth"),),
    "Band4": (
        ("vocals", "model_bs_roformer_ep_317_sdr_12.9755.ckpt"),
        ("drums", "drums_bs_roformer.ckpt"),
        ("bass", "bass_bs_roformer.ckpt"),
    ),
    "Producer5": (
        ("vocals", "model_bs_roformer_ep_317_sdr_12.9755.ckpt"),
        ("drums", "drums_bs_roformer.ckpt"),
        ("bass", "bass_bs_roformer.ckpt"),
        ("piano", "piano_bs_roformer.ckpt"),
    ),
}

#: aliases for raw (non-preset) weight specs so stems keep friendly names.
_RAW_STEM_ALIASES = {"model_bs_roformer_ep_317_sdr_12.9755": "vocals"}
_FAMILY_SUFFIXES = ("_bs_roformer",)


def _stem_for_basename(basename: str) -> str:
    stem = Path(basename).stem.lower()
    if stem in _RAW_STEM_ALIASES:
        return _RAW_STEM_ALIASES[stem]
    for suffix in _FAMILY_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


_SAMPLE_RATE = 44100


def _family_of(basename: str) -> str:
    if basename.endswith(".pth"):
        return "mdx"
    if basename.endswith(".ckpt") and ("bs_roformer" in basename or "roformer" in basename):
        return "bs"
    raise ValueError(
        f"Unsupported weight family for {basename!r}; expected .pth (MDX-Net) or a bs-roformer .ckpt"
    )


def _load_stereo_44k(audio) -> np.ndarray:
    """Return float32 array shaped (2, samples) at 44100 Hz."""

    def _resample(arr: np.ndarray, orig_sr: int) -> np.ndarray:
        import librosa

        return np.ascontiguousarray(
            librosa.resample(arr, orig_sr=orig_sr, target_sr=_SAMPLE_RATE),
            dtype="float32",
        )

    if isinstance(audio, str):
        data, sr = sf.read(audio, dtype="float32", always_2d=True)  # (samples, ch)
        arr = np.ascontiguousarray(data.T).astype("float32")
        if sr != _SAMPLE_RATE:
            arr = _resample(arr, sr)
    else:
        arr = np.asarray(audio, dtype="float32")
        if arr.ndim == 1:
            arr = np.stack([arr, arr])
        elif arr.ndim == 2 and arr.shape[0] > 2:
            # accept (samples, channels) orientation
            arr = arr.T if arr.shape[-1] <= 2 else arr[:2].T
    if arr.shape[0] == 1:
        arr = np.repeat(arr, 2, axis=0)
    return arr


def _cmb_spectrogram_to_wave(spec_m, mp, extra_bins_h=None, extra_bins=None):
    """Band recombination mirroring lib.lib_v5.spec_utils.cmb_spectrogram_to_wave.

    The reference allocates ``spec_s`` with an uninitialized np.ndarray; in a
    long-lived process the reused buffer can hold stale non-finite bins and
    librosa.resample then rejects them. Zero-initialized buffers are safe
    because out-of-crop bins are filtered or zeroed downstream anyway."""

    import librosa

    from lib.lib_v5 import spec_utils

    bands_n = len(mp.param["band"])
    offset = 0
    wave: np.ndarray | None = None
    for d in range(1, bands_n + 1):
        bp = mp.param["band"][d]
        spec_s = np.zeros((2, bp["n_fft"] // 2 + 1, spec_m.shape[2]), dtype=complex)
        h = bp["crop_stop"] - bp["crop_start"]
        spec_s[:, bp["crop_start"]:bp["crop_stop"], :] = spec_m[:, offset:offset + h, :]
        offset += h

        def band_wave() -> np.ndarray:
            return spec_utils.spectrogram_to_wave(
                spec_s,
                bp["hl"],
                mp.param["mid_side"],
                mp.param["mid_side_b2"],
                mp.param["reverse"],
            )

        if d == bands_n:  # higher band
            if extra_bins_h:
                max_bin = bp["n_fft"] // 2
                spec_s[:, max_bin - extra_bins_h:max_bin, :] = extra_bins[:, :extra_bins_h, :]
            if bp["hpf_start"] > 0:
                spec_s = spec_utils.fft_hp_filter(spec_s, bp["hpf_start"], bp["hpf_stop"] - 1)
            wave = band_wave() if wave is None else np.add(wave, band_wave())
        else:
            sr = mp.param["band"][d + 1]["sr"]
            if d == 1:  # lower band
                spec_s = spec_utils.fft_lp_filter(spec_s, bp["lpf_start"], bp["lpf_stop"])
                wave = librosa.resample(
                    band_wave(),
                    orig_sr=bp["sr"],
                    target_sr=sr,
                    res_type="sinc_fastest",
                )
            else:  # mid band
                spec_s = spec_utils.fft_hp_filter(spec_s, bp["hpf_start"], bp["hpf_stop"] - 1)
                spec_s = spec_utils.fft_lp_filter(spec_s, bp["lpf_start"], bp["lpf_stop"])
                wave2 = np.add(wave, band_wave())
                wave = librosa.core.resample(wave2, orig_sr=bp["sr"], target_sr=sr, res_type="scipy")

    return wave.T


class UVR5Engine:
    """Load MDX-Net / BS-RoFormer weights and run source separation."""

    def __init__(self, model_paths=None) -> None:
        self._model_paths = model_paths if model_paths is not None else ModelPaths.from_settings()
        self._loaded: dict[str, dict] = {}  # stem_name -> wrapper
        self._spec: str | None = None
        self._device: str = "cpu"

    # ------------------------------------------------------------------ load

    def _weight_dir(self) -> Path:
        value = getattr(self._model_paths, "uvr5_weights", "") or ""
        if not value:
            raise ValueError("Model path 'uvr5_weights' is not configured")
        base = Path(value)
        return base if base.is_absolute() else PROJECT_ROOT / base

    def _resolve_stems(self, spec: str) -> list[tuple[str, Path]]:
        dir_path = self._weight_dir()
        if spec in PRESET_WEIGHTS:
            items = [(stem, dir_path / name) for stem, name in PRESET_WEIGHTS[spec]]
            resolved = []
            for stem, path in items:
                if not path.exists():
                    raise ValueError(f"Model file missing: {path.name} (expected under {dir_path})")
                _family_of(path.name)  # validate the extension early
                resolved.append((stem, path))
            return resolved

        # raw weight spec: with or without extension
        base = Path(spec).name
        if not (base.endswith(".pth") or base.endswith(".ckpt")):
            for ext in (".ckpt", ".pth"):
                candidate = dir_path / f"{spec}{ext}"
                if candidate.exists():
                    base = candidate.name
                    break
        path = dir_path / base
        stem = _stem_for_basename(path.name)
        items = [(stem, path)]
        resolved = []
        for stem_name, p in items:
            if not p.exists():
                raise ValueError(f"Model file missing: {p.name} (expected under {dir_path})")
            _family_of(p.name)  # validate the extension early
            resolved.append((stem_name, p))
        return resolved

    def load(self, spec: str, device: str = "cpu") -> None:
        from core.device_utils import resolve_pytorch_device

        requested = device if device in ("cuda", "rocm", "directml", "cpu") else "auto"
        dev = str(resolve_pytorch_device(requested)).lower()
        bare = "cuda" if dev.startswith("cuda") else "cpu"

        new_loaded: dict[str, dict] = {}

        for stem, path in self._resolve_stems(spec):
            family = _family_of(path.name)
            wrapper = self._load_mdx(path, dev) if family == "mdx" else self._load_bs(path, bare)
            new_loaded[stem] = wrapper
        self._loaded = new_loaded
        self._spec = spec
        self._device = bare

    @staticmethod
    def _ensure_ref_path() -> None:
        ref = str(_UVR5_REF_DIR)
        if os.path.isdir(ref) and ref not in sys.path:
            sys.path.append(ref)

    @staticmethod
    def _unwrap_state(state):
        if isinstance(state, dict):
            for key in ("model_params", "state_dict"):
                if key in state:
                    return state[key]
        return state

    def _load_mdx(self, path: Path, device_str: str) -> dict:
        self._ensure_ref_path()
        from lib.lib_v5 import nets_61968KB as Nets
        from lib.lib_v5.model_param_init import ModelParameters

        mp = ModelParameters(str(_UVR5_REF_DIR / "lib" / "lib_v5" / "modelparams" / "4band_v2.json"))
        model = Nets.CascadedASPPNet(mp.param["bins"] * 2)
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(self._unwrap_state(state))
        model.eval()
        model.to(device_str)
        return {"kind": "mdx", "mp": mp, "model": model}

    def _load_bs(self, path: Path, device_str: str) -> dict:
        self._ensure_ref_path()
        from bs_roformer.bs_roformer import BSRoformer

        base_config = {
            "dim": 512,
            "depth": 12,
            "stereo": True,
            "num_stems": 1,
            "time_transformer_depth": 1,
            "freq_transformer_depth": 1,
            "linear_transformer_depth": 0,
            "freqs_per_bands": tuple(
                [2] * 24 + [4] * 12 + [12] * 8 + [24] * 8 + [48] * 8 + [128, 129]
            ),
            "dim_head": 64,
            "heads": 8,
            "attn_dropout": 0.1,
            "ff_dropout": 0.1,
            "flash_attn": True,
            "dim_freqs_in": 1025,
            "stft_n_fft": 2048,
            "stft_hop_length": 441,
            "stft_win_length": 2048,
            "stft_normalized": False,
            "mask_estimator_depth": 2,
        }

        candidates: list[dict] = [base_config]
        for suffix in (".yaml", ".yml", ".json"):
            config_path = path.with_suffix(suffix)
            if not config_path.exists():
                continue
            override = self._read_bs_config_overrides(config_path)
            candidates.insert(0, {**base_config, **override})
        # some public checkpoints (e.g. drums ep_937) were trained with dim=384;
        # try the default architecture first, then fall back.
        if base_config["dim"] == 512:
            candidates.append({**base_config, "dim": 384})

        import warnings

        state = torch.load(path, map_location="cpu")
        state_dict = self._unwrap_state(state)
        model: object | None = None
        last_error: Exception | None = None
        for config in candidates:
            try:
                candidate_model = BSRoformer(**config)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # per-param size-mismatch noise on mismatched candidates
                    candidate_model.load_state_dict(state_dict, strict=True)
            except (RuntimeError, ValueError, KeyError) as exc:
                last_error = exc
                continue
            model = candidate_model
            break
        if model is None:
            raise RuntimeError(
                f"Could not load BS-RoFormer weights {path.name} with any supported "
                f"architecture; last error: {last_error}"
            ) from last_error

        model.eval()  # type: ignore[union-attr]
        model.to(device_str)  # type: ignore[union-attr]
        return {"kind": "bs", "model": model, "device": device_str}

    @staticmethod
    def _read_bs_config_overrides(config_path: Path) -> dict:
        """Read a per-model config file (UVR5 style: top-level 'model' mapping)."""

        import json as _json

        text = config_path.read_text(encoding="utf-8")
        if config_path.suffix == ".json":
            raw = _json.loads(text)
        else:
            import yaml  # noqa: WPS433 (reference implementation uses PyYAML too)

            raw = yaml.safe_load(text) or {}
        overrides = dict(raw.get("model") or {}) if isinstance(raw, dict) else {}
        for key in ("freqs_per_bands",):
            value = overrides.get(key)
            if isinstance(value, list):
                overrides[key] = tuple(int(item) for item in value)
        return {key: value for key, value in overrides.items() if value is not None}

    def is_loaded(self) -> bool:
        return self._spec is not None and bool(self._loaded)

    # ------------------------------------------------------------------- run

    def run(
        self,
        audio,
        out_dir: str | Path,
        intensity: float = 0.0,
        progress_cb=None,
    ) -> dict[str, str]:
        """Runs separation for every loaded stem; returns {stem_name: wav path}.

        ``progress_cb(fraction, message)`` (optional) reports coarse progress:
        0.0 at audio-load start, one step per demixed stem across [0.1, 0.9),
        0.95 before writing and 1.0 on completion."""

        def _report(fraction: float, message: str) -> None:
            if progress_cb is not None:
                try:
                    progress_cb(float(fraction), str(message))
                except Exception:
                    pass

        if not self.is_loaded():
            raise RuntimeError("Engine is not loaded; call load(spec, device) before run()")
        value = float(intensity)
        if not 0.0 <= value <= 20.0:
            raise ValueError(f"intensity must be within [0, 20], got {value}")

        _report(0.0, "load audio")
        mix = _load_stereo_44k(audio)
        if mix.size == 0 or mix.shape[-1] == 0:
            raise ValueError("audio file contains no samples")
        out_path = Path(out_dir)
        os.makedirs(out_path, exist_ok=True)
        base = Path(str(audio)).stem if isinstance(audio, str) else "separation"

        stems = list(self._loaded)
        total = len(stems)
        estimates: dict[str, np.ndarray] = {}
        for index, stem_name in enumerate(stems):
            _report(0.1 + 0.8 * index / total, f"demix {stem_name}")
            estimate = self._demix(self._loaded[stem_name], mix, value)
            estimates[stem_name] = estimate

        # align all arrays to the shortest common length (stft/istft framing can
        # drift by a few frames relative to the source)
        n_out = min([mix.shape[-1]] + [int(e.shape[-1]) for e in estimates.values()])
        mix_aligned = mix[..., :n_out]
        estimates = {k: v[..., :n_out] for k, v in estimates.items()}

        result: dict[str, np.ndarray] = {}
        if len(estimates) == 1 and "vocals" in estimates:
            est = estimates["vocals"]
            wrapper = self._loaded["vocals"]
            if wrapper["kind"] == "mdx":
                # MDX target is the accompaniment -> estimate is instrumental
                result["vocals"] = np.clip(mix_aligned - est, -1.0, 1.0)
                result["accompaniment"] = np.clip(est, -1.0, 1.0)
            else:
                # BS target_instrument == vocals -> estimate is the vocal stem
                result["vocals"] = np.clip(est, -1.0, 1.0)
                result["accompaniment"] = np.clip(mix_aligned - est, -1.0, 1.0)
        elif len(estimates) == 1:
            stem_name, est = next(iter(estimates.items()))
            wrapper = self._loaded[stem_name]
            result[stem_name] = np.clip(est if wrapper["kind"] == "bs" else mix_aligned - est, -1.0, 1.0)
            other_name = "accompaniment" if stem_name == "vocals" else "other"
            result[other_name] = np.clip(mix_aligned - est, -1.0, 1.0)
        else:
            # multi-stem preset: each estimate is its own stem (BS family) and
            # the residual mix minus all estimates is the remaining material
            for stem_name, est in estimates.items():
                result[stem_name] = np.clip(est, -1.0, 1.0)
            mixed_stems = sum(estimates.values())
            result["other"] = np.clip(mix_aligned - mixed_stems, -1.0, 1.0)

        _report(0.95, "write stems")
        paths: dict[str, str] = {}
        for stem_name, arr in result.items():
            path = out_path / f"{base}_{stem_name}.wav"
            sf.write(str(path), (np.clip(arr.T, -1.0, 1.0) * 32768).astype("int16"), _SAMPLE_RATE)
            paths[stem_name] = str(path)
        _report(1.0, "done")
        return paths

    def _demix(self, wrapper: dict, mix: np.ndarray, intensity: float) -> np.ndarray:
        if wrapper["kind"] == "mdx":
            return self._demix_mdx(wrapper, mix, intensity)
        return self._demix_bs(wrapper, mix)

    # -------------------------------------------------------------- MDX path

    def _demix_mdx(self, wrapper: dict, mix: np.ndarray, intensity: float) -> np.ndarray:
        import librosa

        from lib.lib_v5 import spec_utils
        from lib.utils import inference

        mp = wrapper["mp"]
        bands_n = len(mp.param["band"])
        X_wave: dict[int, np.ndarray] = {}
        for d in range(bands_n, 0, -1):
            bp = mp.param["band"][d]
            if d == bands_n:
                X_wave[d] = mix.astype("float32", copy=False)
            else:
                X_wave[d] = librosa.resample(
                    X_wave[d + 1],
                    orig_sr=mp.param["band"][d + 1]["sr"],
                    target_sr=bp["sr"],
                    res_type=bp.get("res_type") or "kaiser_fast",
                )

        X_spec_s: dict[int, np.ndarray] = {}
        input_high_end_h = None
        input_high_end = None
        high_end_process = "mirroring"
        for d in range(bands_n, 0, -1):
            bp = mp.param["band"][d]
            X_spec_s[d] = spec_utils.wave_to_spectrogram_mt(
                X_wave[d],
                bp["hl"],
                bp["n_fft"],
                mp.param["mid_side"],
                mp.param["mid_side_b2"],
                mp.param["reverse"],
            )
            if d == bands_n:
                input_high_end_h = (bp["n_fft"] // 2 - bp["crop_stop"]) + (
                    mp.param["pre_filter_stop"] - mp.param["pre_filter_start"]
                )
                input_high_end = X_spec_s[d][
                    :, bp["n_fft"] // 2 - input_high_end_h : bp["n_fft"] // 2, :
                ]

        X_spec_m = spec_utils.combine_spectrograms(X_spec_s, mp)
        aggressiveness = {
            "value": float(intensity) / 100.0,
            "split_bin": mp.param["band"][1]["crop_stop"],
        }
        data = {
            "postprocess": False,
            "tta": False,
            "window_size": 512,
            "high_end_process": high_end_process,
        }
        with torch.no_grad():
            pred, _X_mag, X_phase = inference(X_spec_m, self._device, wrapper["model"], aggressiveness, data)

        if not np.isfinite(pred).all():
            # fp16 autocast can overflow a few magnitude bins; silencing them keeps
            # the reconstruction finite instead of raising inside librosa.resample.
            pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)

        y_spec_m = pred * X_phase  # estimate of the accompaniment target
        if high_end_process.startswith("mirroring"):
            input_high_end_ = spec_utils.mirroring(high_end_process, y_spec_m, input_high_end, mp)
            wav_target = _cmb_spectrogram_to_wave(y_spec_m, mp, input_high_end_h, input_high_end_)
        else:  # pragma: no cover (mirroring is the only configured mode)
            wav_target = _cmb_spectrogram_to_wave(y_spec_m, mp)

        estimate = np.array(wav_target, dtype="float32")
        # cmb_spectrogram_to_wave returns (samples, channels) -> canonicalize to (channels, samples)
        if estimate.ndim == 2 and estimate.shape[0] not in (1, 2):
            estimate = estimate.T
        top_sr = mp.param["sr"]
        if top_sr != _SAMPLE_RATE:
            estimate = librosa.resample(estimate, orig_sr=top_sr, target_sr=_SAMPLE_RATE)
        return estimate

    # --------------------------------------------------------------- BS path

    def _demix_bs(self, wrapper: dict, mix: np.ndarray) -> np.ndarray:
        import torch.nn as nn

        model = wrapper["model"]
        device = wrapper["device"]
        C = 352800  # chunk_size at 44.1k
        step = C // 2  # num_overlap == 2
        border = C - step
        fade_size = C // 10

        length_init = mix.shape[-1]
        x = np.ascontiguousarray(mix, dtype="float32")
        if length_init > 2 * border and border > 0:
            x = nn.functional.pad(torch.from_numpy(x), (border, border), mode="reflect").numpy()

        fadein = torch.linspace(0, 1, fade_size)
        fadeout = torch.linspace(1, 0, fade_size)
        window_middle = torch.ones(C)
        window_middle[-fade_size:] *= fadeout
        window_middle[:fade_size] *= fadein
        window_start = torch.ones(C)
        window_start[-fade_size:] *= fadeout
        window_finish = torch.ones(C)
        window_finish[:fade_size] *= fadein

        mix_t = torch.tensor(x, dtype=torch.float32)
        amp_ctx = torch.amp.autocast("cuda", enabled=True) if device.startswith("cuda") else nullcontext()
        result = torch.zeros((1,) + mix_t.shape, dtype=torch.float32)
        counter = torch.zeros_like(result)

        with amp_ctx:
            with torch.inference_mode():
                i = 0
                batch_data: list[torch.Tensor] = []
                batch_locations: list[tuple[int, int]] = []
                while i < mix_t.shape[1]:
                    part = mix_t[:, i : i + C].to(device)
                    length = part.shape[-1]
                    if length < C:
                        if length > C // 2 + 1:
                            part = nn.functional.pad(part, (0, C - length), mode="reflect")
                        else:
                            part = nn.functional.pad(
                                part, (0, C - length, 0, 0), mode="constant", value=0
                            )
                    batch_data.append(part)
                    batch_locations.append((i, length))
                    i += step

                    if len(batch_data) >= 4 or i >= mix_t.shape[1]:
                        arr = torch.stack(batch_data, dim=0)
                        out = model(arr).float()
                        window = window_start if i - step == 0 else (
                            window_finish if i >= mix_t.shape[1] else window_middle
                        )
                        for j, (start, l) in enumerate(batch_locations):
                            result[..., start : start + l] += out[j][..., :l].cpu() * window[..., :l]
                            counter[..., start : start + l] += window[..., :l]
                        batch_data = []
                        batch_locations = []

        estimated = (result / counter).squeeze(0).numpy().astype("float32")
        # silence non-finite samples instead of clipping inf to full-scale clicks
        np.nan_to_num(estimated, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        if length_init > 2 * border and border > 0:
            estimated = estimated[..., border : -border]
        return estimated

    # ------------------------------------------------------------- lifecycle

    def unload(self) -> None:
        for stem in list(self._loaded):
            del self._loaded[stem]
        self._spec = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


__all__ = ["PRESET_WEIGHTS", "UVR5Engine"]
