"""Voice clone engine + orchestrator/page step runners (wave E-3/E-4/E-5).

``VoiceCloneEngine`` drives the GPT-SoVITS v2Pro/v2ProPlus inference chain
refactored into staged steps (S3.4 interface)::

    load -> preprocess (text frontend + BERT) -> t2s (prompt semantic tokens)
    -> sovits (decode) -> [vocoder: v3/v4 only, none for v2Pro] -> save

The reference ``TTS`` class is used as the weight holder; because torchaudio's
base ``load`` backend needs torchcodec+FFmpeg DLLs (unavailable), the
reference-audio spectrogram is computed by this engine (soundfile +
``torchaudio.transforms.Resample`` + the reference ``spectrogram_torch``),
and injected into the prompt cache alongside the CNHuBERT prompt semantic
codes. Everything else goes through the reference data flow.
"""

from __future__ import annotations

import gc
import pathlib

import numpy as np

from engines.tts._ref import build_reference_config, import_tts_reference, resolve_model_paths

STAGES_TOTAL = 5  # load | preprocess | t2s | sovits | save


class VoiceCloneEngine:
    def __init__(self) -> None:
        self._tts = None
        self.version = ""
        self.device = "cpu"
        self.is_half = False

    # -- lifecycle ---------------------------------------------------------

    def load(self, version: str = "v2Pro", device="auto", model_paths=None) -> dict:
        """Load the GSV chain for ``version``; v2Pro/v2ProPlus supported."""
        from core.device_utils import resolve_pytorch_device

        if version not in ("v2Pro", "v2ProPlus"):
            raise ValueError(f"Unsupported TTS version '{version}' (v2Pro / v2ProPlus)")
        resolved = str(resolve_pytorch_device(device))
        paths = resolve_model_paths(model_paths)
        key = "v2proplus" if version == "v2ProPlus" else "v2pro"
        t2s_path = paths[f"t2s_{key}"]
        sovits_path = paths[f"sovits_{key}"]
        for label, p in (
            ("t2s", t2s_path),
            ("sovits", sovits_path),
            ("bert", paths["bert"]),
            ("hubert", paths["hubert"]),
        ):
            if not pathlib.Path(p).exists():
                raise FileNotFoundError(f"TTS model missing ({label}): {p}")

        is_half = resolved.startswith("cuda")
        TTS, _ = import_tts_reference()

        from engines.tts._ref import GSV_ROOT, ensure_reference_imported

        ensure_reference_imported()
        # sv.py binds ``sv_path`` to a cwd-relative constant at import time;
        # rebind to the absolute checkout path (it is read at load time).
        import sv as _sv_ref

        _sv_ref.sv_path = str(
            GSV_ROOT
            / "GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt"
        )

        config = build_reference_config(
            version,
            resolved,
            t2s_path=str(t2s_path),
            sovits_path=str(sovits_path),
            bert_path=str(paths["bert"]),
            hubert_path=str(paths["hubert"]),
            is_half=is_half,
        )
        import os as _os

        _previous = _os.getcwd()
        try:
            # TTS_Config writes GPT_SoVITS/configs/tts_infer.yaml relative to
            # cwd; keep that side effect inside the reference checkout.
            _os.chdir(str(GSV_ROOT))
            self._tts = TTS(config)
        finally:
            _os.chdir(_previous)
        # The GSV v2Pro checkpoints ship fp16; when we run in fp32 (is_half=False,
        # e.g. CPU/directml) the reference leaves some sub-models (T2S, VITS
        # SynthesizerTrn, SV) in fp16, which clashes with the fp32 inputs. Cast every
        # reference sub-module to one consistent precision to match ``is_half``.
        # This is an engine-side fix only; the vendored GSV source is untouched.
        import torch as _torch

        _cast = _torch.float32 if not is_half else _torch.float16
        for _attr in dir(self._tts):
            _mod = getattr(self._tts, _attr, None)
            if isinstance(_mod, _torch.nn.Module):
                try:
                    _mod.to(dtype=_cast)
                except Exception:
                    pass
            elif isinstance(_mod, (list, tuple)):
                for _m in _mod:
                    if isinstance(_m, _torch.nn.Module):
                        try:
                            _m.to(dtype=_cast)
                        except Exception:
                            pass

        self.version = str(getattr(self._tts.configs, "version", version))
        self.device = resolved
        self.is_half = is_half
        return {"version": self.version, "device": resolved, "is_half": is_half}

    def unload(self) -> None:
        self._tts = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # -- inference ----------------------------------------------------------

    def set_reference(self, ref_audio: str, prompt_text: str = "", prompt_lang: str = "zh") -> None:
        """Prompt side for *ref_audio*: HuBERT semantic codes + spectrogram.

        ``prompt_text`` (transcript of the reference audio) is optional; when
        given, its phonemes/BERT features are cached for the T2S prompt.
        """
        if self._tts is None:
            raise RuntimeError("VoiceCloneEngine.load() must be called before set_reference().")
        import soundfile as sf
        import torch

        from engines.tts._ref import ensure_reference_imported

        ensure_reference_imported()
        from module.mel_processing import spectrogram_torch

        ref = pathlib.Path(ref_audio)
        if not ref.is_file():
            raise FileNotFoundError(f"reference audio not found: {ref_audio!r}")

        # prompt semantic codes (librosa-based in the reference; no torchaudio.load)
        self._tts._set_prompt_semantic(str(ref))

        config = self._tts.configs
        data, raw_sr = sf.read(str(ref), dtype="float32", always_2d=True)
        audio = torch.from_numpy(data.T).to(config.device).float()
        if audio.shape[0] > 1:
            audio = audio.mean(0, keepdim=True)
        maxx = audio.abs().max()
        if float(maxx) > 1:
            audio = audio / min(2.0, float(maxx))

        from TTS_infer_pack.TTS import resample

        spec_audio = audio
        if int(raw_sr) != int(config.sampling_rate):
            spec_audio = resample(audio, int(raw_sr), int(config.sampling_rate), config.device)
        spec = spectrogram_torch(
            spec_audio,
            config.filter_length,
            config.sampling_rate,
            config.hop_length,
            config.win_length,
            center=False,
        )
        if config.is_half:
            spec = spec.half()
        audio16 = None
        if self.is_v2pro():
            audio16 = resample(spec_audio, int(config.sampling_rate), 16000, config.device)
            if config.is_half:
                audio16 = audio16.half()

        self._tts.prompt_cache["refer_spec"] = [(spec, audio16)]
        self._tts.prompt_cache["ref_audio_path"] = str(ref)

        if prompt_text:
            from engines.tts._ref import reference_workdir
            from TTS_infer_pack.text_segmentation_method import splits

            prompt_text = prompt_text.strip("\n")
            if prompt_text and prompt_text[-1] not in splits:
                prompt_text += "。" if prompt_lang != "en" else "."
            if self._tts.prompt_cache.get("prompt_text") != prompt_text:
                with reference_workdir():
                    phones, bert_features, norm_text = self._tts.text_preprocessor.segment_and_extract_feature_for_text(
                        prompt_text, prompt_lang, config.version
                    )
                self._tts.prompt_cache["prompt_text"] = prompt_text
                self._tts.prompt_cache["prompt_lang"] = prompt_lang
                self._tts.prompt_cache["phones"] = phones
                self._tts.prompt_cache["bert_features"] = bert_features
                self._tts.prompt_cache["norm_text"] = norm_text
        else:
            self._tts.prompt_cache["prompt_text"] = ""
            self._tts.prompt_cache["prompt_lang"] = ""
            self._tts.prompt_cache["phones"] = None
            self._tts.prompt_cache["bert_features"] = None
            self._tts.prompt_cache["norm_text"] = None

    def is_v2pro(self) -> bool:
        return self.version in ("v2Pro", "v2ProPlus")

    def clone(
        self,
        text: str,
        lang: str = "zh",
        *,
        ref_audio: str | None = None,
        prompt_text: str = "",
        prompt_lang: str = "zh",
        speed: float = 1.0,
        seed: int = -1,
        out_path: str | None = None,
        progress_cb=None,
        log_cb=None,
        cancel_checker=None,
    ) -> tuple[int, np.ndarray]:
        """Synthesize *text* with the reference voice; returns ``(sr, int16 data)``.

        Stages are reported through ``progress_cb(fraction, stage)`` with the
        stage names ``preprocess`` / ``t2s`` / ``sovits`` / ``vocoder`` /
        ``save`` and an optional ``log_cb`` gets ``Stage x/N ...`` lines.
        """
        import torch

        from core.exceptions import CancellationError

        if self._tts is None:
            raise RuntimeError("VoiceCloneEngine.load() must be called before clone().")
        if ref_audio:
            self.set_reference(ref_audio, prompt_text=prompt_text, prompt_lang=prompt_lang or lang)
        if not text or not text.strip():
            raise ValueError("target text is empty")
        speed = float(speed)
        if not 0.5 <= speed <= 1.5:
            raise ValueError(f"speed out of range [0.5, 1.5]: {speed}")

        tts = self._tts
        config = tts.configs
        precision = tts.precision
        from TTS_infer_pack.TTS import set_seed as _reference_seed

        _reference_seed(int(seed))
        stages = ["preprocess", "t2s", "sovits"] + (["vocoder"] if config.use_vocoder else []) + ["save"]
        total = len(stages)

        def check_cancelled() -> None:
            if callable(cancel_checker) and cancel_checker():
                raise CancellationError("tts cancelled")

        def progress(frac: float, stage: str) -> None:
            if progress_cb is not None:
                try:
                    progress_cb(float(frac), stage)
                except Exception:
                    pass

        def log(message: str) -> None:
            if log_cb is not None:
                try:
                    log_cb(str(message))
                except Exception:
                    pass

        log(f"Stage 1/{STAGES_TOTAL} load (device={self.device} half={self.is_half})")

        # -- preprocess ------------------------------------------------------
        index = 0
        progress(0.0, stages[index])
        log(f"Stage {index + 1}/{total} preprocess")
        prompt_used = bool(str(tts.prompt_cache.get("prompt_text") or ""))
        from engines.tts._ref import reference_workdir

        with reference_workdir():
            data = tts.text_preprocessor.preprocess(text.strip(), lang, "cut5", config.version)
        if not data:
            log("empty phoneme sequence; returning silence")
            sr = 32000
            return sr, np.zeros(int(sr), dtype=np.int16)
        batches, batch_index_list = tts.to_batch(
            data,
            prompt_data=tts.prompt_cache if prompt_used else None,
            batch_size=1,
            threshold=0.75,
            split_bucket=(speed == 1.0),
            device=config.device,
            precision=precision,
        )
        check_cancelled()

        refer_specs = [
            spec.to(dtype=precision, device=config.device) for spec, _audio in tts.prompt_cache["refer_spec"]
        ]
        sv_emb = None
        if self.is_v2pro():
            sv_emb = [
                tts.sv_model.compute_embedding3(audio) for _spec, audio in tts.prompt_cache["refer_spec"]
            ] or None

        output_sr = int(config.sampling_rate)
        fragments: list[list[torch.Tensor]] = []

        # -- t2s + sovits -----------------------------------------------------
        for item in batches:
            check_cancelled()
            prompt = None
            if prompt_used:
                prompt = (
                    tts.prompt_cache["prompt_semantic"]
                .expand(len(item["all_phones"]), -1)
                .to(config.device)
            )

            index = 1
            progress((index - 0.5) / total, stages[index])
            log(f"Stage {index + 1}/{total} t2s")
            # Coerce the T2S inputs to the model's precision so the half-weight
            # model never sees float32 tensors (the reference `to_batch` returns
            # lists of float32 tensors even when `precision` is fp16). This is a
            # wrapper-level fix only; the vendored GSV source is left untouched.
            def _coerce(x):
                if isinstance(x, torch.Tensor):
                    return x.to(dtype=t2s_dtype) if x.is_floating_point() else x
                if isinstance(x, (list, tuple)):
                    return [_coerce(y) for y in x]
                return x

            t2s_dtype = precision if isinstance(precision, torch.dtype) else (
                torch.float16 if self.is_half else torch.float32
            )
            _phones = _coerce(item["all_phones"])
            _feats = _coerce(item["all_bert_features"])
            if prompt is not None:
                prompt = _coerce(prompt)
            pred_semantic_list, idx_list = tts.t2s_model.model.infer_panel_batch_infer(
                _phones,
                item["all_phones_len"],
                prompt,
                _feats,
                top_k=15,
                top_p=1.0,
                temperature=1.0,
                early_stop_num=float(config.hz * (config.max_sec or 30)),
                max_len=int(item["max_len"]),
                repetition_penalty=1.35,
            )
            check_cancelled()
            trimmed = [t[-i:] for t, i in zip(pred_semantic_list, idx_list)]

            index = 2
            progress(index / total, stages[index])
            log(f"Stage {index + 1}/{total} sovits")
            if speed == 1.0:
                batch_fragments = tts.vits_model.decode(
                    torch.cat(trimmed).unsqueeze(0).unsqueeze(0).to(config.device),
                    torch.cat(item["phones"]).unsqueeze(0).to(config.device),
                    refer_specs,
                    speed=1.0,
                    sv_emb=sv_emb,
                ).detach()[0, 0, :]
                upsample = 1
                import math

                upsample = math.prod(tts.vits_model.upsample_rates)
                lens = [t.shape[0] * 2 * upsample for t in trimmed]
                ends = [sum(lens[: i + 1]) for i in range(len(lens))]
                ends.insert(0, 0)
                batch_fragments = [batch_fragments[ends[i - 1]:ends[i]] for i in range(1, len(ends))]
            else:
                batch_fragments = []
                for t, p in zip(trimmed, item["phones"]):
                    batch_fragments.append(
                        tts.vits_model.decode(
                            t.unsqueeze(0).unsqueeze(0), p.unsqueeze(0), refer_specs, speed=speed, sv_emb=sv_emb
                        ).detach()[0, 0, :]
                    )
            check_cancelled()
            fragments.append(batch_fragments)

        # -- vocoder (v3/v4 only; v2Pro skips) ---------------------------------
        if config.use_vocoder:
            index = 3
            progress(index / total, stages[index])
            log(f"Stage {index + 1}/{total} vocoder")
            raise NotImplementedError(
                "V3/V4 vocoder synthesis is not implemented yet; use v2Pro / v2ProPlus."
            )

        # -- postprocess + save -----------------------------------------------
        index = len(stages) - 1
        progress((index + 0.5) / total, stages[index])
        log(f"Stage {index + 1}/{total} save")
        sr, audio = tts.audio_postprocess(
            fragments,
            output_sr,
            batch_index_list,
            1.0 if speed == 1.0 else speed,
            split_bucket=(speed == 1.0),
            fragment_interval=0.0,
        )
        check_cancelled()
        if out_path:
            import soundfile as sf

            pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(out_path), audio, int(sr))
            log(f"wrote {out_path} ({len(audio) / sr:.2f}s)")
        progress(1.0, "save")
        return int(sr), audio


def run_tts(ctx, cfg, progress_cb=None) -> None:
    """Orchestrator step entry (see core.orchestrator.STEP_ENTRYPOINTS)."""
    ref = str(getattr(cfg, "ref_audio", "") or "")
    text = str(getattr(cfg, "tts_text", "") or "")
    if not ref or not pathlib.Path(ref).is_file():
        raise FileNotFoundError(f"tts: reference audio not found: {ref!r}")
    if not text:
        raise ValueError("tts: target text is empty")

    from core.config import PROJECT_ROOT

    out_dir = pathlib.Path(
        getattr(cfg, "output_dir", "")
        or (pathlib.Path(getattr(cfg, "audio_path", "") or "").parent or PROJECT_ROOT) / "outputs" / "tts"
    )
    out_path = out_dir / f"{pathlib.Path(ref).stem}.wav"
    key = str(getattr(cfg, "tts_version", "v2Pro") or "v2Pro")

    engine = VoiceCloneEngine()
    try:
        summary = engine.load(key, device=getattr(cfg, "device", "auto"), model_paths=getattr(cfg, "model_paths", None))
        sr, audio = engine.clone(
            text,
            lang=str(getattr(cfg, "tts_lang", "zh") or "zh"),
            ref_audio=ref,
            prompt_text=str(getattr(cfg, "ref_text", "") or ""),
            prompt_lang=str(getattr(cfg, "ref_lang", "zh") or "zh"),
            speed=float(getattr(cfg, "tts_speed", 1.0)),
            out_path=str(out_path),
            progress_cb=progress_cb,
            cancel_checker=getattr(cfg, "cancel_checker", None),
        )
    finally:
        engine.unload()
    ctx.tts_audio_path = str(out_path)
    ctx.results["tts"] = {
        "path": str(out_path),
        "sr": int(sr),
        "duration": round(len(audio) / int(sr), 3),
        "version": summary["version"],
    }


def run_tts_job(ctx: dict, cfg, progress_cb) -> dict:
    """GUI page job body (JobRunner): clones the target text to ``cfg.output_dir``.

    ``ctx`` carries the ref/input metadata; returns the same dict with
    ``output`` (wav path), ``sr``, ``duration_ms`` added.
    """
    from core.config import PROJECT_ROOT

    ref = str(ctx.get("ref") or getattr(cfg, "ref_audio", "") or "")
    output_dir = str(getattr(cfg, "output_dir", "") or (PROJECT_ROOT / "outputs" / "tts"))
    if not pathlib.Path(ref).is_file():
        raise FileNotFoundError(f"reference audio not found: {ref!r}")
    if not ctx.get("text"):
        raise ValueError("target text is empty")

    engine = VoiceCloneEngine()
    try:
        summary = engine.load(
            str(getattr(cfg, "version", "v2Pro") or "v2Pro"),
            device=str(getattr(cfg, "device", "auto") or "auto"),
            model_paths=getattr(cfg, "model_paths", None),
        )
        out_path = pathlib.Path(output_dir) / f"{pathlib.Path(ref).stem}.wav"
        sr, audio = engine.clone(
            str(ctx.get("text")),
            lang=str(getattr(cfg, "lang", "zh") or "zh"),
            ref_audio=ref,
            speed=float(getattr(cfg, "speed", 1.0)),
            out_path=str(out_path),
            progress_cb=progress_cb,
            log_cb=getattr(cfg, "log_cb", None),
            cancel_checker=getattr(cfg, "cancel_checker", None),
        )
    finally:
        engine.unload()
    ctx["output"] = str(out_path)
    ctx["sr"] = int(sr)
    ctx["duration_ms"] = int(round(len(audio) * 1000 / int(sr)))
    ctx["version"] = summary["version"]
    return ctx
