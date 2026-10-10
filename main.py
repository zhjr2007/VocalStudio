"""VocalStudio command-line entry point."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def _build_config(args) -> PipelineConfig:
    from core.config import ModelPaths, PipelineConfig, PipelineStep

    cfg = PipelineConfig()
    audio = getattr(args, "audio", None)
    if audio is not None:
        cfg.audio_path = audio
    mapping = {
        "output_dir": "output_dir",
        "device": "device",
        "lang": "language",
        "model": "separation_model",
        "agg": "separation_agg",
        "slicing": "slicing_method",
        "slice_min": "slice_min_sec",
        "slice_max": "slice_max_sec",
        "bpm": "tempo",
        "quant": "quantization_mode",
        "engine": "asr_engine",
        "speakers": "num_speakers",
        "tts_text": "tts_text",
        "tts_version": "tts_version",
        "ref_audio": "ref_audio",
        "speed": "tts_speed",
    }
    for arg_name, cfg_field in mapping.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            setattr(cfg, cfg_field, value)

    formats = getattr(args, "formats", None)
    if formats:
        cfg.output_formats = [f.strip().lower().lstrip(".") for f in formats.split(",")]

    cfg.enable_separation = bool(getattr(args, "separate", False)) or getattr(args, "model", None) is not None
    cfg.enable_diarization = bool(cfg.num_speakers > 0)
    if getattr(args, "no_punc", False):
        cfg.enable_punctuation = False
    if getattr(args, "srt", False):
        cfg.output_srt = True
    if getattr(args, "lyrics", None):
        cfg.original_lyrics = args.lyrics
        cfg.lyric_output_mode = "manual"

    cfg.model_paths = ModelPaths.from_settings()

    steps_arg = getattr(args, "steps", None)
    if steps_arg:
        cfg.pipeline_steps = [PipelineStep(s.strip()) for s in steps_arg.split(",")]
    elif not cfg.enable_separation and not cfg.enable_diarization:
        command_defaults = {
            "transcribe": ["vocal2midi"],
            "asr": ["speech_asr"],
            "tts": ["tts"],
            "detect": [],
        }
        key = getattr(args, "func_key", None)
        if key in command_defaults:
            cfg.pipeline_steps = [PipelineStep(s) for s in command_defaults[key]]
    return cfg


def _print_result(ctx) -> None:
    from core.context import NoteInfo

    print("--- Result ---")
    for label, value in (
        ("audio_type", ctx.audio_type.value if isinstance(ctx.audio_type, object) else ""),
        ("vocals", ctx.vocals_path),
        ("accompaniment", ctx.accompaniment_path),
        ("notes", f"{len(ctx.notes)} notes" if isinstance(ctx.notes, list) else None),
        ("asr_text", (ctx.asr_text or "")[:200]),
        ("srt", ctx.srt_path),
        ("tts_audio", ctx.tts_audio_path),
    ):
        text = "" if value is None else str(value)
        print(f"  {label}: {text}")


def cmd_info(args) -> int:
    from core.config import ModelPaths
    from core.device_utils import check_python_environment, get_gpu_info

    env = check_python_environment()
    gpu = get_gpu_info()
    print("VocalStudio environment")
    print(f"  python : {env['python']}")
    print(f"  torch  : {env['torch']} (cuda={env['torch_cuda']}, hip={env['torch_hip']})")
    print(f"  nvidia-smi: {env['nvidia_smi'] or 'not found'}")
    if gpu["torch_cuda_available"]:
        print(
            f"  GPU    : {gpu['name']} | {gpu['backend']} "
            f"sm_{gpu['compute_capability'].replace('.', '')} | "
            f"{gpu['total_vram_gb']:.0f}GB total / {gpu['free_vram_gb']:.1f}GB free"
        )
    else:
        print("  GPU    : CUDA not available")

    paths = ModelPaths.from_settings()
    missing, ready = [], []
    for name in paths.__dataclass_fields__:
        value = getattr(paths, name)
        if not value:
            continue
        p = Path(value)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        (ready if p.exists() else missing).append(f"{name} -> {p}")
    print(f"\nModel paths ({len(ready)} ready, {len(missing)} missing):")
    for line in ready:
        print(f"  [ok]   {line}")
    for line in missing:
        print(f"  [MISS] {line}")
    return 0


def cmd_detect(args) -> int:
    from core.config import PipelineConfig
    from core.content_detector import ContentDetector

    if not args.audio or not Path(args.audio).exists():
        print("error: --audio required and must exist", file=sys.stderr)
        return 1
    detector = ContentDetector()
    audio_type = detector.detect(args.audio)
    print(f"content type: {audio_type.value}")
    import librosa

    y, sr = librosa.load(str(args.audio), sr=16000, mono=True)
    metrics = detector.measure(y, sr)
    if metrics is not None:
        for key, val in metrics.to_dict().items():
            print(f"  {key}: {val}")
    return 0


def _run(args) -> int:
    from core.config import PipelineConfig  # noqa: F401

    cfg = _build_config(args)
    if getattr(cfg, "audio_path", "") and not Path(cfg.audio_path).exists():
        print(f"error: audio file not found: {cfg.audio_path}", file=sys.stderr)
        return 1
    from core.orchestrator import VocalStudioOrchestrator

    def progress(fraction, message):
        print(f"{fraction * 100:5.1f}%  {message}")

    try:
        ctx = VocalStudioOrchestrator(cfg).with_progress(progress).execute()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2 if type(exc).__name__ == "CancellationError" else 1
    _print_result(ctx)
    out = Path(args.output_dir or cfg.output_dir or (Path(cfg.audio_path).parent / "outputs"))
    ctx.results["output_dir"] = str(out)
    print(f"  output: {out}")
    return 0


def cmd_separate(args) -> int:
    if not getattr(args, "audio", None):
        print("error: --audio required", file=sys.stderr)
        return 1
    audio = Path(str(args.audio))
    if not audio.exists():
        print(f"error: audio file not found: {args.audio}", file=sys.stderr)
        return 1
    out_dir = args.output_dir or str(audio.parent / "outputs" / "separation")

    from engines.separation import UVR5Engine

    def progress(fraction, message):
        print(f"progress {float(fraction):.2f} {message}", flush=True)

    engine = UVR5Engine()
    try:
        print("stage fetch", flush=True)
        engine.load(args.model or "Karaoke", device=args.device)
        print("stage analyze", flush=True)
        paths = engine.run(str(audio), out_dir, intensity=args.agg, progress_cb=progress)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        engine.unload()
    for stem_name, path in paths.items():
        print(f"{stem_name}: {path}", flush=True)
    return 0


def cmd_transcribe(args) -> int:
    args.func_key = "transcribe"
    if not getattr(args, "audio", None):
        print("error: --audio required", file=sys.stderr)
        return 1
    from core.config import PipelineStep

    steps = [PipelineStep.VOCAL2MIDI]
    if getattr(args, "separate", False):
        steps.insert(0, PipelineStep.SEPARATION)
    args.steps = ",".join(s.value for s in steps)
    return _run(args)


def cmd_asr(args) -> int:
    args.func_key = "asr"
    if not getattr(args, "audio", None):
        print("error: --audio required", file=sys.stderr)
        return 1
    from core.config import PipelineStep

    steps = []
    if getattr(args, "separate", False):
        steps.append(PipelineStep.SEPARATION)
    if getattr(args, "speakers", 0):
        steps.append(PipelineStep.DIARIZATION)
    steps.append(PipelineStep.SPEECH_ASR)
    args.steps = ",".join(s.value for s in steps)
    return _run(args)


def cmd_tts(args) -> int:
    from core.config import ModelPaths, PipelineConfig

    if not getattr(args, "text", None):
        print("error: --text required", file=sys.stderr)
        return 1
    cfg = PipelineConfig(
        audio_path=getattr(args, "ref_audio", "") or "",
        output_dir=args.output_dir or str(PROJECT_ROOT / "outputs"),
        device=args.device,
        enable_tts=True,
        tts_version=args.version,
        tts_text=args.text,
        tts_lang=args.lang or "zh",
        tts_speed=float(getattr(args, "speed", None) or 1.0),
        ref_audio=getattr(args, "ref_audio", "") or "",
        ref_text=args.ref_text or "",
        ref_lang=args.ref_lang or args.lang or "zh",
    )
    cfg.model_paths = ModelPaths.from_settings()
    from core.config import PipelineStep
    from core.orchestrator import VocalStudioOrchestrator

    try:
        ctx = VocalStudioOrchestrator(cfg).with_progress(lambda f, m: print(f"{f*100:5.1f}%  {m}")).execute()
    except Exception as exc:
        if os.environ.get("VS_DEBUG") == "1":
            import traceback as _tb

            _tb.print_exc()
        print(f"error: {exc}", file=sys.stderr)
        return 1
    _print_result(ctx)
    return 0


def cmd_pipeline(args) -> int:
    args.func_key = "pipeline"
    if not getattr(args, "audio", None):
        print("error: --audio required", file=sys.stderr)
        return 1
    if not getattr(args, "steps", None):
        cfg_probe = PipelineConfig()
        from core.config import PipelineStep

        steps = []
        if args.separate:
            steps.append(PipelineStep.SEPARATION)
        if args.speakers:
            steps.extend([PipelineStep.DIARIZATION])
        steps.append(PipelineStep.CONTENT_DETECTION)
        if not args.asr_only:
            steps.append(PipelineStep.VOCAL2MIDI)
        if not args.vocal2midi_off:
            steps.append(PipelineStep.SPEECH_ASR)
        if args.tts_text:
            steps.append(PipelineStep.TTS)
        args.steps = ",".join(s.value for s in dict.fromkeys(steps))
    return _run(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vocalstudio", description="VocalStudio unified CLI")
    sub = parser.add_subparsers(dest="command")

    p_info = sub.add_parser("info", help="show environment and model path status")
    p_info.set_defaults(func=cmd_info)

    p_detect = sub.add_parser("detect", help="detect content type (singing/speech/mixed)")
    p_detect.add_argument("--audio", required=True)
    p_detect.set_defaults(func=cmd_detect)

    common_audio = [
        ("--audio"),
        ("--output-dir"),
        ("--device", {"default": "auto", "choices": ["auto", "cuda", "rocm", "directml", "cpu"]}),
        ("--lang"),
    ]

    def add_common(p, need_audio: bool):
        if need_audio:
            p.add_argument("--audio")
        p.add_argument("--output-dir")
        p.add_argument("--device", default="auto", choices=["auto", "cuda", "rocm", "directml", "cpu"])
        p.add_argument("--lang", help="zh / ja / en (default auto)")

    p_sep = sub.add_parser("separate", help="vocal separation (UVR5)")
    add_common(p_sep, True)
    p_sep.add_argument("--out", dest="output_dir", help="output directory (same as --output-dir; default <audio parent>/outputs/separation)")
    p_sep.add_argument("--model", default="Karaoke", help="preset name (Karaoke/FullVocal/Band4/Producer5) or weight file name")
    p_sep.add_argument("--agg", type=int, default=0)
    p_sep.set_defaults(func=cmd_separate)

    p_tr = sub.add_parser("transcribe", help="vocal to MIDI (Vocal2Midi pipeline)")
    add_common(p_tr, True)
    p_tr.add_argument("--lyrics", help="reference lyrics for alignment")
    p_tr.add_argument("--slicing", default=None, choices=["default", "smart", "heuristic", "grid"])
    p_tr.add_argument("--slice-min", type=float, default=None)
    p_tr.add_argument("--slice-max", type=float, default=None)
    p_tr.add_argument("--bpm", type=float, default=120.0)
    p_tr.add_argument("--quant", default="bayes", choices=["simple", "smart", "bayes", "dp"])
    p_tr.add_argument("--formats", default="mid,txt", help="comma list: mid,txt,csv,ustx")
    p_tr.add_argument("--separate", action="store_true", help="run separation first")
    p_tr.set_defaults(func=cmd_transcribe)

    p_asr = sub.add_parser("asr", help="speech recognition (FunASR)")
    add_common(p_asr, True)
    p_asr.add_argument("--engine", default="sensevoice", choices=["sensevoice", "paraformer"])
    p_asr.add_argument("--spk_num", "--speakers", dest="speakers", type=int, default=0, help="number of speakers for diarization (0=off)")
    p_asr.add_argument("--no-punc", action="store_true")
    p_asr.add_argument("--srt", action="store_true")
    p_asr.add_argument("--fmt", dest="formats", default=None, help="comma list of output formats: txt,srt,json (default txt)")
    p_asr.add_argument("--separate", action="store_true", help="run separation first")
    p_asr.set_defaults(func=cmd_asr)

    p_tts = sub.add_parser("tts", help="voice clone / TTS (GPT-SoVITS)")
    add_common(p_tts, False)
    p_tts.add_argument("--ref-audio")
    p_tts.add_argument("--ref-text", default="")
    p_tts.add_argument("--ref-lang", default=None)
    p_tts.add_argument("--text", required=True)
    p_tts.add_argument("--version", default="v2Pro", choices=["v2Pro", "v2ProPlus"])
    p_tts.add_argument("--speed", type=float, default=1.0, help="speed factor (0.5~1.5, default 1.0)")
    p_tts.set_defaults(func=cmd_tts)

    p_pl = sub.add_parser("pipeline", help="run a full pipeline")
    add_common(p_pl, True)
    p_pl.add_argument("--steps", default=None, help="comma list from: separation,diarization,content_detection,vocal2midi,speech_asr,tts")
    p_pl.add_argument("--model")
    p_pl.add_argument("--agg", type=int, default=0)
    p_pl.add_argument("--speakers", type=int, default=0)
    p_pl.add_argument("--separate", action="store_true")
    p_pl.add_argument("--lyrics", default=None)
    p_pl.add_argument("--slicing", default=None, choices=["default", "smart", "heuristic", "grid"])
    p_pl.add_argument("--quant", default=None, choices=["simple", "smart", "bayes", "dp"])
    p_pl.add_argument("--bpm", type=float, default=120.0)
    p_pl.add_argument("--formats", default="mid,txt")
    p_pl.add_argument("--engine", default="sensevoice", choices=["sensevoice", "paraformer"])
    p_pl.add_argument("--srt", action="store_true")
    p_pl.add_argument("--no-punc", action="store_true")
    p_pl.add_argument("--asr-only", dest="asr_only", action="store_true")
    p_pl.add_argument("--vocal2midi-off", dest="vocal2midi_off", action="store_true")
    p_pl.add_argument("--tts-text", dest="tts_text", default=None)
    p_pl.add_argument("--tts-version", dest="tts_version", default=None, choices=["v2Pro", "v2ProPlus"])
    p_pl.add_argument("--ref-audio", dest="ref_audio", default=None)
    p_pl.add_argument("--speed", type=float, default=None, help="TTS speed factor (0.5~1.5)")
    p_pl.set_defaults(func=cmd_pipeline)

    return parser


def main(argv: list[str] | None = None) -> int:
    if os.environ.get("VOCALSTUDIO_SMOKE") == "1":
        # frozen self-check probe: report the bundled torch build, then exit 0
        try:
            import torch as _torch

            print(f"SMOKE_TORCH {_torch.__version__} cuda={_torch.cuda.is_available()}")
        except Exception as exc:  # torch is optional for the CLI boot probe
            print(f"SMOKE_TORCH unavailable ({type(exc).__name__})")
        print("SMOKE_FROZEN_PASS")
        return 0
    args = build_parser().parse_args(argv)
    if not getattr(args, "func", None):
        build_parser().print_help()
        return 1
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
