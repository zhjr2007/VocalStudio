"""Separation engine package (wave B-5/B-6)."""

from pathlib import Path

from core.config import ModelPaths
from .uvr5_engine import PRESET_WEIGHTS, UVR5Engine

#: Class resolved by `engines.get_engine("separation")`.
engine_class = UVR5Engine


def run_separation(ctx, cfg, progress_cb) -> None:
    """Orchestrator entry point for the separation step (wave B-6)."""
    model_paths = getattr(cfg, "model_paths", None) or ModelPaths()
    if not getattr(model_paths, "uvr5_weights", ""):
        model_paths = ModelPaths.from_settings()
    engine = UVR5Engine(model_paths)
    try:
        spec = cfg.separation_model or "Karaoke"
        device = getattr(cfg, "device", None) or "auto"
        engine.load(spec, device=device)
        out_dir = Path(cfg.output_dir) if getattr(cfg, "output_dir", "") else (
            Path(ctx.audio_path).parent / "outputs" / "separation"
        )
        paths = engine.run(str(ctx.audio_path), out_dir, intensity=cfg.separation_agg)
    finally:
        engine.unload()

    ctx.vocals_path = paths.get("vocals")
    ctx.accompaniment_path = paths.get("accompaniment")
    extra = {name: path for name, path in paths.items() if name not in ("vocals", "accompaniment")}
    if extra:
        ctx.results["separation_stems"] = extra
    progress_cb(1.0, f"wrote {len(paths)} stem(s)")


__all__ = ["PRESET_WEIGHTS", "UVR5Engine", "engine_class", "run_separation"]
