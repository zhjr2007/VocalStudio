"""Vocal-to-MIDI pipeline runner (wave C1-9): pitch-only (+ optional chords).

Stages, logged as ``Stage x/N <name>`` lines (N = 5, or 6 when chord analysis
is enabled)::

    load   - GAME model suite via GameEngine (ONNX/pytorch backend)
    slice  - RMS/smart/heuristic/grid slicing of the full waveform
    pitch  - per-chunk note extraction with GAME (global onsets)
    quant  - optional BPM-aware grid quantization
    chord  - (cfg.enable_chords only) madmom chord/key/meter analysis,
             written as <key>.chords.json + <key>.chords.csv
    export - selected note formats via note_io.save_notes

:func:`run` is the core entry point used by the GUI page job;
:func:`run_vocal2midi` is the orchestrator step entry resolved from
``core.orchestrator.STEP_ENTRYPOINTS["vocal2midi"]``. ``cfg`` is duck-typed:
any object exposing the attributes below works (PipelineConfig, the page's
V2MidiJobCfg, ...); missing attributes fall back to the reference defaults.
"""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np

from core.config import PROJECT_ROOT

# Reference defaults (Vocal2Midi auto_lyric_hybrid __main__), used when cfg
# does not carry the attribute.
_DEFAULTS: dict[str, Any] = {
    "slicing_method": "default",
    "slice_min_sec": 5.0,
    "slice_max_sec": 10.0,
    "tempo": 120.0,
    "quantization_mode": "bayes",
    "quantization_step": 16,
    "output_formats": ("mid", "txt"),
    "language": "auto",
    "device": "auto",
    "seg_threshold": 0.2,
    "seg_radius": 0.02,
    "est_threshold": 0.2,
    "batch_size": 4,
    "t0": 0.0,
    "nsteps": 8,
    "enable_chords": False,
    "chord_hpss": False,
    "min_chord_beats": 0.0,
    "output_lyrics": False,
    "reference_lyrics": "",
}

_GAME_DEFAULT_REL = pathlib.Path("Vocal2Midi") / "experiments" / "GAME-1.0.3-medium-onnx"
_RMVPE_DEFAULT_REL = pathlib.Path("Vocal2Midi") / "experiments" / "RMVPE" / "rmvpe.onnx"


def _attr(cfg: Any, name: str) -> Any:
    value = getattr(cfg, name, None) if cfg is not None else None
    return _DEFAULTS[name] if value is None else value


def resolve_model_paths(model_paths=None) -> tuple[pathlib.Path, pathlib.Path]:
    """Resolve (GAME model dir, RMVPE onnx path) from settings or repo defaults."""
    from core.config import ModelPaths

    mp = model_paths if model_paths is not None else ModelPaths.from_settings()

    def pick(name: str, default_rel: pathlib.Path) -> pathlib.Path:
        value = getattr(mp, name, "") or ""
        if value:
            p = pathlib.Path(value)
            return p if p.is_absolute() else PROJECT_ROOT / p
        return PROJECT_ROOT / default_rel

    return pick("game", _GAME_DEFAULT_REL), pick("rmvpe", _RMVPE_DEFAULT_REL)


def run(audio: np.ndarray, sr: int, cfg=None, progress_cb=None) -> dict[str, Any]:
    """Run the pitch-only (+ optional chord) vocal-to-MIDI pipeline.

    Args:
        audio: mono or multi-channel waveform (any numeric dtype).
        sr: sample rate of ``audio``.
        cfg: duck-typed config (see module docstring); also reads optional
            ``out_dir`` / ``key`` / ``model_paths`` / ``log_cb`` /
            ``cancel_checker`` attributes.
        progress_cb: ``callback(fraction, stage_name)``; fractions are the
            stage boundaries (monotone, ending at 1.0).

    Returns a dict with ``notes`` (list[NoteInfo]), ``files`` (format -> path),
    ``chord_files`` ({'json','csv'} -> path or {}), ``chords`` (songData-style
    result or None), plus run metadata.
    """
    from engines.vocal2midi.game_engine import GameEngine
    from engines.vocal2midi.note_io import normalize_output_formats, save_notes
    from engines.vocal2midi.quantization import quantize_notes, should_apply_quantization
    from engines.vocal2midi.slicer import slice as slice_audio

    waveform = np.asarray(audio, dtype=np.float32)
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1).astype(np.float32)

    out_dir = pathlib.Path(getattr(cfg, "out_dir", "") or (pathlib.Path.cwd() / "outputs" / "vocal2midi"))
    key = str(getattr(cfg, "key", "") or "vocal2midi")
    formats = normalize_output_formats(_attr(cfg, "output_formats"))
    enable_chords = bool(_attr(cfg, "enable_chords"))
    output_lyrics = bool(_attr(cfg, "output_lyrics"))
    lyric_stages = ["lyric"] if output_lyrics else []
    stages = ["load", "slice", "pitch", "quant"] + lyric_stages + (["chord"] if enable_chords else []) + ["export"]
    total = len(stages)

    def log(message: str) -> None:
        callback = getattr(cfg, "log_cb", None) if cfg is not None else None
        if callable(callback):
            try:
                callback(str(message))
            except Exception:
                pass

    def check_cancelled() -> None:
        checker = getattr(cfg, "cancel_checker", None) if cfg is not None else None
        if callable(checker) and checker():
            from core.exceptions import CancellationError

            raise CancellationError("vocal2midi cancelled")

    def progress(fraction: float, stage: str) -> None:
        if progress_cb is not None:
            try:
                progress_cb(float(fraction), stage)
            except Exception:
                pass

    game_dir, rmvpe_path = resolve_model_paths(getattr(cfg, "model_paths", None))
    cancel_checker = getattr(cfg, "cancel_checker", None) if cfg is not None else None

    # -- Stage 1/5 load -------------------------------------------------------
    index = 0
    progress(index / total, stages[index])
    log(f"Stage {index + 1}/{total} load")
    game = GameEngine()
    try:
        backend = game.load(game_dir, device=str(_attr(cfg, "device")))
        check_cancelled()

        # -- Stage 2/5 slice ---------------------------------------------------
        index += 1
        progress(index / total, stages[index])
        log(f"Stage {index + 1}/{total} slice")
        chunks = slice_audio(
            waveform,
            int(sr),
            str(_attr(cfg, "slicing_method")),
            length=float(_attr(cfg, "slice_min_sec")),
            max_len_sec=float(_attr(cfg, "slice_max_sec")),
        )
        log(f"sliced into {len(chunks)} chunk(s)")
        check_cancelled()

        # -- Stage 3/5 pitch -----------------------------------------------------
        index += 1
        progress(index / total, stages[index])
        log(f"Stage {index + 1}/{total} pitch")
        t0 = float(_attr(cfg, "t0"))
        nsteps = max(1, int(_attr(cfg, "nsteps")))
        ts = [t0 + i * (1.0 - t0) / nsteps for i in range(nsteps)]  # reference schedule
        notes = game.infer(
            chunks,
            int(sr),
            ts=ts,
            seg_threshold=float(_attr(cfg, "seg_threshold")),
            seg_radius=float(_attr(cfg, "seg_radius")),
            est_threshold=float(_attr(cfg, "est_threshold")),
            batch_size=int(_attr(cfg, "batch_size")),
            language=None if str(_attr(cfg, "language")) == "auto" else str(_attr(cfg, "language")),
            cancel_checker=cancel_checker,
        )
        notes.sort(key=lambda n: n.onset)
        log(f"{len(notes)} note(s) extracted (backend={game.backend})")
        check_cancelled()

        # -- Stage 4/5 quant -----------------------------------------------------
        index += 1
        progress(index / total, stages[index])
        log(f"Stage {index + 1}/{total} quant")
        tempo = float(_attr(cfg, "tempo"))
        mode = str(_attr(cfg, "quantization_mode"))
        step = int(_attr(cfg, "quantization_step"))
        if should_apply_quantization(mode, step):
            quantize_notes(notes, tempo=tempo, step=step, mode=mode)
            log(f"quantized with mode={mode} step={step} bpm={tempo:g}")
        else:
            log("quantization skipped (mode/step)")
        check_cancelled()

        # -- Stage 5/7 lyric (optional) -------------------------------------------
        chunk_logs: list[str] = []
        lyric_note = None
        if output_lyrics:
            index += 1
            progress(index / total, stages[index])
            log(f"Stage {index + 1}/{total} lyric")
            from engines.vocal2midi.lyric_pipeline import run_lyric_alignment, write_asr_match_log

            _per_chunk, chunk_logs = run_lyric_alignment(
                chunks,
                int(sr),
                cfg,
                notes,
                progress_cb=progress,
                cancel_checker=cancel_checker,
                log_cb=log,
            )
            _match_log = write_asr_match_log(chunk_logs, out_dir, key)
            lyric_note = f"lyrics aligned for {len(_per_chunk)} chunk(s); log={_match_log}"
            log(lyric_note)
            check_cancelled()

        # -- Stage 5/6 chord (optional) -------------------------------------------
        chords = None
        chord_files: dict[str, pathlib.Path] = {}
        if enable_chords:
            index += 1
            progress(index / total, stages[index])
            log(f"Stage {index + 1}/{total} chord")
            from engines.chords import ChordEngine, write_chord_outputs

            engine = ChordEngine()
            try:
                chords = engine.analyze(
                    waveform,
                    int(sr),
                    hpss=bool(_attr(cfg, "chord_hpss")),
                    min_chord_beats=float(_attr(cfg, "min_chord_beats")),
                    cancel_checker=cancel_checker,
                )
            finally:
                engine.unload()
            chord_files = write_chord_outputs(chords, out_dir, key)
            log(
                f"chords: key={chords['key']} bpm={chords['bpm']} "
                f"time_sig={chords['time_sig']} changes={len(chords['chordPositions'])}"
            )
            check_cancelled()

        # -- Stage 6/6 export ------------------------------------------------------
        index += 1
        progress(index / total, stages[index])
        log(f"Stage {index + 1}/{total} export")
        rmvpe_result = None
        if "ustx" in formats:
            from engines.vocal2midi.rmvpe_engine import RmvpeEngine

            rmvpe = RmvpeEngine()
            try:
                rmvpe.load(rmvpe_path, device=str(_attr(cfg, "device")))
                out = rmvpe.infer(waveform, int(sr))
                # save_ustx only reads .midi_pitch and .time_step_seconds.
                class _RmvpeView:  # noqa: D106 - tiny attribute view for the reference exporter
                    midi_pitch = out["midi_pitch"]
                    time_step_seconds = out["time_step_seconds"]

                rmvpe_result = _RmvpeView()
            finally:
                rmvpe.unload()
        files = save_notes(notes, out_dir, key, formats, tempo=tempo, rmvpe_result=rmvpe_result)
        log(f"wrote {', '.join(sorted(files)) or 'no formats'} to {out_dir}")
    finally:
        game.unload()

    progress(1.0, "done")
    return {
        "notes": notes,
        "files": {fmt: path for fmt, path in files.items()},
        "chord_files": chord_files,
        "chords": chords,
        "n_chunks": len(chunks),
        "asr_match_log": [] if not output_lyrics else chunk_logs,
        "backend": game.backend,
    }


def run_vocal2midi(ctx, cfg, progress_cb=None) -> None:
    """Orchestrator step entry (see core.orchestrator.STEP_ENTRYPOINTS).

    Reads the active vocals (or source audio) from ``ctx``, runs :func:`run`,
    and stores the notes on ``ctx.notes`` plus files/chords under
    ``ctx.results["vocal2midi"]``.
    """
    import soundfile as sf

    audio_path = ctx.active_vocals_path or getattr(cfg, "audio_path", "") or ""
    if not audio_path or not pathlib.Path(audio_path).exists():
        raise FileNotFoundError(f"vocal2midi: audio file not found: {audio_path!r}")

    frames, rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
    waveform = np.mean(frames, axis=1).astype(np.float32) if frames.shape[1] > 1 else frames[:, 0].copy()

    out_dir = pathlib.Path(getattr(cfg, "output_dir", "") or (pathlib.Path(audio_path).parent / "outputs" / "vocal2midi"))
    key = pathlib.Path(audio_path).stem or "vocal2midi"

    class _RunCfg:  # noqa: D106 - forwards PipelineConfig attributes to run()
        def __getattr__(self, name):
            if name in ("out_dir", "key"):
                return {"out_dir": out_dir, "key": key}[name]
            return getattr(cfg, name)

    result = run(waveform, int(rate), _RunCfg(), progress_cb)
    ctx.notes = list(result["notes"])
    ctx.results["vocal2midi"] = {
        "files": {fmt: str(path) for fmt, path in result["files"].items()},
        "chord_files": {name: str(path) for name, path in result["chord_files"].items()},
        "chords": result["chords"],
        "n_chunks": result["n_chunks"],
        "backend": result["backend"],
    }
