"""Pipeline orchestration: step ordering, execution, cancellation."""

from __future__ import annotations

import importlib

from .config import AudioType, PipelineConfig, PipelineStep
from .context import PipelineContext
from .exceptions import CancellationError, ModelNotFoundError, VocalStudioError

STEP_ORDER: tuple[PipelineStep, ...] = (
    PipelineStep.SEPARATION,
    PipelineStep.DIARIZATION,
    PipelineStep.CONTENT_DETECTION,
    PipelineStep.VOCAL2MIDI,
    PipelineStep.SPEECH_ASR,
    PipelineStep.TTS,
)

STEP_ENTRYPOINTS: dict[str, tuple[str, str]] = {
    "separation": ("engines.separation", "run_separation"),
    "diarization": ("engines.speaker", "run_diarization"),
    "vocal2midi": ("engines.vocal2midi.pipeline", "run_vocal2midi"),
    "speech_asr": ("engines.speech_asr", "run_speech_asr"),
    "tts": ("engines.tts.voice_clone", "run_tts"),
}


class VocalStudioOrchestrator:
    """Runs enabled pipeline steps in canonical order against a shared context.

    Engine entry points follow the convention:
        run_<step>(ctx: PipelineContext, cfg: PipelineConfig, progress_cb) -> None
    where progress_cb(fraction: float, message: str = "") reports step-local progress.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.steps: list[PipelineStep] = []
        self._cancelled = False
        self.progress_callback = None

    # -- construction ------------------------------------------------------

    def build_steps(self) -> list[PipelineStep]:
        cfg = self.config
        if cfg.pipeline_steps:
            normalized = []
            for value in cfg.pipeline_steps:
                try:
                    step = value if isinstance(value, PipelineStep) else PipelineStep(str(value))
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"Unknown pipeline step: {value!r}") from exc
                normalized.append(step)
            wanted = set(normalized)
            steps = [s for s in STEP_ORDER if s in wanted]
        else:
            steps = []
            if cfg.enable_separation:
                steps.append(PipelineStep.SEPARATION)
            if cfg.enable_diarization:
                steps.append(PipelineStep.DIARIZATION)
            if cfg.auto_detect_content:
                steps.append(PipelineStep.CONTENT_DETECTION)
            if cfg.enable_vocal2midi:
                steps.append(PipelineStep.VOCAL2MIDI)
            if cfg.enable_speech_asr:
                steps.append(PipelineStep.SPEECH_ASR)
            if cfg.enable_tts:
                steps.append(PipelineStep.TTS)
        self.steps = steps
        return steps

    # -- execution ---------------------------------------------------------

    def with_progress(self, callback):
        self.progress_callback = callback
        return self

    def cancel(self) -> None:
        self._cancelled = True

    def execute(
        self,
        context: PipelineContext | None = None,
    ) -> PipelineContext:
        cfg = self.config
        steps = self.build_steps()
        if not steps:
            raise VocalStudioError("No pipeline steps are enabled")
        ctx = context or PipelineContext(audio_path=cfg.audio_path)
        if not ctx.audio_path and cfg.audio_path:
            ctx.audio_path = cfg.audio_path
        cfg.validate()

        total = len(steps)
        for index, step in enumerate(steps):
            self._ensure_cancelled(step)
            skipped_reason = self._skip_reason(step, ctx)
            if skipped_reason:
                ctx.set_step_status(step.value, "skipped", skipped_reason)
                continue
            label = f"[{index + 1}/{total}] {step.value}"
            self._notify(index / total, f"{label} started")
            ctx.set_step_status(step.value, "running")
            self._run_step(ctx, step)
            ctx.set_step_status(step.value, "done")
            self._notify((index + 1) / total, f"{label} done")
        return ctx

    # -- internals ----------------------------------------------------------

    def _notify(self, fraction: float, message: str) -> None:
        if callable(self.progress_callback):
            try:
                self.progress_callback(fraction, message)
            except Exception:
                pass

    def _ensure_cancelled(self, step: PipelineStep) -> None:
        checker = self.config.cancel_checker
        cancelled = self._cancelled
        if not cancelled and callable(checker):
            try:
                cancelled = bool(checker())
            except Exception:
                pass
        if cancelled:
            raise CancellationError("Pipeline cancelled", step=step.value)

    def _skip_reason(self, step: PipelineStep, ctx: PipelineContext) -> str | None:
        if step is PipelineStep.VOCAL2MIDI and ctx.audio_type is AudioType.SPEECH:
            return "content detected as speech"
        if step is PipelineStep.SPEECH_ASR and ctx.audio_type is AudioType.SINGING:
            return "content detected as singing"
        if step is PipelineStep.TTS and (not self.config.ref_audio or not self.config.tts_text):
            return "reference audio / target text missing"
        return None

    def _run_step(self, ctx: PipelineContext, step: PipelineStep) -> None:
        cfg = self.config
        if step is PipelineStep.CONTENT_DETECTION:
            from .content_detector import ContentDetector

            detector = ContentDetector()
            source = ctx.active_vocals_path or cfg.audio_path
            ctx.audio_type = detector.detect(source)
            metrics = self._pitch_metrics(detector, source)
            if metrics is not None:
                ctx.results["pitch_metrics"] = metrics.to_dict()
            return

        module_name, fn_name = STEP_ENTRYPOINTS[step.value]
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            raise VocalStudioError(
                f"Engine for step '{step.value}' is not available (missing module '{exc.name}'). "
                "This stage may not be built yet.",
                step=step.value,
            ) from exc
        runner = getattr(module, fn_name)

        def _progress(fraction: float, message: str = "") -> None:
            base = self.steps.index(step) / max(len(self.steps), 1)
            span = 1.0 / max(len(self.steps), 1)
            self._notify(base + span * fraction, f"{step.value}: {message}")

        try:
            runner(ctx, cfg, _progress)
        except FileNotFoundError as exc:
            raise ModelNotFoundError(
                f"model missing for step '{step.value}': {exc}",
                step=step.value,
                model=str(exc),
            ) from exc
        except ValueError as exc:
            if "missing" in str(exc).lower():
                raise ModelNotFoundError(
                    f"model missing for step '{step.value}': {exc}",
                    step=step.value,
                    model=str(exc),
                ) from exc
            raise

    @staticmethod
    def _pitch_metrics(detector: "ContentDetector", source: str):
        import librosa

        y, sr = librosa.load(str(source), sr=16000, mono=True)
        return detector.measure(y, sr)

