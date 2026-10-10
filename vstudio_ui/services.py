"""Business-layer adapter for the new UI.

Every call the interface makes into ``core`` goes through this module. Two reasons:

1. **One implementation site for the frozen contract.** ``ui-design/contract.md``
   lists six contact surfaces; if each page imports ``core`` itself, the contract
   is enforced in eight places and understood in none. Here it is enforced once.
2. **Nothing slow may run on the GUI thread.** ``detect_system_hardware`` shells
   out to WMI and ``auto_select_device`` imports torch; both take seconds on a
   cold process. Every function below is therefore written to be safe to call
   from a worker thread and is handed to :class:`vstudio_ui.jobs.JobHandle`
   rather than called inline. A frozen window is the classic symptom of getting
   this wrong, and it is invisible in unit tests.

Nothing here mutates business state: these are read-only probes plus the
``as_job`` adapter. Writes (settings, running an engine) get their own entry
point when the page that needs them is built.

Every probe swallows its own exceptions and reports them as data — a diagnostic
page that crashes because the thing it diagnoses is broken is useless.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
import json
import sys
import os
import importlib.util

from core import app_settings, config as config_module, device_utils
from core.config import ModelPaths, PipelineConfig
from core.orchestrator import STEP_ORDER, VocalStudioOrchestrator

__all__ = [
    "ASR_ENGINES",
    "CHECK_ORDER",
    "DEVICE_CHOICES",
    "LANGUAGE_CHOICES",
    "PIPELINE_STEPS",
    "PREFERENCE_FIELDS",
    "QUANTIZATION_MODES",
    "SEPARATION_MODELS",
    "SLICING_METHODS",
    "TTS_VERSIONS",
    "Check",
    "Preferences",
    "RunRequest",
    "RunResult",
    "RuntimeSnapshot",
    "SelfCheckReport",
    "StepOutcome",
    "TrainingRequest",
    "TrainingSpec",
    "as_job",
    "build_config",
    "build_training_spec",
    "load_preferences",
    "probe_runtime",
    "run_pipeline_job",
    "run_self_check",
    "run_training_job",
    "save_preferences",
    "short_error",
    "suggest_output_dir",
    "training_tasks",
]

#: Stage names, in the order :func:`run_self_check` visits them. Pages map these
#: onto localised labels; keeping i18n out of this module is deliberate.
CHECK_ORDER: tuple[str, ...] = ("hardware", "backend", "device", "models", "settings")

#: The pipeline steps the UI may enable, in the orchestrator's canonical order.
#: Taken from the business layer rather than hard-coded, so a new step shows up
#: here instead of silently having no page that can reach it.
PIPELINE_STEPS: tuple[str, ...] = tuple(step.value for step in STEP_ORDER)

# Option vocabularies, mirrored from the business layer.
#
# Pages must not import ``core``, but a combo box needs the list of legal values —
# and a hard-coded list in a page is how the UI ends up offering a choice that
# ``PipelineConfig.validate()`` rejects at run time. Reading them from the same
# tuples the validator uses makes that impossible.
DEVICE_CHOICES: tuple[str, ...] = tuple(device_utils.RUNTIME_DEVICE_CHOICES)
LANGUAGE_CHOICES: tuple[str, ...] = tuple(config_module.SUPPORTED_LANGUAGES)
SEPARATION_MODELS: tuple[str, ...] = tuple(config_module.SEPARATION_MODELS)
SLICING_METHODS: tuple[str, ...] = tuple(config_module.SLICING_METHODS)
QUANTIZATION_MODES: tuple[str, ...] = tuple(config_module.QUANTIZATION_MODES)
TTS_VERSIONS: tuple[str, ...] = tuple(config_module.TTS_VERSIONS)
ASR_ENGINES: tuple[str, ...] = tuple(config_module.ASR_ENGINES)

#: Languages the *interface* can be shown in, which is not the same list as
#: ``LANGUAGE_CHOICES`` above: that one is the pipeline's input-language set
#: (``auto`` / ``zh`` / ``ja`` / ``en``) and belongs to the task pages. Offering
#: "auto" as a UI language would be meaningless, and offering "ja" would select a
#: translation table that does not exist.
UI_LANGUAGES: tuple[str, ...] = tuple(app_settings.LANGUAGE_CHOICES)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def short_error(exc: BaseException) -> str:
    """One-line, bounded description of an exception (never a traceback)."""
    text = " ".join(str(exc).split()) or exc.__class__.__name__
    return text if len(text) <= 160 else text[:157] + "..."


def _report(progress_cb: Callable[[float, str], None] | None, fraction: float, stage: str) -> None:
    """Best-effort progress report; a broken callback must not fail the probe."""
    if progress_cb is None:
        return
    try:
        progress_cb(float(fraction), stage)
    except Exception:
        pass


def as_job(fn: Callable[[Callable | None], object]):
    """Adapt ``fn(progress_cb)`` to ``JobRunner``'s ``fn(ctx, cfg, progress_cb)``.

    The frozen signature takes three arguments because engines need a context and
    a config. A probe needs neither, and writing ``lambda ctx, cfg, cb: ...`` at
    every call site is how the arguments get swapped by accident.
    """

    def job(_ctx=None, _cfg=None, progress_cb=None):
        return fn(progress_cb)

    return job


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RuntimeSnapshot:
    """What the business layer can say about this machine right now."""

    cpu: str = ""
    #: ``(name, vram_gb)`` per adapter **as WMI reports it**. WMI lists every
    #: adapter — including emulator virtual displays — in arbitrary order, and its
    #: ``AdapterRAM`` is a 32-bit field that saturates at 4 GB. It is a fallback
    #: and a tooltip, never the headline.
    gpus: tuple[tuple[str, float], ...] = ()
    #: The adapter torch will actually compute on. Authoritative; empty when torch
    #: is unavailable (the packaged frontend excludes it).
    gpu_name: str = ""
    vram_total_gb: float = 0.0
    vram_free_gb: float = 0.0
    #: ``rocm`` / ``cuda`` / ``zluda`` / ``directml`` / ``cpu``.
    backend: str = "cpu"
    #: ``"torch"`` when *backend* came from the runtime, ``"wmi"`` when it is only
    #: a guess from adapter names. Surfaced because the two disagree in a
    #: predictable way: an AMD iGPU next to an NVIDIA card makes the WMI heuristic
    #: answer ``rocm`` while torch answers ``cuda``.
    backend_source: str = "wmi"
    torch_version: str = ""
    device: str = ""
    device_setting: str = "auto"
    models_total: int = 0
    models_missing: tuple[str, ...] = ()
    error: str = ""

    @property
    def gpu_names(self) -> tuple[str, ...]:
        return tuple(name for name, _vram in self.gpus)

    @property
    def models_ready(self) -> int:
        return max(0, self.models_total - len(self.models_missing))

    @property
    def models_ok(self) -> bool:
        return self.models_total > 0 and not self.models_missing

    @property
    def has_vram(self) -> bool:
        return self.vram_total_gb > 0


@dataclass(frozen=True)
class Check:
    """One stage of :func:`run_self_check`."""

    key: str
    ok: bool
    detail: str = ""


@dataclass(frozen=True)
class SelfCheckReport:
    checks: tuple[Check, ...] = ()
    snapshot: RuntimeSnapshot = field(default_factory=RuntimeSnapshot)

    @property
    def ok(self) -> bool:
        return bool(self.checks) and all(check.ok for check in self.checks)


@dataclass(frozen=True)
class Preferences:
    """Startup preferences read from ``settings/user.json`` (read-only)."""

    theme: str = "dark"
    language: str = "zh"
    device: str = "auto"
    font_scale: float = 1.0


# ---------------------------------------------------------------------------
# probes (worker-thread safe)
# ---------------------------------------------------------------------------


def probe_runtime(progress_cb: Callable[[float, str], None] | None = None) -> RuntimeSnapshot:
    """Read hardware, backend, resolved device and model readiness.

    Blocking: expect a WMI round trip, a torch import and a CUDA context
    initialisation. Call it through :class:`~vstudio_ui.jobs.JobHandle`.

    The backend is taken from torch when torch can answer, and only falls back to
    the WMI heuristic otherwise. That ordering matters: this machine has an AMD
    iGPU *and* an NVIDIA card, and the heuristic answers ``rocm`` while the real
    device is ``cuda`` — see ``RuntimeSnapshot.backend_source``.
    """
    _report(progress_cb, 0.0, "hardware")

    cpu = ""
    gpus: tuple[tuple[str, float], ...] = ()
    error = ""
    try:
        hardware = device_utils.detect_system_hardware()
        cpu = str(hardware.get("cpu") or "")
        gpus = tuple(
            (str(name), float(vram or 0.0)) for name, vram in (hardware.get("gpus") or ())
        )
    except Exception as exc:  # noqa: BLE001 - reported as data
        error = short_error(exc)

    _report(progress_cb, 0.35, "backend")
    gpu_name = ""
    vram_total_gb = 0.0
    vram_free_gb = 0.0
    torch_version = ""
    backend = "cpu"
    backend_source = "wmi"
    try:
        gpu_info = device_utils.get_gpu_info() or {}
        gpu_name = str(gpu_info.get("name") or "")
        vram_total_gb = float(gpu_info.get("total_vram_gb") or 0.0)
        vram_free_gb = float(gpu_info.get("free_vram_gb") or 0.0)
        torch_version = str(gpu_info.get("torch_version") or "")
        reported = str(gpu_info.get("backend") or "none")
        if reported not in ("", "none"):
            backend, backend_source = reported, "torch"
    except Exception as exc:  # noqa: BLE001
        error = error or short_error(exc)

    if backend_source == "wmi":
        try:
            backend = device_utils.default_backend()
        except Exception as exc:  # noqa: BLE001
            backend = "cpu"
            error = error or short_error(exc)

    _report(progress_cb, 0.65, "device")
    try:
        device = device_utils.auto_select_device()
    except Exception as exc:  # noqa: BLE001
        device = ""
        error = error or short_error(exc)

    device_setting = "auto"
    try:
        device_setting = str(app_settings.get_settings().device or "auto")
    except Exception:  # noqa: BLE001 - a missing user.json is normal
        pass

    _report(progress_cb, 0.9, "models")
    models_total = len(ModelPaths.__dataclass_fields__)
    models_missing: tuple[str, ...] = ()
    try:
        models_missing = missing_model_paths()
    except Exception as exc:  # noqa: BLE001
        models_total = 0
        error = error or short_error(exc)

    _report(progress_cb, 1.0, "")
    return RuntimeSnapshot(
        cpu=cpu,
        gpus=gpus,
        gpu_name=gpu_name,
        vram_total_gb=vram_total_gb,
        vram_free_gb=vram_free_gb,
        backend=backend,
        backend_source=backend_source,
        torch_version=torch_version,
        device=device,
        device_setting=device_setting,
        models_total=models_total,
        models_missing=models_missing,
        error=error,
    )


def run_self_check(progress_cb: Callable[[float, str], None] | None = None) -> SelfCheckReport:
    """Walk all six contract surfaces and report per-stage results.

    This is the page's "can this machine actually do the work" button. It reports
    rather than raises, so a partially broken install still produces a useful
    screen instead of an empty error dialog.
    """
    checks: list[Check] = []

    _report(progress_cb, 0.0, "hardware")
    cpu = ""
    gpus: tuple = ()
    try:
        hardware = device_utils.detect_system_hardware()
        cpu = str(hardware.get("cpu") or "")
        gpus = tuple(hardware.get("gpus") or ())
        checks.append(Check("hardware", bool(cpu), cpu or "unknown"))
    except Exception as exc:  # noqa: BLE001
        checks.append(Check("hardware", False, short_error(exc)))

    _report(progress_cb, 0.3, "backend")
    gpu_name = ""
    vram_total_gb = 0.0
    vram_free_gb = 0.0
    torch_version = ""
    backend, backend_source = "cpu", "wmi"
    try:
        gpu_info = device_utils.get_gpu_info() or {}
        gpu_name = str(gpu_info.get("name") or "")
        vram_total_gb = float(gpu_info.get("total_vram_gb") or 0.0)
        vram_free_gb = float(gpu_info.get("free_vram_gb") or 0.0)
        torch_version = str(gpu_info.get("torch_version") or "")
        reported = str(gpu_info.get("backend") or "none")
        if reported not in ("", "none"):
            backend, backend_source = reported, "torch"
    except Exception:  # noqa: BLE001 - falling back to the WMI guess is a valid answer
        pass

    backend_error = ""
    if backend_source == "wmi":
        try:
            backend = device_utils.default_backend()
        except Exception as exc:  # noqa: BLE001
            backend_error = short_error(exc)
    checks.append(
        Check(
            "backend",
            not backend_error,
            f"{backend} ({backend_source})" if not backend_error else backend_error,
        )
    )

    _report(progress_cb, 0.55, "device")
    try:
        device = device_utils.auto_select_device()
        checks.append(Check("device", True, device))
    except Exception as exc:  # noqa: BLE001
        device = ""
        checks.append(Check("device", False, short_error(exc)))

    _report(progress_cb, 0.78, "models")
    models_total = len(ModelPaths.__dataclass_fields__)
    models_missing: tuple[str, ...] = ()
    try:
        models_missing = missing_model_paths()
        checks.append(
            Check("models", not models_missing, f"{models_total - len(models_missing)}/{models_total}")
        )
    except Exception as exc:  # noqa: BLE001
        models_total = 0
        checks.append(Check("models", False, short_error(exc)))

    _report(progress_cb, 0.92, "settings")
    try:
        app_settings.get_settings().validate()
        checks.append(Check("settings", True, "settings/user.json"))
    except Exception as exc:  # noqa: BLE001
        checks.append(Check("settings", False, short_error(exc)))

    _report(progress_cb, 1.0, "")
    snapshot = RuntimeSnapshot(
        cpu=cpu,
        gpus=tuple((str(n), float(v or 0.0)) for n, v in gpus),
        gpu_name=gpu_name,
        vram_total_gb=vram_total_gb,
        vram_free_gb=vram_free_gb,
        backend=backend,
        backend_source=backend_source,
        torch_version=torch_version,
        device=device,
        models_total=models_total,
        models_missing=models_missing,
    )
    return SelfCheckReport(checks=tuple(checks), snapshot=snapshot)


def load_preferences() -> Preferences:
    """Read startup preferences. Never raises: defaults are a valid answer.

    Theme mapping note: ``AppSettings`` still carries the first-generation
    choices ``auto`` / ``light`` / ``dark`` / ``rhine``. The new UI has exactly
    two themes and does no OS-theme detection yet, so ``auto`` and ``rhine`` both
    read as ``light``.
    """
    try:
        settings = app_settings.get_settings()
    except Exception:  # noqa: BLE001
        return Preferences()

    language = settings.language if settings.language in ("zh", "en") else "zh"
    try:
        font_scale = float(settings.font_scale)
    except (TypeError, ValueError):
        font_scale = 1.0
    return Preferences(
        theme="light" if settings.theme in ("light", "rhine") else "dark",
        language=language,
        device=str(settings.device or "auto"),
        font_scale=font_scale,
    )


#: The preference names :func:`save_preferences` will write. Anything else is a
#: typo, and a typo that silently does nothing is the failure mode this whole
#: module exists to prevent.
PREFERENCE_FIELDS: tuple[str, ...] = ("device", "theme", "language", "font_scale")


def save_preferences(**changes) -> Preferences:
    """Persist *changes* to ``settings/user.json``; return the resulting state.

    This is the **only** writer of user settings in the new UI, which is what
    makes "the header button and the settings page cannot disagree about what is
    on disk" true by construction.

    Three properties, in order of how much they matter:

    1. **Validate before writing.** The candidate object is checked first, so an
       out-of-range font scale or an unknown device raises *without* having
       overwritten a good file with a broken one. Mutating the cached object and
       saving afterwards would leave the cache poisoned as well.
    2. **Unknown names raise.** A misspelled preference would otherwise look
       saved and be gone at the next launch.
    3. **Only the named fields change.** Callers pass one control's value, not a
       whole settings snapshot, so two pages cannot clobber each other's fields
       with stale reads.
    """
    unknown = [name for name in changes if name not in PREFERENCE_FIELDS]
    if unknown:
        raise ValueError(f"unknown preference {unknown[0]!r}")

    current = app_settings.get_settings()
    candidate = app_settings.AppSettings(**{**current.to_dict(), **changes})
    candidate.validate()  # raises on garbage; nothing has been written yet
    candidate.save()  # refreshes the cache for this path
    return load_preferences()


# ---------------------------------------------------------------------------
# running work
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunRequest:
    """What a page wants the business layer to do, in business-layer terms.

    Deliberately *not* a ``PipelineConfig``: a config is a mutable 40-field object
    that a worker thread mutates (``cancel_checker``, ``log_cb``), while this is a
    frozen description a page can build and hand over. :func:`build_config` is the
    one-way conversion, so pages never touch the mutable object at all.
    """

    audio_path: str = ""
    output_dir: str = ""
    device: str = "inherit"
    language: str = "auto"
    #: ``PipelineStep`` values to run, in canonical order. Empty means "derive from
    #: the enable_* flags in *options*", which is the orchestrator's other path.
    steps: tuple[str, ...] = ()
    output_formats: tuple[str, ...] = ("mid", "txt")
    #: The long tail: any remaining ``PipelineConfig`` field name -> value.
    #: Kept as pairs rather than a dict so the dataclass stays hashable.
    options: tuple[tuple[str, object], ...] = ()


@dataclass(frozen=True)
class StepOutcome:
    """One pipeline step's fate, as recorded by the orchestrator in the context."""

    step: str
    status: str = ""  # done | skipped | running
    detail: str = ""

    @property
    def skipped(self) -> bool:
        return self.status == "skipped"


@dataclass(frozen=True)
class RunResult:
    """A finished run, flattened out of ``PipelineContext``.

    Paths are plain strings (empty when the step did not run) rather than
    ``None``, so a page can render any field without a truthiness dance.
    """

    steps: tuple[StepOutcome, ...] = ()
    audio_type: str = ""
    detected_language: str = ""
    vocals_path: str = ""
    accompaniment_path: str = ""
    denoised_vocals_path: str = ""
    notes_count: int = 0
    asr_text: str = ""
    srt_path: str = ""
    tts_audio_path: str = ""
    speakers: tuple[str, ...] = ()
    #: Directory the run wrote to.
    output_dir: str = ""
    #: Files an external-process step reported (training checkpoints today).
    checkpoints: tuple[str, ...] = ()
    #: Exit code when the step ran an external process; ``None`` when it did not.
    exit_code: int | None = None
    error: str = ""
    files: tuple[tuple[str, str], ...] = ()
    backends: tuple[tuple[str, str], ...] = ()

    @property
    def ok(self) -> bool:
        return not self.error

    @property
    def skipped(self) -> tuple[StepOutcome, ...]:
        return tuple(outcome for outcome in self.steps if outcome.skipped)

    @property
    def ran(self) -> tuple[StepOutcome, ...]:
        return tuple(outcome for outcome in self.steps if not outcome.skipped)

    def outcome(self, step: str) -> StepOutcome | None:
        """What happened to *step*, or ``None`` if it was never part of this run.

        Pages need this to avoid reporting a number that was never computed: a
        ``notes_count`` of 0 means "no notes", but only if the MIDI step actually
        ran. Printing 0 because the step was absent reads as a finding.
        """
        for entry in self.steps:
            if entry.step == step:
                return entry
        return None

    def did_run(self, step: str) -> bool:
        """True when *step* ran and was not skipped."""
        entry = self.outcome(step)
        return entry is not None and not entry.skipped

    def produced_paths(self) -> tuple[tuple[str, str], ...]:
        """``(label, path)`` for every file the run wrote, in a stable order."""
        candidates = (
            ("vocals", self.vocals_path),
            ("accompaniment", self.accompaniment_path),
            ("denoised", self.denoised_vocals_path),
            ("srt", self.srt_path),
            ("tts", self.tts_audio_path),
        )
        seen = set()
        result = []
        for label, path in (*candidates, *self.files, *(("checkpoint", p) for p in self.checkpoints)):
            if path and path not in seen:
                result.append((label, path)); seen.add(path)
        return tuple(result)


def suggest_output_dir(audio_path: str, kind: str) -> str:
    """Default output directory, matching the CLI's documented convention.

    ``main.py separate`` documents ``<audio parent>/outputs/separation``; the same
    shape is used for every step so a user who has run the CLI recognises it.
    """
    kind = str(kind or "output").strip() or "output"
    if not audio_path:
        return ""
    try:
        return str(Path(audio_path).parent / "outputs" / kind)
    except (OSError, ValueError):
        return ""


def build_config(request: RunRequest) -> PipelineConfig:
    """Turn a :class:`RunRequest` into the mutable config the engines expect.

    Three things here are not obvious and all three are load-bearing:

    1. **``model_paths`` must be filled in.** Engines read it as
       ``getattr(cfg, "model_paths", None) or ModelPaths.from_settings()`` — and an
       empty ``ModelPaths()`` is **truthy**, so the ``or`` short-circuits and the
       engine gets blank paths. Only the separation engine has its own fallback.
    2. **``log_cb`` must exist before the job starts.** ``JobRunner`` installs its
       log forwarder only if the attribute is already there, and it does that
       *before* calling the job body — too late for the body to add it. Without
       this line the log pane stays empty forever.
    3. **Unknown option names raise.** A page that sets a knob the config does not
       have would otherwise appear to work while doing nothing.
    """
    cfg = PipelineConfig(
        audio_path=str(request.audio_path or ""),
        output_dir=str(request.output_dir or ""),
        device=resolve_device_preference(request.device),
        language=str(request.language or "auto"),
        output_formats=list(request.output_formats or ()),
        pipeline_steps=list(request.steps or ()),
    )
    try:
        cfg.model_paths = ModelPaths.from_settings()
    except Exception:  # noqa: BLE001 - an unreadable settings file is reported by the engine
        pass

    known = PipelineConfig.__dataclass_fields__
    owned_by_runner = ("cancel_checker", "log_cb", "model_paths", "pipeline_steps")
    for name, value in _option_items(request.options):
        # Check ownership first: ``log_cb`` is not a config field at all, so testing
        # membership in ``known`` first would report it as a typo rather than as
        # "the runner owns this".
        if name in owned_by_runner:
            raise ValueError(f"option {name!r} is owned by the runner, not a page")
        if name not in known:
            raise ValueError(f"unknown pipeline option {name!r}")
        setattr(cfg, name, value)

    cfg.log_cb = None
    return cfg


def _option_items(options) -> Iterable[tuple[str, object]]:
    """Accept either ``((name, value), ...)`` or a plain mapping."""
    if not options:
        return ()
    if isinstance(options, Mapping):
        return tuple(options.items())
    return tuple(options)


def run_pipeline_job(_ctx=None, cfg=None, progress_cb=None) -> RunResult:
    """``JobRunner`` body: run *cfg*'s enabled steps and return a :class:`RunResult`.

    Written for the frozen signature, so it is handed straight to
    ``JobHandle.start(services.run_pipeline_job, None, cfg)``. That pairing is what
    makes cancellation work: ``JobRunner.cancel()`` installs ``cfg.cancel_checker``
    and the orchestrator polls it at every step boundary.

    ``CancellationError`` is deliberately **not** caught here. ``JobRunner`` maps it
    to ``done(None, True)``; catching it would turn "the user pressed Cancel" into
    "the job finished", and the page would report success.
    """
    if cfg is None:
        raise ValueError("run_pipeline_job needs a config")

    runtime = preflight(cfg)
    # Not every engine creates its own output directory (separation does not), and
    # a run that dies on "no such directory" after loading a 300 MB model wastes
    # the load. Doing it here means it happens on the worker thread, once.
    if cfg.output_dir:
        try:
            Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        except OSError:
            pass  # let the engine report it with its own, more specific error

    from core.context import PipelineContext
    context = PipelineContext(audio_path=getattr(cfg, 'audio_path', ''))
    class ServiceOrchestrator(VocalStudioOrchestrator):
        def _run_step(self, ctx, step):
            super()._run_step(ctx, step)
            if step.value == 'separation':
                _normalize_separation(ctx, cfg)
                if getattr(cfg, 'enable_denoise', False):
                    _run_denoise(ctx, cfg)
    orchestrator = ServiceOrchestrator(cfg)
    def report(fraction, message=''):
        # Normalize raw vocal-weight stem names before the next step reads them.
        _normalize_separation(context, cfg)
        if progress_cb: progress_cb(fraction, message)
    orchestrator.with_progress(report)
    context = orchestrator.execute(context)
    _normalize_separation(context, cfg)
    result = _result_from_context(context, cfg.output_dir)
    if runtime:
        result = replace(result, backends=tuple((s.step, runtime['onnx'] if s.step == 'vocal2midi' else 'CPU' if s.step == 'content_detection' else runtime['pytorch']) for s in result.ran))
    return result


def _normalize_separation(context, cfg):
    stems = context.results.get('separation_stems', {})
    spec = str(getattr(cfg, 'separation_model', '')).lower()
    if not context.vocals_path and (spec.startswith('hp2_') or spec.startswith('hp5_') or spec.startswith('vr-deecho')):
        context.vocals_path = stems.get(spec) or stems.get(spec.replace('-', '_'))
    if not context.accompaniment_path:
        context.accompaniment_path = stems.get('other')


def _run_denoise(context, cfg):
    """Optional FRCRN enhancement after separation; same device as the task."""
    import torch
    import librosa
    import numpy as np
    import soundfile as sf
    from modelscope.models.audio.ans.frcrn import FRCRN
    from core.exceptions import CancellationError
    def check():
        if cfg.cancel_checker and cfg.cancel_checker(): raise CancellationError('Denoise cancelled')
    check()
    model = cfg.model_paths.resolve('denoise')/'speech_frcrn_ans_cirm_16k'
    device = device_utils.resolve_pytorch_device(cfg.device)
    output = Path(cfg.output_dir)/(Path(context.active_vocals_path).stem+'_denoised.wav')
    if cfg.log_cb: cfg.log_cb(f'Denoise: FRCRN / {device}')
    parameters = json.loads((model/'configuration.json').read_text(encoding='utf-8'))['model']
    enhancer = FRCRN(**parameters)
    # The downloaded artifact is a state dictionary, never executable model code.
    enhancer.load_state_dict(torch.load(model/'pytorch_model.bin', map_location='cpu', weights_only=True), strict=True)
    enhancer.to(device).eval()
    try:
        audio, sr = sf.read(context.active_vocals_path, dtype='float32', always_2d=True)
        audio = librosa.resample(audio.mean(axis=1), orig_sr=sr, target_sr=16000)
        if not audio.size: raise ValueError('Denoise input is empty')
        rms = float(np.sqrt(np.mean(audio**2)))
        if rms:
            audio = audio * (10**(-25/20)/rms)
            power = audio**2; loud = power[power > power.mean()]
            if loud.size: audio = audio * (10**(-25/20)/float(np.sqrt(loud.mean())))
        cleaned = np.zeros_like(audio)
        # Bounded windows keep this optional stage within the 8 GB task budget.
        window, margin = 4*16000, 4000
        with torch.inference_mode():
            for start in range(0, len(audio), window):
                check()
                left, right = max(0, start-margin), min(len(audio), start+window+margin)
                chunk = audio[left:right]
                aligned = ((max(16000, len(chunk))+319)//320)*320
                padded = np.pad(chunk, (0, aligned-len(chunk)))
                values = enhancer(torch.from_numpy(padded).unsqueeze(0).to(device))[4][0].cpu().numpy()
                size = min(window, len(audio)-start)
                cleaned[start:start+size] = values[start-left:start-left+size]
        sf.write(str(output), np.clip(cleaned, -1, 1), 16000)
        check()
        context.denoised_vocals_path = str(output)
    finally:
        del enhancer
        import gc
        gc.collect()
        if device.type == 'cuda':
            import torch
            torch.cuda.empty_cache()


def _result_from_context(context, output_dir="") -> RunResult:
    """Flatten a ``PipelineContext`` into the frozen view the pages render."""
    steps = tuple(
        StepOutcome(
            step=str(name),
            status=str(entry.get("status") or ""),
            detail=str(entry.get("detail") or ""),
        )
        for name, entry in sorted((context.step_status or {}).items())
    )
    audio_type = getattr(context.audio_type, "value", None) or str(context.audio_type or "")
    files = []
    backends = []
    for step, value in (getattr(context, 'results', {}) or {}).items():
        if not isinstance(value, dict): continue
        if step == 'separation_stems':
            files.extend((f'separation: {name}', str(path)) for name, path in value.items() if isinstance(path, (str, Path)))
        for group in ('files', 'chord_files'):
            for name, path in (value.get(group) or {}).items():
                if isinstance(path, (str, Path)):
                    files.append((f'{step}: {name}', str(path)))
        if value.get('backend'):
            backends.append((str(step), str(value['backend'])))
    return RunResult(
        steps=steps,
        audio_type=audio_type,
        detected_language=str(context.detected_language or ""),
        vocals_path=str(context.vocals_path or ""),
        accompaniment_path=str(context.accompaniment_path or ""),
        denoised_vocals_path=str(context.denoised_vocals_path or ""),
        notes_count=len(context.notes or ()),
        asr_text=str(context.asr_text or ""),
        srt_path=str(context.srt_path or ""),
        tts_audio_path=str(context.tts_audio_path or ""),
        speakers=tuple(str(name) for name in (context.speakers or {})),
        output_dir=str(output_dir or getattr(context, 'output_dir', '') or ''),
        files=tuple(files),
        backends=tuple(backends),
    )


# ---------------------------------------------------------------------------
# training (a different beast: its own job body, its own spec, real process kill)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingRequest:
    """A training run, described the same frozen way a pipeline run is."""

    task: str = "sensevoice_singing"
    data_dir: str = ""
    output_dir: str = ""
    #: Defaults to ``cuda`` on purpose. Fine-tuning pins the CUDA GPU regardless
    #: of the app's auto backend, because the adapters set ``CUDA_VISIBLE_DEVICES``
    #: and per-entry GPU args. Inheriting ``auto`` here would silently train on
    #: the CPU.
    device: str = "cuda"
    epochs: int = 100
    batch: int = 4
    lr: float = 5e-4


class TrainingSpec:
    """Mutable spec for ``engines.training.run_training_job``.

    Duck-typed to match ``TrainingJobSpec`` plus the two attributes ``JobRunner``
    needs to be able to set: ``log_cb`` and ``cancel_checker``. Both must exist
    *before* the job starts — ``JobRunner`` only wraps what is already there.
    """

    def __init__(self, task="", data_dir="", params=None, output_dir="", device="cuda") -> None:
        self.task = str(task)
        self.data_dir = str(data_dir)
        self.params = dict(params or {})
        self.output_dir = str(output_dir)
        self.device = str(device or "cuda")
        self.log_cb = None
        self.cancel_checker = None


def training_tasks() -> tuple[str, ...]:
    """Training task ids, read from the adapter table.

    Imported lazily: ``engines.training.adapters`` pulls in the venv/device
    helpers, and this module is imported at application start-up.
    """
    try:
        from engines.training.adapters import TASK_IDS

        return tuple(TASK_IDS)
    except Exception:  # noqa: BLE001 - the page still has to render
        return ("sensevoice_singing",)


def build_training_spec(request: TrainingRequest) -> TrainingSpec:
    """Convert a :class:`TrainingRequest` into the spec the training body expects."""
    return TrainingSpec(
        task=request.task,
        data_dir=request.data_dir,
        params={
            "epochs": int(request.epochs),
            "batch": int(request.batch),
            "lr": float(request.lr),
        },
        output_dir=request.output_dir,
        device=request.device,
    )


def run_training_job(_ctx=None, cfg=None, progress_cb=None) -> RunResult:
    """``JobRunner`` body for a training run, returning a :class:`RunResult`.

    Two differences from :func:`run_pipeline_job` that matter to the page:

    * **A non-zero exit code is not an exception.** The engine's ``stream()``
      records ``exit_code`` and returns normally, so a crashed training run looks
      exactly like a successful one unless the code is checked. It is checked
      here and turned into ``RunResult.error``.
    * **Cancel really kills.** Training runs a subprocess, and its ``cancel()``
      terminates (then kills) it — unlike the pipeline's purely cooperative
      cancellation. That is why the training page can promise a responsive Cancel.
    """
    from engines.training.job_runner import run_training_job as _run

    if cfg is None:
        raise ValueError("run_training_job needs a spec")
    status = training_availability(cfg.task, cfg.device)
    if not status.ready:
        raise RuntimeError(f'{status.state}: {status.detail}')
    if getattr(sys, 'frozen', False):
        raise RuntimeError('Training extension execution is not included in this inference release')

    context: dict = {}
    _run(context, cfg, progress_cb)

    code = context.get("exit_code")
    try:
        exit_code = None if code is None else int(code)
    except (TypeError, ValueError):
        exit_code = None
    output_dir = str(getattr(cfg, "output_dir", "") or "")
    checkpoints = tuple(str(path) for path in (context.get("checkpoints") or ()))
    ok = exit_code == 0

    return RunResult(
        steps=(
            StepOutcome(
                "training",
                "done" if ok else "failed",
                f"exit {exit_code}" if exit_code is not None else "",
            ),
        ),
        output_dir=output_dir,
        checkpoints=checkpoints,
        exit_code=exit_code,
        error="" if ok else f"training exited with code {exit_code}",
    )

# Capability states are shared by the home dashboard and worker preflight.
@dataclass(frozen=True)
class Capability:
    step: str
    state: str = 'ready'
    detail: str = ''


STEP_MODELS = {
    'separation': ('uvr5_weights',),
    'diarization': ('campplus', 'vad'),
    'content_detection': (),
    'vocal2midi': ('game', 'hubertfa', 'rmvpe', 'romaji_asr'),
    'speech_asr': ('sensevoice', 'vad', 'ct_punc'),
    'tts': ('gsv_root', 'chinese_hubert', 'chinese_roberta'),
}
STEP_DEPENDENCIES = {
    'separation': ('torch', 'librosa', 'soundfile', 'onnxruntime'),
    'diarization': ('torch', 'funasr', 'soundfile'),
    'content_detection': ('librosa', 'soundfile'),
    'vocal2midi': ('torch', 'onnxruntime', 'transformers', 'mido', 'librosa'),
    'speech_asr': ('torch', 'funasr', 'modelscope', 'soundfile'),
    'tts': ('torch', 'transformers', 'librosa', 'pyopenjtalk', 'soundfile'),
}


def application_dir():
    return Path(sys.executable).resolve().parent if getattr(sys, 'frozen', False) else config_module.PROJECT_ROOT


def resolve_device_preference(page_device='inherit', global_device=None):
    value = page_device or 'inherit'
    if value == 'inherit':
        value = global_device if global_device is not None else load_preferences().device
    return device_utils.normalize_runtime_device(value)


def capabilities(cfg=None):
    paths = getattr(cfg, 'model_paths', None) or ModelPaths.from_settings()
    result = []
    for step in PIPELINE_STEPS:
        deps = [name for name in STEP_DEPENDENCIES.get(step, ()) if importlib.util.find_spec(name) is None]
        if deps:
            result.append(Capability(step, 'dependency_missing', ', '.join(deps))); continue
        required = list(STEP_MODELS.get(step, ()))
        if step == 'speech_asr' and getattr(cfg, 'asr_engine', '') == 'paraformer':
            required = ['paraformer', 'vad', 'ct_punc']
        if step == 'tts':
            prefix = 'gsv_v2proplus' if getattr(cfg, 'tts_version', '') == 'v2ProPlus' else 'gsv_v2pro'
            required += [prefix+'_gpt', prefix+'_sovits']
        if step == 'separation' and getattr(cfg, 'enable_denoise', False): required.append('denoise')
        missing = []
        for name in required:
            try:
                path = paths.resolve(name)
                if not path.exists() or (name == 'denoise' and not (path/'speech_frcrn_ans_cirm_16k'/'pytorch_model.bin').is_file()): missing.append(name)
            except (OSError, ValueError): missing.append(name)
        if missing:
            result.append(Capability(step, 'model_missing', ', '.join(missing))); continue
        problem = device_problem(getattr(cfg, 'device', 'inherit'), step)
        result.append(Capability(step, 'device_unsupported' if problem else 'ready', problem))
    return tuple(result)


def device_problem(device, step):
    requested = resolve_device_preference(device)
    if step == 'content_detection' or requested in ('auto', 'cpu'): return ''
    if requested == 'directml' and step != 'vocal2midi':
        return 'DirectML supports ONNX MIDI; select Auto or ROCm for PyTorch models'
    if requested == 'cuda' and device_utils.detect_rocm(): return 'CUDA preference is incompatible with this ROCm runtime'
    try:
        resolved = device_utils.resolve_pytorch_device(requested)
        if resolved.type == 'cuda':
            import torch
            if not torch.cuda.is_available(): return f'{requested}: GPU is unavailable'
    except RuntimeError as exc: return str(exc)
    if step == 'vocal2midi':
        providers, note = device_utils.ort_execution_providers(requested)
        if providers[0] == 'CPUExecutionProvider': return note
    return ''


def preflight(cfg):
    cfg.validate()
    if not Path(cfg.audio_path).is_file(): raise ValueError('Input must be an audio file')
    if not cfg.output_dir: raise ValueError('Output directory is required')
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    # Check write permission before any model allocation. Remove only this probe.
    import tempfile
    with tempfile.NamedTemporaryFile(prefix='.vstudio-', dir=output): pass
    selected = set(cfg.pipeline_steps or ())
    for status in capabilities(cfg):
        if status.step in selected and status.state != 'ready':
            raise RuntimeError(f'{status.step}: {status.state}: {status.detail}')
    if 'tts' in selected:
        if not cfg.tts_text.strip(): raise ValueError('TTS target text is required')
        if not cfg.ref_audio or not Path(cfg.ref_audio).is_file(): raise ValueError('TTS reference audio is required')
    requested = resolve_device_preference(cfg.device)
    # Do not accept a CUDA preference against an AMD runtime: torch calls both
    # devices cuda, but this preference explicitly means the NVIDIA backend.
    if requested == 'cuda' and device_utils.detect_rocm():
        raise RuntimeError('CUDA preference is incompatible with this ROCm runtime; select ROCm or Auto')
    device = device_utils.resolve_pytorch_device(requested)
    if device.type == 'cuda':
        import torch
        if not torch.cuda.is_available(): raise RuntimeError(f'{requested}: GPU is unavailable')
    providers, note = device_utils.ort_execution_providers(requested)
    if 'vocal2midi' in selected and requested in ('rocm', 'cuda', 'directml', 'zluda') and providers[0] == 'CPUExecutionProvider':
        raise RuntimeError(f'ONNX GPU backend is unavailable: {note}; install the GPU runtime or explicitly select CPU')
    if getattr(cfg, 'log_cb', None):
        from vstudio_runtime import runtime_policy
        actual = device_utils.auto_select_device() if requested == 'auto' else requested
        cfg.log_cb(f'Runtime: {runtime_policy()} / {actual} ({device}); ONNX {providers}. {note}')
    from vstudio_runtime import runtime_policy
    providers, _ = device_utils.ort_execution_providers(requested)
    if requested == 'directml' and not any(p in providers for p in ('DmlExecutionProvider', 'DMLExecutionProvider')):
        raise RuntimeError('DirectML is not installed in this runtime; select an available backend')
    return {'pytorch': 'PyTorch CPU' if device.type == 'cpu' else runtime_policy(), 'onnx': 'ONNX '+', '.join(providers)}


@dataclass(frozen=True)
class ExtensionStatus:
    state: str
    detail: str = ''
    manifest: str = ''
    python: str = ''
    entrypoint: str = ''

    @property
    def ready(self): return self.state == 'ready'


def training_extension_status(task='', backend='', root=None):
    """Schema 1 extension contract. Never execute code while detecting it."""
    root = Path(root) if root is not None else application_dir()
    manifest = root / 'extensions' / 'training' / 'manifest.json'
    if not manifest.is_file(): return ExtensionStatus('extension_missing', str(manifest))
    try:
        data = json.loads(manifest.read_text(encoding='utf-8-sig'))
        if data.get('schema_version') != 1 or data.get('api_version') != '1':
            return ExtensionStatus('version_mismatch', 'Requires schema 1 / API 1', str(manifest))
        if not isinstance(data.get('version'), str) or not data['version']:
            raise ValueError('Extension version is required')
        if backend and backend not in data.get('backends', []):
            return ExtensionStatus('device_unsupported', backend, str(manifest))
        if task and task not in data.get('tasks', []):
            return ExtensionStatus('task_unsupported', task, str(manifest))
        resolved = []
        for name in ('python', 'entrypoint'):
            path = (manifest.parent / data[name]).resolve()
            if not path.is_relative_to(manifest.parent.resolve()): raise ValueError('Extension paths must stay inside its directory')
            if not path.is_file(): return ExtensionStatus('dependency_missing', str(path), str(manifest))
            resolved.append(str(path))
        return ExtensionStatus('ready', data['version'], str(manifest), *resolved)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return ExtensionStatus('invalid_manifest', short_error(exc), str(manifest))


def training_availability(task='', device='cuda'):
    torch = sys.modules.get('torch')
    hip = bool(getattr(getattr(torch, 'version', None), 'hip', None))
    backend = 'rocm' if hip else device
    status = training_extension_status(task, backend)
    if hip:
        return status if not status.ready else ExtensionStatus('device_unsupported', 'AMD training adaptation is not included in this release', status.manifest)
    if getattr(sys, 'frozen', False):
        return status if not status.ready else ExtensionStatus('device_unsupported', 'This inference release only detects training extensions', status.manifest)
    # Preserve the existing source-tree CUDA / CPU training adapters.
    return ExtensionStatus('ready', 'Source training adapters')


def missing_model_paths():
    paths = ModelPaths.from_settings()
    missing = list(paths.missing())
    for name in paths.__dataclass_fields__:
        if name in missing: continue
        try:
            path = paths.resolve(name)
            if not path.exists() or (path.is_dir() and not any(p.is_file() and p.suffix not in ('.pyc',) and not p.name.startswith('.') for p in path.rglob('*'))): missing.append(name)
        except (OSError, ValueError): missing.append(name)
    return tuple(missing)
