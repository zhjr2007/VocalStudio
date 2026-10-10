"""Vocal separation page (UVR5). M1: full layout per build-checklist S1.1;
B-7: start button -> JobRunner(UVR5Engine) job with fetch/analyze/separate
stages, then dual-track waveform comparison + player on the default vocal lane."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import soundfile as sf

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from gui.widgets.audio_input import AudioInputArea
from gui.widgets.audio_player import AudioPlayerWidget
from gui.widgets.stem_mixer import StemMixerWidget
from gui.widgets.info_card import InfoCard
from gui.widgets.job_runner import JobRunner
from gui.widgets.preset_cards import PRESET_ORDER, PresetGrid
from gui.widgets.waveform_view import TRACK_COLORS, WaveformView, compute_peaks

USAGE_KEYS = ("separation_usage_1", "separation_usage_2", "separation_usage_3")
MODEL_KEYS = (
    "separation_models_1",
    "separation_models_2",
    "separation_models_3",
    "separation_models_4",
)

STAGE_KEYS = ("tier_standard", "tier_hq", "tier_ultra")
ETA_KEYS = ("tier_eta_standard", "tier_eta_hq", "tier_eta_ultra")

JOB_STAGE_LABEL_KEYS = {"fetch": "stage_fetch", "analyze": "stage_analyze", "separate": "stage_separate"}
TRACK_LABEL_KEYS = {"vocals": "track_vocals", "accompaniment": "track_accomp"}

BACKEND_EXE_NAME = "VocalStudioBackend.exe"


def _backend_exe() -> str | None:
    """Standalone backend executable path, if one ships next to the frontend.

    The frozen frontend is a slim GUI-only exe; separation then runs in
    ``VocalStudioBackend.exe`` as a hidden child process so no console window
    appears. The backend ships either next to the frontend exe or in a sibling
    one-dir folder (``VocalStudioBackend/VocalStudioBackend.exe``). In
    source/dev mode this returns None unless VOCALSTUDIO_BACKEND points at an
    executable."""
    override = os.environ.get("VOCALSTUDIO_BACKEND")
    if override:
        return str(override) if Path(override).is_file() else None
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).parent
        for candidate in (
            exe_dir / BACKEND_EXE_NAME,
            exe_dir / "VocalStudioBackend" / BACKEND_EXE_NAME,
        ):
            if candidate.is_file():
                return str(candidate)
    return None

class SeparationJobCfg:
    """Attributes consumed by :func:`run_separation_job`; JobRunner wraps the
    settable ``log_cb`` / ``cancel_checker`` before calling fn."""

    def __init__(self, spec, device="auto", intensity=0.0, denoise=False, output_dir="") -> None:
        self.spec = str(spec)
        self.device = str(device or "auto")
        self.intensity = float(intensity)
        self.denoise = bool(denoise)
        self.output_dir = str(output_dir or "")
        self.log_cb = None
        self.cancel_checker = lambda: False


def run_separation_job(ctx, cfg, progress_cb):
    """Job body for JobRunner: runs UVR5 separation on ``ctx["input"]``.

    When a standalone backend exe is available (:func:`_backend_exe`) the job
    runs it as a hidden child process; otherwise the engine runs in-process.
    Stage names advance fetch -> analyze -> separate; emitted fractions are
    monotone and end at 1.0 under "separate". Written stem paths land in
    ``ctx["stems"]`` (dict name -> wav path) and the source duration in
    ``ctx["duration_ms"]`` before returning ``ctx``."""

    from core.exceptions import CancellationError

    input_path = str(ctx.get("input") or "")
    if not input_path or not Path(input_path).exists():
        raise FileNotFoundError(f"input audio not found: {input_path!r}")

    def check_cancelled() -> None:
        checker = getattr(cfg, "cancel_checker", None)
        if callable(checker) and checker():
            raise CancellationError("separation cancelled")

    def log(message) -> None:
        callback = getattr(cfg, "log_cb", None)
        if callable(callback):
            try:
                callback(str(message))
            except Exception:
                pass

    log(
        f"preset={cfg.spec} device={cfg.device} intensity={cfg.intensity:g}"
        f" denoise={'on' if cfg.denoise else 'off'}"
    )

    backend = _backend_exe()
    if backend is not None:
        _run_via_backend(backend, input_path, cfg, ctx, check_cancelled, log, progress_cb)
        return ctx

    from engines.separation.uvr5_engine import UVR5Engine

    progress_cb(0.0, "fetch")
    engine = UVR5Engine()
    try:
        engine.load(cfg.spec, device=cfg.device)
        check_cancelled()
        progress_cb(0.28, "fetch")

        progress_cb(0.34, "analyze")
        frames, rate = sf.read(str(input_path), dtype="float32", always_2d=True)
        ctx["duration_ms"] = int(frames.shape[0] * 1000 / int(rate)) if rate else 0
        check_cancelled()
        progress_cb(0.46, "analyze")

        progress_cb(0.52, "separate")
        paths = engine.run(str(input_path), cfg.output_dir, intensity=float(cfg.intensity))
    finally:
        engine.unload()

    ctx["stems"] = {name: str(path) for name, path in paths.items()}
    progress_cb(1.0, "separate")
    return ctx


def _run_via_backend(backend, input_path, cfg, ctx, check_cancelled, log, progress_cb) -> None:
    """Runs separation in the standalone backend exe (hidden child process).

    The backend is a windowed PyInstaller build of ``main.py``; it reports over
    stdout: ``stage <name>`` markers, ``progress <0..1> <message>`` lines, one
    ``<stem>: <path>`` line per written stem and ``error: ...`` on failure.
    Progress fractions map onto the page's "separate" window [0.52, 1.0]."""
    command = [
        backend,
        "separate",
        "--audio", str(input_path),
        "--model", cfg.spec,
        "--agg", str(int(round(float(cfg.intensity)))),
        "--device", cfg.device or "auto",
        "--out", str(cfg.output_dir),
    ]

    progress_cb(0.0, "fetch")
    check_cancelled()

    # analyze stage: read the source for duration (same as the in-process path)
    progress_cb(0.34, "analyze")
    frames, rate = sf.read(str(input_path), dtype="float32", always_2d=True)
    ctx["duration_ms"] = int(frames.shape[0] * 1000 / int(rate)) if rate else 0
    check_cancelled()

    import queue as _queue

    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        creationflags=creationflags,
    )

    eof = object()
    pending: "_queue.Queue" = _queue.Queue()

    def _reader() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            pending.put(line.rstrip("\n"))
        pending.put(eof)

    reader = threading.Thread(target=_reader, daemon=True)
    reader.start()

    last_fraction = 0.46
    stems: dict[str, str] = {}
    errors: list[str] = []
    stem_re = re.compile(r"^([A-Za-z0-9_]+):\s*(.+)$")

    try:
        while True:
            check_cancelled()
            try:
                item = pending.get(timeout=0.5)
            except _queue.Empty:
                continue
            if item is eof:
                break
            line = str(item).strip()
            if not line or line.startswith("stage "):
                continue
            if line.startswith("progress "):
                parts = line.split(" ", 2)
                try:
                    fraction = float(parts[1])
                except (IndexError, ValueError):
                    continue
                target = max(last_fraction, min(1.0, 0.52 + 0.48 * fraction))
                if target > last_fraction:
                    progress_cb(target, "separate")
                    last_fraction = target
                message = parts[2] if len(parts) > 2 else ""
                if message:
                    log(message)
            elif line.startswith("error:"):
                errors.append(line)
                log(line)
            else:
                match = stem_re.match(line)
                if match and Path(match.group(2).strip()).is_file():
                    stems[match.group(1)] = match.group(2).strip()
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                process.kill()

    reader.join(timeout=2.0)
    exit_code = process.wait()
    if exit_code != 0:
        detail = errors[-1][len("error:"):].strip() if errors else f"exit code {exit_code}"
        raise RuntimeError(f"backend separation failed: {detail}")
    if not stems:
        raise RuntimeError("backend returned no stems")

    ctx["stems"] = {name: str(path) for name, path in stems.items()}
    progress_cb(1.0, "separate")

def preset_required_weights(preset: str) -> tuple[str, ...]:
    """Weight file basenames (under the uvr5_weights dir) a preset needs."""
    if preset == "Karaoke":
        return ("HP5_only_main_vocal.pth",)
    if preset == "FullVocal":
        return ("HP2_all_vocals.pth",)
    band = (
        "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        "drums_bs_roformer.ckpt",
        "bass_bs_roformer.ckpt",
    )
    if preset == "Band4":
        return band
    if preset == "Producer5":
        return band + ("piano_bs_roformer.ckpt",)
    return ()


def is_preset_ready(preset: str, model_paths=None) -> bool:
    from core.config import ModelPaths

    if model_paths is None:
        model_paths = ModelPaths.from_settings()
    uvr5_dir_value = getattr(model_paths, "uvr5_weights", "") or ""
    if not uvr5_dir_value:
        return False
    uvr5_dir = Path(uvr5_dir_value)
    if not uvr5_dir.is_absolute():
        from core.config import PROJECT_ROOT

        uvr5_dir = PROJECT_ROOT / uvr5_dir
    return all((uvr5_dir / name).exists() for name in preset_required_weights(preset))


class SeparationPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self._title_label = QLabel(self)
        layout.addWidget(self._title_label)

        self._input_area = AudioInputArea(self)
        layout.addWidget(self._input_area, 2)

        grid_row = QHBoxLayout()
        self._preset_title = QLabel(self)
        title_font = self._preset_title.font()
        title_font.setBold(True)
        self._preset_title.setFont(title_font)
        grid_row.addWidget(self._preset_title)
        grid_row.addStretch(1)
        layout.addLayout(grid_row)

        self._presets = PresetGrid(self)
        layout.addWidget(self._presets, 3)

        option_row = QHBoxLayout()
        stage_label = QLabel(self)
        option_row.addWidget(stage_label)
        self._stage_label = stage_label
        self._stage_combo = QComboBox(self)
        for _key in STAGE_KEYS:
            self._stage_combo.addItem("")
        option_row.addWidget(self._stage_combo)
        self._denoise_check = QCheckBox(self)
        option_row.addStretch(1)
        option_row.addWidget(self._denoise_check)
        layout.addLayout(option_row)

        intensity_row = QHBoxLayout()
        self._intensity_label = QLabel(self)
        intensity_row.addWidget(self._intensity_label)
        self._intensity_slider = QSlider(Qt.Horizontal, self)
        self._intensity_slider.setRange(0, 20)
        self._intensity_slider.setValue(0)
        intensity_row.addWidget(self._intensity_slider)
        self._value_label = QLabel(self)
        intensity_row.addStretch(1)
        intensity_row.addWidget(self._value_label)
        layout.addLayout(intensity_row)

        dir_row = QHBoxLayout()
        self._dir_label = QLabel(self)
        dir_row.addWidget(self._dir_label)
        self._dir_edit = QLineEdit(self)
        dir_row.addWidget(self._dir_edit, 1)
        self._browse_button = QPushButton(self)
        dir_row.addWidget(self._browse_button)
        layout.addLayout(dir_row)

        start_row = QHBoxLayout()
        start_row.addStretch(1)
        self._start_button = QPushButton(self)
        start_row.addWidget(self._start_button)
        layout.addLayout(start_row)

        progress_row = QHBoxLayout()
        self._progress_label = QLabel(self)
        progress_row.addWidget(self._progress_label)
        self._progress_bar = QProgressBar(self)
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        progress_row.addWidget(self._progress_bar, 1)
        self._status_stage_label = QLabel(self)
        progress_row.addWidget(self._status_stage_label)
        layout.addLayout(progress_row)

        self._result_title = QLabel(self)
        result_font = self._result_title.font()
        result_font.setBold(True)
        self._result_title.setFont(result_font)
        layout.addWidget(self._result_title)

        self._waveform = WaveformView(self)
        layout.addWidget(self._waveform, 3)

        track_row = QHBoxLayout()
        self._track_combo = QComboBox(self)
        track_row.addWidget(self._track_combo)
        track_row.addStretch(1)
        layout.addLayout(track_row)

        self._player = AudioPlayerWidget(self)
        layout.addWidget(self._player, 2)

        self._stem_mixer = StemMixerWidget(self)
        layout.addWidget(self._stem_mixer, 3)
        self._player.loop_changed.connect(self._on_player_loop_changed)

        self._job_runner: JobRunner | None = None
        self._progress_events: list[tuple[float, str]] = []
        self._last_progress: float = 0.0
        self._result_ctx: dict | None = None
        self._lane_names: list[str] = []
        self._stage_raw: str = ""
        self._failure_text: str = ""

        self._intensity_slider.valueChanged.connect(
            lambda value: self._value_label.setText(str(value))
        )
        self._dir_edit.textChanged.connect(lambda _text: self._update_start_enabled())
        self._browse_button.clicked.connect(self._on_browse_clicked)
        self._start_button.clicked.connect(self.start)
        self._track_combo.currentIndexChanged.connect(self._on_track_changed)
        self._update_start_enabled()

        self._eta_label = QLabel(self)
        self._stage_combo.currentIndexChanged.connect(self._update_eta)
        layout.addWidget(self._eta_label)

        self._usage_card = InfoCard("info_usage", USAGE_KEYS, self)
        self._models_card = InfoCard("info_models", MODEL_KEYS, self)
        layout.addWidget(self._usage_card, 3)
        layout.addWidget(self._models_card, 4)

        self.refresh_presets()
        self.retranslate()

    def retranslate(self) -> None:
        from gui.i18n import tr

        self._title_label.setText(tr("separation"))
        self._input_area.retranslate()
        self._preset_title.setText(tr("preset_grid_title"))
        self._stage_label.setText(tr("stage_label"))
        for index, key in enumerate(STAGE_KEYS):
            text = tr(key)
            if self._stage_combo.itemText(index) != text:
                self._stage_combo.setItemText(index, text)
        self._denoise_check.setText(tr("denoise"))
        self._intensity_label.setText(tr("intensity"))
        self._value_label.setText(str(self._intensity_slider.value()))
        self._dir_label.setText(tr("output_dir"))
        self._browse_button.setText(tr("browse"))
        self._start_button.setText(tr("start"))
        self._update_eta()
        self._progress_label.setText(tr("progress"))
        self._result_title.setText(tr("result_title"))
        if not self.is_running():
            if self._stage_raw in JOB_STAGE_LABEL_KEYS:
                self._status_stage_label.setText(tr(JOB_STAGE_LABEL_KEYS[self._stage_raw]))
            elif self._stage_raw == "done":
                self._status_stage_label.setText(tr("separation_done"))
            elif self._stage_raw == "failed":
                text = tr("separation_failed") + (f": {self._failure_text}" if self._failure_text else "")
                self._status_stage_label.setText(text)
            else:
                self._status_stage_label.setText(tr("stage_idle"))
        for index, name in enumerate(self._lane_names):
            key = TRACK_LABEL_KEYS.get(name, name)
            self._track_combo.setItemText(index, tr(key, name))
        self._waveform.retranslate()
        self._player.retranslate()
        self._stem_mixer.retranslate()
        self._presets.retranslate()
        self._usage_card.retranslate()
        self._models_card.retranslate()

    def refresh_presets(self) -> None:
        for name in PRESET_ORDER:
            self._presets.set_enabled(name, is_preset_ready(name))
        selected = self._presets.selected_preset()
        if selected is None or not self._presets.is_enabled(selected):
            for name in PRESET_ORDER:
                if self._presets.is_enabled(name):
                    self._presets.select(name)
                    break

    def preset_grid(self) -> PresetGrid:
        return self._presets

    def selected_preset(self) -> str | None:
        return self._presets.selected_preset()

    def stage_index(self) -> int:
        return self._stage_combo.currentIndex()

    def set_stage_index(self, index: int) -> None:
        if 0 <= index < self._stage_combo.count():
            self._stage_combo.setCurrentIndex(index)

    def _update_eta(self, *_args) -> None:
        from gui.i18n import tr

        index = self._stage_combo.currentIndex()
        if 0 <= index < len(ETA_KEYS):
            self._eta_label.setText(tr(ETA_KEYS[index]))

    def denoise_enabled(self) -> bool:
        return self._denoise_check.isChecked()

    def set_denoise(self, enabled: bool) -> None:
        self._denoise_check.setChecked(bool(enabled))

    def intensity(self) -> int:
        return int(self._intensity_slider.value())

    def set_intensity(self, value: int) -> None:
        self._intensity_slider.setValue(int(value))

    def output_dir(self) -> str:
        return self._dir_edit.text()

    def set_output_dir(self, path: str) -> None:
        self._dir_edit.setText(str(path))

    def _update_start_enabled(self) -> None:
        self._start_button.setEnabled(bool(self.output_dir().strip()))

    def _on_browse_clicked(self) -> None:
        from gui.i18n import tr

        chosen = QFileDialog.getExistingDirectory(
            self, tr("browse"), self.output_dir() or str(Path.home())
        )
        if chosen:
            self.set_output_dir(chosen)

    def config(self) -> dict:
        return {
            "preset": self.selected_preset(),
            "intensity": int(self._intensity_slider.value()),
            "denoise": bool(self._denoise_check.isChecked()),
            "output_dir": self.output_dir(),
        }

    def add_files(self, paths: list[str]) -> int:
        return self._input_area.add_files(paths)

    def accepted_count(self) -> int:
        return self._input_area.accepted_count()

    def rows(self) -> list[tuple[str, bool]]:
        return self._input_area.rows()

    def selected_index(self) -> int:
        return self._input_area.selected_index()

    def select(self, index: int) -> None:
        self._input_area.select(index)

    def input_area(self) -> AudioInputArea:
        return self._input_area

    # -- job flow (wave B-7) ------------------------------------------------

    def waveform_view(self) -> WaveformView:
        return self._waveform

    def audio_player(self) -> AudioPlayerWidget:
        return self._player

    def stem_mixer(self) -> StemMixerWidget:
        return self._stem_mixer

    def track_combo(self) -> QComboBox:
        return self._track_combo

    def is_running(self) -> bool:
        runner = self._job_runner
        return bool(runner is not None and runner.isRunning())

    def progress_events(self) -> list[tuple[float, str]]:
        return list(self._progress_events)

    def last_progress(self) -> float:
        return self._last_progress

    def stage_sequence(self) -> list[str]:
        seen: list[str] = []
        for _value, stage in self._progress_events:
            if stage and stage not in seen:
                seen.append(stage)
        return seen

    def result_paths(self) -> dict[str, str]:
        ctx = self._result_ctx or {}
        stems = ctx.get("stems") or {}
        return {name: str(path) for name, path in stems.items()}

    def start(self) -> JobRunner | None:
        """Kicks off the separation job; a fresh page run is one JobRunner."""
        if self.is_running():
            return self._job_runner
        rows = [path for path, ok in self.input_area().rows() if ok]
        selected = self.input_area().selected_path()
        audio_path = ""
        if selected and Path(selected).is_file():
            audio_path = str(selected)
        elif rows:
            audio_path = str(rows[0])
        if not audio_path or not Path(audio_path).is_file():
            return None

        from core.app_settings import get_settings

        cfg = SeparationJobCfg(
            spec=self.selected_preset() or "Karaoke",
            device=get_settings().device or "auto",
            intensity=float(self.intensity()),
            denoise=bool(self.denoise_enabled()),
            output_dir=self.output_dir(),
        )
        ctx: dict = {"input": audio_path, "preset": cfg.spec}
        runner = JobRunner(run_separation_job, ctx=ctx, cfg=cfg, parent=self)
        self._job_runner = runner
        self._progress_events = []
        self._last_progress = 0.0
        self._result_ctx = None
        self._lane_names = []
        self._stage_raw = ""
        self._failure_text = ""
        runner.progress.connect(self._on_job_progress)
        runner.done.connect(self._on_job_done)
        runner.failed.connect(self._on_job_failed)
        runner.start_job()
        self._start_button.setEnabled(False)
        return runner

    def _on_job_progress(self, value: float, stage: str) -> None:
        from gui.i18n import tr

        fraction = max(0.0, min(1.0, float(value)))
        self._progress_events.append((fraction, str(stage)))
        self._last_progress = fraction
        self._progress_bar.setValue(int(round(fraction * 100)))
        key = JOB_STAGE_LABEL_KEYS.get(str(stage))
        self._stage_raw = str(stage) if key else self._stage_raw
        self._status_stage_label.setText(tr(key, str(stage)) if key else (str(stage) or ""))

    def _on_job_done(self, result, cancelled: bool) -> None:
        from gui.i18n import tr

        self._start_button.setEnabled(True)
        if not result or cancelled:
            return
        stems = {name: str(path) for name, path in ((result or {}).get("stems") or {}).items()}
        ordered: list[str] = []
        if "vocals" in stems:
            ordered.append("vocals")
        for name in ("accompaniment", "other"):
            if len(ordered) < 2 and name in stems:
                ordered.append(name)
        if not ordered:
            ordered = list(stems)[:2]

        tracks: list[dict] = []
        duration_ms = int((result or {}).get("duration_ms") or 0)
        for index, name in enumerate(ordered):
            data, rate = sf.read(str(stems[name]), dtype="float32", always_2d=True)
            if not tracks and int(rate) > 0:
                duration_ms = max(duration_ms, int(data.shape[0] * 1000 / int(rate)))
            mono = np.mean(data, axis=1).astype("float32") if data.shape[1] > 1 else data[:, 0].copy()
            peaks = compute_peaks(mono, int(rate), 512)
            tracks.append(
                {"label": name, "peaks": peaks, "color": TRACK_COLORS[index % len(TRACK_COLORS)]}
            )
        self._result_ctx = dict(result or {})
        self._waveform.set_waveforms(tracks)
        self._waveform.set_duration(duration_ms)

        self._lane_names = list(ordered)
        self._track_combo.blockSignals(True)
        try:
            self._track_combo.clear()
            for name in ordered:
                key = TRACK_LABEL_KEYS.get(name, name)
                self._track_combo.addItem(tr(key, name), str(stems[name]))
            if ordered:
                self._track_combo.setCurrentIndex(0)
        finally:
            self._track_combo.blockSignals(False)
        if ordered:
            self._player.set_file(str(stems[ordered[0]]))

        # live stem mixer: show every stem (2-track or multi-stem)
        all_names: list[str] = []
        if "vocals" in stems:
            all_names.append("vocals")
        for name in ("accompaniment", "other"):
            if name in stems and name not in all_names:
                all_names.append(name)
        for name in stems:
            if name not in all_names:
                all_names.append(name)
        stem_meta = [{"name": name, "path": stems[name]} for name in all_names]
        self._stem_mixer.set_stems(stem_meta)
        self._stage_raw = "done"
        self._status_stage_label.setText(tr("separation_done"))

    def _on_player_loop_changed(self) -> None:
        start, end = self._player.loop_bounds()
        if self._player.loop_enabled and end > start:
            self._waveform.set_loop(start, end)
        else:
            self._waveform.clear_loop()

    def _on_job_failed(self, exc) -> None:
        from gui.i18n import tr

        self._start_button.setEnabled(True)
        self._failure_text = str(exc)
        self._stage_raw = "failed"
        self._status_stage_label.setText(
            tr("separation_failed") + (f": {exc}" if str(exc) else "")
        )

    def _on_track_changed(self, index: int) -> None:
        if not 0 <= index < len(self._lane_names):
            return
        name = self._lane_names[index]
        stems = self.result_paths()
        path = stems.get(name) or str(self._track_combo.itemData(index) or "")
        if path:
            self._player.set_file(path)
