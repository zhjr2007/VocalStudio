"""Chord / key / meter analysis engine (wave C1-7).

Built on madmom 0.16.1 (pip package; all model weights ship inside the
installed package under ``madmom/models``). The py3.13/numpy2/scipy
compatibility shims from :mod:`engines.chords.compat` are installed by the
package ``__init__`` before any ``madmom`` import happens here, so this
module can be imported directly as well.

:meth:`ChordEngine.analyze` runs three independent madmom pipelines on the
(optionally HPSS-filtered) waveform:

- chords  - CNN chord features (10 fps) + CRF recognition -> segments
            ``(start, end, label)``; labels are mapped to compact names
            ("C", "Am"); ``N`` (no-chord) segments are dropped.
- key     - CNN key recognition -> 24-way prediction; the argmax label is a
            string like ``"C major"`` and the confidence is its probability.
- meter   - RNN downbeat activations (100 fps) + DBN tracking for 3/4 and
            4/4; the HMM with the highest log-probability wins (madmom's own
            selection rule); bpm is the median of ``60 / beat interval``.

The result uses the "songData"-style shape required by wave-plan C1-7::

    {"key": "C major" | None, "confidence": float, "bpm": float | None,
     "time_sig": "4/4" | None,
     "chordPositions": [{"position": <seconds>, "value": <compact label>}, ...]}

madmom is a pure-numpy implementation: inference always runs on the CPU.
"""

from __future__ import annotations

import gc
import json
import pathlib
from typing import Any

import numpy as np

# The package __init__ installs the compat shims before this module is
# imported; importing madmom here is therefore safe on py3.13/numpy2/scipy>=1.12.
from madmom.audio.signal import Signal
from madmom.features.chords import CNNChordFeatureProcessor, CRFChordRecognitionProcessor
from madmom.features.downbeats import (
    DBNDownBeatTrackingProcessor,
    RNNDownBeatProcessor,
    _process_dbn,
)
from madmom.features.key import CNNKeyRecognitionProcessor, key_prediction_to_label

SAMPLE_RATE = 44100
CHORD_FPS = 10.0
BEAT_FPS = 100.0

_ROOTS = ("A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#")

# The 24 compact chord names the engine may emit (12 major + 12 minor).
LEGAL_CHORD_LABELS: frozenset[str] = frozenset(
    [root for root in _ROOTS] + [f"{root}m" for root in _ROOTS]
)


def madmom_label_to_compact(label: str) -> str | None:
    """Map a madmom chord label to the compact name ("C:maj"->"C", "A:min"->"Am")."""
    if not label or label == "N":
        return None
    root, _, quality = str(label).partition(":")
    if quality == "maj":
        return root
    if quality == "min":
        return f"{root}m"
    return None  # defensive: unknown label shapes are treated as no-chord


def _empty_result() -> dict[str, Any]:
    return {
        "key": None,
        "confidence": 0.0,
        "bpm": None,
        "time_sig": None,
        "chordPositions": [],
    }


def _track_downbeats(tracker: DBNDownBeatTrackingProcessor, activations: np.ndarray) -> tuple[list[float], str | None]:
    """NumPy-2-safe replica of ``DBNDownBeatTrackingProcessor.process``.

    madmom 0.16.1 builds ``np.asarray([(path, log_prob), ...])`` from HMM paths
    of differing lengths: an object array on numpy<2, but a ValueError on
    numpy>=2. We run the identical decoding and beat conversion while reading
    the log-probabilities directly out of the result tuples. Returns
    ``(beat_seconds, time_sig)``; ``time_sig`` is None when no activation
    reaches the tracker threshold.
    """
    first = 0
    if tracker.threshold:
        idx = np.nonzero(activations >= tracker.threshold)[0]
        if idx.any():
            first = int(np.min(idx))
            last = min(len(activations), int(np.max(idx)) + 1)
        else:
            last = first
        activations = activations[first:last]
    if not activations.size or not np.any(activations):
        return [], None

    results = [_process_dbn((hmm, activations)) for hmm in tracker.hmms]
    log_probs = [float(log_prob) for _path, log_prob in results]
    best = int(np.argmax(log_probs))  # madmom's own meter selection rule
    path, _log_prob = results[best]
    time_sig: str | None = f"{int(tracker.beats_per_bar[best])}/4"

    st = tracker.hmms[best].transition_model.state_space
    om = tracker.hmms[best].observation_model
    positions = st.state_positions[path]
    beat_numbers = positions.astype(int) + 1
    if getattr(tracker, "correct", True):
        beats = np.empty(0, dtype=np.int64)
        beat_range = om.pointers[path] >= 1
        change_idx = np.nonzero(np.diff(beat_range.astype(np.int64)))[0] + 1
        if beat_range[0]:
            change_idx = np.r_[0, change_idx]
        if beat_range[-1]:
            change_idx = np.r_[change_idx, beat_range.size]
        if change_idx.any():
            for left, right in change_idx.reshape((-1, 2)):
                peak = int(np.argmax(activations[left:right]) // 2 + left)
                beats = np.hstack((beats, [peak]))
    else:
        beats = np.nonzero(np.diff(beat_numbers))[0] + 1
    if not len(beats):
        return [], time_sig
    seconds = (np.asarray(beats, dtype=np.float64) + first) / float(tracker.fps)
    return [float(t) for t in seconds], time_sig


def _apply_chord_floor(positions: list[dict[str, Any]], min_chord_beats: float, bpm: float | None) -> list[dict[str, Any]]:
    """Drop chord changes closer than ``min_chord_beats`` to the previous one.

    A no-op when ``min_chord_beats <= 0`` or the tempo is unknown; the first
    position always survives.
    """
    if min_chord_beats <= 0 or bpm is None or bpm <= 0:
        return positions
    min_gap_sec = float(min_chord_beats) * 60.0 / float(bpm)
    kept: list[dict[str, Any]] = []
    for pos in positions:
        if kept and (pos["position"] - kept[-1]["position"]) < min_gap_sec:
            continue
        kept.append(pos)
    return kept


class ChordEngine:
    """Loads the madmom chord/key/downbeat models and analyzes a waveform."""

    def __init__(self) -> None:
        self.loaded = False
        self._chord_features = None
        self._chord_crf = None
        self._key_proc = None
        self._downbeat_rnn = None
        self._dbn_tracker = None

    # -- lifecycle ---------------------------------------------------------

    def load(self) -> None:
        """Build the five madmom processors (models come from the pip package)."""
        import librosa  # noqa: F401 - ensures the resample/hpss deps are importable

        self._chord_features = CNNChordFeatureProcessor()
        self._chord_crf = CRFChordRecognitionProcessor(fps=CHORD_FPS)
        self._key_proc = CNNKeyRecognitionProcessor()
        self._downbeat_rnn = RNNDownBeatProcessor()
        self._dbn_tracker = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=BEAT_FPS)
        self.loaded = True

    def unload(self) -> None:
        """Release the processor objects (and their cached model weights)."""
        for name in ("_chord_features", "_chord_crf", "_key_proc", "_downbeat_rnn", "_dbn_tracker"):
            setattr(self, name, None)
        self.loaded = False
        gc.collect()

    # -- analysis ----------------------------------------------------------

    def analyze(
        self,
        audio: np.ndarray,
        sr: int,
        *,
        hpss: bool = False,
        min_chord_beats: float = 0.0,
        cancel_checker=None,
    ) -> dict[str, Any]:
        """Analyze chords / key / meter for ``audio`` (see module docstring).

        Args:
            audio: mono or multi-channel waveform (any numeric dtype).
            sr: sample rate of ``audio``.
            hpss: when True, only the harmonic part of a librosa HPSS split is
                analyzed (helps on mixed material; costs one extra pass).
            min_chord_beats: minimum chord length in beats; changes closer than
                this to the previous change are dropped (0 = keep all).
            cancel_checker: optional zero-arg callable raising/returning truthy
                between sub-stages (cooperative cancellation, JobRunner style).
        """
        if not self.loaded:
            raise RuntimeError("ChordEngine.load() must be called before analyze().")

        waveform = np.asarray(audio, dtype=np.float32)
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=1).astype(np.float32)
        if waveform.size == 0 or int(sr) <= 0:
            return _empty_result()

        import librosa

        rate = int(sr)
        if rate != SAMPLE_RATE:
            waveform = librosa.resample(waveform, orig_sr=rate, target_sr=SAMPLE_RATE).astype(np.float32)
            rate = SAMPLE_RATE
        if hpss:
            waveform = librosa.hpss(waveform)[0].astype(np.float32)

        def _check() -> None:
            if cancel_checker is not None and cancel_checker():
                from core.exceptions import CancellationError

                raise CancellationError("chord analysis cancelled")

        signal = Signal(waveform, sample_rate=rate)

        # -- chords ---------------------------------------------------------
        segments = self._chord_crf(self._chord_features(signal))
        positions: list[dict[str, Any]] = []
        for start, _end, label in segments:
            compact = madmom_label_to_compact(label)
            if compact is None:
                continue
            positions.append({"position": round(float(start), 3), "value": compact})
        _check()

        # -- key -------------------------------------------------------------
        prediction = np.asarray(self._key_proc(signal), dtype=np.float64).ravel()
        if prediction.size:
            key_label = key_prediction_to_label(prediction)
            confidence = float(np.max(prediction))
        else:  # pragma: no cover - madmom always returns a full vector
            key_label, confidence = None, 0.0
        _check()

        # -- downbeats / meter / bpm ------------------------------------------
        activations = np.asarray(self._downbeat_rnn(signal), dtype=np.float32)
        beats, time_sig = _track_downbeats(self._dbn_tracker, activations) if activations.size else ([], None)
        bpm: float | None = None
        if len(beats) >= 2:
            diffs = np.diff(np.asarray(beats))
            diffs = diffs[diffs > 0]
            if diffs.size:
                bpm = round(float(np.median(60.0 / diffs)), 1)
        _check()

        positions = _apply_chord_floor(positions, float(min_chord_beats or 0.0), bpm)
        return {
            "key": key_label,
            "confidence": confidence,
            "bpm": bpm,
            "time_sig": time_sig,
            "chordPositions": positions,
        }


def write_chord_outputs(result: dict[str, Any], out_dir: str | pathlib.Path, key: str) -> dict[str, pathlib.Path]:
    """Write the songData-style result as ``<key>.chords.json`` + chord-sheet CSV.

    Returns a mapping ``{"json": path, "csv": path}``. The CSV is a plain
    chord sheet with one row per change: ``position_sec,value``.
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{key}.chords.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_lines = ["position_sec,value"]
    for pos in result.get("chordPositions", []):
        csv_lines.append(f"{pos['position']:.3f},{pos['value']}")
    csv_path = out_dir / f"{key}.chords.csv"
    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    return {"json": json_path, "csv": csv_path}
