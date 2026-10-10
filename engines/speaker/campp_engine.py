"""CAM++ speaker diarization engine (wave D-3).

Pipeline: FSMN-VAD segmentation -> per-segment CAM++ embeddings ->
agglomerative cosine clustering -> per-speaker merged 16 kHz mono wav files.

Both models load through funasr ``AutoModel(disable_update=True)`` from the
local directories configured in ``settings/defaults.json`` (fields
``campplus`` / ``vad``), so no network access is required at runtime.
"""

from __future__ import annotations

import gc
import pathlib
from typing import Any

import numpy as np

SAMPLE_RATE = 16000

#: In auto mode (spk_num <= 0) clusters keep merging while their average
#: cosine similarity stays at or above this threshold. 0.72 was tuned on
#: the 90-minute lecture recordings (temp/asr_sociology experiments 10-18;
#: the maintained pipeline copy lives in scripts/asr_experiment/):
#: per-segment embeddings at ~0.65 fragment the dominant speaker into
#: hundreds of clusters, while 12 s window embeddings at 0.72 keep one
#: dominant (teacher) cluster covering ~98 % of the lecture, plus sparse
#: short student/noise fragments. Higher values (0.75/0.78) fragment the
#: teacher again; lower values (0.65) swallow the roll-call mix.
AUTO_MERGE_SIMILARITY = 0.72

#: CAM++ is tuned for clips of about 12 s or less. A short VAD segment is
#: embedded from one padded window; a longer one is embedded from
#: start/middle/end windows (each at most WINDOW_MS) whose normalized
#: embeddings are averaged. A single middle window of a multi-minute
#: segment can fall on a pause or a prosody shift and pull the whole
#: segment out of the speaker cluster (observed on recording 0902: the
#: 10:59 and 45:21 teacher blocks).
WINDOW_MS = 12000
WINDOW_PAD_MS = 2000


def _resolve_model_path(name: str, value: str | pathlib.Path) -> pathlib.Path:
    """Resolve a model directory; empty values fall back to settings/defaults.json."""
    from core.config import PROJECT_ROOT, ModelPaths

    if not value:
        paths = ModelPaths.from_settings()
        value = getattr(paths, name, "") or ""
    if not value:
        raise ValueError(
            f"diarization model path '{name}' is not configured (settings/defaults.json)"
        )
    p = pathlib.Path(value)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _cosine_similarity_matrix(embs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = embs / norms
    return unit @ unit.T


def _agglomerate(
    sim: np.ndarray, target_clusters: int | None = None, threshold: float | None = None
) -> list[list[int]]:
    """Average-linkage agglomeration on a cosine similarity matrix.

    Repeatedly merges the most similar pair of clusters until either
    ``target_clusters`` remain (fixed-k mode) or the best remaining merge
    drops below ``threshold`` (auto mode). Returns index lists ordered by
    earliest member so labels follow time order.

    Vectorized merge updates: merging two clusters rewrites one similarity
    row as the size-weighted average of the two old rows, which keeps
    average-linkage semantics while cutting the naive per-pair
    fancy-indexing scan (O(n^3) Python calls) to O(n^2) work.
    """
    n = sim.shape[0]
    m = sim.astype(np.float64, copy=True)
    sizes = np.ones(n, dtype=np.float64)
    members: dict[int, list[int]] = {i: [i] for i in range(n)}
    alive: list[int] = list(range(n))
    while len(alive) > 1:
        if target_clusters is not None and len(alive) <= target_clusters:
            break
        am = m[np.ix_(alive, alive)]
        np.fill_diagonal(am, -2.0)
        i, j = np.unravel_index(int(np.argmax(am)), am.shape)
        s = float(am[i, j])
        if threshold is not None and s < threshold:
            break
        ci, cj = int(alive[i]), int(alive[j])
        ni, nj = sizes[ci], sizes[cj]
        others = [k for k in alive if k != cj]
        m[ci, others] = (ni * m[ci, others] + nj * m[cj, others]) / (ni + nj)
        m[others, ci] = m[ci, others]
        sizes[ci] = ni + nj
        members[ci].extend(members[cj])
        del members[cj]
        alive.remove(cj)
    clusters = [members[k] for k in alive]
    clusters.sort(key=lambda c: min(c))
    return clusters


def cluster_embeddings(embs: np.ndarray, spk_num: int) -> np.ndarray:
    """Cluster speaker embeddings; returns integer labels 0..K-1 (time order)."""
    n = embs.shape[0]
    if n == 0:
        return np.zeros(0, dtype=int)
    sim = _cosine_similarity_matrix(embs.astype(np.float64))
    if spk_num >= 1:
        clusters = _agglomerate(sim, target_clusters=min(int(spk_num), n))
    else:
        clusters = _agglomerate(sim, threshold=AUTO_MERGE_SIMILARITY)
    labels = np.zeros(n, dtype=int)
    for label, members in enumerate(clusters):
        labels[members] = label
    return labels


class CAMPPEngine:
    """CAM++ speaker-embedding + FSMN-VAD diarization engine."""

    def __init__(self) -> None:
        self.spk_model: Any = None
        self.vad_model: Any = None
        self.model_path: str = ""
        self.vad_path: str = ""
        self.device: str = "auto"

    # -- lifecycle ---------------------------------------------------------

    def load(self, model_path: str = "", vad_path: str = "", device: str = "auto") -> None:
        """Load CAM++ + FSMN-VAD via funasr AutoModel (no network updates)."""
        from core.device_utils import resolve_pytorch_device
        from funasr import AutoModel

        spk_dir = _resolve_model_path("campplus", model_path)
        vad_dir = _resolve_model_path("vad", vad_path)
        if not spk_dir.is_dir():
            raise FileNotFoundError(f"CAM++ model directory not found: {spk_dir}")
        if not vad_dir.is_dir():
            raise FileNotFoundError(f"VAD model directory not found: {vad_dir}")

        dev = str(resolve_pytorch_device(device))
        funasr_device = "cuda" if dev.startswith("cuda") else "cpu"

        self.model_path = str(spk_dir)
        self.vad_path = str(vad_dir)
        self.device = device
        self.spk_model = AutoModel(model=str(spk_dir), disable_update=True, device=funasr_device)
        self.vad_model = AutoModel(model=str(vad_dir), disable_update=True, device=funasr_device)

    def unload(self) -> None:
        self.spk_model = None
        self.vad_model = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - torch is always present in the venv
            pass

    @property
    def is_loaded(self) -> bool:
        return self.spk_model is not None and self.vad_model is not None

    # -- inference ---------------------------------------------------------

    def _vad_segments(self, audio_path: str, duration_ms: int) -> list[tuple[int, int]]:
        """FSMN-VAD speech segments as (start_ms, end_ms), time-ordered."""
        result = self.vad_model.generate(input=audio_path, batch_size_s=300)
        raw: list[list[float]] = []
        for item in result or []:
            value = item.get("value") if isinstance(item, dict) else None
            if value:
                raw.extend([list(seg) for seg in value])
        segments: list[tuple[int, int]] = []
        for s, e in sorted(raw):
            s2 = max(0, min(int(s), duration_ms))
            e2 = max(0, min(int(e), duration_ms))
            if e2 > s2:
                segments.append((s2, e2))
        # No speech detected: treat the whole file as one segment so that
        # diarization still yields at least one speaker.
        return segments or [(0, max(duration_ms, 1))]

    def _embed_window(self, waveform: np.ndarray, start_ms: int, end_ms: int) -> np.ndarray:
        """Embed one [start_ms, end_ms) time window with CAM++ (192-dim)."""
        a = max(0, int(round(start_ms * SAMPLE_RATE / 1000)))
        b = min(int(round(end_ms * SAMPLE_RATE / 1000)), waveform.shape[0])
        seg = waveform[a:b]
        if seg.size < SAMPLE_RATE // 4:  # pad very short segments to >=250 ms
            seg = np.pad(seg, (0, max(1, SAMPLE_RATE // 4 - seg.size)), mode="constant")
        res = self.spk_model.generate(input=seg.astype(np.float32))
        return np.asarray(res[0]["spk_embedding"].detach().cpu().numpy(), dtype=np.float64).reshape(-1)

    def _embed_segments(self, waveform: np.ndarray, segments: list[tuple[int, int]]) -> np.ndarray:
        """One CAM++ embedding (192-dim) per VAD segment.

        Short segments use a single padded +/-2 s window; segments longer
        than WINDOW_MS use start/middle/end windows and the mean of the
        normalized window embeddings (multi-window embedding).
        """
        total_ms = int(round(waveform.shape[0] * 1000 / SAMPLE_RATE))
        embs = []
        for start_ms, end_ms in segments:
            length = end_ms - start_ms
            if length <= WINDOW_MS:
                ws, we = max(0, start_ms - WINDOW_PAD_MS), min(total_ms, end_ms + WINDOW_PAD_MS)
                if we - ws > WINDOW_MS:
                    mid = (start_ms + end_ms) // 2
                    ws = max(0, mid - WINDOW_MS // 2)
                    we = min(total_ms, mid + WINDOW_MS // 2)
                windows = [(ws, we)]
            else:
                mid = (start_ms + end_ms) // 2
                windows = [(start_ms, start_ms + WINDOW_MS),
                           (mid - WINDOW_MS // 2, mid + WINDOW_MS // 2),
                           (end_ms - WINDOW_MS, end_ms)]
            rows = np.vstack([self._embed_window(waveform, ws, we) for ws, we in windows])
            norms = np.linalg.norm(rows, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embs.append((rows / norms).mean(axis=0))
        return np.vstack(embs)

    def diarize(self, audio, sr: int | None = None, spk_num: int = 3, out_dir=None) -> dict:
        """Diarize ``audio`` (path to a wav file).

        Args:
            audio: path of the input audio (any sample rate; resampled to 16 kHz).
            sr: source sample-rate hint (informational only).
            spk_num: expected speaker count. >=1 forces that many clusters
                (capped at the number of VAD segments); <=0 = automatic.
            out_dir: when given, per-speaker merged audio is written here as
                16 kHz mono ``speaker_<id>.wav`` files.

        Returns:
            {spk_id: {"segments": [(start_ms, end_ms), ...], "audio_path": str | None}}
            with spk_ids ordered by first appearance in time.
        """
        if not self.is_loaded:
            raise RuntimeError("CAMPPEngine.load() must be called before diarize()")

        import librosa
        import soundfile as sf

        audio = pathlib.Path(audio)
        waveform, _ = librosa.load(str(audio), sr=SAMPLE_RATE, mono=True)
        duration_ms = int(round(waveform.shape[0] * 1000 / SAMPLE_RATE))

        segments = self._vad_segments(str(audio), duration_ms)
        embs = self._embed_segments(waveform, segments)
        labels = cluster_embeddings(embs, spk_num)

        # Group segment indices by label; compact ids in first-appearance order.
        id_of_label: dict[int, int] = {}
        groups: dict[int, list[tuple[int, int]]] = {}
        for idx, seg in enumerate(segments):
            lab = int(labels[idx])
            if lab not in id_of_label:
                id_of_label[lab] = len(id_of_label)
            groups.setdefault(id_of_label[lab], []).append(seg)

        speakers: dict[int, dict] = {}
        for spk_id in sorted(groups):
            segs = sorted(groups[spk_id])
            audio_path = None
            if out_dir is not None:
                out = pathlib.Path(out_dir)
                out.mkdir(parents=True, exist_ok=True)
                chunks = [
                    waveform[int(round(s * SAMPLE_RATE / 1000)) : int(round(e * SAMPLE_RATE / 1000))]
                    for s, e in segs
                ]
                merged = np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, dtype=np.float32)
                audio_path = str(out / f"speaker_{spk_id}.wav")
                sf.write(audio_path, merged, SAMPLE_RATE)
            speakers[spk_id] = {"segments": segs, "audio_path": audio_path}
        return speakers


def run_diarization(ctx, cfg, progress_cb=None) -> None:
    """Orchestrator entry point for the diarization step (wave D-3).

    Sets ``ctx.speakers`` ({spk_id: {segments, audio_path}}), the flattened
    time-ordered ``ctx.speaker_segments`` and ``ctx.results["diarization"]``.
    """
    from core.config import ModelPaths

    model_paths = getattr(cfg, "model_paths", None) or ModelPaths.from_settings()
    audio_path = ctx.active_vocals_path or getattr(cfg, "audio_path", "") or ""
    if not audio_path or not pathlib.Path(audio_path).exists():
        raise FileNotFoundError(f"diarization: audio file not found: {audio_path!r}")

    base_out = pathlib.Path(getattr(cfg, "output_dir", "") or (pathlib.Path(audio_path).parent / "outputs"))
    out_dir = base_out / "diarization"
    spk_num = int(getattr(cfg, "num_speakers", 0) or 0)

    engine = CAMPPEngine()
    try:
        if progress_cb is not None:
            progress_cb(0.05, "loading CAM++/VAD models")
        engine.load(
            model_path=getattr(model_paths, "campplus", "") or "",
            vad_path=getattr(model_paths, "vad", "") or "",
            device=getattr(cfg, "device", "auto"),
        )
        if progress_cb is not None:
            progress_cb(0.25, "models ready")
        speakers = engine.diarize(audio_path, spk_num=spk_num, out_dir=out_dir)
    finally:
        engine.unload()

    ctx.speakers = speakers
    all_segments = [(s, e) for info in speakers.values() for s, e in info["segments"]]
    ctx.speaker_segments = sorted(all_segments)
    ctx.results["diarization"] = {
        "num_speakers": len(speakers),
        "requested": spk_num,
        "files": {str(k): v["audio_path"] for k, v in speakers.items()},
    }
    if progress_cb is not None:
        progress_cb(1.0, f"{len(speakers)} speaker(s)")
