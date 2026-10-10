"""Speech ASR engine (wave D-4): SenseVoice / Paraformer + CT punctuation.

Pipeline: FSMN-VAD segmentation -> per-chunk ASR through funasr ``AutoModel``
(``disable_update=True``, local model dirs from ``settings/defaults.json``)
-> optional CT-Transformer punctuation restore applied per chunk (the proven
FunASR api_server pattern).

SenseVoice raw output carries header tokens such as
``<|zh|><|NEUTRAL|><|Speech|>你好世界``; they are parsed into language /
emotion / event and stripped so the returned text is clean (no ``<|...|>``
residue, no emoji). Paraformer returns plain Chinese text directly.
"""

from __future__ import annotations

import gc
import json
import pathlib
import re
from collections import Counter
from typing import Any

import numpy as np

SAMPLE_RATE = 16000

#: Header token vocabularies emitted by SenseVoice (mirrors funasr's
#: postprocess_utils emo_dict / event_dict / lang_dict keys).
LANG_TAGS = {"auto", "zh", "en", "ja", "yue", "ko", "nospeech"}
EMOTION_TAGS = {"NEUTRAL", "HAPPY", "SAD", "ANGRY", "FEARFUL", "DISGUSTED", "SURPRISED"}
EVENT_TAGS = {"Speech", "BGM", "Applause", "Laughter", "Cry", "Sneeze", "Breath", "Cough"}

_TAG_RE = re.compile(r"<\|([^|]*)\|>")


def _resolve_model_path(name: str, value: str | pathlib.Path) -> pathlib.Path:
    """Resolve a model directory; empty values fall back to settings/defaults.json."""
    from core.config import PROJECT_ROOT, ModelPaths

    if not value:
        paths = ModelPaths.from_settings()
        value = getattr(paths, name, "") or ""
    if not value:
        raise ValueError(f"ASR model path '{name}' is not configured (settings/defaults.json)")
    p = pathlib.Path(value)
    return p if p.is_absolute() else PROJECT_ROOT / p


def parse_sensevoice_tags(raw: str) -> tuple[str, str, str]:
    """Split raw SenseVoice output into (language, emotion, event).

    Missing categories come back as empty strings; unknown tokens are ignored.
    """
    tags = _TAG_RE.findall(raw or "")
    language = next((t for t in tags if t.lower() in LANG_TAGS), "")
    emotion = next((t for t in tags if t in EMOTION_TAGS), "")
    event = next((t for t in tags if t in EVENT_TAGS), "")
    return language, emotion, event


def strip_tags(raw: str) -> str:
    """Remove every ``<|...|>`` token from raw ASR output."""
    return _TAG_RE.sub("", raw or "").strip()


def _has_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


class SpeechASREngine:
    """SenseVoice / Paraformer speech recognition with FSMN-VAD chunking."""

    def __init__(self) -> None:
        self.asr_model: Any = None
        self.vad_model: Any = None
        self.punc_model: Any = None
        self.engine_name: str = ""
        self.device: str = "auto"
        self.model_path: str = ""
        self.vad_path: str = ""
        self.punc_path: str = ""

    # -- lifecycle ---------------------------------------------------------

    def load(
        self,
        engine: str = "sensevoice",
        model_path: str = "",
        vad_path: str = "",
        punc_path=None,
        device: str = "auto",
    ) -> None:
        """Load ASR + FSMN-VAD (+ optional CT punctuation) via funasr AutoModel.

        ``punc_path=None`` auto-resolves the CT model from settings/defaults.json;
        an explicit ``""`` skips loading the punctuation model entirely.
        """
        from core.device_utils import resolve_pytorch_device
        from funasr import AutoModel

        engine = (engine or "sensevoice").lower()
        if engine not in ("sensevoice", "paraformer"):
            raise ValueError(f"Unsupported ASR engine '{engine}'. Choices: sensevoice, paraformer")

        asr_dir = _resolve_model_path(engine, model_path)  # ModelPaths field == engine name
        vad_dir = _resolve_model_path("vad", vad_path)
        if not asr_dir.is_dir():
            raise FileNotFoundError(f"ASR model directory not found: {asr_dir}")
        if not vad_dir.is_dir():
            raise FileNotFoundError(f"VAD model directory not found: {vad_dir}")

        dev = str(resolve_pytorch_device(device))
        funasr_device = "cuda" if dev.startswith("cuda") else "cpu"

        self.engine_name = engine
        self.model_path = str(asr_dir)
        self.vad_path = str(vad_dir)
        self.device = device
        self.asr_model = AutoModel(model=str(asr_dir), disable_update=True, device=funasr_device)
        self.vad_model = AutoModel(model=str(vad_dir), disable_update=True, device=funasr_device)

        # Punctuation model is optional: load it only when configured and present.
        self.punc_model = None
        self.punc_path = ""
        if punc_path != "":  # "" means "skip punctuation model" explicitly
            try:
                candidate = _resolve_model_path("ct_punc", punc_path or "")
            except ValueError:
                candidate = None
            if candidate is not None and candidate.is_dir():
                self.punc_model = AutoModel(model=str(candidate), disable_update=True, device=funasr_device)
                self.punc_path = str(candidate)

    def unload(self) -> None:
        self.asr_model = None
        self.vad_model = None
        self.punc_model = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - torch is always present in the venv
            pass

    @property
    def is_loaded(self) -> bool:
        return self.asr_model is not None and self.vad_model is not None

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
        # transcription still yields text.
        return segments or [(0, max(duration_ms, 1))]

    def _asr_chunk(self, waveform: np.ndarray, start_ms: int, end_ms: int) -> str:
        """Run the ASR model on one VAD chunk; returns raw text (tags included)."""
        a = int(round(start_ms * SAMPLE_RATE / 1000))
        b = min(int(round(end_ms * SAMPLE_RATE / 1000)), waveform.shape[0])
        seg = waveform[a:b]
        if seg.size < SAMPLE_RATE // 4:  # pad very short chunks to >=250 ms
            seg = np.pad(seg, (0, max(1, SAMPLE_RATE // 4 - seg.size)), mode="constant")
        res = self.asr_model.generate(input=seg.astype(np.float32))
        parts: list[str] = []
        for item in res or []:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
        return "".join(parts)

    def _restore_punctuation(self, text: str) -> str:
        """Apply the CT punctuation model to plain (tag-free) text."""
        if self.punc_model is None or not text.strip():
            return text
        res = self.punc_model.generate(input=text)
        parts: list[str] = []
        for item in res or []:
            if isinstance(item, dict):
                t = item.get("text")
                if t:
                    parts.append(str(t))
        punctuated = "".join(parts).strip()
        return punctuated or text

    def transcribe(self, audio_path, enable_punctuation: bool = True, progress_cb=None) -> dict:
        """Transcribe ``audio_path`` (any sample rate; resampled to 16 kHz).

        Returns:
            {"text": str, "segments": [(start_ms, end_ms, text), ...],
             "emotion": str, "event": str, "language": str}

        ``text`` is clean (no ``<|...|>`` tokens); punctuation is restored per
        chunk when enabled and the CT model was loaded. ``segments`` holds one
        entry per VAD speech chunk that produced text, in time order.
        """
        if not self.is_loaded:
            raise RuntimeError("SpeechASREngine.load() must be called before transcribe()")

        import librosa

        audio_path = str(audio_path)
        waveform, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        duration_ms = int(round(waveform.shape[0] * 1000 / SAMPLE_RATE))
        vad_segments = self._vad_segments(audio_path, duration_ms)

        segments: list[tuple[int, int, str]] = []
        languages: list[str] = []
        emotions: list[str] = []
        events: list[str] = []
        total = len(vad_segments)
        for idx, (start_ms, end_ms) in enumerate(vad_segments):
            raw = self._asr_chunk(waveform, start_ms, end_ms)
            lang, emo, ev = parse_sensevoice_tags(raw)
            text = strip_tags(raw)
            if enable_punctuation:
                text = self._restore_punctuation(text).strip()
            languages.append(lang)
            emotions.append(emo)
            events.append(ev)
            if text:
                segments.append((start_ms, end_ms, text))
            if progress_cb is not None and total > 0:
                try:
                    progress_cb((idx + 1) / total, f"chunk {idx + 1}/{total}")
                except Exception:
                    pass

        # Join chunk texts; avoid inserting spaces between CJK chunks.
        full_text = ""
        for _, _, text in segments:
            if not full_text:
                full_text = text
                continue
            sep = "" if (_has_cjk(full_text[-1:]) and _has_cjk(text[:1])) else " "
            full_text += sep + text
        full_text = full_text.strip()

        emo_counter = Counter(e for e in emotions if e and e != "NEUTRAL")
        ev_counter = Counter(v for v in events if v and v != "Speech")
        return {
            "text": full_text,
            "segments": segments,
            "emotion": emo_counter.most_common(1)[0][0] if emo_counter else "",
            "event": ev_counter.most_common(1)[0][0] if ev_counter else "",
            "language": next((l for l in languages if l), ""),
        }


# -- output helpers ----------------------------------------------------------


def format_srt_time(ms: int) -> str:
    """Milliseconds -> SRT timecode ``HH:MM:SS,mmm``."""
    ms = max(0, int(ms))
    return f"{ms // 3600000:02d}:{(ms % 3600000) // 60000:02d}:{(ms % 60000) // 1000:02d},{ms % 1000:03d}"


def build_srt(segments: list[tuple[int, int, str]]) -> str:
    """Render (start_ms, end_ms, text) tuples as an SRT document."""
    blocks = []
    for index, (start_ms, end_ms, text) in enumerate(segments, start=1):
        blocks.append(f"{index}\n{format_srt_time(start_ms)} --> {format_srt_time(end_ms)}\n{text}")
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def assign_speaker(speakers: dict, start_ms: int, end_ms: int) -> int:
    """Speaker id with the largest overlap to [start_ms, end_ms]; -1 when none."""
    best_spk, best_overlap = -1, 0
    for spk_id, info in (speakers or {}).items():
        segs = info.get("segments", []) if isinstance(info, dict) else []
        overlap = sum(max(0, min(end_ms, e) - max(start_ms, s)) for s, e in segs)
        if overlap > best_overlap:
            try:
                spk_value = int(spk_id)
            except (TypeError, ValueError):
                continue
            best_spk, best_overlap = spk_value, overlap
    return best_spk


# -- orchestrator entry point -------------------------------------------------


def run_speech_asr(ctx, cfg, progress_cb=None) -> None:
    """Orchestrator entry point for the speech_asr step (wave D-4).

    Writes ``<out>/asr/<stem>.txt|.srt|.json`` per requested formats and sets
    ``ctx.asr_text``, ``ctx.srt_path`` and ``ctx.results["speech_asr"]``.
    """
    from core.config import ModelPaths

    model_paths = getattr(cfg, "model_paths", None) or ModelPaths.from_settings()
    audio_path = ctx.active_vocals_path or getattr(cfg, "audio_path", "") or ""
    if not audio_path or not pathlib.Path(audio_path).exists():
        raise FileNotFoundError(f"speech_asr: audio file not found: {audio_path!r}")

    engine_name = str(getattr(cfg, "asr_engine", "sensevoice") or "sensevoice").lower()
    enable_punc = bool(getattr(cfg, "enable_punctuation", True))

    base_out = pathlib.Path(getattr(cfg, "output_dir", "") or (pathlib.Path(audio_path).parent / "outputs"))
    out_dir = base_out / "asr"

    engine = SpeechASREngine()
    try:
        if progress_cb is not None:
            progress_cb(0.05, f"loading {engine_name} model")
        engine.load(
            engine=engine_name,
            model_path=getattr(model_paths, engine_name, "") or "",
            vad_path=getattr(model_paths, "vad", "") or "",
            # None -> auto-resolve CT model from settings; "" -> skip explicitly.
            punc_path=(getattr(model_paths, "ct_punc", "") or (None if enable_punc else "")),
            device=getattr(cfg, "device", "auto"),
        )
        if progress_cb is not None:
            progress_cb(0.25, "models ready")

        def _asr_progress(fraction: float, message: str = "") -> None:
            if progress_cb is not None:
                progress_cb(0.25 + 0.65 * fraction, f"transcribing {message}".strip())

        result = engine.transcribe(audio_path, enable_punctuation=enable_punc, progress_cb=_asr_progress)
    finally:
        engine.unload()

    speakers = getattr(ctx, "speakers", None) or {}
    segment_dicts = []
    for start_ms, end_ms, text in result["segments"]:
        segment_dicts.append(
            {
                "start_ms": int(start_ms),
                "end_ms": int(end_ms),
                "text": text,
                "speaker": assign_speaker(speakers, start_ms, end_ms),
            }
        )

    ctx.asr_text = result["text"]
    if not getattr(ctx, "detected_language", "") and result.get("language"):
        ctx.detected_language = result["language"]

    formats = {str(f).lower().lstrip(".") for f in (getattr(cfg, "output_formats", None) or [])}
    if getattr(cfg, "output_srt", False):
        formats.add("srt")
    if getattr(cfg, "output_json", False):
        formats.add("json")
    wanted = [f for f in ("txt", "srt", "json") if f in formats] or ["txt"]

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = pathlib.Path(audio_path).stem
    files: dict[str, str] = {}
    if "txt" in wanted:
        path = out_dir / f"{stem}.txt"
        path.write_text(result["text"], encoding="utf-8")
        files["txt"] = str(path)
    if "srt" in wanted:
        path = out_dir / f"{stem}.srt"
        path.write_text(build_srt(result["segments"]), encoding="utf-8")
        ctx.srt_path = str(path)
        files["srt"] = str(path)
    if "json" in wanted:
        payload = {
            "engine": engine_name,
            "language": result.get("language") or getattr(cfg, "language", "") or "",
            "emotion": result.get("emotion", ""),
            "event": result.get("event", ""),
            "text": result["text"],
            "segments": segment_dicts,
            "speakers": {str(k): v for k, v in speakers.items()},
        }
        path = out_dir / f"{stem}.json"
        path.write_text(json_dumps(payload), encoding="utf-8")
        files["json"] = str(path)

    ctx.results["speech_asr"] = {
        "engine": engine_name,
        "language": result.get("language", ""),
        "emotion": result.get("emotion", ""),
        "event": result.get("event", ""),
        "num_segments": len(segment_dicts),
        "text_length": len(result["text"]),
        "files": files,
    }
    if progress_cb is not None:
        progress_cb(1.0, f"{len(segment_dicts)} segment(s)")


def json_dumps(payload) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)
