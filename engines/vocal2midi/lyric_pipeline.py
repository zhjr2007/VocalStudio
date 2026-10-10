"""Lyric alignment composer (wave C2-5).

Turns sliced vocal chunks into per-note lyrics by chaining:

    Qwen3-ASR transcript -> .lab + HubertFA forced alignment -> LyricMatcher

This is the piece the vendored ``auto_lyric_hybrid`` drives through a daemon
subprocess worker; on this host the Qwen runtime cannot spawn inside that
daemon, so we load ASR/HFA **in-process** (the GUI JobRunner QThread) and
orchestrate the same stages here.

The composer:

1. transcribes each chunk with :class:`Qwen3AsrEngine`;
2. writes a ``.lab`` next to a temp wav and runs :class:`HfaEngine.align`;
3. aligns the transcript to the (optional) reference lyrics with
   :class:`LyricMatcherEngine.align`;
4. maps each aligned word interval back onto the chunk's extracted notes by
   time overlap, filling ``note.lyric``;
5. returns a per-chunk asr_match_log (same shape as the baseline artifacts).
"""

from __future__ import annotations

import pathlib
from typing import Any

from core.config import PROJECT_ROOT

_V2M_REF_DIR = PROJECT_ROOT / "Vocal2Midi"

_IGNORE_TOKENS = {"SP", "AP", "EP", "br", "sil", "pau"}


def _ensure_ref_path() -> None:
    ref = str(_V2M_REF_DIR)
    if _V2M_REF_DIR.is_dir() and ref not in sys.path:
        sys.path.append(ref)


def _is_lexical(text: str) -> bool:
    return str(text or "").strip().upper() not in _IGNORE_TOKENS


def align_chunk_notes(notes, words, chunk_start: float) -> None:
    """Fill ``note.lyric`` on ``notes`` from HFA ``words`` by time overlap.

    ``words`` is a list of ``{text, start, end}`` in chunk-local seconds;
    ``chunk_start`` shifts them onto the global timeline. Each note keeps the
    lyric of the lexical word covering its onset (or the following one).
    """
    intervals = [
        (float(w["start"]), float(w["end"]), str(w["text"]))
        for w in (words or [])
        if _is_lexical(w["text"])
    ]
    if not intervals or not notes:
        return
    for note in notes:
        onset = float(getattr(note, "onset", 0.0))
        local = max(0.0, onset - chunk_start)
        chosen = ""
        for start, end, text in intervals:
            if start <= local <= end:
                chosen = text
                break
        if not chosen:
            # fall back to the interval starting nearest the onset
            nearest = min(intervals, key=lambda iv: abs(iv[0] - local))
            if nearest[0] <= local:
                chosen = nearest[2]
        if chosen:
            note.lyric = chosen


def run_lyric_alignment(
    chunks,
    sr: int,
    cfg,
    notes,
    progress_cb=None,
    cancel_checker=None,
    log_cb=None,
) -> tuple[dict[str, Any], list[str]]:
    """Align lyrics onto ``notes`` for the given ``chunks``.

    Returns ``(per_chunk, chunk_logs)`` where ``per_chunk`` maps chunk index to
    ``{asr, matched_lyric, words}`` and ``chunk_logs`` is the [[stem] ... ] list
    written to ``<key>_asr_match_log.txt``.
    """
    from .hfa_engine import HfaEngine
    from .lyric_matcher import LyricMatcherEngine
    from .qwen3asr_engine import Qwen3AsrEngine

    language = str(getattr(cfg, "language", "auto") or "zh")
    reference = str(getattr(cfg, "reference_lyrics", "") or "")
    device = str(getattr(cfg, "device", "auto") or "cpu")
    model_paths = getattr(cfg, "model_paths", None)

    def check() -> None:
        if callable(cancel_checker) and cancel_checker():
            from core.exceptions import CancellationError

            raise CancellationError("lyric alignment cancelled")

    def log(msg: str) -> None:
        if callable(log_cb):
            try:
                log_cb(str(msg))
            except Exception:
                pass

    def progress(frac: float, stage: str) -> None:
        if callable(progress_cb):
            try:
                progress_cb(float(frac), stage)
            except Exception:
                pass

    asr = Qwen3AsrEngine()
    hfa = HfaEngine()
    matcher = LyricMatcherEngine("zh" if language in ("auto", "zh") else language)
    if reference and reference.strip():
        matcher.bind_reference(reference)

    per_chunk: dict[str, Any] = {}
    chunk_logs: list[str] = []

    try:
        asr.load(model_paths, device=device)
        hfa.load(model_paths, device=device)
        progress(0.0, "lyric")

        for idx, chunk in enumerate(chunks):
            check()
            stem = f"chunk_{idx}"
            audio = chunk["waveform"]
            # 1) ASR
            asr_text = asr.transcribe(audio, int(chunk.get("sr") or sr), language=language)
            log(f"[{stem}] ASR: {asr_text[:80]!r}")
            check()

            # 2) HFA forced alignment (uses the ASR text as the .lab)
            if asr_text.strip():
                words = []
                try:
                    pred = hfa.align(audio, int(chunk.get("sr") or sr), asr_text, language=language)
                    if pred:
                        for _stem, p in pred.items():
                            words = HfaEngine.words_from_pred(p)
                            break
                except Exception as exc:  # noqa: BLE001 - alignment optional
                    log(f"[{stem}] HFA failed ({str(exc)[:80]}); using ASR words")
            else:
                words = []

            # 3) matcher (reference / auto)
            aligned = matcher.align(asr_text, reference_lyrics=reference)
            words = aligned.words if not words else [
                {"text": w.text, "start": w.start, "end": w.end} for w in words
            ]
            check()

            # 4) back-fill lyrics onto this chunk's notes
            chunk_notes = [n for n in notes if float(getattr(n, "onset", 0.0)) >= float(chunk["start"]) - 0.05
                           and float(getattr(n, "onset", 0.0)) < float(chunk["end"]) + 0.05]
            align_chunk_notes(chunk_notes, words, float(chunk["start"]))

            per_chunk[stem] = {
                "asr": asr_text,
                "matched_lyric": aligned.matched_lyric,
                "confidence": aligned.confidence,
                "words": words,
            }
            chunk_logs.append(
                "\n".join(
                    [
                        f"[{stem}]",
                        f"ASR Output: {asr_text or '[Empty or Failed]'}",
                        f"Words: {', '.join(w['text'] for w in words) or '(none)'}",
                        f"Matched: {aligned.matched_lyric or '(no reference)'}",
                        f"Confidence: {aligned.confidence:.3f}",
                        "Status: OK",
                    ]
                )
                + "\n"
            )
            progress((idx + 1) / max(1, len(chunks)), "lyric")
        return per_chunk, chunk_logs
    finally:
        asr.unload()
        hfa.unload()


def write_asr_match_log(chunk_logs: list[str], out_dir: str | pathlib.Path, key: str) -> pathlib.Path:
    """Write the baseline-shaped ``<key>_asr_match_log.txt``."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{key}_asr_match_log.txt"
    path.write_text("\n".join(chunk_logs), encoding="utf-8")
    return path
