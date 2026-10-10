"""Benchmark each VocalStudio model on the active compute backend and run the
two capstone workflows. Usage: python scripts/benchmark_models.py [--device auto|cuda|rocm|cpu] [--json out.json]

- capability (a): speaker diarization -> per-speaker ASR transcript.
- capability (b): music source separation -> vocal/instrumental + multi-stem.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.config import PROJECT_ROOT, ModelPaths
from core import device_utils as du

BENCH = ROOT / "temp" / "bench"
BENCH.mkdir(parents=True, exist_ok=True)

SPEECH = BENCH / "lecture_60s.wav"      # multi-speaker classroom recording (derived, 60s)
SPEECH2 = ROOT / "temp" / "d3_speech_18s.wav"  # 18s two-speaker speech
DUET = BENCH / "duet_60s.wav"           # two-singer song excerpt (derived, 60s)
REF_WAV = ROOT / "experiments" / "speaker" / "iic" / "speech_campplus_sv_zh-cn_16k-common" / "examples" / "speaker1_a_cn_16k.wav"

results: dict = {}


def _resolve_device(requested: str) -> tuple[str, str]:
    """Return (backend_label, device_str_for_engines)."""
    if requested in ("cuda", "rocm", "cpu"):
        # user forced a backend
        if requested == "rocm" and (du.detect_rocm() or du.detect_zluda()):
            return "rocm", "cuda"
        if requested == "rocm":
            return "rocm(missing)", "cpu"
        return requested, requested
    # auto: report the effective backend per the target order rocm->zluda->cuda->cpu
    if du.detect_rocm():
        return "rocm", "cuda"
    if du.detect_zluda():
        return "zluda", "cuda"
    if du.cuda_available():
        return "cuda", "cuda"
    return "cpu", "cpu"


def _time(fn, *a, **kw):
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    return (time.perf_counter() - t0), out


def _bench_load_run(label, load, run, *run_args, **run_kw):
    """Time load + a single run; returns dict."""
    rec = {}
    try:
        dt, mod = _time(load)
        rec["load_s"] = round(dt, 2)
    except Exception as e:
        rec["error"] = f"load: {type(e).__name__}: {e}"
        return rec
    try:
        dt, out = _time(run, *run_args, **run_kw)
        rec["run_s"] = round(dt, 2)
        aux = run_kw.get("_note")
        if aux:
            rec["note"] = aux
    except Exception as e:
        rec["error"] = f"run: {type(e).__name__}: {e}"
    return rec


def bench_speaker_asr(device: str):
    """capability (a): diarization -> per-speaker ASR."""
    from engines.speaker.campp_engine import CAMPPEngine
    from engines.speech_asr.asr_engine import SpeechASREngine

    src = str(SPEECH if SPEECH.exists() else SPEECH2)
    rec = {"audio": src}
    eng = CAMPPEngine()
    dt, _ = _time(eng.load, device=device)
    rec["diarize_load_s"] = round(dt, 2)
    dn, out = _time(eng.diarize, src, None, 4, str(BENCH / "spk_diar"))
    rec["diarize_run_s"] = round(dn, 2)
    rec["speakers"] = sorted(out) if isinstance(out, dict) else out
    per_spk = {}
    asr = SpeechASREngine()
    dt, _ = _time(asr.load, engine="sensevoice", device=device)
    rec["asr_load_s"] = round(dt, 2)
    for spk in (sorted(out) if isinstance(out, dict) else []):
        info = out[spk]
        wav = info["audio_path"] if isinstance(info, dict) else (info[0] if isinstance(info, list) else info)
        if not Path(str(wav)).exists():
            continue
        dn, res = _time(asr.transcribe, str(wav), True, None)
        per_spk[str(spk)] = {"wav": str(wav), "asr_s": round(dn, 2),
                             "text": (res or {}).get("text", "")[:80]}
    rec["per_speaker"] = per_spk
    eng.unload(); asr.unload()
    return rec


def bench_asr_engines(device: str):
    from engines.speech_asr.asr_engine import SpeechASREngine
    src = str(SPEECH if SPEECH.exists() else SPEECH2)
    out = {}
    for eng_name in ("sensevoice", "paraformer"):
        e = SpeechASREngine()
        try:
            dt, _ = _time(e.load, engine=eng_name, device=device)
            load_s = round(dt, 2)
        except Exception as ex:
            out[eng_name] = {"error": f"load: {type(ex).__name__}: {ex}"}
            continue
        try:
            dn, res = _time(e.transcribe, src, True, None)
            out[eng_name] = {"load_s": load_s, "asr_s": round(dn, 2),
                             "text": (res or {}).get("text", "")[:100]}
        except Exception as ex:
            out[eng_name] = {"load_s": load_s, "error": f"asr: {type(ex).__name__}: {ex}"}
        e.unload()
    return out


def bench_separation(device: str, spec: str = "Band4"):
    from engines.separation.uvr5_engine import UVR5Engine
    if not DUET.exists():
        return {"error": "no duet test clip"}
    e = UVR5Engine(ModelPaths.from_settings())
    dt, _ = _time(e.load, spec, device=device)
    rec = {"preset": spec, "load_s": round(dt, 2)}
    try:
        dn, paths = _time(e.run, str(DUET), str(BENCH / "sep"), 20)
        rec["run_s"] = round(dn, 2)
        rec["stems"] = {k: str(v) for k, v in (paths or {}).items()}
    except Exception as ex:
        rec["error"] = f"run: {type(ex).__name__}: {ex}"
    e.unload()
    return rec


def bench_vocal2midi(device: str, vocals: Path | None = None):
    from engines.vocal2midi.pipeline import run
    if vocals is None or not Path(vocals).exists():
        return {"error": "no vocals input"}
    try:
        import soundfile as sf
        y, sr = sf.read(str(vocals), always_2d=True)
        y = y.mean(axis=1) if y.ndim == 2 else y
    except Exception as ex:
        return {"error": f"load audio: {type(ex).__name__}: {ex}"}
    class _Cfg:
        out_dir = str(BENCH / "v2m")
        bpm = 120
        key = 1
    dn, out = _time(run, np.asarray(y, dtype=np.float32), int(sr), _Cfg(), None)
    return {"audio_len_s": round(len(y) / sr, 2), "run_s": round(dn, 2),
            "midi_pitch_len": len((out or {}).get("midi_pitch", [])),
            "note_count": len((out or {}).get("notes", []))}


def bench_tts(device: str):
    from engines.tts.voice_clone import VoiceCloneEngine
    if not REF_WAV.exists():
        return {"error": "no ref wav"}
    e = VoiceCloneEngine()
    dt, _ = _time(e.load, "v2Pro", "auto")
    rec = {"load_s": round(dt, 2)}
    try:
        dn, _out = _time(e.clone, "你好，VocalStudio。", "zh", ref_audio=str(REF_WAV))
        rec["clone_s"] = round(dn, 2)
    except Exception as ex:
        rec["error"] = f"clone: {type(ex).__name__}: {ex}"
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    backend, dev = _resolve_device(args.device)
    results["backend"] = backend
    results["engine_device"] = dev
    results["torch"] = {"cuda": du.cuda_available(), "rocm": du.detect_rocm(), "zluda": du.detect_zluda()}
    print(f"[backend] {backend} | engines use device={dev} | torch cuda={results['torch']['cuda']}")

    results["capability_a_speaker_asr"] = bench_speaker_asr(dev)
    results["asr_models"] = bench_asr_engines(dev)
    results["separation"] = {s: bench_separation(dev, s) for s in ("Karaoke", "Band4", "Producer5")}
    vocals = None
    sep = results["separation"].get("Band4", {})
    if isinstance(sep.get("stems"), dict):
        st = sep["stems"]
        vocals = st.get("vocals") or st.get("accompaniment")
    results["vocal2midi"] = bench_vocal2midi(dev, Path(vocals) if vocals else None)
    results["tts"] = bench_tts(dev)

    if args.json:
        Path(args.json).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[json] {args.json}")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
