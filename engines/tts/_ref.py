"""GPT-SoVITS inference chain, refactored into four thin wrappers.

Each module wraps one reference component (all lazy-imported from
``GPT-SoVITS-v2pro-20250604-nvidia50/GPT_SoVITS`` under the reference's own
cwd so its ``os.getcwd()``-relative model paths keep resolving):

- :mod:`engines.tts.hubert` - CNHuBERT content encoder
- :mod:`engines.tts.t2s_model` - Text2Semantic (GPT) decoder
- :mod:`engines.tts.sovits_model` - VITS decoder (v1/v2/v2Pro/v2ProPlus/v3/v4)
- :mod:`engines.tts.vocoder` - BigVGAN / Generator vocoder (v3/v4 only)

:mod:`engines.tts.voice_clone` combines them in ``VoiceCloneEngine.clone``
(S3.4 interface) with staged progress: load -> preprocess -> t2s -> sovits
(-> vocoder on v3/v4) -> save.
"""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path

from core.config import PROJECT_ROOT

GSV_ROOT = PROJECT_ROOT / "GPT-SoVITS-v2pro-20250604-nvidia50"
GSV_PKG = GSV_ROOT / "GPT_SoVITS"
STUBS_DIR = Path(__file__).resolve().parent / "stubs"

_imported = False


def _ensure_import_paths() -> None:
    """Put everything the reference chain imports on ``sys.path``.

    Order matters: our stubs first (pytorch_lightning / torchmetrics /
    matplotlib, real x_transformers vendored), then the GSV package dir (for
    ``AR``, ``module``, ``TTS_infer_pack``, ...), then GSV_ROOT (for ``tools``
    and the ``GPT_SoVITS.<x>`` absolute imports used by f5_tts).
    """
    for base in (STUBS_DIR, GSV_PKG, GSV_ROOT):
        text = str(base)
        if text not in sys.path:
            sys.path.insert(0, text)


def ensure_reference_imported() -> None:
    """Import the reference ``TTS_infer_pack.TTS`` under the GSV cwd.

    The reference module captures ``os.getcwd()`` at import time and later
    resolves ``GPT_SoVITS/pretrained_models/...`` paths relative to it, so the
    first import (and every import attempt) must happen with GSV_ROOT as cwd.
    The original cwd is restored afterwards; the already-imported module keeps
    working because every later path resolution is absolute.
    """
    global _imported
    if _imported:
        return
    _ensure_import_paths()
    import nltk

    nltk_dir = str(PROJECT_ROOT / "nltk_data")
    if nltk_dir not in nltk.data.path:
        nltk.data.path.insert(0, nltk_dir)
    previous = os.getcwd()
    try:
        os.chdir(str(GSV_ROOT))
        # Frozen launch: transformers' lazy __getattr__ resolves
        # "Wav2Vec2FeatureExtractor" through a RELATIVE import
        # (importlib.util class_to_module path) which fails under PyInstaller's
        # frozen importer. Pre-import the concrete wav2vec2 feature extractor
        # and pin it on the transformers namespace to bypass the lazy lookup.
        try:
            import transformers as _transformers

            import transformers.models.wav2vec2.feature_extraction_wav2vec2 as _wf

            if not hasattr(_transformers, "Wav2Vec2FeatureExtractor"):
                _transformers.Wav2Vec2FeatureExtractor = _wf.Wav2Vec2FeatureExtractor
        except Exception as _preload_exc:  # pragma: no cover - source mode falls through
            if os.environ.get("VS_DEBUG") == "1":
                import traceback as _tb

                print("[tts._ref] wav2vec2 preload failed:", repr(_preload_exc))
                _tb.print_exc()
        # The reference tools/audio_sr package ships empty (no __init__.py);
        # TTS.py imports AP_BWE at module level. v2Pro/v2ProPlus never invoke
        # super-sampling, so a fail-fast stub is injected into sys.modules.
        try:
            import types as _types

            _audio_sr = _types.ModuleType("tools.audio_sr")
            if not hasattr(_audio_sr, "AP_BWE"):

                class _AP_BWE:  # pragma: no cover - never instantiated for v2Pro
                    def __init__(self, device, cls):
                        raise FileNotFoundError(
                            "AP_BWE super-sampling weights are not installed"
                        )

            _audio_sr.AP_BWE = _AP_BWE
            sys.modules["tools.audio_sr"] = _audio_sr
        except Exception as _stub_exc:  # pragma: no cover
            if os.environ.get("VS_DEBUG") == "1":
                print("[tts._ref] audio_sr stub failed:", repr(_stub_exc))
        # inflect (used by wordsegment inside the GSV english G2P) decorates
        # its engine functions with typeguard @typechecked, which needs
        # inspect.getsource() — unavailable on frozen bytecode. A stub
        # typechecked (sys.modules wins before any finder) keeps inflect
        # working without runtime type checks.
        try:
            import types as _types2

            def _typechecked(*args, **kwargs):  # noqa: ANN001 - stub decorator
                if args and len(args) == 1 and callable(args[0]):
                    return args[0]

                def _decorator(func):
                    return func

                return _decorator

            _typeguard = _types2.ModuleType("typeguard")
            _typeguard.typechecked = _typechecked
            sys.modules["typeguard"] = _typeguard
        except Exception as _tg_exc:  # pragma: no cover
            if os.environ.get("VS_DEBUG") == "1":
                print("[tts._ref] typeguard stub failed:", repr(_tg_exc))
        import TTS_infer_pack  # noqa: F401 - triggers all reference imports
    finally:
        os.chdir(previous)
    _imported = True


@contextlib.contextmanager
def reference_workdir():
    """Run a reference-code snippet with the GSV checkout as the cwd.

    The text frontend resolves ``GPT_SoVITS/text/G2PWModel`` and the SV weights
    relative to the GSV root; the side effect is reverted on exit.
    """
    previous = os.getcwd()
    try:
        os.chdir(str(GSV_ROOT))
        yield
    finally:
        os.chdir(previous)


def import_tts_reference():
    """Return the reference ``TTS``/``TTS_Config`` classes (lazy import hook)."""
    ensure_reference_imported()
    from TTS_infer_pack.TTS import TTS, TTS_Config

    return TTS, TTS_Config


def resolve_model_paths(model_paths=None) -> dict:
    """Absolute GSV weight paths from settings (ModelPaths duck type)."""
    from core.config import ModelPaths

    mp = model_paths if model_paths is not None else ModelPaths.from_settings()

    def pick(name: str) -> Path:
        value = getattr(mp, name, "") or ""
        path = Path(value)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path

    return {
        "bert": pick("chinese_roberta"),
        "hubert": pick("chinese_hubert"),
        "t2s_v2pro": pick("gsv_v2pro_gpt"),
        "sovits_v2pro": pick("gsv_v2pro_sovits"),
        "t2s_v2proplus": pick("gsv_v2proplus_gpt"),
        "sovits_v2proplus": pick("gsv_v2proplus_sovits"),
    }


def build_reference_config(
    version: str,
    device: str,
    *,
    t2s_path: str,
    sovits_path: str,
    bert_path: str,
    hubert_path: str,
    is_half: bool,
) -> dict:
    """Reference ``TTS_Config`` payload (the yaml ``custom`` section shape).

    Passing a plain key/value dict does NOT work: ``TTS_Config.__init__``
    falls back to the default ``v2`` section unless the dict carries
    ``{"custom": {...}}`` (the tts_infer.yaml layout).
    """
    return {
        "custom": {
            "device": str(device),
            "is_half": bool(is_half),
            "version": str(version),
            "t2s_weights_path": str(t2s_path),
            "vits_weights_path": str(sovits_path),
            "bert_base_path": str(bert_path),
            "cnhuhbert_base_path": str(hubert_path),
        }
    }
