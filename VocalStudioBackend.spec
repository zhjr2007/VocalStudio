# -*- mode: python ; coding: utf-8 -*-
"""Complete inference onedir bundle: new PyQt5 UI and unchanged CLI dispatch. Uses the active build environment (ROCm 10 by default); training remains an optional detection interface."""
from pathlib import Path
import importlib.util
import importlib.metadata

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules, copy_metadata

ROOT = Path(SPECPATH)  # noqa: F821 (provided by the PyInstaller spec context)

# dest must be "settings" (a directory source is flattened into dest), so the
# frozen layout matches core.config's PROJECT_ROOT/settings/defaults.json.
datas = [("settings/defaults.json", "settings")]
binaries = []
hiddenimports = []
AMD = importlib.util.find_spec('rocm_sdk') is not None

# UVR5 reference implementation. The engine appends this directory to sys.path
# at runtime, so bundle the code (not the weights) under its exact
# PROJECT_ROOT-relative path inside _MEIPASS.
REF_REL = Path("GPT-SoVITS-v2pro-20250604-nvidia50") / "tools" / "uvr5"
ref_dir = ROOT / REF_REL
if ref_dir.is_dir():
    for f in sorted(ref_dir.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(ref_dir)
        parts = rel.parts
        if parts[0] == "uvr5_weights" or "__pycache__" in parts:
            continue
        dest = REF_REL / Path(*parts[:-1])
        datas.append((str(f), str(dest).replace("\\", "/")))

# third-party imports performed by the reference modules at runtime only
hiddenimports += collect_submodules("einops") + collect_submodules("rotary_embedding_torch")
hiddenimports += ["tqdm", "packaging"]

# orchestrator step entry points are resolved via importlib (string names)
hiddenimports += ["engines.separation", "engines.vocal2midi.pipeline", "engines.speech_asr", "engines.speaker", "engines.tts.voice_clone"]
hiddenimports += ['PyQt5.QtMultimedia', 'funasr', 'modelscope', 'mido', 'pyopenjtalk']
binaries += collect_dynamic_libs('onnxruntime')  # includes DirectML.dll for the AMD ONNX path
for package in ('funasr', 'modelscope', 'pyopenjtalk', 'madmom', 'g2p_en', 'torchcodec'):
    hiddenimports += collect_submodules(package)
    datas += collect_data_files(package, include_py_files=package in ('funasr', 'modelscope'))
# MIDI imports reference code via PROJECT_ROOT/Vocal2Midi at runtime.
for f in (ROOT/'Vocal2Midi'/'inference').rglob('*'):
    if f.is_file() and '__pycache__' not in f.parts and f.suffix in ('.py', '.json', '.yaml', '.yml', '.txt'):
        datas.append((str(f), str(f.relative_to(ROOT).parent)))
# Dynamic backend/plugin and version probes use distribution metadata.
for dist in importlib.metadata.distributions():
    name = dist.metadata.get('Name')
    if name:
        try: datas += copy_metadata(name)
        except Exception: pass
if AMD:
    for package in ('rocm_sdk', 'rocm_sdk_core', 'rocm_sdk_libraries', 'rocm_bootstrap', '_rocm_sdk_core', '_rocm_sdk_libraries'):
        hiddenimports += collect_submodules(package)
        datas += collect_data_files(package, include_py_files=True)
        binaries += collect_dynamic_libs(package)
    # Device-specific wheels merge these resources into the base torch package.
    import torch
    torch_root = Path(torch.__file__).parent
    for folder in ('.kpack', 'lib'):
        for f in (torch_root/folder).rglob('*'):
            if f.is_file() and f.suffix.lower() not in ('.pyc', '.lib'):
                dest = Path('torch')/f.relative_to(torch_root).parent
                datas.append((str(f), str(dest)))

# the launcher imports both front ends inside function bodies. PyInstaller does
# follow those, but naming them costs nothing and survives a refactor that moves
# the import behind a getattr() or a plugin table.
hiddenimports += ["vstudio_launcher", "vstudio_runtime", "vstudio_path_compat", "vstudio_ui", "vstudio_ui.app", "vstudio_ui.pages", '_jieba_fast_functions_py3']

# TTS chain deps (GSV TTS_infer_pack imports transformers/peft/… inside the
# frozen app; onnxruntime is pulled by the g2pw onnx converter).
hiddenimports += collect_submodules("peft") + collect_submodules("einx")
hiddenimports += collect_submodules("fast_langdetect") + collect_submodules("split_lang")
hiddenimports += [
    "transformers",
    "transformers.models.bert",
    "transformers.models.roberta",
    "transformers.models.auto",
    "transformers.models.wav2vec2",
    "transformers.models.wav2vec2.feature_extraction_wav2vec2",
    "tokenizers",
    "loguru",
    "opencc",
    "cn2an",
    "jieba_fast",
    "nltk",
    "fasttext_predict",
    "colorlog",
    "robust_downloader",
    "ffmpeg",
]

# transformers 5.x probes torchcodec via importlib.metadata at import time
# (wav2vec2 feature extractor); the metadata must exist in the frozen layout.
try:
    datas += copy_metadata("torchcodec")
except Exception:  # pragma: no cover - package not installed
    pass
hiddenimports += ["torchcodec", "budoux"]
datas += collect_data_files("budoux")
# jieba_fast ships compiled submodules (.pyd in the package dir): the static
# analysis misses them, so copy the whole package side-by-side
hiddenimports += collect_submodules("jieba_fast")
try:
    datas += collect_data_files("jieba_fast")
except Exception:  # pragma: no cover
    pass
# wordsegment (GSV english.py) needs its bundled unigram/bigram data files
hiddenimports += ["wordsegment", "g2p_en"]
try:
    datas += collect_data_files("wordsegment")
except Exception:  # pragma: no cover
    pass
try:
    datas += collect_data_files("g2p_en")  # checkpoint20.npz (~25MB)
except Exception:  # pragma: no cover
    pass

# G2P English corpus for the GSV text frontend (runtime nltk.data.path lookup)
if (ROOT / "nltk_data").is_dir():
    datas.append(("nltk_data", "nltk_data"))

# engines/tts stubs: pytorch_lightning / torchmetrics / matplotlib stubs and the
# vendored x_transformers are inserted into sys.path at runtime by _ref.py
if (ROOT / "engines" / "tts" / "stubs").is_dir():
    for f in sorted((ROOT / "engines" / "tts" / "stubs").rglob("*.py")):
        rel = f.relative_to(ROOT / "engines" / "tts" / "stubs")
        dest = Path("engines/tts/stubs") / rel.parent
        datas.append((str(f), str(dest).replace("\\", "/")))

# GPT-SoVITS reference implementation (TTS chain + its own package layout).
# Code only: weights (pretrained_models, G2PWModel, uvr5_weights) stay external
# and are copied next to _internal by build_portable.bat.
GSV_ROOT = Path("GPT-SoVITS-v2pro-20250604-nvidia50")
TTS_SRC_REL = GSV_ROOT / "GPT_SoVITS"
TTS_SKIP_PREFIXES = ("pretrained_models", "G2PWModel", "uvr5_weights", "__pycache__")
for sub in (
    "AR", "module", "TTS_infer_pack", "BigVGAN", "f5_tts", "text",
    "configs", "eres2net", "feature_extractor",
):
    src = ROOT / TTS_SRC_REL / sub
    if not src.is_dir():
        continue
    for f in sorted(src.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(ROOT / TTS_SRC_REL)
        if any(part in TTS_SKIP_PREFIXES for part in rel.parts):
            continue
        dest = TTS_SRC_REL / rel.parent
        datas.append((str(f), str(dest).replace("\\", "/")))
for single in ("sv.py", "process_ckpt.py", "utils.py", "utils_ckpt.py"):
    src = ROOT / TTS_SRC_REL / single
    if src.is_file():
        datas.append((str(src), str(TTS_SRC_REL).replace("\\", "/")))
# the tools/i18n + tools/audio_sr packages (reference imports at runtime)
for tools_sub in ("i18n", "audio_sr"):
    src = ROOT / GSV_ROOT / "tools" / tools_sub
    if not src.is_dir():
        continue
    for f in sorted(src.rglob("*")):
        if f.is_file() and "__pycache__" not in f.parts:
            rel = f.relative_to(ROOT / GSV_ROOT)
            dat_dir = GSV_ROOT / rel.parent
            datas.append((str(f), str(dat_dir).replace("\\", "/")))

# CUDA 12 native libs for torch cu128 (nvidia.* wheels); without these the
# frozen exe reports cuda=False.
for pkg in (() if AMD else (
    "nvidia.cublas",
    "nvidia.cuda_runtime",
    "nvidia.cudnn",
    "nvidia.cufft",
    "nvidia.cuda_nvrtc",
    "nvidia.nvjitlink",
)):
    try:
        binaries += collect_dynamic_libs(pkg)
    except Exception:  # pragma: no cover - package not installed in this venv
        pass

a = Analysis(
    ["vstudio_launcher.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['scripts/pyi_runtime.py'],
    # PyQt5 is no longer excluded: the new UI is the bundle's default front end.
    # qfluentwidgets stays out — the new UI is PyQt5-only on purpose, and pulling
    # it in would resurrect the styling fight the rebuild was about.
    excludes=["qfluentwidgets", "tensorflow", "tensorboard", "matplotlib.tests", "pytest"],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="VocalStudioBackend",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # keep the large CUDA/torch DLLs intact and build fast
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="VocalStudioBackend",
)
