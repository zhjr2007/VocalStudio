"""Device resolution for CUDA / ROCm / DirectML / CPU backends."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

RUNTIME_DEVICE_CHOICES = ("auto", "cuda", "rocm", "zluda", "directml", "cpu")

_LEGACY_ALIASES = {
    "": "auto",
    "dml": "directml",
    "gpu": "auto",
    "nvidia": "cuda",
    "amd": "rocm",
    "xpu": "cpu",
}

_HW_CACHE: dict | None = None

# Set to "1" by scripts/run_with_zluda.bat; the authoritative ZLUDA marker.
_ZLUDA_ENV_VAR = "ZLUDA_ACTIVE"

_CIM_COMMAND = (
    "$cpu=(Get-CimInstance Win32_Processor | Select-Object -First 1).Name; "
    "Write-Output ('CPU:' + $cpu); "
    "Get-CimInstance Win32_VideoController | ForEach-Object { Write-Output ('GPU:' + $_.Name + '|' + [string]$_.AdapterRAM) }"
)


def detect_system_hardware() -> dict:
    """Report machine CPU/GPU names via WMI (Windows). Cached per process."""
    global _HW_CACHE
    if _HW_CACHE is not None:
        return _HW_CACHE
    cpu_name = ""
    gpus: list[tuple[str, float]] = []
    if sys.platform.startswith("win"):
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command", _CIM_COMMAND],
                capture_output=True, text=True, timeout=25,
            ).stdout
            for line in out.splitlines():
                if line.startswith("CPU:"):
                    cpu_name = line[4:].strip()
                elif line.startswith("GPU:"):
                    name, _, ram_text = line[4:].partition("|")
                    try:
                        vram_gb = round(int(ram_text) / 1024**3, 2)
                    except ValueError:
                        vram_gb = 0.0
                    gpus.append((name.strip(), vram_gb))
        except Exception:
            pass
    if not cpu_name:
        cpu_name = platform.processor() or "CPU"
    _HW_CACHE = {"cpu": cpu_name, "gpus": gpus}
    return _HW_CACHE


def _gpu_names() -> list[str]:
    return [name.lower() for name, _vram in detect_system_hardware()["gpus"]]


def _has_nvidia_gpu() -> bool:
    return any(("nvidia" in n or "rtx" in n or "geforce" in n) for n in _gpu_names())


def _has_amd_gpu() -> bool:
    return any(("radeon" in n or "amd" in n) for n in _gpu_names())


def default_backend() -> str:
    """Pick a concrete backend from the detected hardware (no torch needed).

    Priority follows the recorded target order: rocm / zluda (AMD) > cuda
    (NVIDIA) > cpu.  ZLUDA cannot be confirmed without torch, so an AMD GPU is
    reported as ``rocm`` here; the refined ``zluda`` label is decided at
    runtime by :func:`detect_zluda` / :func:`auto_select_device`.
    """
    if _has_amd_gpu():
        return "rocm"
    if _has_nvidia_gpu():
        return "cuda"
    return "cpu"


def normalize_runtime_device(device: str | None) -> str:
    key = (device or "auto").strip().lower()
    key = _LEGACY_ALIASES.get(key, key)
    if key not in RUNTIME_DEVICE_CHOICES:
        raise ValueError(
            f"Unknown runtime device '{device}'. Choices: {', '.join(RUNTIME_DEVICE_CHOICES)}"
        )
    return key


def _import_torch():
    import torch

    return torch


def _torch_looks_real(module) -> bool:
    """PyInstaller 'excluded' torch can end up as a stub module in sys.modules.

    It imports fine but carries neither ``__version__`` nor ``cuda``; treat
    those shells as unavailable so the frontend exe falls back cleanly.
    """
    return bool(getattr(module, "__version__", None) and hasattr(module, "cuda"))


def torch_available() -> bool:
    try:
        return _torch_looks_real(_import_torch())
    except ImportError:
        return False


def detect_rocm() -> bool:
    if not torch_available():
        return False
    torch = _import_torch()
    return torch.cuda.is_available() and bool(torch.version.hip)


def cuda_available() -> bool:
    if not torch_available():
        return False
    torch = _import_torch()
    return torch.cuda.is_available() and not torch.version.hip


def _zluda_env_active() -> bool:
    """True when launched under the ZLUDA launcher (sets ZLUDA_ACTIVE=1)."""
    return os.environ.get(_ZLUDA_ENV_VAR, "").strip().lower() in ("1", "true", "yes", "on")


def detect_zluda() -> bool:
    """True when a CUDA torch build exposes an AMD GPU through ZLUDA.

    ZLUDA is a drop-in ``nvcuda`` so torch stays a stock CUDA wheel
    (``torch.version.hip`` unset) yet reports cuda available.  The launcher
    sets ``ZLUDA_ACTIVE=1`` (scripts/run_with_zluda.bat) as the authoritative
    marker; the legacy heuristic (CUDA torch + AMD dGPU + no NVIDIA) covers
    launcher builds that predate the marker.
    """
    if not cuda_available():
        return False
    if _zluda_env_active():
        return _has_amd_gpu()
    return _has_amd_gpu() and not _has_nvidia_gpu()


def preferred_pytorch_gpu_index() -> int:
    """This ROCm 10 release targets RX 7600M XT, regardless of iGPU ordering."""
    torch = _import_torch()
    if sys.platform == 'win32' and str(getattr(torch, '__version__', '')).startswith('2.13.0+rocm10.0.0'):
        for index in range(torch.cuda.device_count()):
            if 'RX 7600M XT' in torch.cuda.get_device_name(index): return index
        raise RuntimeError('This ROCm 10 package requires an available AMD Radeon RX 7600M XT; refusing to use an unsupported GPU')
    return 0


def resolve_pytorch_device(device: str | None = "auto"):
    dev = normalize_runtime_device(device)
    torch = _import_torch()
    def native_rocm_device():
        index = preferred_pytorch_gpu_index()
        if sys.platform == 'win32' and str(getattr(torch, '__version__', '')).startswith('2.13.0+rocm10.0.0'):
            # Current GPU is thread-local. Qt workers must also bind it because
            # vendor models sometimes use bare "cuda" internally.
            torch.cuda.set_device(index)
        return torch.device("cuda", index)
    if dev == "cpu":
        return torch.device("cpu")
    if dev == "cuda":
        if not cuda_available():
            raise RuntimeError(
                "CUDA requested but not available. Check the torch cu12x build and driver."
            )
        return torch.device("cuda", 0)
    if dev == "zluda":
        if detect_zluda():
            return torch.device("cuda", 0)
        raise RuntimeError(
            "ZLUDA requested but not detected. An AMD GPU must be exposed as CUDA "
            "under the ZLUDA launcher (scripts/run_with_zluda.bat)."
        )
    if dev == "rocm":
        if detect_rocm():
            return native_rocm_device()
        if detect_zluda():
            # Under the ZLUDA launcher an AMD GPU emulates NVIDIA CUDA.
            return torch.device("cuda", 0)
        raise RuntimeError(
            "ROCm requested but not available. Install the ROCm (hip) torch "
            "build (venv-rocm; wheels from repo.amd.com, see docs/zluda-notes.md) "
            "or run under the ZLUDA launcher (scripts/run_with_zluda.bat). "
            "Refusing to fall back to a CUDA (NVIDIA) device."
        )
    if dev == "directml":
        # DirectML is an ORT execution provider; torch engines fall back to CPU
        return torch.device("cpu")
    # auto: rocm -> zluda -> cuda -> cpu (directml is ORT-only)
    if detect_rocm():
        return native_rocm_device()
    if detect_zluda():
        return torch.device("cuda", 0)
    if cuda_available():
        return torch.device("cuda", 0)
    return torch.device("cpu")


def auto_select_device() -> str:
    if detect_rocm():
        return "rocm"
    if detect_zluda():
        return "zluda"
    if cuda_available():
        return "cuda"
    return "cpu"


def get_gpu_info() -> dict:
    info = {
        "backend": "none",
        "name": None,
        "total_vram_gb": 0.0,
        "free_vram_gb": 0.0,
        "compute_capability": None,
        "cuda_version": None,
        "torch_version": None,
        "torch_cuda_available": False,
    }
    if not torch_available():
        return info
    torch = _import_torch()
    info["torch_version"] = torch.__version__
    if torch.cuda.is_available():
        index = preferred_pytorch_gpu_index()
        info["torch_cuda_available"] = True
        if torch.version.hip:
            info["backend"] = "rocm"
        elif detect_zluda():
            info["backend"] = "zluda"
        else:
            info["backend"] = "cuda"
        info["name"] = torch.cuda.get_device_name(index)
        info['device_index'] = index
        props = torch.cuda.get_device_properties(index)
        info["total_vram_gb"] = round(props.total_memory / 1024**3, 2)
        free, _total = torch.cuda.mem_get_info(index)
        info["free_vram_gb"] = round(free / 1024**3, 2)
        info["compute_capability"] = f"{props.major}.{props.minor}"
        info["cuda_version"] = torch.version.cuda
    return info


def vram_status() -> tuple[float, float]:
    if not torch_available():
        return 0.0, 0.0
    torch = _import_torch()
    if not torch.cuda.is_available():
        return 0.0, 0.0
    free, total = torch.cuda.mem_get_info(preferred_pytorch_gpu_index())
    return free / 1024**3, total / 1024**3


def check_python_environment() -> dict:
    import sys

    torch = _import_torch() if torch_available() else None
    return {
        "python": sys.version,
        "torch": torch.__version__ if torch else None,
        "torch_cuda": torch.version.cuda if torch and torch.version.cuda else None,
        "torch_hip": torch.version.hip if torch and torch.version.hip else None,
        "nvidia_smi": _find_smi(),
    }


def _find_smi() -> str | None:
    for name in ("nvidia-smi",):
        path = shutil.which(name)
        if path:
            return path
    return None


# ---------------------------------------------------------------------------
# onnxruntime integration (used by the ORT-backed engines)
# ---------------------------------------------------------------------------

_ORT_MODULE = None


def bootstrap_onnxruntime():
    """Import onnxruntime with the nvidia cu12 DLL dirs registered first.

    The pinned onnxruntime-gpu 1.x build links cublas/cudnn/cufft from the
    `nvidia` pip packages; on Windows those native libs must be on the DLL
    search path before ORT is imported or the CUDA EP fails to initialize.
    Under a native ROCm (hip) torch build those packages do not exist and
    the CUDA EP must never be initialized, so registration is skipped unless
    the process is running under the ZLUDA launcher.
    """
    global _ORT_MODULE
    if _ORT_MODULE is not None:
        return _ORT_MODULE
    if sys.platform.startswith("win") and not (detect_rocm() and not _zluda_env_active()):
        try:
            import site

            roots = list(site.getsitepackages()) + [Path(sys.executable).parent.parent / "Lib" / "site-packages"]
            for root in dict.fromkeys(str(r) for r in roots if r):
                nvidia_dir = Path(root) / "nvidia"
                if not nvidia_dir.is_dir():
                    continue
                for bin_dir in sorted(nvidia_dir.glob("*/bin")):
                    try:
                        os.add_dll_directory(str(bin_dir))
                    except OSError:
                        pass
        except Exception:
            pass
    import onnxruntime

    _ORT_MODULE = onnxruntime
    return onnxruntime


def ort_execution_providers(device: str | None = "auto") -> tuple[list[str], str]:
    """Pick ORT execution providers for a runtime device choice.

    Returns ``(providers, note)``; the CPU EP is always appended as the last
    fallback entry and ``note`` explains any silent fallback so engines can
    surface it in their logs.
    """
    dev = normalize_runtime_device(device)
    ort = bootstrap_onnxruntime()
    available = set(ort.get_available_providers())
    cuda_ep = "CUDAExecutionProvider" if "CUDAExecutionProvider" in available else None
    dml_ep = next((name for name in ('DmlExecutionProvider', 'DMLExecutionProvider') if name in available), None)

    if dev == "zluda":
        # ZLUDA exposes nvcuda; ORT sees a CUDA device.
        if cuda_ep:
            return [cuda_ep, "CPUExecutionProvider"], "ZLUDA (emulated CUDA EP)"
        return ["CPUExecutionProvider"], "ZLUDA not available for ORT; using CPU"
    if dev == "cuda":
        if cuda_ep:
            return [cuda_ep, "CPUExecutionProvider"], ""
        return ["CPUExecutionProvider"], f"CUDA EP not installed; using CPU (onnxruntime {ort.__version__})"
    if dev == "rocm":
        # ORT has no Windows ROCm EP. Under ZLUDA the AMD GPU shows up as an
        # emulated CUDA device; otherwise stay on CPU and never touch NVIDIA.
        if detect_zluda():
            if cuda_ep:
                return [cuda_ep, "CPUExecutionProvider"], "ROCm via ZLUDA (emulated CUDA EP)"
            return ["CPUExecutionProvider"], "ZLUDA not available for ORT; using CPU"
        if detect_rocm():
            if dml_ep:
                return [dml_ep, 'CPUExecutionProvider'], 'ROCm PyTorch + DirectML ONNX GPU; unsupported ONNX operators may use CPU'
            return ["CPUExecutionProvider"], "ROCm torch active; DirectML is missing, using CPU"
        return ["CPUExecutionProvider"], "ROCm unavailable for ORT; using CPU"
    if dev == "directml":
        if dml_ep:
            return [dml_ep, "CPUExecutionProvider"], ""
        return (
            ["CPUExecutionProvider"],
            "DML EP not installed (running onnxruntime-gpu build); use scripts/switch_ort_directml.bat for the DirectML flavor.",
        )
    if dev == "cpu":
        return ["CPUExecutionProvider"], ""
    # auto: rocm-aware — a native ROCm torch build must never pull in the
    # CUDA EP, because that would initialize the NVIDIA GPU.
    if detect_rocm():
        if dml_ep:
            return [dml_ep, "CPUExecutionProvider"], "auto under ROCm torch -> DML (no Windows ORT ROCm EP)"
        return ["CPUExecutionProvider"], "auto under ROCm torch -> CPU (no Windows ORT ROCm EP)"
    if cuda_ep:
        return [cuda_ep, "CPUExecutionProvider"], ""
    if dml_ep:
        return [dml_ep, "CPUExecutionProvider"], ""
    return ["CPUExecutionProvider"], "auto -> CPU"


def create_ort_session(model_path, providers):
    """Create a sequential GPU-first session and fail if GPU initialization fails."""
    ort = bootstrap_onnxruntime()
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.enable_mem_pattern = False
    options.enable_cpu_mem_arena = False
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    configured = list(providers)
    if providers[0] in ('DmlExecutionProvider', 'DMLExecutionProvider'):
        from core.directml_adapter import select_adapter
        torch = _import_torch()
        name = torch.cuda.get_device_name(preferred_pytorch_gpu_index()) if detect_rocm() and torch.cuda.is_available() else ''
        adapter = select_adapter(name, 0x1002 if detect_rocm() else None)
        configured[0] = (providers[0], {'device_id': str(adapter['index'])})
        print(f"[ONNX DirectML] GPU={adapter['name']} adapter={adapter['index']} model={Path(model_path).name}", flush=True)
    profile = os.environ.get('VOCALSTUDIO_ORT_PROFILE_DIR')
    if profile:
        Path(profile).mkdir(parents=True, exist_ok=True)
        options.enable_profiling = True
        options.profile_file_prefix = str(Path(profile)/Path(model_path).stem)
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=configured)
    if providers[0] != 'CPUExecutionProvider' and session.get_providers()[0] != providers[0]:
        raise RuntimeError(f'ONNX GPU initialization failed: requested {providers[0]}, got {session.get_providers()}')
    session.disable_fallback()  # an EP failure must not silently restart the entire model on CPU
    return session
