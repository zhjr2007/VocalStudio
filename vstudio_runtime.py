"""Shared UI/portable-CLI runtime policy; core engine signatures stay unchanged."""
import os
import sys
from pathlib import Path

_configured = False
_dll_handles = []


def configure_runtime():
    global _configured
    if _configured: return
    from vstudio_path_compat import configure_unicode_model_paths
    configure_unicode_model_paths()
    root = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parent))
    codecs = root/'tools'/'ffmpeg'/'bin'
    if codecs.is_dir():
        os.environ['PATH'] = str(codecs)+os.pathsep+os.environ.get('PATH', '')
        if hasattr(os, 'add_dll_directory'): _dll_handles.append(os.add_dll_directory(str(codecs)))
    import torch
    if sys.platform == 'win32' and torch.version.hip and str(torch.__version__).startswith('2.13.0+rocm10.0.0'):
        from core.device_utils import preferred_pytorch_gpu_index
        torch.cuda.set_device(preferred_pytorch_gpu_index())
        # MIOpen BatchNorm JIT currently discovers host MSVC headers. Native
        # ATen GPU kernels avoid that external SDK dependency without CPU fallback.
        # The override is for diagnosing a fully provisioned developer machine.
        torch.backends.cudnn.enabled = os.environ.get('VOCALSTUDIO_MIOPEN') == '1'
    _configured = True


def runtime_policy():
    import torch
    if torch.version.hip:
        return 'PyTorch ROCm / '+('MIOpen' if torch.backends.cudnn.enabled else 'native GPU kernels')
    return 'PyTorch CUDA' if torch.version.cuda else 'PyTorch CPU'
