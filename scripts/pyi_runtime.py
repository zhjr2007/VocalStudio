"""Frozen runtime: local codecs, retained DLL handles and usable CLI stdio."""
import os
import sys
from pathlib import Path
from vstudio_path_compat import configure_unicode_model_paths
configure_unicode_model_paths()

root = Path(sys._MEIPASS)
_dll_handles = []
for folder in (root/'tools'/'ffmpeg'/'bin', root/'_rocm_sdk_core'/'bin', root/'_rocm_sdk_libraries'/'bin', root/'torch'/'lib'):
    if folder.is_dir():
        os.environ['PATH'] = str(folder) + os.pathsep + os.environ.get('PATH', '')
        if hasattr(os, 'add_dll_directory'): _dll_handles.append(os.add_dll_directory(str(folder)))
os.environ.setdefault('NLTK_DATA', str(root/'nltk_data'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
# Windowed PyInstaller replaces stdout/stderr with None. Reconnect redirected
# standard handles (or the parent console for CLI calls), avoiding print crashes.
if sys.platform == 'win32':
    import ctypes
    import msvcrt
    ctypes.windll.kernel32.GetStdHandle.restype = ctypes.c_void_p
    inherited_stdout = ctypes.windll.kernel32.GetStdHandle(-11)
    if len(sys.argv) > 1 and not inherited_stdout: ctypes.windll.kernel32.AttachConsole(-1)
    for name, handle_id, flags in (('stdout', -11, os.O_WRONLY), ('stderr', -12, os.O_WRONLY), ('stdin', -10, os.O_RDONLY)):
        if getattr(sys, name) is not None: continue
        kernel = ctypes.windll.kernel32
        kernel.GetStdHandle.restype = ctypes.c_void_p
        handle = kernel.GetStdHandle(handle_id)
        if handle and handle != ctypes.c_void_p(-1).value:
            try:
                fd = msvcrt.open_osfhandle(handle, flags)
                setattr(sys, name, os.fdopen(fd, 'r' if name == 'stdin' else 'w', encoding='utf-8', errors='replace', buffering=1))
                continue
            except OSError: pass
        setattr(sys, name, open(os.devnull, 'r' if name == 'stdin' else 'w', encoding='utf-8'))
