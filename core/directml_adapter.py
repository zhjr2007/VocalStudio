"""DXGI adapter identity for ONNX DirectML; never assume GPU 0 is the AMD GPU."""
import ctypes
import sys
import uuid


def adapters():
    if sys.platform != 'win32': return []
    class GUID(ctypes.Structure):
        _fields_ = [('data', ctypes.c_ubyte * 16)]
    class LUID(ctypes.Structure):
        _fields_ = [('low', ctypes.c_uint), ('high', ctypes.c_int)]
    class DESC(ctypes.Structure):
        _fields_ = [('name', ctypes.c_wchar * 128), ('vendor', ctypes.c_uint),
                    ('device', ctypes.c_uint), ('subsystem', ctypes.c_uint), ('revision', ctypes.c_uint),
                    ('video_memory', ctypes.c_size_t), ('system_memory', ctypes.c_size_t),
                    ('shared_memory', ctypes.c_size_t), ('luid', LUID), ('flags', ctypes.c_uint)]
    def call(pointer, index, result, args, *values):
        table = ctypes.cast(pointer, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        return ctypes.WINFUNCTYPE(result, ctypes.c_void_p, *args)(table[index])(pointer, *values)
    factory = ctypes.c_void_p()
    iid = GUID(); iid.data[:] = uuid.UUID('770aae78-f26f-4dba-a829-253c83d1b387').bytes_le
    create = ctypes.WinDLL('dxgi').CreateDXGIFactory1
    create.argtypes = [ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p)]
    create.restype = ctypes.c_long
    if create(ctypes.byref(iid), ctypes.byref(factory)) < 0: return []
    result = []
    try:
        for index in range(32):
            adapter = ctypes.c_void_p()
            hr = call(factory, 12, ctypes.c_long, [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)], index, ctypes.byref(adapter))
            if hr < 0: break
            try:
                desc = DESC()
                if call(adapter, 10, ctypes.c_long, [ctypes.POINTER(DESC)], ctypes.byref(desc)) >= 0:
                    result.append(dict(index=index, name=desc.name, vendor=desc.vendor, vram=desc.video_memory, software=bool(desc.flags & 2)))
            finally: call(adapter, 2, ctypes.c_ulong, [])
    finally: call(factory, 2, ctypes.c_ulong, [])
    return result


def select_adapter(preferred_name='', vendor=None):
    devices = [item for item in adapters() if not item['software']]
    if preferred_name:
        match = next((item for item in devices if item['name'].casefold() == preferred_name.casefold()), None)
        if match: return match
        raise RuntimeError(f'DirectML cannot find requested GPU: {preferred_name}')
    matches = [item for item in devices if vendor is None or item['vendor'] == vendor]
    if not matches: raise RuntimeError('No compatible DirectML GPU adapter found')
    return max(matches, key=lambda item:item['vram'])
