"""Tests for the recorded backend auto-detect order: rocm -> zluda -> cuda/directml -> cpu."""

from __future__ import annotations

import pytest

import core.device_utils as du


class _FakeTorch:
    class _Dev:
        def __init__(self, s):
            self._s = s

        def __repr__(self):
            return self._s

    @staticmethod
    def device(*args):
        return _FakeTorch._Dev("cuda:0" if "cuda" in args[0] else "cpu")


def _patch_hw(monkeypatch, amd=False, nvidia=False):
    patches = {
        "_has_amd_gpu": lambda: amd,
        "_has_nvidia_gpu": lambda: nvidia,
    }
    for name, fn in patches.items():
        monkeypatch.setattr(du, name, fn)


def test_auto_select_prefers_native_rocm(monkeypatch):
    monkeypatch.setattr(du, "detect_rocm", lambda: True)
    monkeypatch.setattr(du, "detect_zluda", lambda: False)
    monkeypatch.setattr(du, "cuda_available", lambda: False)
    assert du.auto_select_device() == "rocm"


def test_auto_select_detects_zluda_on_amd_only(monkeypatch):
    # A CUDA build exposing an AMD-only dGPU == ZLUDA; beats plain cuda.
    monkeypatch.setattr(du, "detect_rocm", lambda: False)
    monkeypatch.setattr(du, "detect_zluda", lambda: True)
    monkeypatch.setattr(du, "cuda_available", lambda: True)
    assert du.auto_select_device() == "zluda"


def test_auto_select_falls_to_cuda_when_nvidia_present(monkeypatch):
    # NVIDIA present -> not ZLUDA, select cuda.
    monkeypatch.setattr(du, "detect_rocm", lambda: False)
    monkeypatch.setattr(du, "detect_zluda", lambda: False)
    monkeypatch.setattr(du, "cuda_available", lambda: True)
    assert du.auto_select_device() == "cuda"


def test_auto_select_falls_to_cpu(monkeypatch):
    monkeypatch.setattr(du, "detect_rocm", lambda: False)
    monkeypatch.setattr(du, "detect_zluda", lambda: False)
    monkeypatch.setattr(du, "cuda_available", lambda: False)
    assert du.auto_select_device() == "cpu"


def test_detect_zluda_requires_amd_and_no_nvidia(monkeypatch):
    monkeypatch.setattr(du, "cuda_available", lambda: True)
    _patch_hw(monkeypatch, amd=True, nvidia=False)
    assert du.detect_zluda() is True
    _patch_hw(monkeypatch, amd=True, nvidia=True)
    assert du.detect_zluda() is False
    _patch_hw(monkeypatch, amd=False, nvidia=False)
    assert du.detect_zluda() is False


def test_detect_zluda_false_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(du, "cuda_available", lambda: False)
    _patch_hw(monkeypatch, amd=True, nvidia=False)
    assert du.detect_zluda() is False


def test_default_backend_amd_rocm_cuda_cpu(monkeypatch):
    _patch_hw(monkeypatch, amd=True, nvidia=False)
    assert du.default_backend() == "rocm"
    _patch_hw(monkeypatch, amd=False, nvidia=True)
    assert du.default_backend() == "cuda"
    _patch_hw(monkeypatch, amd=False, nvidia=False)
    assert du.default_backend() == "cpu"


def test_resolve_zluda_rejected_when_not_detected(monkeypatch):
    monkeypatch.setattr(du, "_import_torch", lambda: _FakeTorch)
    monkeypatch.setattr(du, "detect_zluda", lambda: False)
    with pytest.raises(RuntimeError):
        du.resolve_pytorch_device("zluda")


def test_resolve_zluda_returns_cuda_when_detected(monkeypatch):
    monkeypatch.setattr(du, "_import_torch", lambda: _FakeTorch)
    monkeypatch.setattr(du, "detect_zluda", lambda: True)
    dev = du.resolve_pytorch_device("zluda")
    assert "cuda" in str(dev)


def test_resolve_auto_order_rocm_before_cuda(monkeypatch):
    monkeypatch.setattr(du, "_import_torch", lambda: _FakeTorch)
    monkeypatch.setattr(du, "detect_rocm", lambda: True)
    monkeypatch.setattr(du, "detect_zluda", lambda: False)
    monkeypatch.setattr(du, "cuda_available", lambda: True)
    assert "cuda" in str(du.resolve_pytorch_device("auto"))


def test_resolve_auto_order_cpu(monkeypatch):
    monkeypatch.setattr(du, "_import_torch", lambda: _FakeTorch)
    monkeypatch.setattr(du, "detect_rocm", lambda: False)
    monkeypatch.setattr(du, "detect_zluda", lambda: False)
    monkeypatch.setattr(du, "cuda_available", lambda: False)
    assert "cpu" in str(du.resolve_pytorch_device("auto"))


def test_ort_zluda_uses_cuda_ep_when_available(monkeypatch):
    class _FakeORT:
        @staticmethod
        def get_available_providers():
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    monkeypatch.setattr(du, "bootstrap_onnxruntime", lambda: _FakeORT)
    providers, note = du.ort_execution_providers("zluda")
    assert providers[0] == "CUDAExecutionProvider"
    assert "ZLUDA" in note


def test_ort_zluda_falls_back_to_cpu(monkeypatch):
    class _FakeORT:
        @staticmethod
        def get_available_providers():
            return ["CPUExecutionProvider"]

    monkeypatch.setattr(du, "bootstrap_onnxruntime", lambda: _FakeORT)
    providers, note = du.ort_execution_providers("zluda")
    assert providers == ["CPUExecutionProvider"]
    assert "ZLUDA" in note


# ---------------------------------------------------------------------------
# NV isolation: a rocm request must never land on a CUDA (NVIDIA) device
# ---------------------------------------------------------------------------


def test_rocm_request_never_falls_back_to_nvidia_cuda(monkeypatch):
    # No hip torch, no ZLUDA, plain CUDA visible -> hard refusal.
    monkeypatch.setattr(du, "_import_torch", lambda: _FakeTorch)
    monkeypatch.setattr(du, "detect_rocm", lambda: False)
    monkeypatch.setattr(du, "detect_zluda", lambda: False)
    monkeypatch.setattr(du, "cuda_available", lambda: True)
    with pytest.raises(RuntimeError, match="ROCm"):
        du.resolve_pytorch_device("rocm")


def test_rocm_request_uses_hip_device_when_native_rocm(monkeypatch):
    monkeypatch.setattr(du, "_import_torch", lambda: _FakeTorch)
    monkeypatch.setattr(du, "detect_rocm", lambda: True)
    monkeypatch.setattr(du, "detect_zluda", lambda: False)
    monkeypatch.setattr(du, "cuda_available", lambda: False)
    assert "cuda" in str(du.resolve_pytorch_device("rocm"))


def test_detect_zluda_env_marker_beats_nvidia_presence(monkeypatch):
    # ZLUDA_ACTIVE=1 (set by run_with_zluda.bat) wins even with an NVIDIA GPU
    # physically connected; without it the old heuristic applies.
    monkeypatch.setattr(du, "cuda_available", lambda: True)
    _patch_hw(monkeypatch, amd=True, nvidia=True)
    monkeypatch.delenv(du._ZLUDA_ENV_VAR, raising=False)
    assert du.detect_zluda() is False
    monkeypatch.setenv(du._ZLUDA_ENV_VAR, "1")
    assert du.detect_zluda() is True


def test_ort_rocm_never_uses_cuda_ep_without_zluda(monkeypatch):
    class _FakeORT:
        @staticmethod
        def get_available_providers():
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    monkeypatch.setattr(du, "bootstrap_onnxruntime", lambda: _FakeORT)
    monkeypatch.setattr(du, "detect_zluda", lambda: False)
    monkeypatch.setattr(du, "detect_rocm", lambda: True)
    providers, _note = du.ort_execution_providers("rocm")
    assert providers == ["CPUExecutionProvider"]


def test_ort_auto_skips_cuda_ep_under_native_rocm(monkeypatch):
    class _FakeORT:
        @staticmethod
        def get_available_providers():
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    monkeypatch.setattr(du, "bootstrap_onnxruntime", lambda: _FakeORT)
    monkeypatch.setattr(du, "detect_rocm", lambda: True)
    monkeypatch.setattr(du, "detect_zluda", lambda: False)
    providers, note = du.ort_execution_providers("auto")
    assert providers == ["CPUExecutionProvider"]
    assert "ROCm" in note


@pytest.mark.parametrize('choice', ['auto', 'rocm', 'directml'])
def test_amd_onnx_prefers_real_directml_provider_spelling(monkeypatch, choice):
    class ORT:
        @staticmethod
        def get_available_providers(): return ['CUDAExecutionProvider', 'DmlExecutionProvider', 'CPUExecutionProvider']
    monkeypatch.setattr(du, 'bootstrap_onnxruntime', lambda:ORT)
    monkeypatch.setattr(du, 'detect_rocm', lambda:True)
    monkeypatch.setattr(du, 'detect_zluda', lambda:False)
    assert du.ort_execution_providers(choice)[0] == ['DmlExecutionProvider', 'CPUExecutionProvider']


def test_directml_adapter_matches_requested_gpu_among_multiple_cards(monkeypatch):
    from core import directml_adapter as dml
    monkeypatch.setattr(dml, 'adapters', lambda:[dict(index=0,name='NVIDIA RTX',vendor=0x10de,vram=16,software=False), dict(index=1,name='AMD Radeon RX 7600M XT',vendor=0x1002,vram=8,software=False),dict(index=2,name='AMD 780M',vendor=0x1002,vram=1,software=False)])
    assert dml.select_adapter('AMD Radeon RX 7600M XT')['index'] == 1
    assert dml.select_adapter(vendor=0x1002)['index'] == 1
    with pytest.raises(RuntimeError, match='requested GPU'): dml.select_adapter('Unavailable GPU')


def test_rocm10_selects_target_discrete_gpu_after_integrated_gpu(monkeypatch):
    from types import SimpleNamespace
    names = ['AMD Radeon 780M Graphics', 'AMD Radeon RX 7600M XT']
    fake=SimpleNamespace(__version__='2.13.0+rocm10.0.0',cuda=SimpleNamespace(device_count=lambda:len(names),get_device_name=lambda i:names[i]))
    monkeypatch.setattr(du,'_import_torch',lambda:fake)
    monkeypatch.setattr(du.sys,'platform','win32')
    assert du.preferred_pytorch_gpu_index()==1
    names[:]=['AMD Radeon 780M Graphics']
    with pytest.raises(RuntimeError,match='RX 7600M XT'): du.preferred_pytorch_gpu_index()


def test_rocm10_binds_worker_current_device_to_target(monkeypatch):
    from types import SimpleNamespace
    bound=[]
    fake=SimpleNamespace(__version__='2.13.0+rocm10.0.0', device=lambda kind,index:(kind,index),cuda=SimpleNamespace(device_count=lambda:2,get_device_name=lambda i:['AMD Radeon 780M Graphics','AMD Radeon RX 7600M XT'][i],set_device=bound.append))
    monkeypatch.setattr(du,'_import_torch',lambda:fake)
    monkeypatch.setattr(du.sys,'platform','win32')
    monkeypatch.setattr(du,'detect_rocm',lambda:True)
    assert du.resolve_pytorch_device('rocm')==('cuda',1)
    assert bound==[1]
