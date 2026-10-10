# VocalStudio ROCm 10 便携版

目标：Windows / AMD Radeon RX 7600M XT 8 GB / gfx1102。发行状态以 `build-manifest.json` 为准；`UNVERIFIED` 表示尚未完成真机验收，`VERIFIED_RX7600M_XT` 表示随包 `amd-acceptance.json` 的必需用例已通过。

双击 `VocalStudioBackend.exe` 启动新版界面。整体移动本目录，保留 `_internal` 中的模型和运行库。

支持的模型默认优先 GPU：PyTorch 使用 ROCm，MIDI ONNX 使用 DirectML，并匹配同一块 RX 7600M XT。显式 GPU 后端不可用时提示错误。不支持的 ONNX 算子及音频读写、重采样、规则分析等辅助工作使用 CPU。各步骤实际后端显示在结果与日志中。

页面“跟随全局”读取设置；页面显式选择优先。旧 CUDA / DirectML 偏好与当前 PyTorch 任务不兼容时会提示，请选择 Auto 或 ROCm。全局偏好修改只影响后续任务。

PowerShell 命令行示例：

```powershell
& .\VocalStudioBackend.exe info
& .\VocalStudioBackend.exe separate --audio 'D:\Music\song.wav' --device rocm --model Karaoke --output-dir 'D:\Out\vocals'
```

八个页面支持中英文、浅深主题和字号缩放。运行时可切换页面并统一取消；关闭窗口会协作取消并等待线程结束。

当前包仅提供训练扩展安装状态、版本、后端和任务支持检测，不提供 AMD 训练。跨显示器 DPI 和人工拖拽贴边仍待验收；单显示器边缘缩放、工作区最大化/还原、最小尺寸已验证。

- `amd-acceptance.json`：最终便携验收记录；`validation/`：日志与 UI / 生命周期证据。
- `build-manifest.json`：模型资源清单、可执行文件 SHA256 与发行状态。
- `release-notes.md`：环境、依赖来源、实际验证范围。
- `build-support/`：依赖锁定、构建脚本和本地原生 wheel 来源。
