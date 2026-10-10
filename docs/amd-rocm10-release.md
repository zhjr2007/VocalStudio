# VocalStudio AMD ROCm 10 发行与验收记录

日期：2026-10-10。目标平台：Windows / AMD Radeon RX 7600M XT 8 GB（gfx1102）。

当前状态：**已通过目标显卡清洁环境真机推理与迁移验收（`VERIFIED_RX7600M_XT`）**。
便携包 8 项必需用例全部通过，不依赖开发环境显卡掩码。训练扩展仅提供检测接口，不交付 AMD 训练适配。

## 环境与 GPU 策略

独立环境 `venv-rocm10`（由 `scripts/prepare_rocm10.ps1` 创建），不替换既有 CUDA 环境。

| 组件 | 固定版本 |
| --- | --- |
| Python | 3.13 |
| ROCm 产品 | 10.0.0 |
| PyTorch | 2.13.0+rocm10.0.0 |
| torchaudio | 2.11.0.2+rocm10.0.0 |
| ONNX Runtime | onnxruntime-directml 1.24.4 |
| GPU | AMD Radeon RX 7600M XT，gfx1102 |

策略：所有支持的模型优先使用 GPU —— PyTorch 模型走 ROCm；ONNX MIDI 走 DirectML，并通过 DXGI
精确匹配目标独显（不能默认 adapter 0，也不能用"CUDA EP 存在"推断 AMD 可用）。

同时存在核显与独显的机器上，PyTorch 默认枚举顺序可能与目标不符；固定发行按显卡名称选择
目标独显，并在执行线程绑定当前 GPU，不依赖外部 `HIP_VISIBLE_DEVICES` 掩码。

显式 GPU 选择缺少对应后端时在运行前报错；ONNX GPU session 初始化失败时拒绝整模型静默回退
CPU，不支持的算子可由 CPU 执行。音频读写、重采样、分段与规则分析使用 CPU；madmom
和弦/调性/节拍分析为 NumPy CPU 实现。结果面板与日志显示各步骤实际后端。

Windows MIOpen BatchNorm JIT 依赖宿主 MSVC/UCRT 头文件，缺 SDK 时会失败；本包固定版本默认
关闭 MIOpen 路径，使用 ATen 原生 ROCm GPU kernels（不是 CPU 回退），策略见
`vstudio_runtime.py`，UI 与便携 CLI 共用。开发诊断可设置 `VOCALSTUDIO_MIOPEN=1`。

## 源码环境真实推理（真机）

用 6 秒歌声、约 10.46 秒测试语音、6 秒参考语音执行，素材未上传到外部服务。

| 用例 | 结果 |
| --- | --- |
| 张量分配、运算 | 通过（ROCm） |
| 人声分离 HP5 | 通过（ROCm native GPU，两个 WAV） |
| MIDI | 通过（8.98 s） |
| SenseVoice ASR | 通过（ROCm，完整文本及 TXT / SRT / JSON） |
| CAM++ / VAD 说话人识别 | 通过 |
| GPT-SoVITS v2Pro TTS | 通过 |
| FRCRN 去噪 | 通过（16 kHz WAV） |
| UI 歌声流水线 | 通过（分离 → 内容检测 → MIDI / ASR） |
| UI 语音流水线 | 通过（说话人识别 → 内容检测 → ASR → TTS） |

真实输出只证明程序与推理路径可运行，不代表识别准确率或合成音质已经做数据集评测；
其他模型与参数组合未逐一验证。

## 依赖与资产

- 入口：无参数启动新 UI；子命令由 `main.py` CLI 处理。单目录包包含 22 项模型配置对应的权重、
  词典、FFmpeg shared、NLTK 资源、ROCm / gfx1102 kpack / DLL 与分发元数据。
- 去噪权重（FRCRN）：官方 ModelScope 端点不稳定时使用公开镜像，锁定 revision 与 SHA256，
  详见 `docs/denoise-model-provenance.json`；只读取配置并以 `weights_only=True` 加载 state
  dictionary，不执行镜像仓库代码。
- 三个 Python 3.13 原生依赖（pyopenjtalk、madmom、jieba_fast）由本地 wheel 重打包，
  流程与来源见 `scripts/repack_native_wheels.py`。
- 可重建文件：
  - `requirements-rocm10-torch.txt`：AMD 官方 wheel 索引及固定版本；
  - `requirements-rocm10-inference.txt` / `constraints-rocm10.txt` / `requirements-rocm10.lock.txt`：推理依赖与锁定；
  - `scripts/prepare_rocm10.ps1`：创建独立环境，不改动其他虚拟环境；
  - `VocalStudioBackend.spec` / `scripts/pyi_runtime.py`：完整依赖与运行库收集，窗口程序 CLI 标准输入输出修复；
  - `scripts/assemble_portable_assets.py`：单一模型清单，拒绝空模型目录，复制资产并生成 SHA256 清单；
  - `build_portable.bat`：默认 AMD ROCm 10；`cuda` 参数切换 CUDA 环境；
  - `scripts/validate_amd_release.py`：清除开发环境变量与 PATH，按顺序执行真实验收。
- 模型及需保留物理源码的资源清单共 3,739 个文件、17,452,843,881 bytes，另计 Python / DLL /
  ROCm 运行时；最终数量以随包 `build-manifest.json` 为准。

## 自动与界面验证

- 含动效完整回归：**353 passed，1 skipped**（394.00 s）。
- 无显卡掩码的真实 UI 生命周期 3 项通过：不兼容 GPU 偏好只发一次 failed；AMD MIDI 取消与
  运行中关闭均只发一次 cancelled done（约 2.59 s / 2.37 s），线程释放后才关闭窗口。
- 界面矩阵：8 页 × 浅/深 × 中/英 × 100%/150%/200%（96 张截图）通过；1440×900 基线按屏幕
  可用高度调整。
- 窗口实测：原生 WS_THICKFRAME、边缘拖拽缩放、双击最大化/还原、工作区最大化、最小尺寸通过。
- Windows 多媒体后端修复：试听组件改为点击播放时才创建 QMediaPlayer 并加载媒体；显示或
  选择输入/结果不会打开音频硬件。
- 新增 PyQt5 动效：320 ms 切页淡入位移、300 ms 导航回弹、150 ms 悬停/聚焦渐变、380 ms 点击
  涟漪、240 ms 参数/日志展开；动画可打断可反转；`VOCALSTUDIO_REDUCED_MOTION=1` 可关闭。
- 跨显示器 DPI 与人工拖拽贴边尚未实测。

## 便携包迁移验证

清空 Python / Conda / venv、CUDA / ROCm / ROCR / HIP / HSA 环境变量与 PATH（仅保留随包 FFmpeg
与系统目录），将整个目录迁移至包外工作目录后执行全部用例：

| 最终便携用例 | 结果 | 用时 |
| --- | --- | --- |
| UI 开窗 | 通过 | 15.01 s |
| CLI / 硬件 / 22 项模型 | 通过 | 4.54 s |
| 人声分离 | 通过 | 11.52 s |
| MIDI | 通过 | 8.98 s |
| ASR + 说话人 | 通过 | 58.6 s |
| TTS | 通过 | 34.93 s |
| 歌声流水线 | 通过 | 42.2 s |
| 语音流水线 | 通过 | 36.16 s |

MIDI 日志精确标识 RX 7600M XT / DXGI；ONNX profile 记录 10,915 次 DirectML GPU node events。
PyTorch 自动选择同一块独显，未使用外部 HIP 掩码。

发行闸门要求：全部必需用例通过、实际 DirectML GPU 事件存在、清洁环境策略匹配，且源目录与
迁移目录的二进制 SHA256 和资产清单一致。最终可执行文件 SHA256：

```
5b5e4fc95ef0dffa4818810062cd148af9924f453188a89bb18cc3bb9c366732
```

交付目录 `dist/VocalStudioBackend/`：整体保留或移动本目录，双击 `VocalStudioBackend.exe`
启动新界面；带子命令时使用 CLI。整个便携目录约 20.9 GiB（含模型与运行库）。

验证范围限定为上述硬件与已列出的模型 / 用例；AMD 训练按方案只提供扩展检测。
