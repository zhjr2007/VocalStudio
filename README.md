# VocalStudio

一站式人声处理工作台：**人声分离 → 内容识别（ASR / 歌词对齐）→ 人声转 MIDI → 声音克隆 TTS**，
在同一个应用里完成；提供 PyQt5 桌面界面与统一 CLI。

One-stop vocal processing workbench: **separation → recognition → MIDI → TTS** in a single
PyQt5 desktop app with a unified CLI.

## 功能 / Features

| 页面 | 引擎 | 说明 |
|---|---|---|
| 总览 Overview | 硬件与模型状态检测 | 后端可用性一览 |
| 人声分离 Separation | UVR5 BS-RoFormer + MDX（双轨 / 多轨） | 人声、伴奏、鼓、贝斯、钢琴等多轨拆分 |
| 人声转 MIDI | GAME（音高）+ RMVPE + 量化 + madmom 和弦/调性 | 支持歌词对齐（Qwen3-ASR + HubertFA） |
| ASR | SenseVoice / Paraformer + CAM++ 说话人分离 | 输出 TXT / SRT / JSON |
| 声音克隆 TTS | GPT-SoVITS v2Pro / v2ProPlus | 参考音频克隆，支持中 / 英 / 日 |
| 流水线 Pipeline | 端到端编排（歌声 / 语音两条分支） | 一次跑完多个阶段 |
| 训练 Training | FunASR SenseVoice 微调 + GSV s1 + CAM++ / GAME 适配器 | AMD 下仅提供扩展检测 |

界面：第二代界面 `vstudio_ui/`（`run_ui.bat` 启动，默认）；第一代 PyQt-Fluent 界面保留在
`gui/`（`run.bat`）。

## 硬件与后端 / Backends

后端自动识别顺序：**ROCm → ZLUDA → CUDA → DirectML → CPU**（实现见 `core/device_utils.py`）。
参考验证环境：Windows / AMD Radeon RX 7600M XT（ROCm 10 + DirectML）；NVIDIA 走 CUDA 12.8
（见 `install.bat`）。

## 快速开始 / Quick start

需要 **Python 3.13**（Windows）。

```bat
quickstart.bat          :: 一键：建环境 → 装依赖（默认 CPU 版 PyTorch）→ 下载可用模型 → 启动界面
quickstart.bat cuda     :: NVIDIA 显卡使用 CUDA 版 PyTorch
:: AMD 显卡（ROCm 10）：
powershell -ExecutionPolicy Bypass -File scripts\prepare_rocm10.ps1
```

手动方式：

```bat
install.bat             :: 建 venv + 安装 pinned 依赖（CUDA 12.8）
venv\Scripts\python scripts\download_models.py --only sensevoice campplus
run_ui.bat              :: 启动新界面
:: 或 CLI：
venv\Scripts\python main.py info
venv\Scripts\python main.py separate --audio materials\song.mp3 --model Karaoke
venv\Scripts\python main.py tts --ref-audio ref.wav --text "你好，VocalStudio。"
venv\Scripts\python main.py pipeline --steps separation,content_detection,vocal2midi
```

另见同目录的 [`快速开始.md`](快速开始.md)（分享给非开发用户的说明）。

## 模型权重说明 / Model weights

仓库**不包含**模型权重（体积原因）。`scripts\download_models.py` 只能自动下载 2 个 ModelScope
模型（`sensevoice`、`campplus`）；其余权重需自行获取并放入外部目录，保持相对布局
（路径表见 `settings/defaults.json`，全部相对 PROJECT_ROOT、可移植）：

- `GPT-SoVITS-v2pro-20250604-nvidia50/` — UVR5 分离权重、GSV TTS 预训练（来自 GPT-SoVITS 发行包）
- `Vocal2Midi/experiments/` — GAME 系列 ONNX、RMVPE
- `experiments/funasr|speaker/iic/…` — FunASR / CAM++ 权重（这两个可自动下载）

校验完整性：`venv\Scripts\python scripts\download_models.py --verify`（SHA256 清单）。

## 便携打包 / Portable build

```bat
build_portable.bat        :: 默认 ROCm 10（AMD）；build_portable.bat cuda 使用 CUDA 环境
```

产出 `dist\VocalStudioBackend\`（onedir，模型与运行库在 `_internal`），双击
`VocalStudioBackend.exe` 启动；AMD 发行验收流程见 `docs/amd-rocm10-release.md`。

## 测试 / Tests

```bat
venv\Scripts\python -m pytest tests\ -q
```

部分用例需要本地权重或测试素材（`materials/`），缺失时会自动跳过。

## 目录结构 / Layout

| 路径 | 内容 |
|---|---|
| `vstudio_ui/` | 第二代界面（默认；任务系统、动效、主题） |
| `gui/` | 第一代 PyQt-Fluent 界面（i18n zh/en） |
| `core/` | 配置 / 设备后端 / 编排器 / 异常 |
| `engines/` | 分离、vocal2midi（含和弦）、说话人、ASR、TTS、训练 |
| `scripts/` | 模型下载、打包、验收、快照等工具 |
| `tests/` | pytest 套件 |
| `docs/` | 许可与溯源、便携说明、发行记录 |

## 致谢 / Acknowledgements

本项目的架构灵感、模型权重与设计参考来自以下开源项目，特此致谢：

- [Vocal2Midi](https://github.com/Xiantaidu/Vocal2Midi) — 人声转 MIDI 的数据流与
  GAME / RMVPE 权重（Apache-2.0）
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) — 声音克隆 TTS 与 UVR5 分离权重
- [FunASR](https://github.com/modelscope/FunASR) — 语音识别（SenseVoice / Paraformer / CAM++）
- [Ultimate Vocal Remover (UVR5)](https://github.com/Anjok07/ultimatevocalremovergui) — 分离模型与思路
- 以及 PyQt-Fluent-Widgets、madmom、librosa、ModelScope、HuggingFace 等众多开源库与社区

`engines/` 是在理解上述项目实现后的**自主重写**；上游检出与权重仅用作架构参照与只读推理。
各来源的许可与署名详见 [docs/licensing.md](docs/licensing.md)。

## 许可 / License

[Apache License 2.0](LICENSE)。第三方组件许可见 [docs/licensing.md](docs/licensing.md)。

---

**开发方式**：本项目以 **vibe coding**（AI 辅助编程）方式开发——由 AI 编程代理协作编写与
验证，作者负责方向、验收与打磨。

*This project is developed in a **vibe coding** style: built and verified with AI coding
agents, steered and polished by the author.*
