# Licensing & provenance

> 2026-09-06 确定，作为本项目代码与发布的核心原则。

## 目标
- `projects/` 下的 vendored 源项目（`Vocal2Midi`、`FunASR-V3`、
  `GPT-SoVITS-v2pro-20250604-nvidia50`，均为本地检出、不入本仓库）**仅作为两类用途**：
  1. **架构参照**：用于理解其输入→特征→推理→量化/对齐→输出的数据流与底层实现逻辑；
  2. **推理权重**：在运行时**只读加载**预训练权重（模型文件不属于本仓库代码）。
- **不照抄、不整体引用**其源码文件进 `engines/`。
- `engines/` 及本仓库其余代码 = 深度理解后的**自主重写/原创实现**。
- 本仓库以 **Apache-2.0** 许可发布（见根目录 `LICENSE`）。

## 当前审计（2026-09-06）
- `engines/` 大部分为自研重构；但存在对 vendored 源码的**路径引用**与**直接引用**：
  - **直接导入源码**：`engines/tts/_ref.py`、`engines/tts/voice_clone.py` 直接
    `import TTS_infer_pack.*`（GPT-SoVITS 推理链）。
  - **路径引用 vendored 目录**：`engines/separation/uvr5_engine.py`→`tools/uvr5`；
    `engines/vocal2midi/*`→`Vocal2Midi/...`；`engines/training/adapters.py`→GSV；
    `engines/tts/vocoder.py`→GSV。
- 上述属于"参照其实现/权重"，其中 TTS 的 `import TTS_infer_pack` 为**完全引用**，
  与"不照抄/不整体引用"目标不符 → 列为后续重构方向（见下）。

## 后续（对齐"多自主性"）
- 将 `engines/tts/` 的中文 G2P（G2PW）、T2S、SoVITS、HuBERT 前端的推理链改为
  **自研实现**，仅保留对权重的只读加载；移除对 `TTS_infer_pack` 的直接导入。
- 核对 vendored 各项目的真实许可证（Vocal2Midi=Apache-2.0，FunASR/GPT-SoVITS 见其仓库），
  以决定是否需要 NOTICE/署名；权重部分按各源项目许可保留署名。
- 本仓库以 Apache-2.0 发布；vendored 源码与权重按各源项目自身许可保留署名
  （Vocal2Midi = Apache-2.0；FunASR / GPT-SoVITS 见其仓库）。

## 目录布局说明（本地源码树）
- `projects/` 存放实际的 vendored 源码（`Vocal2Midi`、`FunASR-V3`、
  `GPT-SoVITS-v2pro-20250604-nvidia50`）。
- 项目根目录放置了**目录联接(junction)**（`Vocal2Midi`、`FunASR-V3`、`GPT-SoVITS-v2pro-20250604-nvidia50`
  指向 `projects\` 同名目录），使引擎按根相对路径仍能解析到 vendored 源码/权重。这些 junction 不入 git（被 .gitignore 排除），
  若被删除需重建。
- 引擎经这些路径导入 vendored 推理模块（`tools/uvr5` 的 `bs_roformer`/`lib.lib_v5`、`TTS_infer_pack` 等），
  属"运行时引用"，与"不照抄/完全引用"目标不符 → 见上方后续重写清单。
