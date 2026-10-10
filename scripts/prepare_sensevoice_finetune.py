"""Prepare a tiny SenseVoice fine-tune set (train/val .jsonl) using the local
SenseVoice model to transcribe a few short clips (wave G-4).

Usage::

    venv\\Scripts\\python scripts/prepare_sensevoice_finetune.py --out <dir>

Outputs ``<dir>/train.jsonl`` + ``<dir>/val.jsonl`` (FunASR SenseVoiceCTC
data layout: source / target / text_language / emo_target / event_target /
with_or_wo_itn).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import PROJECT_ROOT  # noqa: E402

CLIPS = (
    # (relative path, language tag expected in the target text)
    ("experiments/funasr/iic/SenseVoiceSmall/example/zh.mp3", "zh"),
    ("experiments/speaker/iic/speech_campplus_sv_zh-cn_16k-common/examples/speaker1_a_cn_16k.wav", "zh"),
    ("experiments/speaker/iic/speech_campplus_sv_zh-cn_16k-common/examples/speaker1_b_cn_16k.wav", "zh"),
    ("experiments/funasr/iic/SenseVoiceSmall/example/ja.mp3", "ja"),
)
VAL_CLIPS = (
    ("experiments/speaker/iic/speech_campplus_sv_zh-cn_16k-common/examples/speaker2_a_cn_16k.wav", "zh"),
)


def transcribe(model_dir: Path, path: Path) -> str:
    from funasr import AutoModel

    model = AutoModel(model=str(model_dir), device="cuda:0" if _cuda() else "cpu", disable_update=True)
    result = model.generate(input=str(path))
    return result[0]["text"] if result else ""


def _cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="prepare sensevoice finetune jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-asr", action="store_true", help="use placeholder text instead of ASR")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    from core.config import ModelPaths

    mp = ModelPaths.from_settings()
    sensevoice = Path(mp.sensevoice)
    if not sensevoice.is_absolute():
        sensevoice = PROJECT_ROOT / sensevoice

    def build(rel, lang, key_id: int) -> dict:
        path = PROJECT_ROOT / rel
        if not path.is_file():
            sys.stderr.write(f"skip missing {path}\n")
            return None
        target = f"clip {key_id}" if args.no_asr else transcribe(sensevoice, path)
        if "<|" in target:  # strip emo/lang/event tokens, keep the plain text
            target = " ".join(part for part in target.split() if "<|" not in part)
        return {
            "key": f"g_smoke_{key_id:02d}",
            "text_language": f"<|{lang}|>",
            "emo_target": "<|NEUTRAL|>",
            "event_target": "<|Speech|>",
            "with_or_wo_itn": "<|woitn|>",
            "target": target,
            "source": str(path.resolve()),
        }

    train_items, val_items = [], []
    for index, (rel, lang) in enumerate(CLIPS, start=1):
        item = build(rel, lang, index)
        if item:
            train_items.append(item)
    for index, (rel, lang) in enumerate(VAL_CLIPS, start=1):
        item = build(rel, lang, 90 + index)
        if item:
            val_items.append(item)

    for name, items in (("train.jsonl", train_items), ("val.jsonl", val_items)):
        path = out_dir / name
        path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items), encoding="utf-8")
        print(f"{name}: {len(items)} clips -> {path}")
    if not train_items:
        sys.stderr.write("no clips prepared\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
