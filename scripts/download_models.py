"""Download or verify VocalStudio model weights.

Usage:
    venv\\Scripts\\python scripts\\download_models.py            # download all pending
    venv\\Scripts\\python scripts\\download_models.py --only sensevoice campplus
    venv\\Scripts\\python scripts\\download_models.py --verify   # hash-check every manifest entry

``--verify`` checks the SHA256 of the manifest files below (exit 1 for any
mismatch/missing file); use it after copying a portable install or hardware
refresh.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS: dict[str, tuple[str, str]] = {
    # key: (modelscope repo id, target dir relative to project root)
    "sensevoice": ("iic/SenseVoiceSmall", "experiments/funasr/iic/SenseVoiceSmall"),
    "campplus": (
        "iic/speech_campplus_sv_zh-cn_16k-common",
        "experiments/speaker/iic/speech_campplus_sv_zh-cn_16k-common",
    ),
}

#: SHA256 manifest recorded 2026-09-06 from this checkout (see docs/session-log).
MANIFEST: dict[str, str] = {
    "experiments/funasr/iic/SenseVoiceSmall/model.pt":
        "833ca2dcfdf8ec91bd4f31cfac36d6124e0c459074d5e909aec9cabe6204a3ea",
    "experiments/speaker/iic/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin":
        "3388cf5fd3493c9ac9c69851d8e7a8badcfb4f3dc631020c4961371646d5ada8",
    "GPT-SoVITS-v2pro-20250604-nvidia50/tools/uvr5/uvr5_weights/HP5_only_main_vocal.pth":
        "5908891829634926119720241e8573d97cbeb8277110a7512bdb0bd7563258ee",
    "GPT-SoVITS-v2pro-20250604-nvidia50/tools/uvr5/uvr5_weights/HP2_all_vocals.pth":
        "39796caa5db18d7f9382d8ac997ac967bfd85f7761014bb807d2543cc844ef05",
    "GPT-SoVITS-v2pro-20250604-nvidia50/tools/uvr5/uvr5_weights/model_bs_roformer_ep_317_sdr_12.9755.ckpt":
        "414b7ea8e3f57485282886f0bae273a898e02b9ca14524cdefa0fb9f5784e4d1",
    "GPT-SoVITS-v2pro-20250604-nvidia50/tools/uvr5/uvr5_weights/drums_bs_roformer.ckpt":
        "a2e825a03bc908cb04dbd88eddeefbf5147dd1cf1f95cebf453d9dbfabec494b",
    "GPT-SoVITS-v2pro-20250604-nvidia50/tools/uvr5/uvr5_weights/bass_bs_roformer.ckpt":
        "f6c94864adfb73bbb0ca58ec14d58dd0b364549e9fb61433ae51916f3e2f8d0b",
    "GPT-SoVITS-v2pro-20250604-nvidia50/tools/uvr5/uvr5_weights/piano_bs_roformer.ckpt":
        "5b84f37e8d444c8cb30c79d77f613a41c05868ff9c9ac6c7049c00aefae115aa",
    "Vocal2Midi/experiments/GAME-1.0.3-medium-onnx/encoder.onnx":
        "fb00295abf8156806a3bf2477ccadc9dd6a8f464fa4ee1a65c85100ce17e36fe",
    "Vocal2Midi/experiments/GAME-1.0.3-medium-onnx/segmenter.onnx":
        "49c20f890d02f8877be4b092fae3913ec969e9236cf0a91e661e8bf52465ffda",
    "Vocal2Midi/experiments/GAME-1.0.3-medium-onnx/estimator.onnx":
        "2662a01e24548dc3b9f8ebc081b26947873c01eeaa573c9d0a87e7bf2a43642b",
    "Vocal2Midi/experiments/GAME-1.0.3-medium-onnx/dur2bd.onnx":
        "49d10d1ea186e43c51fd2c51cedb01a3852d363f1c7056ae77df50f7e8dd0996",
    "Vocal2Midi/experiments/GAME-1.0.3-medium-onnx/bd2dur.onnx":
        "6672d56b00f27a20e9bfaa92801c15e394bb520e49031d3ed2515b0a6cf27402",
    "Vocal2Midi/experiments/RMVPE/rmvpe.onnx":
        "5370e71ac80af8b4b7c793d27efd51fd8bf962de3a7ede0766dac0befa3660fd",
    "GPT-SoVITS-v2pro-20250604-nvidia50/GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt":
        "732f94e63b148066e24c7f9d2637f3374083e637635f07fbdb695dee20ddbe1f",
    "GPT-SoVITS-v2pro-20250604-nvidia50/GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth":
        "0f8ead815234365edf045c6d86370ed6e4f440e8195be77ff0ea72684ad406a5",
    "GPT-SoVITS-v2pro-20250604-nvidia50/GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth":
        "d42a22bbbf65fb2bbdd45ad6a66841156977db45c7aabe0a6992ff378d9c7d3b",
    "GPT-SoVITS-v2pro-20250604-nvidia50/GPT_SoVITS/pretrained_models/chinese-hubert-base/pytorch_model.bin":
        "24164f129c66499d1346e2aa55f183250c223161ec2770c0da3d3b08cf432d3c",
    "GPT-SoVITS-v2pro-20250604-nvidia50/GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large/pytorch_model.bin":
        "e53a693acc59ace251d143d068096ae0d7b79e4b1b503fa84c9dcf576448c1d8",
    "GPT-SoVITS-v2pro-20250604-nvidia50/GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt":
        "4f5a0bf73c61eb41b174e1bb54e7ee3c83233892be8e0af1f187024e8e581a35",
    "GPT-SoVITS-v2pro-20250604-nvidia50/GPT_SoVITS/pretrained_models/fast_langdetect/lid.176.bin":
        "7e69ec5451bc261cc7844e49e4792a85d7f09c06789ec800fc4a44aec362764e",
    "GPT-SoVITS-v2pro-20250604-nvidia50/GPT_SoVITS/text/G2PWModel/g2pW.onnx":
        "2eb3c71fd95117b2e1abef8d2d0cd78aae894bbe7f0fac105ddc9c32ce63cbd0",
}


def human_size(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024 or unit == "GB":
            return f"{num:.1f}{unit}"
        num /= 1024
    return f"{num:.1f}GB"


def dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def verify_all() -> int:
    failures = 0
    for rel, expected in MANIFEST.items():
        path = PROJECT_ROOT / rel
        if not path.is_file():
            print(f"[MISS] {rel}")
            failures += 1
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        ok = digest.hexdigest().lower() == expected
        print(f"[{ 'ok' if ok else 'BAD'}] {rel} ({human_size(path.stat().st_size)})")
        if not ok:
            failures += 1
    if failures:
        print(f"verify: {failures} problem(s)")
    else:
        print("verify: all manifest entries ok")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", choices=list(MODELS), help="subset to download")
    parser.add_argument("--verify", action="store_true", help="SHA256-check manifest entries only")
    args = parser.parse_args()

    if args.verify:
        return verify_all()

    from modelscope.hub.snapshot_download import snapshot_download

    selected = args.only or list(MODELS)
    failures = 0
    for key in selected:
        repo_id, rel_dir = MODELS[key]
        target = PROJECT_ROOT / rel_dir
        if target.exists() and (target / "config.yaml").exists() and dir_size(target) > 0:
            print(f"[skip] {key}: already exists at {target} ({human_size(dir_size(target))})")
            continue
        print(f"[get ] {key}: {repo_id} -> {target}", flush=True)
        try:
            snapshot_download(repo_id, local_dir=str(target))
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {key}: {exc}")
            failures += 1
            continue
        print(f"[ ok ] {key}: {human_size(dir_size(target))} at {target}")
    if not failures:
        print("verify: use --verify to hash-check everything")
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
