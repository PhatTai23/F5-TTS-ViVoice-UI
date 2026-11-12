"""Download F5-TTS Vietnamese ViVoice checkpoint into model/ (cross-platform)."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = ROOT / "model" / "F5-TTS-Vietnamese-ViVoice"
REPO_ID = "hynt/F5-TTS-Vietnamese-ViVoice"
FILES = ("model_last.pt", "vocab.txt", "config.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ViVoice model into model/")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_DIR,
        help=f"Output directory (default: {DEFAULT_DIR})",
    )
    args = parser.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    for name in FILES:
        try:
            path = hf_hub_download(repo_id=REPO_ID, filename=name, local_dir=str(out))
            print(f"ok {name} -> {path}")
        except Exception as exc:  # noqa: BLE001
            print(f"skip {name}: {type(exc).__name__}: {exc}")

    vocab = out / "vocab.txt"
    cfg = out / "config.json"
    if cfg.is_file() and (not vocab.is_file() or vocab.stat().st_size < 100):
        shutil.copy(cfg, vocab)
        print("copied config.json -> vocab.txt")

    print("done:", sorted(p.name for p in out.iterdir() if p.is_file()))


if __name__ == "__main__":
    main()
