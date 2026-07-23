"""Assemble generated and shared files in the two upload directories."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEPLOY = ROOT / "deploy" / "huggingface"
MODEL_REPO = DEPLOY / "model_repo"
SPACE = DEPLOY / "space"
ARTIFACTS = ROOT / "artifacts" / "t4-run"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def main() -> None:
    weights = MODEL_REPO / "model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError("run convert_checkpoint.py before preparing uploads")

    for name in ("history.json", "results.json", "failures.csv"):
        copy(ARTIFACTS / name, MODEL_REPO / name)
    copy(ROOT / "LICENSE", MODEL_REPO / "LICENSE")
    copy(ROOT / "LICENSE", SPACE / "LICENSE")
    copy(MODEL_REPO / "model.py", SPACE / "model.py")
    copy(MODEL_REPO / "inference.py", SPACE / "inference.py")

    checksum_lines = []
    for name in (
        "model.safetensors",
        "model.safetensors.metadata.json",
        "config.json",
        "tokenizer_config.json",
        "generation_config.json",
        "history.json",
        "results.json",
        "failures.csv",
    ):
        path = MODEL_REPO / name
        checksum_lines.append(f"{sha256(path)}  {name}")
    (MODEL_REPO / "SHA256SUMS.txt").write_text("\n".join(checksum_lines) + "\n")

    print(f"Prepared model repository: {MODEL_REPO}")
    print(f"Prepared Space: {SPACE}")
    print(f"Model weights SHA-256: {sha256(weights)}")


if __name__ == "__main__":
    main()
