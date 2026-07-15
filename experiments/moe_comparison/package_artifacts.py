"""Create a checkpoint-free archive from real completed MoE run artifacts."""

from __future__ import annotations

import argparse
import hashlib
import tarfile
from pathlib import Path


INCLUDED_FILES = {
    "config.json",
    "environment.json",
    "history.jsonl",
    "split_metadata.json",
    "summary.json",
    "failure.json",
}


def package(results_dir: Path, output: Path) -> None:
    files = sorted(path for path in results_dir.rglob("*") if path.name in INCLUDED_FILES)
    if not files:
        raise ValueError(f"no real result artifacts found below {results_dir}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=path.relative_to(results_dir))
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(f"{digest}  {output.name}\n")
    print(f"Packaged {len(files)} files in {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    package(args.results_dir, args.output)


if __name__ == "__main__":
    main()
