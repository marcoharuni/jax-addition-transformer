"""Inspect the trusted release pickle in the isolated compatibility environment."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import jax
import numpy as np


def path_name(path: tuple[object, ...]) -> str:
    parts = [
        str(getattr(entry, "key", getattr(entry, "idx", getattr(entry, "name", entry))))
        for entry in path
    ]
    return ".".join(parts[:-1] if parts[-1] == "value" else parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()

    checkpoint = pickle.loads(args.checkpoint.read_bytes())
    print(f"top-level type: {type(checkpoint).__module__}.{type(checkpoint).__qualname__}")
    print(f"top-level keys: {sorted(checkpoint)}")
    print(f"step: {checkpoint['step']}")
    print(f"validation: {checkpoint['validation']}")
    print(f"recorded parameter count: {checkpoint['parameter_count']:,}")
    print(f"params type: {type(checkpoint['params']).__module__}.{type(checkpoint['params']).__qualname__}")

    total = 0
    for path, leaf in jax.tree_util.tree_flatten_with_path(checkpoint["params"])[0]:
        array = np.asarray(leaf)
        total += array.size
        print(f"{path_name(path)}\t{array.shape}\t{array.dtype}\t{array.size}")
    print(f"leaf count: {len(jax.tree.leaves(checkpoint['params']))}")
    print(f"calculated parameter count: {total:,}")
    if total != checkpoint["parameter_count"] or total != 10_000_000:
        raise AssertionError("release parameter count mismatch")


if __name__ == "__main__":
    main()
