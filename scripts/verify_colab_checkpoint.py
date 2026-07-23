#!/usr/bin/env python3
"""Verify the trusted release pickle with the pinned Colab compatibility stack."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import pickle
import sys
import zipfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


ROOT = Path(__file__).resolve().parents[1]
COMPAT_DIR = ROOT / "deploy" / "huggingface" / "compat"
sys.path.insert(0, str(COMPAT_DIR))

from legacy_model import AdditionTransformer, greedy_generate  # noqa: E402


REQUIRED_CASES = (
    (0, 0),
    (1, 9),
    (99, 1),
    (123, 456),
    (347, 928),
    (500, 500),
    (999, 1),
    (999, 999),
)
CORE_DISTRIBUTIONS = ("jax", "jaxlib", "flax", "optax", "numpy")


def path_name(path: tuple[object, ...]) -> str:
    parts = [
        str(getattr(entry, "key", getattr(entry, "idx", getattr(entry, "name", entry))))
        for entry in path
    ]
    if parts[-1] == "value":
        parts = parts[:-1]
    return ".".join(parts)


def prompts_for(pairs: tuple[tuple[int, int], ...]) -> np.ndarray:
    prompts = np.empty((len(pairs), 12), dtype=np.int32)
    for row, (a, b) in enumerate(pairs):
        prompts[row] = (
            a // 100,
            (a // 10) % 10,
            a % 10,
            10,
            11,
            10,
            b // 100,
            (b // 10) % 10,
            b % 10,
            10,
            12,
            10,
        )
    return prompts


def expected_internal(a: int, b: int) -> list[int]:
    total = a + b
    return [total % 10, total // 10 % 10, total // 100 % 10, total // 1000 % 10]


def load_release_checkpoint(release_zip: Path) -> dict[str, object]:
    with zipfile.ZipFile(release_zip) as archive:
        matches = [
            name
            for name in archive.namelist()
            if Path(name).name == "exact_10m_checkpoint.pkl"
        ]
        if len(matches) != 1:
            raise RuntimeError(
                "release ZIP must contain exactly one exact_10m_checkpoint.pkl"
            )
        # The caller must supply the trusted repository release. Pickle is unsafe
        # for untrusted input.
        return pickle.loads(archive.read(matches[0]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Restore and generate with the trusted v0.1.0 release pickle."
    )
    parser.add_argument(
        "release_zip",
        type=Path,
        help="trusted jax-addition-transformer-v0.1.0-t4-run.zip",
    )
    args = parser.parse_args()

    config = json.loads((ROOT / "configs" / "colab-runtime.json").read_text())
    expected_versions = {
        name: config["versions"][name] for name in CORE_DISTRIBUTIONS
    }
    resolved_versions = {
        name: importlib.metadata.version(name) for name in CORE_DISTRIBUTIONS
    }
    if resolved_versions != expected_versions:
        raise RuntimeError(
            f"wrong compatibility environment: {resolved_versions}; "
            f"expected {expected_versions}"
        )

    checkpoint = load_release_checkpoint(args.release_zip)
    required_keys = {"params", "step", "validation", "parameter_count"}
    if set(checkpoint) != required_keys:
        raise AssertionError(f"unexpected checkpoint keys: {sorted(checkpoint)}")
    if checkpoint["parameter_count"] != 10_000_000:
        raise AssertionError("checkpoint parameter_count is not 10,000,000")

    fresh_model = AdditionTransformer(nnx.Rngs(params=42))
    graphdef, fresh_state = nnx.split(fresh_model, nnx.Param)
    fresh_shapes = {
        path_name(path): np.asarray(leaf).shape
        for path, leaf in jax.tree_util.tree_flatten_with_path(fresh_state)[0]
    }
    checkpoint_shapes = {
        path_name(path): np.asarray(leaf).shape
        for path, leaf in jax.tree_util.tree_flatten_with_path(checkpoint["params"])[0]
    }
    if checkpoint_shapes != fresh_shapes:
        raise AssertionError("checkpoint tree does not match the exact notebook model")

    parameter_count = sum(
        int(np.asarray(leaf).size)
        for leaf in jax.tree.leaves(checkpoint["params"])
    )
    if parameter_count != 10_000_000:
        raise AssertionError(f"restored {parameter_count:,} parameters")

    restored_model = nnx.merge(
        graphdef,
        jax.tree.map(jnp.asarray, checkpoint["params"]),
    )
    generated, valid = jax.jit(greedy_generate)(
        restored_model,
        jnp.asarray(prompts_for(REQUIRED_CASES)),
    )
    generated = np.asarray(generated)
    if not np.asarray(valid).all():
        raise AssertionError("the restored model generated a non-digit token")

    results = []
    for (a, b), digits in zip(REQUIRED_CASES, generated.tolist(), strict=True):
        expected_digits = expected_internal(a, b)
        if digits != expected_digits:
            raise AssertionError(
                f"{a} + {b}: generated {digits}, expected {expected_digits}"
            )
        answer = "".join(map(str, digits))[::-1].lstrip("0") or "0"
        results.append(
            {
                "expression": f"{a} + {b}",
                "prediction": answer,
                "internal_digits": "".join(map(str, digits)),
            }
        )

    print(
        json.dumps(
            {
                "versions": resolved_versions,
                "backend": jax.default_backend(),
                "checkpoint_keys": sorted(checkpoint),
                "checkpoint_step": checkpoint["step"],
                "parameter_count": parameter_count,
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
