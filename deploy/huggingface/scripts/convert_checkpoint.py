"""Convert the trusted release pickle to deterministic inference-only Safetensors."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from safetensors.numpy import save_file

COMPAT_DIR = Path(__file__).resolve().parents[1] / "compat"
MODEL_REPO_DIR = Path(__file__).resolve().parents[1] / "model_repo"
sys.path.insert(0, str(COMPAT_DIR))
sys.path.insert(0, str(MODEL_REPO_DIR))

from legacy_model import AdditionTransformer as LegacyTransformer  # noqa: E402
from legacy_model import greedy_generate as legacy_generate  # noqa: E402
from model import greedy_generate, validate_weights  # noqa: E402

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


def path_name(path: tuple[object, ...]) -> str:
    parts = [
        str(getattr(entry, "key", getattr(entry, "idx", getattr(entry, "name", entry))))
        for entry in path
    ]
    if parts[-1] == "value":
        parts = parts[:-1]
    return ".".join(parts)


def prompts_for(pairs: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> np.ndarray:
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


def expected_internal(pairs: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> np.ndarray:
    return np.asarray(
        [[total % 10, total // 10 % 10, total // 100 % 10, total // 1000 % 10] for total in (a + b for a, b in pairs)],
        dtype=np.int32,
    )


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    checkpoint = pickle.loads(args.checkpoint.read_bytes())
    if checkpoint["parameter_count"] != 10_000_000:
        raise AssertionError("checkpoint does not record exactly 10,000,000 parameters")

    fresh_model = LegacyTransformer(nnx.Rngs(params=42))
    graphdef, fresh_state = nnx.split(fresh_model, nnx.Param)
    fresh_shapes = {
        path_name(path): np.asarray(leaf).shape
        for path, leaf in jax.tree_util.tree_flatten_with_path(fresh_state)[0]
    }
    arrays = {
        path_name(path): np.asarray(leaf, dtype=np.float32)
        for path, leaf in jax.tree_util.tree_flatten_with_path(checkpoint["params"])[0]
    }
    if {name: value.shape for name, value in arrays.items()} != fresh_shapes:
        raise AssertionError("checkpoint tree does not match a fresh exact notebook architecture")
    validate_weights(arrays)

    restored_model = nnx.merge(graphdef, jax.tree.map(jnp.asarray, checkpoint["params"]))
    prompts = jnp.asarray(prompts_for(REQUIRED_CASES))
    legacy_digits, legacy_valid = jax.jit(legacy_generate)(restored_model, prompts)
    legacy_digits = np.asarray(legacy_digits)
    if not np.asarray(legacy_valid).all():
        raise AssertionError("legacy model generated non-digit tokens")
    np.testing.assert_array_equal(legacy_digits, expected_internal(REQUIRED_CASES))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        dict(sorted(arrays.items())),
        str(args.output),
        metadata={
            "format": "jax-addition-transformer-v1",
            "parameter_count": "10000000",
            "checkpoint_step": str(checkpoint["step"]),
        },
    )

    from safetensors.numpy import load_file

    reloaded = load_file(str(args.output))
    validate_weights(reloaded)
    stable_digits, stable_valid = jax.jit(greedy_generate)(
        {name: jnp.asarray(value) for name, value in reloaded.items()},
        prompts,
    )
    stable_digits = np.asarray(stable_digits)
    if not np.asarray(stable_valid).all():
        raise AssertionError("stable model generated non-digit tokens")
    np.testing.assert_array_equal(stable_digits, legacy_digits)

    metadata = {
        "source_checkpoint_sha256": digest(args.checkpoint),
        "weights_sha256": digest(args.output),
        "parameter_count": sum(value.size for value in reloaded.values()),
        "tensor_count": len(reloaded),
        "checkpoint_step": checkpoint["step"],
        "validation": checkpoint["validation"],
        "required_case_internal_digits": stable_digits.tolist(),
    }
    metadata_path = args.output.with_name(args.output.name + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
