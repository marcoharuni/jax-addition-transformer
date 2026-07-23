#!/usr/bin/env python3
"""Execute notebook 01's real one-step smoke path in an isolated CPU environment."""

from __future__ import annotations

import csv
import json
import math
import pickle
import re
import tempfile
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import optax
from flax import nnx


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "01_build_train_exact_10m.ipynb"

CELL_MARKERS = (
    "SEED = 42",
    'TOKENS = "0123456789 +="',
    "def pair_ids_to_operands(pair_ids):",
    "class HybridBatcher:",
    "def normal_parameter(rngs, shape, scale):",
    "class AdditionTransformer(nnx.Module):",
    "ANSWER_MASK = jnp.arange(MODEL_INPUT_LENGTH)",
    "def greedy_generate(current_model, prompts):",
    "SMOKE_BATCH_SIZE = 8",
)


def main() -> None:
    config = json.loads((ROOT / "configs" / "colab-runtime.json").read_text())
    expected = {
        name: config["versions"][name]
        for name in ("jax", "jaxlib", "flax", "optax", "numpy", "matplotlib")
    }
    import importlib.metadata

    resolved = {
        name: importlib.metadata.version(name)
        for name in expected
    }
    if resolved != expected:
        raise RuntimeError(f"wrong compatibility environment: {resolved}")
    if jax.default_backend() != "cpu":
        raise RuntimeError("this local verifier intentionally requires the CPU backend")

    notebook = json.loads(NOTEBOOK_PATH.read_text())
    code_sources = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    namespace = {
        "csv": csv,
        "json": json,
        "math": math,
        "pickle": pickle,
        "re": re,
        "time": time,
        "Path": Path,
        "jax": jax,
        "jnp": jnp,
        "np": np,
        "optax": optax,
        "nnx": nnx,
    }
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    namespace["plt"] = plt

    with tempfile.TemporaryDirectory(prefix="jax-addition-colab-smoke-") as run_dir:
        for marker in CELL_MARKERS:
            matches = [source for source in code_sources if marker in source]
            if len(matches) != 1:
                raise RuntimeError(
                    f"expected one notebook cell containing {marker!r}, "
                    f"found {len(matches)}"
                )
            source = matches[0].replace(
                'RUN_DIR = Path("/content/jax_addition_run")',
                f"RUN_DIR = Path({run_dir!r})",
            )
            exec(compile(source, f"{NOTEBOOK_PATH.name}:{marker}", "exec"), namespace)

    print(
        json.dumps(
            {
                "versions": resolved,
                "backend": jax.default_backend(),
                "parameter_count": namespace["parameter_count"],
                "smoke_batch_size": namespace["SMOKE_BATCH_SIZE"],
                "smoke_loss": float(namespace["_smoke_metrics"]["loss"]),
                "smoke_evaluation_loss": float(namespace["_smoke_eval_loss"]),
                "smoke_finite": bool(namespace["_smoke_metrics"]["finite"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
