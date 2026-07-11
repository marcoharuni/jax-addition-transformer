"""Orbax PyTree checkpoints and reproducibility metadata."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, UTC
from pathlib import Path

import jax
import numpy as np
import orbax.checkpoint as ocp


def capture_environment(config) -> dict:
    def version(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "operating_system": platform.platform(),
        "jax": jax.__version__,
        "jaxlib": version("jaxlib"),
        "flax": version("flax"),
        "optax": version("optax"),
        "orbax": version("orbax-checkpoint"),
        "numpy": np.__version__,
        "devices": [str(d) for d in jax.devices()],
        "device_count": jax.device_count(),
        "backend": jax.default_backend(),
        "compute_dtype": config.model.compute_dtype,
        "git_commit": commit,
        "configuration_hash": config.fingerprint,
    }


def save_checkpoint(path: str | Path, params, optimizer_state, metadata: dict) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    with ocp.Checkpointer(ocp.StandardCheckpointHandler(use_ocdbt=False)) as checkpointer:
        checkpointer.save(
            path / "state",
            args=ocp.args.StandardSave({"params": params, "optimizer_state": optimizer_state}),
            force=True,
        )
    (path / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def restore_checkpoint(
    path: str | Path, params, optimizer_state, expected_fingerprint: str | None = None
):
    path = Path(path)
    metadata = json.loads((path / "metadata.json").read_text())
    if expected_fingerprint is not None and metadata["config_fingerprint"] != expected_fingerprint:
        raise ValueError("checkpoint configuration fingerprint is incompatible")
    with ocp.Checkpointer(ocp.StandardCheckpointHandler(use_ocdbt=False)) as checkpointer:
        restored = checkpointer.restore(
            path / "state",
            args=ocp.args.StandardRestore({"params": params, "optimizer_state": optimizer_state}),
        )
    return restored["params"], restored["optimizer_state"], metadata
