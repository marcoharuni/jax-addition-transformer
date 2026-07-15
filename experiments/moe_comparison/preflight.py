"""Mandatory hardware and full MoE optimizer-step preflight."""

from __future__ import annotations

import argparse
import platform
from pathlib import Path

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from jax_addition_transformer.config import ExperimentConfig
from jax_addition_transformer.data import tokenize_pairs
from jax_addition_transformer.model import (
    AdditionTransformer,
    active_parameter_proxy,
    assert_parameter_count,
)
from jax_addition_transformer.optimizers import make_optimizer
from jax_addition_transformer.training import create_train_step, tree_all_finite


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    ROOT
    / "experiments/moe_comparison/configs/final/moe_10m_s0175_seed42.json"
)
EXPECTED_TOTAL = 17_942_400
EXPECTED_ACTIVE = 10_006_400


def run_preflight(require_gpu: bool = True) -> dict[str, object]:
    """Construct the exact MoE and execute one finite optimizer update."""

    devices = jax.devices()
    gpu_devices = [device for device in devices if device.platform == "gpu"]
    versions = {
        "python": platform.python_version(),
        "jax": jax.__version__,
        "flax": flax.__version__,
        "optax": optax.__version__,
        "devices": [str(device) for device in devices],
        "gpu": [getattr(device, "device_kind", str(device)) for device in gpu_devices],
    }
    print("Preflight versions:", versions, flush=True)

    if require_gpu and not gpu_devices:
        raise RuntimeError("GPU preflight failed: JAX did not detect a GPU device.")

    experiment = ExperimentConfig.load(CONFIG_PATH)
    model = AdditionTransformer(
        experiment.model,
        rngs=nnx.Rngs(params=experiment.training.seed),
    )
    total = assert_parameter_count(model, EXPECTED_TOTAL)
    active = active_parameter_proxy(model)
    if active != EXPECTED_ACTIVE:
        raise AssertionError(f"expected active proxy {EXPECTED_ACTIVE:,}, found {active:,}")

    graphdef, params = nnx.split(model, nnx.Param)
    optimizer, _ = make_optimizer(experiment.optimizer, params)
    optimizer_state = optimizer.init(params)
    batch_size = experiment.training.batch_size if require_gpu else 1
    tokens = tokenize_pairs(np.arange(batch_size, dtype=np.int64))
    inputs = jnp.asarray(tokens[:, :-1])
    targets = jnp.asarray(tokens[:, 1:])
    logits = nnx.merge(graphdef, params)(inputs)
    if not bool(jnp.all(jnp.isfinite(logits))):
        raise FloatingPointError("GPU preflight produced non-finite forward outputs")

    updated, _, metrics = create_train_step(graphdef, optimizer)(
        params,
        optimizer_state,
        inputs,
        targets,
    )
    jax.block_until_ready(metrics["loss"])
    if not bool(metrics["finite"]):
        raise FloatingPointError("GPU preflight produced non-finite loss or gradients")
    if not bool(tree_all_finite(updated)):
        raise FloatingPointError("GPU preflight produced non-finite updated parameters")

    result = {
        **versions,
        "total_parameters": total,
        "active_parameter_proxy": active,
        "loss": float(metrics["loss"]),
        "gradient_global_norm": float(metrics["gradient_global_norm"]),
        "batch_size": batch_size,
    }
    label = "GPU PREFLIGHT PASSED" if require_gpu else "PREFLIGHT PASSED (CPU allowed)"
    print(f"{label}:", result, flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    run_preflight(require_gpu=not args.allow_cpu)


if __name__ == "__main__":
    main()
