import dataclasses
import json

import jax
import numpy as np
from flax import nnx

from jax_addition_transformer.checkpointing import restore_checkpoint
from jax_addition_transformer.config import (
    ExperimentConfig,
    ModelConfig,
    OptimizerConfig,
    TaskConfig,
    TrainingConfig,
)
from jax_addition_transformer.model import AdditionTransformer
from jax_addition_transformer.optimizers import make_optimizer
from jax_addition_transformer.training import train
from jax_addition_transformer.reporting import generate_report


def _resume_config():
    return ExperimentConfig(
        model=ModelConfig(
            n_layers=1,
            d_model=12,
            n_heads=2,
            n_kv_heads=2,
            d_ff=16,
            max_input_length=9,
            compute_dtype="float32",
        ),
        task=TaskConfig(max_digits=1),
        optimizer=OptimizerConfig(total_steps=10, warmup_steps=1, initial_learning_rate=1e-3),
        training=TrainingConfig(
            train_pairs=60,
            validation_pairs=20,
            test_pairs=20,
            batch_size=8,
            evaluation_batch_size=10,
            max_steps=10,
            validation_interval=5,
            logging_interval=1,
            checkpoint_interval=2,
        ),
    )


def _restore_params(run_dir, config):
    model = AdditionTransformer(config.model, rngs=nnx.Rngs(params=config.training.seed))
    _, params = nnx.split(model, nnx.Param)
    optimizer, _ = make_optimizer(config.optimizer, params)
    optimizer_state = optimizer.init(params)
    return restore_checkpoint(
        run_dir / "checkpoints" / "latest", params, optimizer_state, config.fingerprint
    )[0]


def test_interrupted_resume_matches_uninterrupted_training(tmp_path):
    config = _resume_config()
    uninterrupted = tmp_path / "uninterrupted"
    resumed = tmp_path / "resumed"
    train(config, uninterrupted, smoke_steps=3)
    train(config, resumed, smoke_steps=2)
    train(config, resumed, smoke_steps=3, resume=True)
    left, right = _restore_params(uninterrupted, config), _restore_params(resumed, config)
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        np.testing.assert_array_equal(a, b)
    history = [json.loads(line) for line in (resumed / "history.jsonl").read_text().splitlines()]
    assert [row["step"] for row in history] == [1, 2, 3]


def test_interval_validation_selects_best_checkpoint_and_report(tmp_path):
    config = _resume_config()
    config = dataclasses.replace(
        config,
        training=dataclasses.replace(
            config.training, max_steps=1, validation_interval=1, checkpoint_interval=1
        ),
    )
    run_dir = tmp_path / "validated"
    summary = train(config, run_dir)
    assert summary["best"]["step"] == 1
    assert (run_dir / "checkpoints" / "best" / "state").exists()
    history = json.loads((run_dir / "history.jsonl").read_text())
    assert "validation" in history
    report = generate_report(run_dir).read_text()
    assert "not evaluated" in report and "Best validation loss" in report
