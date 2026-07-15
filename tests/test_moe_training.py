import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from jax_addition_transformer.config import ModelConfig, OptimizerConfig
from jax_addition_transformer.data import tokenize_pairs
from jax_addition_transformer.model import AdditionTransformer
from jax_addition_transformer.optimizers import make_optimizer
from jax_addition_transformer.training import create_train_step


def tiny_moe_model():
    dense = ModelConfig(
        n_layers=1,
        d_model=12,
        n_heads=2,
        n_kv_heads=2,
        d_ff=24,
        compute_dtype="float32",
    )
    config = ModelConfig.matched_moe(
        dense, load_balance_loss_coefficient=0.01, router_z_loss_coefficient=0.001
    )
    return AdditionTransformer(config, rngs=nnx.Rngs(params=11))


def test_one_moe_optimizer_step_reports_separate_losses():
    model = tiny_moe_model()
    graph, params = nnx.split(model, nnx.Param)
    optimizer, _ = make_optimizer(
        OptimizerConfig(total_steps=4, warmup_steps=1, initial_learning_rate=1e-3), params
    )
    state = optimizer.init(params)
    full = tokenize_pairs(np.arange(8))
    step = create_train_step(graph, optimizer)
    updated, _, metrics = step(
        params, state, jnp.asarray(full[:, :-1]), jnp.asarray(full[:, 1:])
    )
    assert bool(metrics["finite"])
    np.testing.assert_allclose(
        metrics["total_optimized_loss"],
        metrics["language_model_loss"]
        + metrics["weighted_load_balancing_loss"]
        + metrics["weighted_router_z_loss"],
    )
    assert metrics["routing_metrics"]
    assert any(
        not np.array_equal(left, right)
        for left, right in zip(jax.tree.leaves(params), jax.tree.leaves(updated), strict=True)
    )


def test_tiny_deterministic_moe_loss_decreases():
    model = tiny_moe_model()
    graph, params = nnx.split(model, nnx.Param)
    optimizer, _ = make_optimizer(
        OptimizerConfig(
            total_steps=30,
            warmup_steps=1,
            initial_learning_rate=0.01,
            peak_learning_rate=0.01,
            final_learning_rate=0.005,
            weight_decay=0.0,
        ),
        params,
    )
    state = optimizer.init(params)
    step = create_train_step(graph, optimizer)
    full = np.repeat(tokenize_pairs(np.array([42])), 16, axis=0)
    inputs, targets = jnp.asarray(full[:, :-1]), jnp.asarray(full[:, 1:])
    losses = []
    for _ in range(30):
        params, state, metrics = step(params, state, inputs, targets)
        losses.append(float(metrics["language_model_loss"]))
    assert losses[-1] < losses[0] * 0.5
