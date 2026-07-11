import jax, jax.numpy as jnp, numpy as np
from flax import nnx
from jax_addition_transformer.config import OptimizerConfig
from jax_addition_transformer.data import tokenize_pairs
from jax_addition_transformer.optimizers import make_optimizer
from jax_addition_transformer.training import create_train_step


def test_train_step_is_finite_and_changes_parameters(tiny_model):
    graph, params = nnx.split(tiny_model, nnx.Param)
    tx, _ = make_optimizer(
        OptimizerConfig(total_steps=10, warmup_steps=1, initial_learning_rate=1e-3), params
    )
    state = tx.init(params)
    full = tokenize_pairs(np.arange(8))
    step = create_train_step(graph, tx)
    new, _, metrics = step(params, state, jnp.asarray(full[:, :-1]), jnp.asarray(full[:, 1:]))
    assert bool(metrics["finite"])
    assert any(
        not np.array_equal(a, b)
        for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(new), strict=True)
    )


def test_tiny_model_overfits_one_example(tiny_model):
    graph, params = nnx.split(tiny_model, nnx.Param)
    config = OptimizerConfig(
        total_steps=40,
        warmup_steps=1,
        initial_learning_rate=0.01,
        peak_learning_rate=0.01,
        final_learning_rate=0.005,
        weight_decay=0.0,
    )
    optimizer, _ = make_optimizer(config, params)
    state = optimizer.init(params)
    step = create_train_step(graph, optimizer)
    full = np.repeat(tokenize_pairs(np.array([42])), 32, axis=0)
    inputs, targets = jnp.asarray(full[:, :-1]), jnp.asarray(full[:, 1:])
    losses = []
    for _ in range(40):
        params, state, metrics = step(params, state, inputs, targets)
        losses.append(float(metrics["loss"]))
    assert losses[-1] < losses[0] * 0.2
