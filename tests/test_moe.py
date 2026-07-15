import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from jax_addition_transformer.checkpointing import restore_checkpoint, save_checkpoint
from jax_addition_transformer.config import ModelConfig, OptimizerConfig
from jax_addition_transformer.model import AdditionTransformer
from jax_addition_transformer.moe import (
    RaggedMoE,
    RouterOutput,
    group_assignments,
    load_balancing_loss,
    reference_moe,
    router_z_loss,
    routing_metrics,
)
from jax_addition_transformer.optimizers import make_optimizer


def moe_config(**changes):
    values = dict(
        n_layers=1,
        d_model=8,
        n_heads=2,
        n_kv_heads=2,
        d_ff=16,
        max_input_length=9,
        compute_dtype="float32",
    )
    values.update(changes)
    return ModelConfig.matched_moe(ModelConfig(**values))


@pytest.mark.parametrize(
    ("compute_dtype", "atol"), [("float32", 1e-6), ("bfloat16", 3e-2)]
)
def test_ragged_output_matches_reference_and_is_finite(compute_dtype, atol):
    module = RaggedMoE(moe_config(compute_dtype=compute_dtype), rngs=nnx.Rngs(params=7))
    x = jax.random.normal(jax.random.key(8), (3, 4, 8))
    actual = module(x)
    expected = reference_moe(module, x)
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=atol)
    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(actual, module(x))


def test_ragged_parameter_and_input_gradients_match_reference():
    module = RaggedMoE(moe_config(), rngs=nnx.Rngs(params=3))
    graph, params = nnx.split(module, nnx.Param)
    x = jax.random.normal(jax.random.key(4), (2, 3, 8))

    def loss(candidate, inputs, reference):
        current = nnx.merge(graph, candidate)
        output = reference_moe(current, inputs) if reference else current(inputs)
        return jnp.sum(jnp.square(output))

    actual_params, actual_x = jax.grad(loss, argnums=(0, 1))(params, x, False)
    expected_params, expected_x = jax.grad(loss, argnums=(0, 1))(params, x, True)
    for actual, expected in zip(
        jax.tree.leaves(actual_params), jax.tree.leaves(expected_params), strict=True
    ):
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(actual_x, expected_x, atol=2e-5, rtol=2e-5)


def test_top_k_and_grouping_invariants_and_scatter_order():
    module = RaggedMoE(moe_config(), rngs=nnx.Rngs(params=0))
    x = jnp.arange(40, dtype=jnp.float32).reshape(5, 1, 8)
    routing = module.router(x)
    grouped = group_assignments(x, routing, module.config.num_experts)
    np.testing.assert_allclose(jnp.sum(routing.selected_weights, axis=-1), 1.0)
    assert routing.selected_experts.shape == (5, 2)
    assert grouped.group_sizes.shape == (4,)
    assert int(grouped.group_sizes.sum()) == 10
    assert np.all(np.diff(np.asarray(grouped.grouped_expert_indices)) >= 0)
    restored = jnp.zeros((5,), dtype=jnp.int32).at[grouped.grouped_token_indices].add(1)
    np.testing.assert_array_equal(restored, 2)


@pytest.mark.parametrize("selected", [jnp.zeros((6, 2), dtype=jnp.int32), None])
def test_empty_experts_and_highly_imbalanced_routes(selected):
    x = jnp.ones((2, 3, 8), dtype=jnp.float32)
    if selected is None:
        selected = jnp.tile(jnp.array([[0, 1]], dtype=jnp.int32), (6, 1))
    routing = RouterOutput(
        selected,
        jnp.full((6, 2), 0.5),
        jnp.full((6, 4), 0.25),
        jnp.zeros((6, 4)),
    )
    grouped = group_assignments(x, routing, 4)
    assert int(grouped.group_sizes.sum()) == 12
    assert np.count_nonzero(np.asarray(grouped.group_sizes) == 0) >= 2


def test_production_path_handles_identical_routing_and_empty_experts():
    module = RaggedMoE(moe_config(), rngs=nnx.Rngs(params=1))
    column_logits = jnp.array([1.0, 0.5, -1.0, -2.0])
    module.router.kernel.value = jnp.tile(column_logits / 8, (8, 1))
    x = jnp.ones((2, 3, 8), dtype=jnp.float32)
    routing = module.router(x)
    grouped = group_assignments(x, routing, 4)
    np.testing.assert_array_equal(grouped.group_sizes, [6, 6, 0, 0])
    np.testing.assert_allclose(module(x), reference_moe(module, x), atol=1e-6, rtol=1e-6)


def test_auxiliary_loss_and_routing_metric_formulas_are_serializable():
    probabilities = jnp.array([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3]], dtype=jnp.float32)
    selected = jnp.array([[0, 1], [1, 2]], dtype=jnp.int32)
    weights = jnp.array([[0.7 / 0.9, 0.2 / 0.9], [2 / 3, 1 / 3]], dtype=jnp.float32)
    logits = jnp.log(probabilities)
    routing = RouterOutput(selected, weights, probabilities, logits)
    assignment_fraction = jnp.array([0.25, 0.5, 0.25])
    expected_balance = 3 * jnp.sum(jnp.mean(probabilities, axis=0) * assignment_fraction)
    np.testing.assert_allclose(load_balancing_loss(routing, 3), expected_balance)
    np.testing.assert_allclose(
        router_z_loss(logits), jnp.mean(jnp.square(jax.nn.logsumexp(logits, axis=-1)))
    )
    metrics = routing_metrics(routing, 3)
    np.testing.assert_array_equal(metrics["assignments_per_expert"], [1, 2, 1])
    np.testing.assert_allclose(metrics["assignment_fraction_per_expert"], assignment_fraction)
    np.testing.assert_allclose(metrics["top1_token_fraction_per_expert"], [0.5, 0.5, 0.0])
    expected_entropy = -jnp.mean(jnp.sum(probabilities * jnp.log(probabilities), axis=-1))
    np.testing.assert_allclose(metrics["router_entropy"], expected_entropy)
    np.testing.assert_allclose(
        metrics["expert_load_coefficient_of_variation"], jnp.std(jnp.array([1.0, 2, 1])) / (4 / 3)
    )
    json.dumps(jax.tree.map(lambda value: np.asarray(value).tolist(), metrics))


def test_production_jaxpr_contains_two_ragged_dot_primitives_and_jits():
    module = RaggedMoE(moe_config(), rngs=nnx.Rngs(params=5))
    x = jnp.ones((2, 3, 8), dtype=jnp.float32)
    jaxpr = str(jax.make_jaxpr(module)(x))
    assert jaxpr.count("ragged_dot_general[") == 2
    assert jax.jit(module)(x).shape == x.shape


def test_moe_checkpoint_round_trip(tmp_path):
    model = AdditionTransformer(moe_config(), rngs=nnx.Rngs(params=9))
    graph, params = nnx.split(model, nnx.Param)
    optimizer, _ = make_optimizer(OptimizerConfig(total_steps=2, warmup_steps=1), params)
    optimizer_state = optimizer.init(params)
    tokens = jnp.arange(18, dtype=jnp.int32).reshape(2, 9) % 13
    before = model(tokens)
    save_checkpoint(
        tmp_path / "checkpoint",
        params,
        optimizer_state,
        {"config_fingerprint": "moe", "step": 0},
    )
    restored, _, metadata = restore_checkpoint(
        tmp_path / "checkpoint", params, optimizer_state, "moe"
    )
    np.testing.assert_array_equal(before, nnx.merge(graph, restored)(tokens))
    assert metadata["step"] == 0
