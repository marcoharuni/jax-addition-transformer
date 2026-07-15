"""Top-k routing and token-preserving expert computation with ``lax.ragged_dot``."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx

from .config import ModelConfig
from .ffn import gelu
from .layers import dtype_from_name, normal_parameter


class RouterOutput(NamedTuple):
    """Router arrays, with token axes flattened to ``[num_tokens, ...]``."""

    selected_experts: jax.Array
    selected_weights: jax.Array
    probabilities: jax.Array
    logits: jax.Array

    @property
    def token_count(self) -> int:
        return self.selected_experts.shape[0]

    @property
    def assignment_count(self) -> int:
        return self.selected_experts.size


class GroupedAssignments(NamedTuple):
    grouped_lhs: jax.Array
    grouped_token_indices: jax.Array
    grouped_expert_indices: jax.Array
    grouped_weights: jax.Array
    group_sizes: jax.Array


class Router(nnx.Module):
    """Bias-free float32 router from ``d_model`` to ``num_experts``."""

    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
        self.kernel = normal_parameter(
            rngs,
            (config.d_model, config.num_experts),
            config.init_scale,
            dtype_from_name(config.parameter_dtype),
        )
        self.top_k = config.top_k

    def __call__(self, x: jax.Array) -> RouterOutput:
        flat = x.reshape(-1, x.shape[-1]).astype(jnp.float32)
        logits = jax.lax.dot_general(
            flat,
            self.kernel.value.astype(jnp.float32),
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        probabilities = jax.nn.softmax(logits, axis=-1).astype(jnp.float32)
        selected_probabilities, selected_experts = jax.lax.top_k(probabilities, self.top_k)
        selected_weights = selected_probabilities / jnp.sum(
            selected_probabilities, axis=-1, keepdims=True
        )
        return RouterOutput(selected_experts, selected_weights, probabilities, logits)


def group_assignments(x: jax.Array, routing: RouterOutput, num_experts: int) -> GroupedAssignments:
    """Duplicate tokens ``top_k`` times and stably group every assignment by expert."""
    flat = x.reshape(-1, x.shape[-1])
    top_k = routing.selected_experts.shape[1]
    token_indices = jnp.repeat(jnp.arange(flat.shape[0], dtype=jnp.int32), top_k)
    expert_indices = routing.selected_experts.reshape(-1).astype(jnp.int32)
    weights = routing.selected_weights.reshape(-1)
    order = jnp.argsort(expert_indices, stable=True)
    grouped_experts = expert_indices[order]
    grouped_tokens = token_indices[order]
    group_sizes = jnp.bincount(expert_indices, length=num_experts).astype(jnp.int32)
    return GroupedAssignments(
        flat[grouped_tokens],
        grouped_tokens,
        grouped_experts,
        weights[order],
        group_sizes,
    )


def _activation(x: jax.Array, kind: str) -> jax.Array:
    if kind == "gelu":
        return gelu(x)
    if kind == "relu":
        return jax.nn.relu(x)
    if kind == "relu2":
        return jnp.square(jax.nn.relu(x))
    raise ValueError(f"unsupported MoE activation {kind!r}")


def load_balancing_loss(routing: RouterOutput, num_experts: int) -> jax.Array:
    """Return E * sum(mean probability * all-top-k assignment fraction)."""
    probability_fraction = jnp.mean(routing.probabilities, axis=0)
    assignment_fraction = jnp.mean(
        jax.nn.one_hot(routing.selected_experts, num_experts, dtype=jnp.float32), axis=(0, 1)
    )
    return num_experts * jnp.sum(probability_fraction * assignment_fraction)


def router_z_loss(logits: jax.Array) -> jax.Array:
    """Mean squared log-normalizer of float32 router logits."""
    return jnp.mean(jnp.square(jax.nn.logsumexp(logits.astype(jnp.float32), axis=-1)))


def routing_metrics(routing: RouterOutput, num_experts: int) -> dict[str, jax.Array]:
    assignments = jnp.bincount(
        routing.selected_experts.reshape(-1), length=num_experts
    ).astype(jnp.int32)
    top1 = jnp.bincount(routing.selected_experts[:, 0], length=num_experts).astype(jnp.int32)
    rank_counts = jnp.stack(
        [
            jnp.bincount(routing.selected_experts[:, rank], length=num_experts)
            for rank in range(routing.selected_experts.shape[1])
        ]
    ).astype(jnp.int32)
    selected_weight_sums = jnp.bincount(
        routing.selected_experts.reshape(-1),
        weights=routing.selected_weights.reshape(-1),
        length=num_experts,
    )
    safe_assignments = jnp.maximum(assignments, 1)
    entropy = -jnp.mean(
        jnp.sum(
            jnp.where(
                routing.probabilities > 0,
                routing.probabilities * jnp.log(routing.probabilities),
                0.0,
            ),
            axis=-1,
        )
    )
    mean_load = jnp.mean(assignments.astype(jnp.float32))
    load_std = jnp.std(assignments.astype(jnp.float32))
    return {
        "assignments_per_expert": assignments,
        "assignment_fraction_per_expert": assignments / routing.assignment_count,
        "mean_router_probability_per_expert": jnp.mean(routing.probabilities, axis=0),
        "mean_selected_routing_weight_per_expert": selected_weight_sums / safe_assignments,
        "selected_weight_sums_per_expert": selected_weight_sums,
        "router_entropy": entropy,
        "minimum_expert_load": jnp.min(assignments),
        "maximum_expert_load": jnp.max(assignments),
        "mean_expert_load": mean_load,
        "maximum_to_mean_load_ratio": jnp.max(assignments) / mean_load,
        "expert_load_coefficient_of_variation": load_std / mean_load,
        "top1_assignments_per_expert": top1,
        "top1_token_fraction_per_expert": top1 / routing.token_count,
        "topk_rank_assignments_per_expert": rank_counts,
        "topk_rank_token_fraction_per_expert": rank_counts / routing.token_count,
        "token_count": jnp.asarray(routing.token_count, dtype=jnp.int32),
        "assignment_count": jnp.asarray(routing.assignment_count, dtype=jnp.int32),
    }


class RaggedMoE(nnx.Module):
    """Token-preserving expert FFN using two actual grouped ragged dot products."""

    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
        dtype = dtype_from_name(config.parameter_dtype)
        self.router = Router(config, rngs=rngs)
        self.up_kernel = normal_parameter(
            rngs,
            (config.num_experts, config.d_model, config.expert_d_ff),
            config.init_scale,
            dtype,
        )
        self.down_kernel = normal_parameter(
            rngs,
            (config.num_experts, config.expert_d_ff, config.d_model),
            config.init_scale,
            dtype,
        )
        self.up_bias = (
            nnx.Param(jnp.zeros((config.num_experts, config.expert_d_ff), dtype))
            if config.use_bias
            else None
        )
        self.down_bias = (
            nnx.Param(jnp.zeros((config.num_experts, config.d_model), dtype))
            if config.use_bias
            else None
        )
        self.config = config

    def __call__(self, x: jax.Array, return_aux: bool = False):
        routing = self.router(x)
        grouped = group_assignments(x, routing, self.config.num_experts)
        compute = dtype_from_name(self.config.compute_dtype)
        lhs = grouped.grouped_lhs.astype(compute)
        up = jax.lax.ragged_dot(
            lhs,
            self.up_kernel.value.astype(compute),
            grouped.group_sizes,
            preferred_element_type=jnp.float32,
        )
        if self.up_bias is not None:
            up = up + self.up_bias.value[grouped.grouped_expert_indices].astype(jnp.float32)
        hidden = _activation(up, self.config.ffn_type).astype(compute)
        down = jax.lax.ragged_dot(
            hidden,
            self.down_kernel.value.astype(compute),
            grouped.group_sizes,
            preferred_element_type=jnp.float32,
        )
        if self.down_bias is not None:
            down = down + self.down_bias.value[grouped.grouped_expert_indices].astype(jnp.float32)
        weighted = down * grouped.grouped_weights[:, None]
        restored = jnp.zeros(
            (routing.token_count, self.config.d_model), dtype=weighted.dtype
        ).at[grouped.grouped_token_indices].add(weighted)
        output = restored.reshape(x.shape)
        if not return_aux:
            return output
        auxiliary = {
            "load_balancing_loss": load_balancing_loss(routing, self.config.num_experts),
            "router_z_loss": router_z_loss(routing.logits),
            "routing_metrics": routing_metrics(routing, self.config.num_experts),
        }
        return output, auxiliary


def reference_moe(module: RaggedMoE, x: jax.Array) -> jax.Array:
    """Readable per-expert reference used only by correctness tests."""
    routing = module.router(x)
    flat = x.reshape(-1, x.shape[-1]).astype(jnp.float32)
    result = jnp.zeros((flat.shape[0], module.config.d_model), dtype=jnp.float32)
    for expert in range(module.config.num_experts):
        up = flat @ module.up_kernel.value[expert].astype(jnp.float32)
        if module.up_bias is not None:
            up = up + module.up_bias.value[expert].astype(jnp.float32)
        down = _activation(up, module.config.ffn_type) @ module.down_kernel.value[
            expert
        ].astype(jnp.float32)
        if module.down_bias is not None:
            down = down + module.down_bias.value[expert].astype(jnp.float32)
        selected = routing.selected_experts == expert
        expert_weight = jnp.sum(jnp.where(selected, routing.selected_weights, 0.0), axis=-1)
        result = result + down * expert_weight[:, None]
    return result.reshape(x.shape)
