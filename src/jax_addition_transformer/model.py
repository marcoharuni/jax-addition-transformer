"""Decoder-only Pre-LayerNorm transformer with directly tied output weights."""

from __future__ import annotations


import jax
import jax.numpy as jnp
from flax import nnx

from .attention import CausalAttention
from .config import ModelConfig
from .ffn import FeedForward
from .layers import dtype_from_name, embedding_lookup, normal_parameter
from .moe import RaggedMoE
from .normalization import make_norm
from .positional import LearnedPositions


class TransformerBlock(nnx.Module):
    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
        dtype = dtype_from_name(config.parameter_dtype)
        self.attention_norm = make_norm(
            config.norm_type, config.d_model, config.norm_epsilon, dtype
        )
        self.attention = CausalAttention(config, rngs=rngs)
        self.ffn_norm = make_norm(config.norm_type, config.d_model, config.norm_epsilon, dtype)
        self.ffn = (
            RaggedMoE(config, rngs=rngs)
            if config.architecture == "moe"
            else FeedForward(config, rngs=rngs)
        )

    def __call__(self, x: jax.Array, return_attention: bool = False, return_aux: bool = False):
        result = self.attention(self.attention_norm(x), return_attention)
        if return_attention:
            attended, probabilities = result
        else:
            attended = result
        x = x + attended
        normalized = self.ffn_norm(x)
        if return_aux and isinstance(self.ffn, RaggedMoE):
            ffn_output, auxiliary = self.ffn(normalized, return_aux=True)
        else:
            ffn_output = self.ffn(normalized)
            auxiliary = None
        x = x + ffn_output
        if return_attention and return_aux:
            return x, probabilities, auxiliary
        if return_attention:
            return x, probabilities
        return (x, auxiliary) if return_aux else x


class ModuleSequence(nnx.Module):
    """NNX-compatible ordered modules without relying on ``nnx.List``."""

    def __init__(self, modules: list[nnx.Module]):
        self.length = len(modules)
        for index, module in enumerate(modules):
            setattr(self, f"layer_{index}", module)

    def __len__(self) -> int:
        return self.length

    def __iter__(self):
        for index in range(self.length):
            yield getattr(self, f"layer_{index}")

    def __getitem__(self, index: int):
        if index < 0:
            index += self.length
        if not 0 <= index < self.length:
            raise IndexError(index)
        return getattr(self, f"layer_{index}")


class AdditionTransformer(nnx.Module):
    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
        dtype = dtype_from_name(config.parameter_dtype)
        self.token_embedding = normal_parameter(
            rngs, (config.vocab_size, config.d_model), config.init_scale, dtype
        )
        self.positions = (
            LearnedPositions(
                config.max_input_length,
                config.d_model,
                rngs=rngs,
                scale=config.init_scale,
                dtype=dtype,
            )
            if config.position_type == "learned"
            else None
        )
        self.blocks = ModuleSequence(
            [TransformerBlock(config, rngs=rngs) for _ in range(config.n_layers)]
        )
        self.final_norm = make_norm(config.norm_type, config.d_model, config.norm_epsilon, dtype)
        self.lm_head = (
            None
            if config.tie_embeddings
            else normal_parameter(
                rngs, (config.d_model, config.vocab_size), config.init_scale, dtype
            )
        )
        self.config = config

    def __call__(
        self, token_ids: jax.Array, return_attention: bool = False, return_aux: bool = False
    ):
        if token_ids.ndim != 2 or token_ids.shape[1] > self.config.max_input_length:
            raise ValueError("token_ids must be [batch, sequence<=max_input_length]")
        x = embedding_lookup(self.token_embedding.value, token_ids)
        if self.positions is not None:
            x = x + self.positions(token_ids.shape[1])
        maps = []
        layer_auxiliary = []
        for block in self.blocks:

            def apply_block(value, current_block=block):
                return current_block(
                    value,
                    return_attention=return_attention,
                    return_aux=return_aux,
                )

            result = (
                jax.checkpoint(apply_block)(x)
                if self.config.uses_rematerialization
                else apply_block(x)
            )
            if return_attention and return_aux:
                x, probabilities, auxiliary = result
                maps.append(probabilities)
                if auxiliary is not None:
                    layer_auxiliary.append(auxiliary)
            elif return_attention:
                x, probabilities = result
                maps.append(probabilities)
            elif return_aux:
                x, auxiliary = result
                if auxiliary is not None:
                    layer_auxiliary.append(auxiliary)
            else:
                x = result
        x = self.final_norm(x)
        output_weight = self.token_embedding.value.T if self.lm_head is None else self.lm_head.value
        logits = jnp.einsum(
            "btd,dv->btv",
            x.astype(jnp.float32),
            output_weight.astype(jnp.float32),
            preferred_element_type=jnp.float32,
        )
        auxiliary = None
        if return_aux:
            if layer_auxiliary:
                auxiliary = {
                    "load_balancing_loss": jnp.mean(
                        jnp.stack([row["load_balancing_loss"] for row in layer_auxiliary])
                    ),
                    "router_z_loss": jnp.mean(
                        jnp.stack([row["router_z_loss"] for row in layer_auxiliary])
                    ),
                    "routing_metrics": tuple(row["routing_metrics"] for row in layer_auxiliary),
                }
            else:
                auxiliary = {
                    "load_balancing_loss": jnp.asarray(0.0, dtype=jnp.float32),
                    "router_z_loss": jnp.asarray(0.0, dtype=jnp.float32),
                    "routing_metrics": (),
                }
        if return_attention and return_aux:
            return logits, maps, auxiliary
        if return_attention:
            return logits, maps
        return (logits, auxiliary) if return_aux else logits


def count_parameters(model: AdditionTransformer) -> int:
    state = nnx.state(model, nnx.Param)
    return sum(int(x.size) for x in jax.tree.leaves(state))


def parameter_table(config: ModelConfig) -> list[tuple[str, int]]:
    bias = 0
    if config.use_bias:
        bias = config.d_model + 2 * config.n_kv_heads * config.head_dim + config.d_model
        if config.architecture == "dense":
            bias += config.d_ff + config.d_model
            if config.ffn_type in {"swiglu", "geglu"}:
                bias += config.d_ff
    attention = (
        config.d_model * config.d_model * 2
        + 2 * config.d_model * config.n_kv_heads * config.head_dim
    )
    if config.architecture == "dense":
        ffn = 2 * config.d_model * config.d_ff + (
            config.d_model * config.d_ff if config.ffn_type in {"swiglu", "geglu"} else 0
        )
        ffn_label = "feed-forward networks"
    else:
        ffn = (
            2 * config.num_experts * config.d_model * config.expert_d_ff
            + config.d_model * config.num_experts
        )
        if config.use_bias:
            ffn += config.num_experts * (config.expert_d_ff + config.d_model)
        ffn_label = "mixture-of-experts networks"
    norms = 4 * config.d_model if config.norm_type == "layernorm" else 2 * config.d_model
    rows = [
        ("attention projections", config.n_layers * attention),
        (ffn_label, config.n_layers * ffn),
        ("block normalizations", config.n_layers * norms),
    ]
    if bias:
        rows.append(("linear biases", config.n_layers * bias))
    rows.append(("token embedding", config.vocab_size * config.d_model))
    rows.append(
        (
            "position embedding",
            config.max_input_length * config.d_model if config.position_type == "learned" else 0,
        )
    )
    rows.append(
        ("final norm", 2 * config.d_model if config.norm_type == "layernorm" else config.d_model)
    )
    if not config.tie_embeddings:
        rows.append(("language-model head", config.d_model * config.vocab_size))
    return rows


def active_parameter_proxy(model: AdditionTransformer) -> int:
    """Count parameters active per token under top-k expert selection.

    All non-expert parameters, including every router, are active. Exactly ``top_k``
    of ``num_experts`` identically shaped expert parameter sets are active per token.
    This is an architectural proxy, not a wall-clock or FLOP measurement.
    """
    total = count_parameters(model)
    if model.config.architecture == "dense":
        return total
    expert_total = 0
    for block in model.blocks:
        expert_total += block.ffn.up_kernel.value.size + block.ffn.down_kernel.value.size
        if block.ffn.up_bias is not None:
            expert_total += block.ffn.up_bias.value.size + block.ffn.down_bias.value.size
    selected_expert_parameters = expert_total * model.config.top_k
    if selected_expert_parameters % model.config.num_experts:
        raise AssertionError("expert parameter tree is not evenly divisible by num_experts")
    return total - expert_total + selected_expert_parameters // model.config.num_experts


def assert_parameter_count(model: AdditionTransformer, expected: int | None = None) -> int:
    actual = count_parameters(model)
    derived = sum(value for _, value in parameter_table(model.config))
    if actual != derived:
        raise AssertionError(f"parameter tree has {actual:,}, derivation has {derived:,}")
    if expected is not None and actual != expected:
        raise AssertionError(f"expected {expected:,} trainable parameters, found {actual:,}")
    if model.config.is_exact_default and actual != 10_000_000:
        raise AssertionError(f"frozen default must have 10,000,000 parameters, found {actual:,}")
    return actual
