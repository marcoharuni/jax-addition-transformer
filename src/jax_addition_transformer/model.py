"""Decoder-only Pre-LayerNorm transformer with directly tied output weights."""

from __future__ import annotations


import jax
import jax.numpy as jnp
from flax import nnx

from .attention import CausalAttention
from .config import ModelConfig
from .ffn import FeedForward
from .layers import dtype_from_name, embedding_lookup, normal_parameter
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
        self.ffn = FeedForward(config, rngs=rngs)

    def __call__(self, x: jax.Array, return_attention: bool = False):
        result = self.attention(self.attention_norm(x), return_attention)
        if return_attention:
            attended, probabilities = result
        else:
            attended = result
        x = x + attended
        x = x + self.ffn(self.ffn_norm(x))
        return (x, probabilities) if return_attention else x


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
        self.blocks = nnx.List(
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

    def __call__(self, token_ids: jax.Array, return_attention: bool = False):
        if token_ids.ndim != 2 or token_ids.shape[1] > self.config.max_input_length:
            raise ValueError("token_ids must be [batch, sequence<=max_input_length]")
        x = embedding_lookup(self.token_embedding.value, token_ids)
        if self.positions is not None:
            x = x + self.positions(token_ids.shape[1])
        maps = []
        for block in self.blocks:
            if return_attention:
                x, probabilities = block(x, True)
                maps.append(probabilities)
            else:
                x = block(x)
        x = self.final_norm(x)
        output_weight = self.token_embedding.value.T if self.lm_head is None else self.lm_head.value
        logits = jnp.einsum(
            "btd,dv->btv",
            x.astype(jnp.float32),
            output_weight.astype(jnp.float32),
            preferred_element_type=jnp.float32,
        )
        return (logits, maps) if return_attention else logits


def count_parameters(model: AdditionTransformer) -> int:
    state = nnx.state(model, nnx.Param)
    return sum(int(x.size) for x in jax.tree.leaves(state))


def parameter_table(config: ModelConfig) -> list[tuple[str, int]]:
    bias = 0
    if config.use_bias:
        bias = (
            config.d_model
            + 2 * config.n_kv_heads * config.head_dim
            + config.d_model
            + config.d_ff
            + config.d_model
        )
        if config.ffn_type in {"swiglu", "geglu"}:
            bias += config.d_ff
    attention = (
        config.d_model * config.d_model * 2
        + 2 * config.d_model * config.n_kv_heads * config.head_dim
    )
    ffn = 2 * config.d_model * config.d_ff + (
        config.d_model * config.d_ff if config.ffn_type in {"swiglu", "geglu"} else 0
    )
    norms = 4 * config.d_model if config.norm_type == "layernorm" else 2 * config.d_model
    rows = [
        ("attention projections", config.n_layers * attention),
        ("feed-forward networks", config.n_layers * ffn),
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
