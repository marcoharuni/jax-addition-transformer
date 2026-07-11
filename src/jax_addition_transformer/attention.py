"""Handwritten MHA, GQA, MQA, causal mask, and float32 attention."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import nnx

from .config import ModelConfig
from .layers import Linear, dtype_from_name
from .positional import apply_rope


def causal_mask(length: int) -> jax.Array:
    return jnp.arange(length)[:, None] >= jnp.arange(length)[None, :]


def scaled_dot_product_attention(
    q: jax.Array, k: jax.Array, v: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Attend with q [B,T,H,D] and k/v [B,T,K,D], repeating KV groups logically."""
    repeats = q.shape[2] // k.shape[2]
    k = jnp.repeat(k, repeats, axis=2)
    v = jnp.repeat(v, repeats, axis=2)
    scores = jnp.einsum(
        "bthd,bshd->bhts",
        q.astype(jnp.float32),
        k.astype(jnp.float32),
        preferred_element_type=jnp.float32,
    )
    scores = scores / math.sqrt(q.shape[-1])
    visible = causal_mask(q.shape[1])[None, None, :, :]
    scores = jnp.where(visible, scores, jnp.finfo(jnp.float32).min)
    probabilities = jax.nn.softmax(scores, axis=-1).astype(jnp.float32)
    output = jnp.einsum(
        "bhts,bshd->bthd", probabilities, v.astype(jnp.float32), preferred_element_type=jnp.float32
    )
    return output, probabilities


class CausalAttention(nnx.Module):
    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
        d, kv = config.d_model, config.n_kv_heads * config.head_dim
        dtype = dtype_from_name(config.parameter_dtype)
        args = {
            "rngs": rngs,
            "scale": config.init_scale,
            "use_bias": config.use_bias,
            "parameter_dtype": dtype,
        }
        self.q_proj = Linear(d, d, **args)
        self.k_proj = Linear(d, kv, **args)
        self.v_proj = Linear(d, kv, **args)
        self.out_proj = Linear(d, d, **args)
        self.config = config

    def __call__(
        self, x: jax.Array, return_attention: bool = False
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        cfg, batch, length = self.config, x.shape[0], x.shape[1]
        compute = dtype_from_name(cfg.compute_dtype)
        q = self.q_proj(x, compute).reshape(batch, length, cfg.n_heads, cfg.head_dim)
        k = self.k_proj(x, compute).reshape(batch, length, cfg.n_kv_heads, cfg.head_dim)
        v = self.v_proj(x, compute).reshape(batch, length, cfg.n_kv_heads, cfg.head_dim)
        if cfg.position_type == "rope":
            q, k = apply_rope(q, cfg.rope_base), apply_rope(k, cfg.rope_base)
        attended, probabilities = scaled_dot_product_attention(q, k, v)
        output = self.out_proj(attended.reshape(batch, length, cfg.d_model), compute)
        return (output, probabilities) if return_attention else output
