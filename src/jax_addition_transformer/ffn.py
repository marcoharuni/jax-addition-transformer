"""Handwritten GELU, ReLU, squared ReLU, SiLU, SwiGLU, and GEGLU FFNs."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from .config import ModelConfig
from .layers import Linear, dtype_from_name


def gelu(x: jax.Array) -> jax.Array:
    return 0.5 * x * (1.0 + jax.lax.erf(x / jnp.sqrt(2.0)))


def silu(x: jax.Array) -> jax.Array:
    return x * jax.nn.sigmoid(x)


class FeedForward(nnx.Module):
    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
        dtype = dtype_from_name(config.parameter_dtype)
        args = {
            "rngs": rngs,
            "scale": config.init_scale,
            "use_bias": config.use_bias,
            "parameter_dtype": dtype,
        }
        self.in_proj = Linear(config.d_model, config.d_ff, **args)
        self.gate_proj = (
            Linear(config.d_model, config.d_ff, **args)
            if config.ffn_type in {"swiglu", "geglu"}
            else None
        )
        self.out_proj = Linear(config.d_ff, config.d_model, **args)
        self.config = config

    def __call__(self, x: jax.Array) -> jax.Array:
        compute = dtype_from_name(self.config.compute_dtype)
        hidden = self.in_proj(x, compute)
        kind = self.config.ffn_type
        if kind == "gelu":
            activated = gelu(hidden)
        elif kind == "relu":
            activated = jax.nn.relu(hidden)
        elif kind == "relu2":
            activated = jnp.square(jax.nn.relu(hidden))
        elif kind == "swiglu":
            activated = silu(hidden) * self.gate_proj(x, compute)
        else:
            activated = gelu(hidden) * self.gate_proj(x, compute)
        return self.out_proj(activated, compute)
