"""Handwritten LayerNorm and RMSNorm."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx


class LayerNorm(nnx.Module):
    def __init__(self, features: int, epsilon: float = 1e-5, dtype: jnp.dtype = jnp.float32):
        self.scale = nnx.Param(jnp.ones((features,), dtype))
        self.bias = nnx.Param(jnp.zeros((features,), dtype))
        self.epsilon = epsilon

    def __call__(self, x: jax.Array) -> jax.Array:
        x32 = x.astype(jnp.float32)
        mean = jnp.mean(x32, axis=-1, keepdims=True)
        variance = jnp.mean(jnp.square(x32 - mean), axis=-1, keepdims=True)
        return (x32 - mean) * jax.lax.rsqrt(
            variance + self.epsilon
        ) * self.scale.value + self.bias.value


class RMSNorm(nnx.Module):
    def __init__(self, features: int, epsilon: float = 1e-5, dtype: jnp.dtype = jnp.float32):
        self.scale = nnx.Param(jnp.ones((features,), dtype))
        self.epsilon = epsilon

    def __call__(self, x: jax.Array) -> jax.Array:
        x32 = x.astype(jnp.float32)
        return (
            x32
            * jax.lax.rsqrt(jnp.mean(jnp.square(x32), axis=-1, keepdims=True) + self.epsilon)
            * self.scale.value
        )


def make_norm(kind: str, features: int, epsilon: float, dtype: jnp.dtype) -> nnx.Module:
    return (
        LayerNorm(features, epsilon, dtype)
        if kind == "layernorm"
        else RMSNorm(features, epsilon, dtype)
    )
