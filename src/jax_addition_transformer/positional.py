"""Learned absolute positions and rotary position encoding."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from .layers import normal_parameter


class LearnedPositions(nnx.Module):
    def __init__(self, length: int, width: int, *, rngs: nnx.Rngs, scale: float, dtype: jnp.dtype):
        self.embedding = normal_parameter(rngs, (length, width), scale, dtype)

    def __call__(self, length: int) -> jax.Array:
        return self.embedding.value[:length]


def apply_rope(x: jax.Array, base: float = 10_000.0) -> jax.Array:
    """Rotate adjacent feature pairs; x is [B, T, H, D]."""
    width = x.shape[-1]
    if width % 2:
        raise ValueError("RoPE head dimension must be even")
    positions = jnp.arange(x.shape[1], dtype=jnp.float32)
    frequencies = base ** (-jnp.arange(0, width, 2, dtype=jnp.float32) / width)
    angles = positions[:, None] * frequencies[None, :]
    cos, sin = jnp.cos(angles)[None, :, None, :], jnp.sin(angles)[None, :, None, :]
    even, odd = x[..., 0::2].astype(jnp.float32), x[..., 1::2].astype(jnp.float32)
    return jnp.stack((even * cos - odd * sin, even * sin + odd * cos), axis=-1).reshape(x.shape)
