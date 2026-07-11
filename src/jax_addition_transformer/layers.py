"""Handwritten parameter initialization, embedding lookup, and projection."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx


def dtype_from_name(name: str) -> jnp.dtype:
    try:
        return {"float16": jnp.float16, "float32": jnp.float32, "bfloat16": jnp.bfloat16}[name]
    except KeyError:
        raise ValueError(f"unsupported dtype {name!r}") from None


def normal_parameter(
    rngs: nnx.Rngs, shape: tuple[int, ...], scale: float, dtype: jnp.dtype
) -> nnx.Param:
    """Initialize W_ij ~ Normal(0, scale^2), explicitly."""
    return nnx.Param(
        (jax.random.normal(rngs.params(), shape, dtype=jnp.float32) * scale).astype(dtype)
    )


class Linear(nnx.Module):
    def __init__(
        self,
        inputs: int,
        outputs: int,
        *,
        rngs: nnx.Rngs,
        scale: float = 0.02,
        use_bias: bool = False,
        parameter_dtype: jnp.dtype = jnp.float32,
    ):
        self.kernel = normal_parameter(rngs, (inputs, outputs), scale, parameter_dtype)
        self.bias = nnx.Param(jnp.zeros((outputs,), parameter_dtype)) if use_bias else None

    def __call__(self, x: jax.Array, compute_dtype: jnp.dtype = jnp.float32) -> jax.Array:
        lhs, rhs = x.astype(compute_dtype), self.kernel.value.astype(compute_dtype)
        y = jax.lax.dot_general(
            lhs, rhs, (((lhs.ndim - 1,), (0,)), ((), ())), preferred_element_type=jnp.float32
        )
        return y + self.bias.value.astype(jnp.float32) if self.bias is not None else y


def embedding_lookup(table: jax.Array, token_ids: jax.Array) -> jax.Array:
    if not jnp.issubdtype(token_ids.dtype, jnp.integer):
        raise TypeError("token IDs must be integers")
    return jnp.take(table, token_ids, axis=0)
