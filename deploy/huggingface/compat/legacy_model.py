"""Exact notebook architecture used to restore the v0.1.0 NNX parameter tree."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import nnx

VOCAB_SIZE = 13
MODEL_INPUT_LENGTH = 15
PROMPT_LENGTH = 12
ANSWER_DIGITS = 4
N_LAYERS = 5
D_MODEL = 320
N_HEADS = 5
HEAD_DIM = 64
D_FF = 2480
PARAM_DTYPE = jnp.float32
COMPUTE_DTYPE = jnp.float16


def normal_parameter(rngs: nnx.Rngs, shape: tuple[int, ...], scale: float) -> nnx.Param:
    values = jax.random.normal(rngs.params(), shape, dtype=jnp.float32) * scale
    return nnx.Param(values.astype(PARAM_DTYPE))


def matrix_multiply(x: jax.Array, weight: jax.Array) -> jax.Array:
    x16 = x.astype(COMPUTE_DTYPE)
    w16 = weight.astype(COMPUTE_DTYPE)
    return jax.lax.dot_general(
        x16,
        w16,
        (((x16.ndim - 1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )


class Linear(nnx.Module):
    def __init__(self, input_width: int, output_width: int, rngs: nnx.Rngs, scale: float = 0.02):
        self.kernel = normal_parameter(rngs, (input_width, output_width), scale)

    def __call__(self, x: jax.Array) -> jax.Array:
        return matrix_multiply(x, self.kernel.get_value())


class LayerNorm(nnx.Module):
    def __init__(self, width: int):
        self.scale = nnx.Param(jnp.ones((width,), dtype=PARAM_DTYPE))
        self.bias = nnx.Param(jnp.zeros((width,), dtype=PARAM_DTYPE))

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x.astype(jnp.float32)
        mean = jnp.mean(x, axis=-1, keepdims=True)
        variance = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
        normalized = (x - mean) * jax.lax.rsqrt(variance + 1e-5)
        return normalized * self.scale.get_value() + self.bias.get_value()


def gelu(x: jax.Array) -> jax.Array:
    return 0.5 * x * (1.0 + jax.lax.erf(x / math.sqrt(2.0)))


CAUSAL_MASK = jnp.tril(jnp.ones((MODEL_INPUT_LENGTH, MODEL_INPUT_LENGTH), dtype=bool))


class CausalMHA(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        residual_scale = 0.02 / math.sqrt(2 * N_LAYERS)
        self.q_proj = Linear(D_MODEL, D_MODEL, rngs)
        self.k_proj = Linear(D_MODEL, D_MODEL, rngs)
        self.v_proj = Linear(D_MODEL, D_MODEL, rngs)
        self.out_proj = Linear(D_MODEL, D_MODEL, rngs, scale=residual_scale)

    def __call__(self, x: jax.Array) -> jax.Array:
        batch, length, _ = x.shape
        q = self.q_proj(x).reshape(batch, length, N_HEADS, HEAD_DIM)
        k = self.k_proj(x).reshape(batch, length, N_HEADS, HEAD_DIM)
        v = self.v_proj(x).reshape(batch, length, N_HEADS, HEAD_DIM)
        scores = jnp.einsum(
            "bthd,bshd->bhts",
            q.astype(jnp.float32),
            k.astype(jnp.float32),
            preferred_element_type=jnp.float32,
        ) / math.sqrt(HEAD_DIM)
        scores = jnp.where(
            CAUSAL_MASK[None, None, :length, :length],
            scores,
            jnp.finfo(jnp.float32).min,
        )
        probabilities = jax.nn.softmax(scores, axis=-1)
        attended = jnp.einsum(
            "bhts,bshd->bthd",
            probabilities,
            v.astype(jnp.float32),
            preferred_element_type=jnp.float32,
        )
        return self.out_proj(attended.reshape(batch, length, D_MODEL))


class FeedForward(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        residual_scale = 0.02 / math.sqrt(2 * N_LAYERS)
        self.up = Linear(D_MODEL, D_FF, rngs)
        self.down = Linear(D_FF, D_MODEL, rngs, scale=residual_scale)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.down(gelu(self.up(x)))


class TransformerBlock(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.attention_norm = LayerNorm(D_MODEL)
        self.attention = CausalMHA(rngs)
        self.ffn_norm = LayerNorm(D_MODEL)
        self.ffn = FeedForward(rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x + self.attention(self.attention_norm(x))
        return x + self.ffn(self.ffn_norm(x))


class AdditionTransformer(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.token_embedding = normal_parameter(rngs, (VOCAB_SIZE, D_MODEL), 0.02)
        self.position_embedding = normal_parameter(rngs, (MODEL_INPUT_LENGTH, D_MODEL), 0.02)
        self.blocks = nnx.List([TransformerBlock(rngs) for _ in range(N_LAYERS)])
        self.final_norm = LayerNorm(D_MODEL)

    def __call__(self, token_ids: jax.Array) -> jax.Array:
        if token_ids.ndim != 2 or token_ids.shape[1] != MODEL_INPUT_LENGTH:
            raise ValueError("token_ids must have shape [batch, 15]")
        x = self.token_embedding.get_value()[token_ids]
        x = x + self.position_embedding.get_value()[None, :, :]
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return jnp.einsum(
            "btd,vd->btv",
            x.astype(jnp.float32),
            self.token_embedding.get_value().astype(jnp.float32),
            preferred_element_type=jnp.float32,
        )


def greedy_generate(model: AdditionTransformer, prompts: jax.Array) -> tuple[jax.Array, jax.Array]:
    buffer = jnp.zeros((prompts.shape[0], MODEL_INPUT_LENGTH), dtype=jnp.int32)
    buffer = buffer.at[:, :PROMPT_LENGTH].set(prompts)

    def generate_digit(current: jax.Array, offset: jax.Array):
        logits = model(current)
        next_token = jnp.argmax(logits[:, PROMPT_LENGTH + offset - 1, :], axis=-1).astype(jnp.int32)
        current = jax.lax.cond(
            offset < ANSWER_DIGITS - 1,
            lambda value: value.at[:, PROMPT_LENGTH + offset].set(next_token),
            lambda value: value,
            current,
        )
        return current, next_token

    _, generated = jax.lax.scan(generate_digit, buffer, jnp.arange(ANSWER_DIGITS))
    generated = jnp.swapaxes(generated, 0, 1)
    return generated, jnp.all(generated < 10, axis=-1)
