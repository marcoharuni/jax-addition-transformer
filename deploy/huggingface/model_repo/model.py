"""Standalone pure-JAX implementation of the released 10M addition transformer."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from safetensors.numpy import load_file

VOCAB_SIZE = 13
MODEL_INPUT_LENGTH = 15
PROMPT_LENGTH = 12
ANSWER_DIGITS = 4
N_LAYERS = 5
D_MODEL = 320
N_HEADS = 5
HEAD_DIM = 64
D_FF = 2480


def expected_shapes() -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {
        "token_embedding": (VOCAB_SIZE, D_MODEL),
        "position_embedding": (MODEL_INPUT_LENGTH, D_MODEL),
        "final_norm.scale": (D_MODEL,),
        "final_norm.bias": (D_MODEL,),
    }
    for layer in range(N_LAYERS):
        prefix = f"blocks.{layer}"
        for projection in ("q_proj", "k_proj", "v_proj", "out_proj"):
            shapes[f"{prefix}.attention.{projection}.kernel"] = (D_MODEL, D_MODEL)
        shapes[f"{prefix}.attention_norm.scale"] = (D_MODEL,)
        shapes[f"{prefix}.attention_norm.bias"] = (D_MODEL,)
        shapes[f"{prefix}.ffn.up.kernel"] = (D_MODEL, D_FF)
        shapes[f"{prefix}.ffn.down.kernel"] = (D_FF, D_MODEL)
        shapes[f"{prefix}.ffn_norm.scale"] = (D_MODEL,)
        shapes[f"{prefix}.ffn_norm.bias"] = (D_MODEL,)
    return shapes


def validate_weights(weights: Mapping[str, np.ndarray | jax.Array]) -> int:
    expected = expected_shapes()
    if set(weights) != set(expected):
        missing = sorted(set(expected) - set(weights))
        extra = sorted(set(weights) - set(expected))
        raise ValueError(f"weight names differ; missing={missing}, extra={extra}")
    total = 0
    for name, shape in expected.items():
        array = np.asarray(weights[name])
        if array.shape != shape:
            raise ValueError(f"{name} has shape {array.shape}; expected {shape}")
        if array.dtype != np.float32:
            raise ValueError(f"{name} has dtype {array.dtype}; expected float32")
        total += array.size
    if total != 10_000_000:
        raise ValueError(f"loaded {total:,} parameters; expected 10,000,000")
    return total


def load_weights(path: str | Path) -> dict[str, jax.Array]:
    arrays = load_file(str(path))
    validate_weights(arrays)
    return {name: jnp.asarray(value) for name, value in arrays.items()}


def _linear(x: jax.Array, weight: jax.Array) -> jax.Array:
    x16, weight16 = x.astype(jnp.float16), weight.astype(jnp.float16)
    return jax.lax.dot_general(
        x16,
        weight16,
        (((x16.ndim - 1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )


def _layer_norm(x: jax.Array, scale: jax.Array, bias: jax.Array) -> jax.Array:
    x = x.astype(jnp.float32)
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(variance + 1e-5) * scale + bias


def _gelu(x: jax.Array) -> jax.Array:
    return 0.5 * x * (1.0 + jax.lax.erf(x / math.sqrt(2.0)))


def forward(weights: Mapping[str, jax.Array], token_ids: jax.Array) -> jax.Array:
    if token_ids.ndim != 2 or token_ids.shape[1] != MODEL_INPUT_LENGTH:
        raise ValueError("token_ids must have shape [batch, 15]")
    x = weights["token_embedding"][token_ids]
    x = x + weights["position_embedding"][None, :, :]
    causal_mask = jnp.tril(jnp.ones((MODEL_INPUT_LENGTH, MODEL_INPUT_LENGTH), dtype=bool))

    for layer in range(N_LAYERS):
        prefix = f"blocks.{layer}"
        normalized = _layer_norm(
            x,
            weights[f"{prefix}.attention_norm.scale"],
            weights[f"{prefix}.attention_norm.bias"],
        )
        q = _linear(normalized, weights[f"{prefix}.attention.q_proj.kernel"]).reshape(
            token_ids.shape[0], MODEL_INPUT_LENGTH, N_HEADS, HEAD_DIM
        )
        k = _linear(normalized, weights[f"{prefix}.attention.k_proj.kernel"]).reshape(
            token_ids.shape[0], MODEL_INPUT_LENGTH, N_HEADS, HEAD_DIM
        )
        v = _linear(normalized, weights[f"{prefix}.attention.v_proj.kernel"]).reshape(
            token_ids.shape[0], MODEL_INPUT_LENGTH, N_HEADS, HEAD_DIM
        )
        scores = jnp.einsum(
            "bthd,bshd->bhts",
            q.astype(jnp.float32),
            k.astype(jnp.float32),
            preferred_element_type=jnp.float32,
        ) / math.sqrt(HEAD_DIM)
        scores = jnp.where(
            causal_mask[None, None, :, :],
            scores,
            jnp.finfo(jnp.float32).min,
        )
        probabilities = jax.nn.softmax(scores, axis=-1)
        attended = jnp.einsum(
            "bhts,bshd->bthd",
            probabilities,
            v.astype(jnp.float32),
            preferred_element_type=jnp.float32,
        ).reshape(token_ids.shape[0], MODEL_INPUT_LENGTH, D_MODEL)
        x = x + _linear(attended, weights[f"{prefix}.attention.out_proj.kernel"])

        normalized = _layer_norm(
            x,
            weights[f"{prefix}.ffn_norm.scale"],
            weights[f"{prefix}.ffn_norm.bias"],
        )
        hidden = _gelu(_linear(normalized, weights[f"{prefix}.ffn.up.kernel"]))
        x = x + _linear(hidden, weights[f"{prefix}.ffn.down.kernel"])

    x = _layer_norm(x, weights["final_norm.scale"], weights["final_norm.bias"])
    return jnp.einsum(
        "btd,vd->btv",
        x.astype(jnp.float32),
        weights["token_embedding"].astype(jnp.float32),
        preferred_element_type=jnp.float32,
    )


def greedy_generate(
    weights: Mapping[str, jax.Array], prompts: jax.Array
) -> tuple[jax.Array, jax.Array]:
    buffer = jnp.zeros((prompts.shape[0], MODEL_INPUT_LENGTH), dtype=jnp.int32)
    buffer = buffer.at[:, :PROMPT_LENGTH].set(prompts)

    def generate_digit(current: jax.Array, offset: jax.Array):
        logits = forward(weights, current)
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
