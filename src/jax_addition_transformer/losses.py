"""Answer-only autoregressive loss and metrics."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def answer_mask(input_length: int = 15, answer_digits: int = 4) -> jax.Array:
    return jnp.arange(input_length) >= input_length - answer_digits


def masked_cross_entropy(
    logits: jax.Array, targets: jax.Array, mask: jax.Array | None = None
) -> tuple[jax.Array, jax.Array]:
    if logits.shape[:-1] != targets.shape:
        raise ValueError("logit and target shapes do not align")
    mask = answer_mask(targets.shape[1]) if mask is None else jnp.asarray(mask, dtype=bool)
    if mask.shape != (targets.shape[1],):
        raise ValueError("mask must have one value per target position")
    log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    losses = -jnp.take_along_axis(log_probs, targets[..., None], axis=-1).squeeze(-1)
    denominator = targets.shape[0] * jnp.sum(mask)
    loss = jnp.sum(jnp.where(mask[None, :], losses, 0.0)) / denominator
    correct = jnp.argmax(logits, axis=-1) == targets
    accuracy = jnp.sum(jnp.where(mask[None, :], correct, False)) / denominator
    return loss, accuracy
