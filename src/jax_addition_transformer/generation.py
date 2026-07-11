"""Fixed-buffer compiled greedy autoregressive generation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .config import TaskConfig
from .data import operands_to_pair_ids, tokenize_pairs
from .tokenizer import decode_internal_answer, format_prompt, parse_expression, validate_operand

DEFAULT_TASK = TaskConfig()


def greedy_generate(
    model, prompts: jax.Array, answer_digits: int = 4
) -> tuple[jax.Array, jax.Array]:
    """Fill a fixed buffer. Future zeros are ordinary `0` tokens, not padding."""
    prompt_length = prompts.shape[1]
    maximum_length = prompt_length + answer_digits - 1
    buffer = jnp.zeros((prompts.shape[0], maximum_length), dtype=jnp.int32)
    buffer = buffer.at[:, :prompt_length].set(prompts)

    def body(current, offset):
        logits = model(current)
        token = jnp.argmax(logits[:, prompt_length + offset - 1, :], axis=-1).astype(jnp.int32)
        current = jax.lax.cond(
            offset < answer_digits - 1,
            lambda value: value.at[:, prompt_length + offset].set(token),
            lambda value: value,
            current,
        )
        return current, token

    _, generated_steps = jax.lax.scan(body, buffer, jnp.arange(answer_digits))
    generated = jnp.swapaxes(generated_steps, 0, 1)
    return generated, jnp.all(generated < 10, axis=-1)


def predict_batch(model_state, a_array, b_array, task: TaskConfig = DEFAULT_TASK):
    a, b = np.asarray(a_array), np.asarray(b_array)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("a_array and b_array must be equally sized vectors")
    for value in np.concatenate((a, b)):
        validate_operand(value, task.max_digits)
    pair_ids = operands_to_pair_ids(a, b, 10**task.max_digits)
    prompt_length = 2 * task.max_digits + 6
    prompts = tokenize_pairs(pair_ids, task.max_digits)[:, :prompt_length]
    return greedy_generate(model_state, jnp.asarray(prompts), task.answer_digits)


def predict_pair(model_state, a: int, b: int, task: TaskConfig = DEFAULT_TASK) -> str:
    generated, valid = predict_batch(model_state, [a], [b], task)
    if not bool(valid[0]):
        return "<invalid>"
    return decode_internal_answer(np.asarray(generated[0]))


def ask(model_state, expression: str, task: TaskConfig = DEFAULT_TASK, debug: bool = False) -> str:
    a, b = parse_expression(expression, task.max_digits)
    prompt = format_prompt(a, b, task.max_digits)
    generated, valid = predict_batch(model_state, [a], [b], task)
    internal = np.asarray(generated[0])
    result = decode_internal_answer(internal) if bool(valid[0]) else "<invalid>"
    if debug:
        return f"prompt={prompt!r} internal={''.join(map(str, internal))} result={result}"
    return result
