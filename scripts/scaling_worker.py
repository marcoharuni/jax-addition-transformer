"""Run one notebook-04 MoE scaling coordinate in an isolated JAX process."""

from __future__ import annotations

import os


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["JAX_COMPILATION_CACHE_DIR"] = "/content/jax_compilation_cache/scaling"
os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "0"
os.environ.pop("JAX_PLATFORMS", None)

import argparse
import json
import math
import pickle
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx


SEED = 42
VOCAB_SIZE = 13
MAX_SEQUENCE_LENGTH = 16
MODEL_INPUT_LENGTH = 15
PROMPT_LENGTH = 12
ANSWER_DIGITS = 4
TRAIN_SIZE = 200_000
VALIDATION_SIZE = 20_000
TEST_SIZE = 780_000
BATCH_SIZE = 2048
EVAL_BATCH_SIZE = 4000
HORIZONS = (50, 125, 300, 750)
PEAK_LR = 1e-3
FINAL_LR = 1e-4
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
PARAM_DTYPE = jnp.float32
COMPUTE_DTYPE = jnp.float16
N_EXPERTS = 4
TOP_K = 2
ROUTER_BALANCE_COEFFICIENT = 1e-2
ROUTER_Z_LOSS_COEFFICIENT = 1e-3

MOE_CONFIGS = {
    "moe_active_0p16m": {
        "dense_reference": "dense_0p16m",
        "n_layers": 2,
        "d_model": 64,
        "n_heads": 1,
        "dense_d_ff": 496,
        "expert_d_ff": 248,
        "dense_parameters": 162_176,
        "stored_parameters": 289_664,
        "active_parameters": 162_688,
    },
    "moe_active_0p64m": {
        "dense_reference": "dense_0p64m",
        "n_layers": 2,
        "d_model": 128,
        "n_heads": 2,
        "dense_d_ff": 992,
        "expert_d_ff": 496,
        "dense_parameters": 643_840,
        "stored_parameters": 1_152_768,
        "active_parameters": 644_864,
    },
    "moe_active_2p16m": {
        "dense_reference": "dense_2p16m",
        "n_layers": 3,
        "d_model": 192,
        "n_heads": 3,
        "dense_d_ff": 1488,
        "expert_d_ff": 744,
        "dense_parameters": 2_164_608,
        "stored_parameters": 3_881_088,
        "active_parameters": 2_166_912,
    },
    "moe_active_10m": {
        "dense_reference": "dense_10m",
        "n_layers": 5,
        "d_model": 320,
        "n_heads": 5,
        "dense_d_ff": 2480,
        "expert_d_ff": 1240,
        "dense_parameters": 10_000_000,
        "stored_parameters": 17_942_400,
        "active_parameters": 10_006_400,
    },
}
DENSE_MODEL_IDS = ("dense_0p16m", "dense_0p64m", "dense_2p16m", "dense_10m")


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def pair_ids_to_operands(pair_ids):
    pair_ids = np.asarray(pair_ids, dtype=np.int32)
    return pair_ids // 1000, pair_ids % 1000


def operand_lengths(values):
    values = np.asarray(values)
    return np.where(values < 10, 1, np.where(values < 100, 2, 3)).astype(np.int8)


def carry_codes(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    units = ((a % 10) + (b % 10) >= 10).astype(np.int8)
    tens = (((a // 10) % 10) + ((b // 10) % 10) + units >= 10).astype(np.int8)
    hundreds = (((a // 100) % 10) + ((b // 100) % 10) + tens >= 10).astype(np.int8)
    return (4 * units + 2 * tens + hundreds).astype(np.int8)


def stratum_codes(a, b):
    return (
        ((operand_lengths(a) - 1) * 3 + (operand_lengths(b) - 1)) * 8 + carry_codes(a, b)
    ).astype(np.int16)


def make_sequences(pair_ids):
    pair_ids = np.asarray(pair_ids, dtype=np.int32)
    a, b = pair_ids_to_operands(pair_ids)
    total = a + b
    sequences = np.empty((len(pair_ids), MAX_SEQUENCE_LENGTH), dtype=np.uint8)
    sequences[:, 0] = a // 100
    sequences[:, 1] = (a // 10) % 10
    sequences[:, 2] = a % 10
    sequences[:, 3:6] = (10, 11, 10)
    sequences[:, 6] = b // 100
    sequences[:, 7] = (b // 10) % 10
    sequences[:, 8] = b % 10
    sequences[:, 9:12] = (10, 12, 10)
    sequences[:, 12] = total % 10
    sequences[:, 13] = (total // 10) % 10
    sequences[:, 14] = (total // 100) % 10
    sequences[:, 15] = (total // 1000) % 10
    return sequences


def largest_remainder(counts, total, capacity):
    ideal = counts.astype(np.float64) * total / counts.sum()
    allocated = np.minimum(np.floor(ideal).astype(np.int64), capacity)
    order = np.lexsort((np.arange(len(counts)), -(ideal - allocated)))
    remaining = total - int(allocated.sum())
    while remaining:
        eligible = order[allocated[order] < capacity[order]]
        take = eligible[:remaining]
        allocated[take] += 1
        remaining -= len(take)
    return allocated


def build_split(seed=SEED):
    pair_ids = np.arange(1_000_000, dtype=np.int32)
    a, b = pair_ids_to_operands(pair_ids)
    codes = stratum_codes(a, b)
    unique_codes, counts = np.unique(codes, return_counts=True)
    train_counts = largest_remainder(counts, TRAIN_SIZE, counts)
    validation_counts = largest_remainder(counts, VALIDATION_SIZE, counts - train_counts)
    rng = np.random.default_rng(seed)
    train_parts, validation_parts, test_parts = [], [], []
    for code_value, train_count, validation_count in zip(
        unique_codes, train_counts, validation_counts, strict=True
    ):
        members = pair_ids[codes == code_value].copy()
        rng.shuffle(members)
        train_parts.append(members[:train_count])
        validation_parts.append(members[train_count : train_count + validation_count])
        test_parts.append(members[train_count + validation_count :])
    train_ids = np.concatenate(train_parts)
    validation_ids = np.concatenate(validation_parts)
    test_ids = np.concatenate(test_parts)
    rng.shuffle(train_ids)
    rng.shuffle(validation_ids)
    rng.shuffle(test_ids)
    assert (len(train_ids), len(validation_ids), len(test_ids)) == (
        TRAIN_SIZE,
        VALIDATION_SIZE,
        TEST_SIZE,
    )
    assert len(np.unique(np.concatenate([train_ids, validation_ids, test_ids]))) == 1_000_000
    return train_ids, validation_ids, test_ids


class HybridBatcher:
    def __init__(self, sequences, strata, batch_size, seed):
        self.sequences = sequences
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.stratum_indices = [
            np.flatnonzero(strata == code) for code in np.unique(strata)
        ]

    def sample(self):
        natural_count = self.batch_size // 2
        balanced_count = self.batch_size - natural_count
        natural = self.rng.integers(0, len(self.sequences), size=natural_count)
        chosen_strata = self.rng.integers(0, len(self.stratum_indices), size=balanced_count)
        balanced = np.empty(balanced_count, dtype=np.int64)
        for stratum in np.unique(chosen_strata):
            locations = np.flatnonzero(chosen_strata == stratum)
            balanced[locations] = self.rng.choice(
                self.stratum_indices[int(stratum)], size=len(locations), replace=True
            )
        indices = np.concatenate([natural, balanced])
        self.rng.shuffle(indices)
        batch = self.sequences[indices].astype(np.int32)
        return batch[:, :-1], batch[:, 1:]


def normal_parameter(rngs, shape, scale):
    values = jax.random.normal(rngs.params(), shape, dtype=jnp.float32) * scale
    return nnx.Param(values.astype(PARAM_DTYPE))


def matrix_multiply(x, weight):
    x16 = x.astype(COMPUTE_DTYPE)
    w16 = weight.astype(COMPUTE_DTYPE)
    return jax.lax.dot_general(
        x16,
        w16,
        (((x16.ndim - 1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )


class Linear(nnx.Module):
    def __init__(self, input_width, output_width, rngs, scale=0.02):
        self.kernel = normal_parameter(rngs, (input_width, output_width), scale)

    def __call__(self, x):
        return matrix_multiply(x, self.kernel.value)


class LayerNorm(nnx.Module):
    def __init__(self, width):
        self.scale = nnx.Param(jnp.ones((width,), dtype=PARAM_DTYPE))
        self.bias = nnx.Param(jnp.zeros((width,), dtype=PARAM_DTYPE))

    def __call__(self, x):
        x = x.astype(jnp.float32)
        mean = jnp.mean(x, axis=-1, keepdims=True)
        variance = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
        return (x - mean) * jax.lax.rsqrt(
            variance + 1e-5
        ) * self.scale.value + self.bias.value


def gelu(x):
    return 0.5 * x * (1.0 + jax.lax.erf(x / math.sqrt(2.0)))


CAUSAL_MASK = jnp.tril(jnp.ones((MODEL_INPUT_LENGTH, MODEL_INPUT_LENGTH), dtype=bool))


class CausalMHA(nnx.Module):
    def __init__(self, config, rngs):
        width = config["d_model"]
        residual_scale = 0.02 / math.sqrt(2 * config["n_layers"])
        self.width = width
        self.heads = config["n_heads"]
        self.head_dim = width // self.heads
        self.q_proj = Linear(width, width, rngs)
        self.k_proj = Linear(width, width, rngs)
        self.v_proj = Linear(width, width, rngs)
        self.out_proj = Linear(width, width, rngs, scale=residual_scale)

    def __call__(self, x):
        batch, length, _ = x.shape
        q = self.q_proj(x).reshape(batch, length, self.heads, self.head_dim)
        k = self.k_proj(x).reshape(batch, length, self.heads, self.head_dim)
        v = self.v_proj(x).reshape(batch, length, self.heads, self.head_dim)
        scores = jnp.einsum(
            "bthd,bshd->bhts",
            q.astype(jnp.float32),
            k.astype(jnp.float32),
            preferred_element_type=jnp.float32,
        ) / math.sqrt(self.head_dim)
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
        return self.out_proj(attended.reshape(batch, length, self.width))


class ModuleSequence(nnx.Module):
    def __init__(self, layers):
        self.length = len(layers)
        for index, layer in enumerate(layers):
            setattr(self, f"layer_{index}", layer)

    def __iter__(self):
        return (getattr(self, f"layer_{index}") for index in range(self.length))


class RaggedTop2MoE(nnx.Module):
    def __init__(self, config, rngs):
        width = config["d_model"]
        expert_width = config["expert_d_ff"]
        residual_scale = 0.02 / math.sqrt(2 * config["n_layers"])
        self.width = width
        self.router = Linear(width, N_EXPERTS, rngs)
        self.up_kernel = normal_parameter(rngs, (N_EXPERTS, width, expert_width), 0.02)
        self.down_kernel = normal_parameter(
            rngs, (N_EXPERTS, expert_width, width), residual_scale
        )

    def __call__(self, x):
        original_shape = x.shape
        flat_x = x.reshape(-1, self.width).astype(jnp.float32)
        token_count = flat_x.shape[0]
        router_logits = self.router(flat_x).astype(jnp.float32)
        router_probabilities = jax.nn.softmax(router_logits, axis=-1)
        top_logits, top_expert_ids = jax.lax.top_k(router_logits, TOP_K)
        top_gate_weights = jax.nn.softmax(top_logits, axis=-1)
        route_token_ids = jnp.repeat(jnp.arange(token_count, dtype=jnp.int32), TOP_K)
        route_expert_ids = top_expert_ids.reshape(-1).astype(jnp.int32)
        route_weights = top_gate_weights.reshape(-1).astype(jnp.float32)
        route_order = jnp.argsort(route_expert_ids, stable=True)
        sorted_token_ids = route_token_ids[route_order]
        sorted_expert_ids = route_expert_ids[route_order]
        sorted_weights = route_weights[route_order]
        sorted_inputs = flat_x[sorted_token_ids].astype(COMPUTE_DTYPE)
        group_sizes = jnp.bincount(sorted_expert_ids, length=N_EXPERTS).astype(jnp.int32)
        hidden = jax.lax.ragged_dot(
            sorted_inputs,
            self.up_kernel.value.astype(COMPUTE_DTYPE),
            group_sizes,
            preferred_element_type=jnp.float32,
        )
        hidden = gelu(hidden)
        sorted_outputs = jax.lax.ragged_dot(
            hidden.astype(COMPUTE_DTYPE),
            self.down_kernel.value.astype(COMPUTE_DTYPE),
            group_sizes,
            preferred_element_type=jnp.float32,
        )
        combined = jnp.zeros((token_count, self.width), dtype=jnp.float32)
        combined = combined.at[sorted_token_ids].add(
            sorted_outputs * sorted_weights[:, None]
        )
        top1_fraction = jnp.mean(
            jax.nn.one_hot(top_expert_ids[:, 0], N_EXPERTS, dtype=jnp.float32), axis=0
        )
        route_fraction = jnp.mean(
            jax.nn.one_hot(route_expert_ids, N_EXPERTS, dtype=jnp.float32), axis=0
        )
        router_importance = jnp.mean(router_probabilities, axis=0)
        balance_loss = N_EXPERTS * jnp.sum(top1_fraction * router_importance)
        z_loss = jnp.mean(jnp.square(jax.nn.logsumexp(router_logits, axis=-1)))
        entropy = -jnp.mean(
            jnp.sum(
                router_probabilities * jnp.log(jnp.maximum(router_probabilities, 1e-9)),
                axis=-1,
            )
        )
        diagnostics = {
            "balance_loss": balance_loss,
            "z_loss": z_loss,
            "router_entropy": entropy,
            "top1_fraction": top1_fraction,
            "route_fraction": route_fraction,
            "router_importance": router_importance,
        }
        return combined.reshape(original_shape), diagnostics


class MoETransformerBlock(nnx.Module):
    def __init__(self, config, rngs):
        self.attention_norm = LayerNorm(config["d_model"])
        self.attention = CausalMHA(config, rngs)
        self.moe_norm = LayerNorm(config["d_model"])
        self.moe = RaggedTop2MoE(config, rngs)

    def __call__(self, x):
        x = x + self.attention(self.attention_norm(x))
        moe_output, diagnostics = self.moe(self.moe_norm(x))
        return x + moe_output, diagnostics


class MoEAdditionTransformer(nnx.Module):
    def __init__(self, config, rngs):
        width = config["d_model"]
        self.token_embedding = normal_parameter(rngs, (VOCAB_SIZE, width), 0.02)
        self.position_embedding = normal_parameter(
            rngs, (MODEL_INPUT_LENGTH, width), 0.02
        )
        self.blocks = ModuleSequence(
            [MoETransformerBlock(config, rngs) for _ in range(config["n_layers"])]
        )
        self.final_norm = LayerNorm(width)

    def __call__(self, token_ids, return_routing=False):
        x = self.token_embedding.value[token_ids]
        x = x + self.position_embedding.value[None, :, :]
        layers = []
        for block in self.blocks:

            def apply_block(value, current_block=block):
                return current_block(value)

            x, diagnostics = jax.checkpoint(apply_block)(x)
            layers.append(diagnostics)
        x = self.final_norm(x)
        logits = jnp.einsum(
            "btd,vd->btv",
            x.astype(jnp.float32),
            self.token_embedding.value.astype(jnp.float32),
            preferred_element_type=jnp.float32,
        )
        if not return_routing:
            return logits
        routing = {key: jnp.stack([layer[key] for layer in layers]) for key in layers[0]}
        for key in ("balance_loss", "z_loss", "router_entropy"):
            routing[key] = jnp.mean(routing[key])
        return logits, routing


def moe_parameter_formulas(config):
    layers, width, expert_width = (
        config["n_layers"],
        config["d_model"],
        config["expert_d_ff"],
    )
    shared = layers * (4 * width * width + 4 * width) + (
        VOCAB_SIZE * width + MODEL_INPUT_LENGTH * width + 2 * width
    )
    router = layers * width * N_EXPERTS
    stored = shared + router + layers * N_EXPERTS * 2 * width * expert_width
    active = shared + router + layers * TOP_K * 2 * width * expert_width
    return stored, active


ANSWER_MASK = jnp.arange(MODEL_INPUT_LENGTH) >= MODEL_INPUT_LENGTH - ANSWER_DIGITS


def answer_loss(logits, targets):
    log_probabilities = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    token_losses = -jnp.take_along_axis(
        log_probabilities, targets[..., None], axis=-1
    ).squeeze(-1)
    denominator = targets.shape[0] * ANSWER_DIGITS
    loss = jnp.sum(jnp.where(ANSWER_MASK[None, :], token_losses, 0.0)) / denominator
    predictions = jnp.argmax(logits, axis=-1)
    accuracy = jnp.sum(
        jnp.where(ANSWER_MASK[None, :], predictions == targets, False)
    ) / denominator
    return loss, accuracy


def greedy_generate(model, prompts):
    buffer = jnp.zeros((prompts.shape[0], MODEL_INPUT_LENGTH), dtype=jnp.int32)
    buffer = buffer.at[:, :PROMPT_LENGTH].set(prompts)

    def generate_digit(current_buffer, offset):
        logits = model(current_buffer)
        next_token = jnp.argmax(logits[:, PROMPT_LENGTH + offset - 1, :], axis=-1)
        current_buffer = jax.lax.cond(
            offset < ANSWER_DIGITS - 1,
            lambda value: value.at[:, PROMPT_LENGTH + offset].set(next_token),
            lambda value: value,
            current_buffer,
        )
        return current_buffer, next_token.astype(jnp.int32)

    _, generated = jax.lax.scan(generate_digit, buffer, jnp.arange(ANSWER_DIGITS))
    generated = jnp.swapaxes(generated, 0, 1)
    return generated, jnp.all(generated < 10, axis=-1)


def make_learning_rate(horizon):
    warmup = max(1, horizon // 10)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=PEAK_LR,
        warmup_steps=warmup,
        decay_steps=horizon - 1,
        end_value=FINAL_LR,
    )
    assert float(schedule(0)) == 0.0
    np.testing.assert_allclose(float(schedule(horizon - 1)), FINAL_LR, rtol=1e-5)
    return schedule


def path_parts(path):
    return tuple(str(getattr(entry, "key", getattr(entry, "idx", entry))) for entry in path)


def all_finite(tree):
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(tree)]))


def build_moe_training(config, horizon):
    model = MoEAdditionTransformer(config, nnx.Rngs(params=SEED))
    real_count = sum(int(leaf.size) for leaf in jax.tree.leaves(nnx.state(model, nnx.Param)))
    stored, active = moe_parameter_formulas(config)
    if real_count != stored or stored != config["stored_parameters"]:
        raise AssertionError(f"stored parameter mismatch: {real_count}, {stored}, {config}")
    if active != config["active_parameters"]:
        raise AssertionError(f"active parameter mismatch: {active}, {config}")
    graphdef, params = nnx.split(model, nnx.Param)
    decay_mask = jax.tree_util.tree_map_with_path(
        lambda path, leaf: leaf.ndim >= 2
        and (
            "attention" in path_parts(path)
            or "up_kernel" in path_parts(path)
            or "down_kernel" in path_parts(path)
        ),
        params,
    )
    learning_rate = make_learning_rate(horizon)
    optimizer = optax.chain(
        optax.clip_by_global_norm(GRAD_CLIP),
        optax.scale_by_adam(b1=0.9, b2=0.99, eps=1e-8),
        optax.masked(optax.add_decayed_weights(WEIGHT_DECAY), decay_mask),
        optax.scale_by_learning_rate(learning_rate),
    )
    optimizer_state = optimizer.init(params)

    @jax.jit
    def train_step(params, optimizer_state, inputs, targets):
        def objective(candidate_params):
            model = nnx.merge(graphdef, candidate_params)
            logits, routing = model(inputs, return_routing=True)
            language_loss, accuracy = answer_loss(logits, targets)
            total = (
                language_loss
                + ROUTER_BALANCE_COEFFICIENT * routing["balance_loss"]
                + ROUTER_Z_LOSS_COEFFICIENT * routing["z_loss"]
            )
            return total, {
                "language_loss": language_loss,
                "answer_token_accuracy": accuracy,
                **routing,
            }

        (loss, auxiliary), gradients = jax.value_and_grad(objective, has_aux=True)(params)
        gradient_norm = optax.global_norm(gradients)
        updates, new_optimizer_state = optimizer.update(gradients, optimizer_state, params)
        new_params = optax.apply_updates(params, updates)
        finite = jnp.isfinite(loss) & all_finite(gradients) & all_finite(new_params)
        safe_params = jax.tree.map(
            lambda new, old: jnp.where(finite, new, old), new_params, params
        )
        safe_state = jax.tree.map(
            lambda new, old: jnp.where(finite, new, old),
            new_optimizer_state,
            optimizer_state,
        )
        return safe_params, safe_state, {
            "loss": loss,
            "gradient_norm": gradient_norm,
            "finite": finite,
            **auxiliary,
        }

    @jax.jit
    def evaluate_batch(params, inputs, targets):
        model = nnx.merge(graphdef, params)
        logits, routing = model(inputs, return_routing=True)
        loss, accuracy = answer_loss(logits, targets)
        return loss, accuracy, routing

    @jax.jit
    def generate_batch(params, prompts):
        return greedy_generate(nnx.merge(graphdef, params), prompts)

    return params, optimizer_state, learning_rate, train_step, evaluate_batch, generate_batch


def evaluate_moe_validation(params, evaluate_batch, generate_batch, validation_ids):
    started = time.perf_counter()
    loss_total = token_total = exact_total = 0.0
    routing_batches = []
    for start in range(0, len(validation_ids), EVAL_BATCH_SIZE):
        ids = validation_ids[start : start + EVAL_BATCH_SIZE]
        sequences = make_sequences(ids).astype(np.int32)
        inputs = jnp.asarray(sequences[:, :-1])
        targets = jnp.asarray(sequences[:, 1:])
        loss, token_accuracy, routing = evaluate_batch(params, inputs, targets)
        generated, valid = generate_batch(params, inputs[:, :PROMPT_LENGTH])
        generated, valid = np.asarray(generated), np.asarray(valid)
        exact = valid & np.all(generated == sequences[:, -ANSWER_DIGITS:], axis=1)
        loss_total += float(loss) * len(ids)
        token_total += float(token_accuracy) * len(ids)
        exact_total += int(exact.sum())
        routing_batches.append(jax.device_get(routing))
    routing = {
        key: np.mean(
            np.stack([np.asarray(batch[key]) for batch in routing_batches]), axis=0
        )
        for key in routing_batches[0]
    }
    return {
        "validation_loss": loss_total / len(validation_ids),
        "validation_token_accuracy": token_total / len(validation_ids),
        "validation_exact_match": exact_total / len(validation_ids),
        "validation_seconds": time.perf_counter() - started,
        "router_entropy": float(routing["router_entropy"]),
        "balance_loss": float(routing["balance_loss"]),
        "z_loss": float(routing["z_loss"]),
        "top1_fraction": np.asarray(routing["top1_fraction"]).tolist(),
        "route_fraction": np.asarray(routing["route_fraction"]).tolist(),
        "router_importance": np.asarray(routing["router_importance"]).tolist(),
    }


def expected_run_config(model_id, config, horizon):
    return {
        "run_id": f"{model_id}_h{horizon:04d}",
        "model_id": model_id,
        "horizon": horizon,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "model": config,
        "experts": N_EXPERTS,
        "top_k": TOP_K,
    }


def checkpoint_parameter_count(params) -> int:
    return sum(int(np.asarray(leaf).size) for leaf in jax.tree.leaves(params))


def run_moe_coordinate(model_id, horizon, run_dir, smoke_test=False):
    config = MOE_CONFIGS[model_id]
    run_config = expected_run_config(model_id, config, horizon)
    run_id = run_config["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    latest_path = run_dir / "latest_checkpoint.pkl"
    best_path = run_dir / "best_checkpoint.pkl"
    history_path = run_dir / "history.json"
    result_path = run_dir / "result.json"
    package_checkpoint = run_dir / "checkpoints" / "latest"

    if result_path.exists() and not smoke_test:
        completed = json.loads(result_path.read_text())
        if completed.get("status") == "complete" and completed.get("run_config") == run_config:
            print(f"skip {run_id}: complete", flush=True)
            return completed
        if completed.get("status") == "complete":
            raise ValueError(f"completed result protocol mismatch: {run_id}")
    if package_checkpoint.exists() and not latest_path.exists():
        raise ValueError(
            f"{run_id} has a package checkpoint, not a notebook-04 pickle checkpoint; "
            "refusing to mix scientific protocols"
        )

    train_ids, validation_ids, _ = build_split()
    train_sequences = make_sequences(train_ids)
    train_a, train_b = pair_ids_to_operands(train_ids)
    train_strata = stratum_codes(train_a, train_b)
    params, optimizer_state, learning_rate, train_step, evaluate_batch, generate_batch = (
        build_moe_training(config, horizon)
    )
    batcher = HybridBatcher(train_sequences, train_strata, BATCH_SIZE, SEED)
    history, start_step = [], 0
    optimizer_seconds = compilation_seconds = 0.0
    if latest_path.exists() and not smoke_test:
        checkpoint = pickle.loads(latest_path.read_bytes())
        if checkpoint.get("run_config") != run_config:
            raise ValueError(f"checkpoint configuration mismatch: {run_id}")
        if checkpoint_parameter_count(checkpoint["params"]) != config["stored_parameters"]:
            raise ValueError(f"checkpoint parameter-count mismatch: {run_id}")
        params = jax.tree.map(jnp.asarray, checkpoint["params"])
        optimizer_state = jax.tree.map(jnp.asarray, checkpoint["optimizer_state"])
        batcher.rng.bit_generator.state = checkpoint["batcher_rng_state"]
        history, start_step = checkpoint["history"], int(checkpoint["step"])
        optimizer_seconds = float(checkpoint["optimizer_seconds"])
        compilation_seconds = float(checkpoint["compilation_seconds"])
        print(f"resume {run_id} after step {start_step}", flush=True)
    else:
        print(f"start {run_id}", flush=True)

    final_step = min(horizon, 1) if smoke_test else horizon
    checkpoint_every, log_every = min(50, horizon), min(25, horizon)
    for step in range(start_step + 1, final_step + 1):
        inputs, targets = batcher.sample()
        before = time.perf_counter()
        params, optimizer_state, metrics = train_step(
            params, optimizer_state, jnp.asarray(inputs), jnp.asarray(targets)
        )
        jax.block_until_ready(metrics["loss"])
        elapsed = time.perf_counter() - before
        optimizer_seconds += elapsed
        if step == start_step + 1:
            compilation_seconds += elapsed
        if step == start_step + 1 or step % log_every == 0 or step == horizon:
            host = jax.device_get(metrics)
            if not bool(host["finite"]):
                raise FloatingPointError(f"non-finite values at {run_id} step {step}")
            record = {
                "step": step,
                "loss": float(host["loss"]),
                "language_loss": float(host["language_loss"]),
                "answer_token_accuracy": float(host["answer_token_accuracy"]),
                "gradient_norm": float(host["gradient_norm"]),
                "balance_loss": float(host["balance_loss"]),
                "z_loss": float(host["z_loss"]),
                "router_entropy": float(host["router_entropy"]),
                "route_fraction": np.asarray(host["route_fraction"]).tolist(),
                "learning_rate": float(learning_rate(step - 1)),
                "step_seconds": elapsed,
            }
            history.append(record)
            print(
                f"{run_id} | step {step:4d}/{horizon} | "
                f"LM {record['language_loss']:.4f} | "
                f"token acc {100 * record['answer_token_accuracy']:6.2f}% | "
                f"entropy {record['router_entropy']:.3f} | {elapsed:.3f}s",
                flush=True,
            )
        if not smoke_test and (step % checkpoint_every == 0 or step == horizon):
            payload = {
                "run_config": run_config,
                "step": step,
                "params": jax.device_get(params),
                "optimizer_state": jax.device_get(optimizer_state),
                "batcher_rng_state": batcher.rng.bit_generator.state,
                "history": history,
                "optimizer_seconds": optimizer_seconds,
                "compilation_seconds": compilation_seconds,
            }
            atomic_write_bytes(
                latest_path, pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
            )
            atomic_write_json(history_path, history)
            print(f"saved checkpoint: {run_id} step {step}", flush=True)

    if smoke_test:
        smoke = {
            "status": "smoke_complete",
            "run_id": run_id,
            "seed": SEED,
            "steps": 1,
            "batch_size": BATCH_SIZE,
            "stored_parameters": config["stored_parameters"],
            "active_parameters": config["active_parameters"],
        }
        atomic_write_json(run_dir / "smoke_result.json", smoke)
        return smoke

    validation = evaluate_moe_validation(params, evaluate_batch, generate_batch, validation_ids)
    examples_seen = horizon * BATCH_SIZE
    result = {
        "status": "complete",
        "run_config": run_config,
        "run_id": run_id,
        "architecture": "moe",
        "model_id": model_id,
        "dense_reference": config["dense_reference"],
        "horizon": horizon,
        "stored_parameters": config["stored_parameters"],
        "active_parameters": config["active_parameters"],
        "examples_seen": examples_seen,
        "input_tokens": examples_seen * MODEL_INPUT_LENGTH,
        "answer_tokens": examples_seen * ANSWER_DIGITS,
        "estimated_training_flops": (
            6 * config["active_parameters"] * examples_seen * MODEL_INPUT_LENGTH
        ),
        "final_training_loss": history[-1]["language_loss"],
        "final_training_token_accuracy": history[-1]["answer_token_accuracy"],
        "optimizer_seconds": optimizer_seconds,
        "compilation_seconds": compilation_seconds,
        **validation,
    }
    atomic_write_bytes(
        best_path,
        pickle.dumps(
            {"run_config": run_config, "params": jax.device_get(params), "validation": validation},
            protocol=pickle.HIGHEST_PROTOCOL,
        ),
    )
    atomic_write_json(result_path, result)
    print(
        f"complete {run_id} | validation loss {validation['validation_loss']:.4f} | "
        f"exact {100 * validation['validation_exact_match']:.2f}%",
        flush=True,
    )
    return result


def preserve_completed_dense(model_id, horizon, run_dir):
    run_id = f"{model_id}_h{horizon:04d}"
    result_path = run_dir / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result.get("status") == "complete" and result.get("run_id") == run_id:
            print(f"skip {run_id}: complete", flush=True)
            return result
    raise RuntimeError(
        "dense training is locked because its sweep is complete; refusing to create a new "
        "result with a different implementation"
    )


def capture_oom_diagnostics(output_dir):
    try:
        snapshot = subprocess.run(
            ["nvidia-smi", "-q"], capture_output=True, text=True, timeout=30, check=False
        )
        (output_dir / "nvidia_smi_oom.txt").write_text(snapshot.stdout + snapshot.stderr)
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        jax.profiler.save_device_memory_profile(str(output_dir / "jax_device_memory.prof"))
    except Exception:
        pass


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("dense", "moe"), required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--horizon", type=int, choices=HORIZONS, required=True)
    parser.add_argument("--seed", type=int, choices=(SEED,), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    valid = DENSE_MODEL_IDS if args.family == "dense" else tuple(MOE_CONFIGS)
    if args.model_id not in valid:
        parser.error(f"{args.model_id!r} is not a {args.family} scaling model")
    if args.smoke_test and args.family != "moe":
        parser.error("--smoke-test is available only for MoE coordinates")
    return args


def main():
    args = parse_arguments()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        versions = {
            "jax": jax.__version__,
            "flax": flax.__version__,
            "optax": optax.__version__,
            "numpy": np.__version__,
        }
        expected_versions = {
            "jax": "0.7.2",
            "flax": "0.12.0",
            "optax": "0.2.6",
            "numpy": "2.3.3",
        }
        if versions != expected_versions:
            raise RuntimeError(
                f"runtime versions do not match notebook 04: {versions} != {expected_versions}"
            )
        if not any(device.platform == "gpu" for device in jax.devices()):
            raise RuntimeError("a GPU runtime is required for scaling workers")
        result = (
            preserve_completed_dense(args.model_id, args.horizon, output_dir)
            if args.family == "dense"
            else run_moe_coordinate(
                args.model_id, args.horizon, output_dir, smoke_test=args.smoke_test
            )
        )
        print(json.dumps(result, indent=2), flush=True)
    except Exception as error:
        message = "".join(traceback.format_exception(error))
        failure = {
            "status": "failed",
            "family": args.family,
            "model_id": args.model_id,
            "horizon": args.horizon,
            "seed": args.seed,
            "error": str(error),
            "traceback": message,
        }
        atomic_write_json(output_dir / "worker_failure.json", failure)
        if "out of memory" in message.lower() or "resource_exhausted" in message.lower():
            capture_oom_diagnostics(output_dir)
        print(message, file=sys.stderr, flush=True)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
