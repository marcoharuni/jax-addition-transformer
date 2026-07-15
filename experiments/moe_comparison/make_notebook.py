"""Generate the clean, standalone Colab T4 MoE training notebook."""

from __future__ import annotations

import hashlib
from pathlib import Path
from textwrap import dedent

import nbformat


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "notebooks" / "03_moe_ragged_dot_t4.ipynb"
DENSE_NOTEBOOK = ROOT / "notebooks" / "01_build_train_exact_10m.ipynb"
COLAB_URL = (
    "https://colab.research.google.com/github/marcoharuni/"
    "jax-addition-transformer/blob/moe-ragged-dot/"
    "notebooks/03_moe_ragged_dot_t4.ipynb"
)


def markdown(text: str):
    return nbformat.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbformat.v4.new_code_cell(dedent(text).strip())


def main() -> None:
    dense = nbformat.read(DENSE_NOTEBOOK, 4)
    dense_source = lambda index: dense.cells[index].source  # noqa: E731

    notebook = nbformat.v4.new_notebook()
    notebook.metadata = {
        "accelerator": "GPU",
        "colab": {"gpuType": "T4", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    notebook.cells = [
        markdown(
            f"""
            # Train the matched 10M-active top-2 MoE with `jax.lax.ragged_dot`

            [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB_URL})

            This is the MoE counterpart of the clean exact-10M dense notebook. It keeps the
            13-token vocabulary, fixed-length addition records, complete one-million-pair
            domain, stratified split, hybrid sampler, attention stack, answer-only objective,
            optimizer, checkpoint flow, greedy evaluation, and T4 precision policy aligned.

            The only architectural substitution is the feed-forward sublayer: four GELU
            experts, top-2 no-drop routing, expert width 1,240, and two grouped
            `jax.lax.ragged_dot` projections per layer. The model stores 17,942,400 parameters
            and has a 10,006,400 active-parameter proxy per token.

            The default run is seed 42 for 175 steps, matching the committed dense
            `dense_10m_s0175_seed42` baseline. Change only `SEED` to 123 or 2026 to produce the
            other independent comparison runs.
            """
        ),
        markdown("## Setup"),
        code(
            '%pip install -q -U "jax[cuda12]==0.7.2" "flax==0.12.0" '
            '"optax==0.2.6" "numpy==2.3.3" "matplotlib==3.10.7"'
        ),
        code(
            """
            import csv
            import json
            import math
            import os
            import pickle
            import re
            import shutil
            import subprocess
            import time
            from pathlib import Path

            CACHE_DIR = Path("/content/jax_compilation_cache/moe_ragged_dot")
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
            os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
            os.environ["JAX_COMPILATION_CACHE_DIR"] = str(CACHE_DIR)
            os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "0"

            import jax
            import jax.numpy as jnp
            import matplotlib.pyplot as plt
            import numpy as np
            import optax
            from flax import nnx

            print("JAX:", jax.__version__)
            print("Flax:", __import__("flax").__version__)
            print("Optax:", optax.__version__)
            print("Devices:", jax.devices())
            subprocess.run(["nvidia-smi"], check=True)

            assert jax.__version__ == "0.7.2"
            assert __import__("flax").__version__ == "0.12.0"
            assert optax.__version__ == "0.2.6"
            assert hasattr(jax.lax, "ragged_dot")
            assert any(device.platform == "gpu" for device in jax.devices()), (
                "Select Runtime → Change runtime type → T4 GPU, restart, and run all cells."
            )
            """
        ),
        code(
            """
            SEED = 42

            VOCAB_SIZE = 13
            MAX_SEQUENCE_LENGTH = 16
            MODEL_INPUT_LENGTH = 15
            PROMPT_LENGTH = 12
            ANSWER_DIGITS = 4

            N_LAYERS = 5
            D_MODEL = 320
            N_HEADS = 5
            HEAD_DIM = 64
            N_EXPERTS = 4
            TOP_K = 2
            EXPERT_D_FF = 1240

            TRAIN_SIZE = 200_000
            VALIDATION_SIZE = 20_000
            TEST_SIZE = 780_000

            BATCH_SIZE = 2048
            EVAL_BATCH_SIZE = 4000
            MAX_STEPS = 175
            LOG_EVERY = 25
            EVAL_EVERY = 175
            CHECKPOINT_EVERY = 25

            PEAK_LR = 1e-3
            FINAL_LR = 1e-4
            WARMUP_STEPS = 17
            WEIGHT_DECAY = 0.1
            GRAD_CLIP = 1.0
            ROUTER_BALANCE_COEFFICIENT = 1e-2
            ROUTER_Z_LOSS_COEFFICIENT = 1e-3

            PARAM_DTYPE = jnp.float32
            COMPUTE_DTYPE = jnp.float16
            START_FRESH = False

            assert D_MODEL == N_HEADS * HEAD_DIM
            assert TOP_K <= N_EXPERTS
            assert TRAIN_SIZE + VALIDATION_SIZE + TEST_SIZE == 1_000_000
            """
        ),
        markdown("## Checkpoint and artifact paths"),
        code(
            """
            from google.colab import drive

            drive.mount("/content/drive")
            RUN_DIR = Path(
                f"/content/drive/MyDrive/jax-addition-transformer/moe-ragged-dot/seed-{SEED}"
            )
            RUN_DIR.mkdir(parents=True, exist_ok=True)

            BEST_CHECKPOINT_PATH = RUN_DIR / "best_checkpoint.pkl"
            LATEST_CHECKPOINT_PATH = RUN_DIR / "latest_checkpoint.pkl"
            HISTORY_PATH = RUN_DIR / "history.json"
            RESULTS_PATH = RUN_DIR / "results.json"
            FAILURES_PATH = RUN_DIR / "failures.csv"
            CONFIG_PATH = RUN_DIR / "config.json"

            RUN_CONFIG = {
                "format_version": 1,
                "seed": SEED,
                "model": {
                    "layers": N_LAYERS,
                    "width": D_MODEL,
                    "heads": N_HEADS,
                    "experts": N_EXPERTS,
                    "top_k": TOP_K,
                    "expert_width": EXPERT_D_FF,
                },
                "training": {
                    "batch_size": BATCH_SIZE,
                    "max_steps": MAX_STEPS,
                    "warmup_steps": WARMUP_STEPS,
                    "peak_learning_rate": PEAK_LR,
                    "final_learning_rate": FINAL_LR,
                },
                "versions": {
                    "jax": jax.__version__,
                    "flax": __import__("flax").__version__,
                    "optax": optax.__version__,
                },
            }

            def atomic_write_bytes(path, payload):
                temporary = path.with_name(path.name + ".tmp")
                temporary.write_bytes(payload)
                os.replace(temporary, path)

            def atomic_write_text(path, text):
                temporary = path.with_name(path.name + ".tmp")
                temporary.write_text(text)
                os.replace(temporary, path)

            atomic_write_text(CONFIG_PATH, json.dumps(RUN_CONFIG, indent=2) + "\\n")
            print("Run directory:", RUN_DIR)
            """
        ),
        markdown("## Data"),
        code(dense_source(6)),
        code(dense_source(7)),
        code(dense_source(8)),
        code(dense_source(9)),
        markdown("## Model primitives"),
        code(
            """
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
                    normalized = (x - mean) * jax.lax.rsqrt(variance + 1e-5)
                    return normalized * self.scale.value + self.bias.value

            def gelu(x):
                return 0.5 * x * (1.0 + jax.lax.erf(x / math.sqrt(2.0)))

            CAUSAL_MASK = jnp.tril(
                jnp.ones((MODEL_INPUT_LENGTH, MODEL_INPUT_LENGTH), dtype=bool)
            )

            class CausalMHA(nnx.Module):
                def __init__(self, rngs):
                    residual_scale = 0.02 / math.sqrt(2 * N_LAYERS)
                    self.q_proj = Linear(D_MODEL, D_MODEL, rngs)
                    self.k_proj = Linear(D_MODEL, D_MODEL, rngs)
                    self.v_proj = Linear(D_MODEL, D_MODEL, rngs)
                    self.out_proj = Linear(D_MODEL, D_MODEL, rngs, scale=residual_scale)

                def __call__(self, x):
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

            class ModuleSequence(nnx.Module):
                def __init__(self, layers):
                    self.length = len(layers)
                    for index, layer in enumerate(layers):
                        setattr(self, f"layer_{index}", layer)

                def __iter__(self):
                    return (getattr(self, f"layer_{index}") for index in range(self.length))
            """
        ),
        markdown("## No-drop top-2 MoE with two ragged projections"),
        code(
            """
            class RaggedTop2MoE(nnx.Module):
                def __init__(self, rngs):
                    residual_scale = 0.02 / math.sqrt(2 * N_LAYERS)
                    self.router = Linear(D_MODEL, N_EXPERTS, rngs)
                    self.up_kernel = normal_parameter(
                        rngs, (N_EXPERTS, D_MODEL, EXPERT_D_FF), 0.02
                    )
                    self.down_kernel = normal_parameter(
                        rngs, (N_EXPERTS, EXPERT_D_FF, D_MODEL), residual_scale
                    )

                def __call__(self, x):
                    original_shape = x.shape
                    flat_x = x.reshape(-1, D_MODEL).astype(jnp.float32)
                    token_count = flat_x.shape[0]

                    router_logits = self.router(flat_x).astype(jnp.float32)
                    router_probabilities = jax.nn.softmax(router_logits, axis=-1)
                    top_logits, top_expert_ids = jax.lax.top_k(router_logits, TOP_K)
                    top_gate_weights = jax.nn.softmax(top_logits, axis=-1)

                    route_token_ids = jnp.repeat(
                        jnp.arange(token_count, dtype=jnp.int32), TOP_K
                    )
                    route_expert_ids = top_expert_ids.reshape(-1).astype(jnp.int32)
                    route_weights = top_gate_weights.reshape(-1).astype(jnp.float32)
                    route_order = jnp.argsort(route_expert_ids, stable=True)
                    sorted_token_ids = route_token_ids[route_order]
                    sorted_expert_ids = route_expert_ids[route_order]
                    sorted_weights = route_weights[route_order]
                    sorted_inputs = flat_x[sorted_token_ids].astype(COMPUTE_DTYPE)
                    group_sizes = jnp.bincount(
                        sorted_expert_ids, length=N_EXPERTS
                    ).astype(jnp.int32)

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

                    combined = jnp.zeros((token_count, D_MODEL), dtype=jnp.float32)
                    combined = combined.at[sorted_token_ids].add(
                        sorted_outputs * sorted_weights[:, None]
                    )

                    top1_mask = jax.nn.one_hot(
                        top_expert_ids[:, 0], N_EXPERTS, dtype=jnp.float32
                    )
                    route_mask = jax.nn.one_hot(
                        route_expert_ids, N_EXPERTS, dtype=jnp.float32
                    )
                    top1_fraction = jnp.mean(top1_mask, axis=0)
                    route_fraction = jnp.mean(route_mask, axis=0)
                    router_importance = jnp.mean(router_probabilities, axis=0)
                    balance_loss = N_EXPERTS * jnp.sum(
                        top1_fraction * router_importance
                    )
                    z_loss = jnp.mean(
                        jnp.square(jax.nn.logsumexp(router_logits, axis=-1))
                    )
                    router_entropy = -jnp.mean(
                        jnp.sum(
                            router_probabilities
                            * jnp.log(jnp.maximum(router_probabilities, 1e-9)),
                            axis=-1,
                        )
                    )
                    routing = {
                        "balance_loss": balance_loss,
                        "z_loss": z_loss,
                        "router_entropy": router_entropy,
                        "top1_fraction": top1_fraction,
                        "route_fraction": route_fraction,
                        "router_importance": router_importance,
                        "group_sizes": group_sizes,
                    }
                    return combined.reshape(original_shape), routing
            """
        ),
        markdown("## MoE Transformer and exact parameter accounting"),
        code(
            """
            class TransformerBlock(nnx.Module):
                def __init__(self, rngs):
                    self.attention_norm = LayerNorm(D_MODEL)
                    self.attention = CausalMHA(rngs)
                    self.moe_norm = LayerNorm(D_MODEL)
                    self.moe = RaggedTop2MoE(rngs)

                def __call__(self, x):
                    x = x + self.attention(self.attention_norm(x))
                    moe_output, routing = self.moe(self.moe_norm(x))
                    return x + moe_output, routing

            class AdditionMoETransformer(nnx.Module):
                def __init__(self, rngs):
                    self.token_embedding = normal_parameter(
                        rngs, (VOCAB_SIZE, D_MODEL), 0.02
                    )
                    self.position_embedding = normal_parameter(
                        rngs, (MODEL_INPUT_LENGTH, D_MODEL), 0.02
                    )
                    self.blocks = ModuleSequence(
                        [TransformerBlock(rngs) for _ in range(N_LAYERS)]
                    )
                    self.final_norm = LayerNorm(D_MODEL)

                def __call__(self, token_ids, return_routing=False):
                    assert token_ids.ndim == 2
                    assert token_ids.shape[1] == MODEL_INPUT_LENGTH
                    x = self.token_embedding.value[token_ids]
                    x = x + self.position_embedding.value[None, :, :]

                    per_layer_routing = []
                    for block in self.blocks:
                        def apply_block(value, current_block=block):
                            return current_block(value)

                        x, routing = jax.checkpoint(apply_block)(x)
                        per_layer_routing.append(routing)

                    x = self.final_norm(x)
                    logits = jnp.einsum(
                        "btd,vd->btv",
                        x.astype(jnp.float32),
                        self.token_embedding.value.astype(jnp.float32),
                        preferred_element_type=jnp.float32,
                    )
                    if not return_routing:
                        return logits
                    routing = {
                        key: jnp.stack([layer[key] for layer in per_layer_routing])
                        for key in per_layer_routing[0]
                    }
                    for key in ("balance_loss", "z_loss", "router_entropy"):
                        routing[key] = jnp.mean(routing[key])
                    return logits, routing

            model = AdditionMoETransformer(nnx.Rngs(params=SEED))
            parameter_state = nnx.state(model, nnx.Param)
            parameter_count = sum(
                int(leaf.size) for leaf in jax.tree.leaves(parameter_state)
            )
            parameter_rows = [
                ("attention projections", N_LAYERS * 4 * D_MODEL * D_MODEL),
                ("expert up/down matrices", N_LAYERS * N_EXPERTS * 2 * D_MODEL * EXPERT_D_FF),
                ("router matrices", N_LAYERS * D_MODEL * N_EXPERTS),
                ("block LayerNorms", N_LAYERS * 4 * D_MODEL),
                ("token embedding / tied head", VOCAB_SIZE * D_MODEL),
                ("position embedding", MODEL_INPUT_LENGTH * D_MODEL),
                ("final LayerNorm", 2 * D_MODEL),
            ]
            active_proxy = (
                N_LAYERS * 4 * D_MODEL * D_MODEL
                + N_LAYERS * TOP_K * 2 * D_MODEL * EXPERT_D_FF
                + N_LAYERS * D_MODEL * N_EXPERTS
                + N_LAYERS * 4 * D_MODEL
                + VOCAB_SIZE * D_MODEL
                + MODEL_INPUT_LENGTH * D_MODEL
                + 2 * D_MODEL
            )
            for name, count in parameter_rows:
                print(f"{name:31} {count:>11,}")
            print("-" * 45)
            print(f"{'stored total':31} {parameter_count:>11,}")
            print(f"{'active-parameter proxy':31} {active_proxy:>11,}")

            assert sum(count for _, count in parameter_rows) == 17_942_400
            assert parameter_count == 17_942_400
            assert active_proxy == 10_006_400
            """
        ),
        markdown("## Mandatory `ragged_dot` correctness preflight"),
        code(
            """
            group_sizes = jnp.asarray([3, 0, 5, 8], dtype=jnp.int32)
            key_x, key_w = jax.random.split(jax.random.key(7))
            lhs = jax.random.normal(key_x, (16, 16), dtype=jnp.float16)
            rhs = jax.random.normal(
                key_w, (N_EXPERTS, 16, 12), dtype=jnp.float16
            )

            @jax.jit
            def preflight_ragged_dot(lhs, rhs, sizes):
                return jax.lax.ragged_dot(
                    lhs, rhs, sizes, preferred_element_type=jnp.float32
                )

            actual = preflight_ragged_dot(lhs, rhs, group_sizes)
            offsets = np.concatenate([[0], np.cumsum(np.asarray(group_sizes))])
            reference = jnp.concatenate([
                lhs[offsets[e] : offsets[e + 1]].astype(jnp.float32)
                @ rhs[e].astype(jnp.float32)
                for e in range(N_EXPERTS)
            ])
            np.testing.assert_allclose(actual, reference, rtol=1e-2, atol=1e-2)

            def preflight_objective(test_lhs, test_rhs):
                output = jax.lax.ragged_dot(
                    test_lhs,
                    test_rhs,
                    group_sizes,
                    preferred_element_type=jnp.float32,
                )
                return jnp.mean(jnp.square(output))

            lhs_grad, rhs_grad = jax.grad(
                preflight_objective, argnums=(0, 1)
            )(lhs, rhs)
            assert bool(jnp.all(jnp.isfinite(lhs_grad)))
            assert bool(jnp.all(jnp.isfinite(rhs_grad)))
            print("ragged_dot forward, empty-group, and gradient checks passed")
            """
        ),
        markdown("## Train"),
        code(
            """
            ANSWER_MASK = (
                jnp.arange(MODEL_INPUT_LENGTH) >= MODEL_INPUT_LENGTH - ANSWER_DIGITS
            )

            def answer_loss(logits, targets):
                log_probabilities = jax.nn.log_softmax(
                    logits.astype(jnp.float32), axis=-1
                )
                token_losses = -jnp.take_along_axis(
                    log_probabilities, targets[..., None], axis=-1
                ).squeeze(-1)
                denominator = targets.shape[0] * ANSWER_DIGITS
                loss = jnp.sum(
                    jnp.where(ANSWER_MASK[None, :], token_losses, 0.0)
                ) / denominator
                predictions = jnp.argmax(logits, axis=-1)
                accuracy = jnp.sum(
                    jnp.where(ANSWER_MASK[None, :], predictions == targets, False)
                ) / denominator
                return loss, accuracy

            graphdef, params = nnx.split(model, nnx.Param)

            def path_parts(path):
                return tuple(
                    str(getattr(entry, "key", getattr(entry, "idx", entry)))
                    for entry in path
                )

            def should_decay(path, leaf):
                parts = path_parts(path)
                return leaf.ndim >= 2 and (
                    "attention" in parts
                    or "up_kernel" in parts
                    or "down_kernel" in parts
                )

            decay_mask = jax.tree_util.tree_map_with_path(should_decay, params)
            learning_rate = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=PEAK_LR,
                warmup_steps=WARMUP_STEPS,
                decay_steps=MAX_STEPS,
                end_value=FINAL_LR,
            )
            optimizer = optax.chain(
                optax.clip_by_global_norm(GRAD_CLIP),
                optax.scale_by_adam(b1=0.9, b2=0.99, eps=1e-8),
                optax.masked(optax.add_decayed_weights(WEIGHT_DECAY), decay_mask),
                optax.scale_by_learning_rate(learning_rate),
            )
            optimizer_state = optimizer.init(params)

            def all_finite(tree):
                return jnp.all(jnp.stack([
                    jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(tree)
                ]))

            @jax.jit
            def train_step(params, optimizer_state, inputs, targets):
                def objective(candidate_params):
                    current_model = nnx.merge(graphdef, candidate_params)
                    logits, routing = current_model(inputs, return_routing=True)
                    language_model_loss, accuracy = answer_loss(logits, targets)
                    total_loss = (
                        language_model_loss
                        + ROUTER_BALANCE_COEFFICIENT * routing["balance_loss"]
                        + ROUTER_Z_LOSS_COEFFICIENT * routing["z_loss"]
                    )
                    auxiliary = {
                        "language_model_loss": language_model_loss,
                        "answer_token_accuracy": accuracy,
                        **routing,
                    }
                    return total_loss, auxiliary

                (loss, auxiliary), gradients = jax.value_and_grad(
                    objective, has_aux=True
                )(params)
                gradient_norm = optax.global_norm(gradients)
                updates, new_optimizer_state = optimizer.update(
                    gradients, optimizer_state, params
                )
                new_params = optax.apply_updates(params, updates)
                finite = (
                    jnp.isfinite(loss)
                    & all_finite(gradients)
                    & all_finite(new_params)
                )
                safe_params = jax.tree.map(
                    lambda new, old: jnp.where(finite, new, old),
                    new_params,
                    params,
                )
                safe_optimizer_state = jax.tree.map(
                    lambda new, old: jnp.where(finite, new, old),
                    new_optimizer_state,
                    optimizer_state,
                )
                metrics = {
                    "loss": loss,
                    "gradient_norm": gradient_norm,
                    "finite": finite,
                    **auxiliary,
                }
                return safe_params, safe_optimizer_state, metrics
            """
        ),
        code(dense_source(16)),
        markdown("## Restore an interrupted run or initialize a fresh run"),
        code(
            """
            def checkpoint_payload(step, validation=None):
                return {
                    "run_config": RUN_CONFIG,
                    "step": int(step),
                    "params": jax.device_get(params),
                    "optimizer_state": jax.device_get(optimizer_state),
                    "batcher_rng_state": batcher.rng.bit_generator.state,
                    "history": history,
                    "validation_history": validation_history,
                    "best_exact_match": float(best_exact_match),
                    "best_validation_loss": float(best_validation_loss),
                    "validation": validation,
                    "stored_parameter_count": parameter_count,
                    "active_parameter_proxy": active_proxy,
                }

            def save_checkpoint(path, step, validation=None):
                payload = checkpoint_payload(step, validation)
                atomic_write_bytes(
                    path, pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
                )

            def assert_checkpoint_compatible(payload):
                if payload.get("run_config") != RUN_CONFIG:
                    raise ValueError(
                        "Checkpoint configuration mismatch. Set START_FRESH = True "
                        "or select the matching seed/run directory."
                    )
                assert payload["stored_parameter_count"] == 17_942_400
                assert payload["active_parameter_proxy"] == 10_006_400

            if START_FRESH:
                for artifact in (
                    BEST_CHECKPOINT_PATH,
                    LATEST_CHECKPOINT_PATH,
                    HISTORY_PATH,
                    RESULTS_PATH,
                    FAILURES_PATH,
                ):
                    artifact.unlink(missing_ok=True)

            if LATEST_CHECKPOINT_PATH.exists() and not START_FRESH:
                restored = pickle.loads(LATEST_CHECKPOINT_PATH.read_bytes())
                assert_checkpoint_compatible(restored)
                params = jax.tree.map(jnp.asarray, restored["params"])
                optimizer_state = jax.tree.map(
                    jnp.asarray, restored["optimizer_state"]
                )
                batcher.rng.bit_generator.state = restored["batcher_rng_state"]
                history = restored["history"]
                validation_history = restored["validation_history"]
                best_exact_match = restored["best_exact_match"]
                best_validation_loss = restored["best_validation_loss"]
                start_step = restored["step"]
                print(f"Resuming after step {start_step}")
            else:
                history = []
                validation_history = []
                best_exact_match = -1.0
                best_validation_loss = math.inf
                start_step = 0
                print("Starting a fresh run")
            """
        ),
        code(
            """
            compile_seconds = 0.0
            training_started = time.perf_counter()
            last_log_time = training_started
            last_log_step = start_step
            final_step = start_step

            for step in range(start_step + 1, MAX_STEPS + 1):
                inputs, targets = batcher.sample()
                before = time.perf_counter()
                params, optimizer_state, metrics = train_step(
                    params,
                    optimizer_state,
                    jnp.asarray(inputs),
                    jnp.asarray(targets),
                )
                jax.block_until_ready(metrics["loss"])
                step_seconds = time.perf_counter() - before
                if step == start_step + 1:
                    compile_seconds = step_seconds
                final_step = step

                should_log = step == start_step + 1 or step % LOG_EVERY == 0
                if should_log:
                    host = jax.device_get(metrics)
                    assert bool(host["finite"]), f"Non-finite values at step {step}"
                    now = time.perf_counter()
                    completed = step - last_log_step
                    examples_per_second = completed * BATCH_SIZE / (now - last_log_time)
                    record = {
                        "step": step,
                        "loss": float(host["loss"]),
                        "language_model_loss": float(host["language_model_loss"]),
                        "answer_token_accuracy": float(host["answer_token_accuracy"]),
                        "gradient_norm": float(host["gradient_norm"]),
                        "balance_loss": float(host["balance_loss"]),
                        "z_loss": float(host["z_loss"]),
                        "router_entropy": float(host["router_entropy"]),
                        "route_fraction": np.asarray(host["route_fraction"]).tolist(),
                        "learning_rate": float(learning_rate(step - 1)),
                        "step_seconds": step_seconds,
                        "examples_per_second": examples_per_second,
                    }
                    history.append(record)
                    print(
                        f"step {step:4d} | LM {record['language_model_loss']:.4f} | "
                        f"token acc {100 * record['answer_token_accuracy']:6.2f}% | "
                        f"grad {record['gradient_norm']:.3f} | "
                        f"{examples_per_second:,.0f} examples/s"
                    )
                    last_log_time = now
                    last_log_step = step

                validation = None
                if step % EVAL_EVERY == 0 or step == MAX_STEPS:
                    validation = evaluate_validation(params)
                    validation["step"] = step
                    validation_history.append(validation)
                    print(
                        f"validation | loss {validation['loss']:.4f} | "
                        f"token acc {100 * validation['token_accuracy']:6.2f}% | "
                        f"exact match {100 * validation['exact_match']:6.2f}%"
                    )
                    improved = (
                        validation["exact_match"] > best_exact_match
                        or (
                            validation["exact_match"] == best_exact_match
                            and validation["loss"] < best_validation_loss
                        )
                    )
                    if improved:
                        best_exact_match = validation["exact_match"]
                        best_validation_loss = validation["loss"]
                        save_checkpoint(BEST_CHECKPOINT_PATH, step, validation)
                        print("saved best checkpoint")

                if step % CHECKPOINT_EVERY == 0 or step == MAX_STEPS:
                    save_checkpoint(LATEST_CHECKPOINT_PATH, step, validation)
                    atomic_write_text(
                        HISTORY_PATH,
                        json.dumps(
                            {
                                "compile_seconds": compile_seconds,
                                "final_step": step,
                                "train": history,
                                "validation": validation_history,
                            },
                            indent=2,
                        ) + "\\n",
                    )
                    print("saved resumable checkpoint")

            training_seconds = time.perf_counter() - training_started
            print(f"Training segment: {training_seconds / 60:.2f} minutes")
            print(f"Best validation exact match: {100 * best_exact_match:.4f}%")
            """
        ),
        markdown("## Curves and routing diagnostics"),
        code(
            """
            train_steps = [row["step"] for row in history]
            validation_steps = [row["step"] for row in validation_history]
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))

            axes[0, 0].plot(
                train_steps,
                [row["language_model_loss"] for row in history],
                label="training LM loss",
            )
            axes[0, 0].plot(
                validation_steps,
                [row["loss"] for row in validation_history],
                marker="o",
                label="validation LM loss",
            )
            axes[0, 0].set_yscale("log")
            axes[0, 0].set_title("Answer loss")
            axes[0, 0].legend()

            axes[0, 1].plot(
                validation_steps,
                [100 * row["exact_match"] for row in validation_history],
                marker="o",
            )
            axes[0, 1].set_title("Validation greedy exact match")
            axes[0, 1].set_ylabel("%")

            axes[1, 0].plot(
                train_steps, [row["balance_loss"] for row in history], label="balance"
            )
            axes[1, 0].plot(
                train_steps, [row["z_loss"] for row in history], label="z-loss"
            )
            axes[1, 0].set_title("Router regularizers (unweighted)")
            axes[1, 0].legend()

            route_loads = np.asarray([
                np.asarray(row["route_fraction"]).mean(axis=0) for row in history
            ])
            for expert in range(N_EXPERTS):
                axes[1, 1].plot(
                    train_steps, route_loads[:, expert], label=f"expert {expert}"
                )
            axes[1, 1].axhline(1 / N_EXPERTS, linestyle="--", linewidth=1)
            axes[1, 1].set_title("Mean top-2 route fraction")
            axes[1, 1].legend()
            for axis in axes.flat:
                axis.set_xlabel("step")
                axis.grid(alpha=0.2)
            plt.tight_layout()
            plt.show()
            """
        ),
        markdown("## Evaluate"),
        code(
            """
            assert BEST_CHECKPOINT_PATH.exists(), (
                "No best checkpoint exists. Run training through endpoint validation."
            )
            checkpoint = pickle.loads(BEST_CHECKPOINT_PATH.read_bytes())
            assert_checkpoint_compatible(checkpoint)
            params = jax.tree.map(jnp.asarray, checkpoint["params"])
            trained_model = nnx.merge(graphdef, params)
            print("Restored best checkpoint from step", checkpoint["step"])
            print("Validation exact match", checkpoint["validation"]["exact_match"])
            """
        ),
        code(
            dense_source(22).replace(
                '"parameter_count": 10_000_000,',
                '"stored_parameter_count": 17_942_400,\n'
                '        "active_parameter_proxy": 10_006_400,',
            )
        ),
        code(dense_source(23)),
        markdown("## Use the model"),
        code(dense_source(25)),
        code(dense_source(26)),
        code(
            """
            archive = shutil.make_archive(
                "/content/moe_ragged_dot_run", "zip", root_dir=RUN_DIR
            )
            print("Best checkpoint:", BEST_CHECKPOINT_PATH)
            print("Latest checkpoint:", LATEST_CHECKPOINT_PATH)
            print("History:", HISTORY_PATH)
            print("Results:", RESULTS_PATH)
            print("Failures:", FAILURES_PATH)
            print("Archive:", archive)

            from google.colab import files
            files.download(archive)
            """
        ),
    ]

    for index, cell in enumerate(notebook.cells):
        identity = f"{index}:{cell.cell_type}:{cell.source}".encode()
        cell.id = hashlib.sha256(identity).hexdigest()[:8]
        if cell.cell_type == "code":
            cell.execution_count = None
            cell.outputs = []
    nbformat.validate(notebook)
    nbformat.write(notebook, OUTPUT)
    print(f"Generated clean standalone notebook: {OUTPUT}")


if __name__ == "__main__":
    main()
