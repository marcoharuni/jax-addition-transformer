"""Pure-state compiled step and explicit Python training loop."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from .checkpointing import capture_environment, restore_checkpoint, save_checkpoint
from .config import ExperimentConfig
from .data import create_split
from .losses import masked_cross_entropy
from .model import AdditionTransformer, assert_parameter_count
from .optimizers import make_optimizer
from .sampling import HybridSampler
from .evaluation import evaluate_ids


def tree_all_finite(tree) -> jax.Array:
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(tree)]))


def _raise_nonfinite() -> None:
    raise FloatingPointError(
        "non-finite loss, gradients, or updated parameters in compiled training step"
    )


def create_train_step(graphdef, optimizer):
    """Close over static graph/optimizer once; pass only array PyTrees to JIT."""

    @jax.jit
    def train_step(params, optimizer_state, inputs, targets):
        def objective(candidate):
            model = nnx.merge(graphdef, candidate)
            loss, accuracy = masked_cross_entropy(model(inputs), targets)
            return loss, accuracy

        (loss, accuracy), gradients = jax.value_and_grad(objective, has_aux=True)(params)
        gradient_norm = optax.global_norm(gradients)
        finite = jnp.isfinite(loss) & tree_all_finite(gradients)
        updates, optimizer_state = optimizer.update(gradients, optimizer_state, params)
        params = optax.apply_updates(params, updates)
        finite = finite & tree_all_finite(params)

        def fail(_):
            jax.debug.callback(_raise_nonfinite, ordered=True)
            return jnp.int32(0)

        jax.lax.cond(finite, lambda _: jnp.int32(0), fail, operand=None)
        return (
            params,
            optimizer_state,
            {
                "loss": loss,
                "answer_token_accuracy": accuracy,
                "gradient_global_norm": gradient_norm,
                "finite": finite,
            },
        )

    return train_step


def train(
    config: ExperimentConfig,
    run_dir: str | Path,
    smoke_steps: int | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    config_path = run_dir / "config.json"
    latest = run_dir / "checkpoints" / "latest"
    if resume:
        if not latest.exists() or not config_path.exists():
            raise FileNotFoundError(f"cannot resume: {latest} or {config_path} does not exist")
        if ExperimentConfig.load(config_path).fingerprint != config.fingerprint:
            raise ValueError("resume configuration is incompatible with the existing run")
    elif latest.exists():
        raise FileExistsError(
            f"run already contains a latest checkpoint; pass resume=True: {latest}"
        )
    else:
        config.save(config_path)
    split = create_split(
        config.training.seed,
        config.training.train_pairs,
        config.training.validation_pairs,
        10**config.task.max_digits,
    )
    split_metadata = {
        "seed": split.seed,
        "train_size": len(split.train),
        "validation_size": len(split.validation),
        "test_size": len(split.test),
        "domain_size": len(split.train) + len(split.validation) + len(split.test),
        "pair_id_formula": f"{10**config.task.max_digits} * a + b",
        "stratification": "operand length of a, operand length of b, carry bits ordered units-tens-hundreds",
        "allocation": "deterministic largest remainder within strata",
        "fingerprint_algorithm": "SHA-256 over named little-endian int64 split arrays",
        "fingerprint": split.fingerprint,
    }
    (run_dir / "split_metadata.json").write_text(json.dumps(split_metadata, indent=2) + "\n")
    model = AdditionTransformer(config.model, rngs=nnx.Rngs(params=config.training.seed))
    parameter_count = assert_parameter_count(
        model, 10_000_000 if config.model.is_exact_default else None
    )
    graphdef, params = nnx.split(model, nnx.Param)
    optimizer, schedule = make_optimizer(config.optimizer, params)
    optimizer_state = optimizer.init(params)
    step_fn = create_train_step(graphdef, optimizer)
    sampler = HybridSampler.create(
        split.train, config.training.batch_size, config.training.seed, 10**config.task.max_digits
    )
    environment = capture_environment(config)
    (run_dir / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    total_steps = smoke_steps if smoke_steps is not None else config.training.max_steps
    history_path = run_dir / "history.jsonl"
    start_step, best = 0, None
    if resume:
        params, optimizer_state, metadata = restore_checkpoint(
            latest, params, optimizer_state, config.fingerprint
        )
        start_step = int(metadata["step"])
        sampler.restore_state(metadata["sampler_state"])
        best = metadata["best"]
        history = metadata["history"]
        history_path.write_text("".join(json.dumps(record) + "\n" for record in history))
    else:
        history = []
        history_path.write_text("")
    started = time.perf_counter()
    compile_seconds = None
    step_durations = []
    for step in range(start_step + 1, total_steps + 1):
        inputs, targets = sampler.sample_batch(config.task.max_digits)
        before = time.perf_counter()
        params, optimizer_state, metrics = step_fn(
            params, optimizer_state, jnp.asarray(inputs), jnp.asarray(targets)
        )
        jax.block_until_ready(metrics["loss"])
        duration = time.perf_counter() - before
        step_durations.append(duration)
        if compile_seconds is None:
            compile_seconds = duration
        checkpoint_interval = 1 if smoke_steps is not None else config.training.checkpoint_interval
        checkpoint_due = step % checkpoint_interval == 0
        validation_due = smoke_steps is None and step % config.training.validation_interval == 0
        should_record = (
            step == start_step + 1
            or step == total_steps
            or checkpoint_due
            or validation_due
            or step % config.training.logging_interval == 0
        )
        if not should_record:
            continue
        host = {name: float(value) for name, value in jax.device_get(metrics).items()}
        if not bool(host["finite"]):
            raise FloatingPointError(f"non-finite loss, gradients, or parameters at step {step}")
        record = {
            "step": step,
            **host,
            "learning_rate": float(schedule(step - 1)),
            "step_seconds": duration,
            "examples_per_second": config.training.batch_size / duration,
            "answer_tokens_per_second": config.training.batch_size
            * config.task.answer_digits
            / duration,
        }
        save_as_best = False
        if validation_due:
            validation_model = nnx.merge(graphdef, params)
            validation, _ = evaluate_ids(
                validation_model,
                split.validation,
                config.training.evaluation_batch_size,
                config.task.max_digits,
            )
            candidate = {
                "step": step,
                "greedy_exact_match": validation["greedy_exact_match"],
                "loss": validation["teacher_forced_loss"],
            }
            record["validation"] = validation
            save_as_best = best is None or (
                candidate["greedy_exact_match"] > best["greedy_exact_match"]
                or (
                    candidate["greedy_exact_match"] == best["greedy_exact_match"]
                    and candidate["loss"] < best["loss"]
                )
            )
            if save_as_best:
                best = candidate
        history.append(record)
        with history_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        if step == start_step + 1 or step % config.training.logging_interval == 0:
            print(json.dumps(record))
        if save_as_best:
            save_checkpoint(
                run_dir / "checkpoints" / "best",
                params,
                optimizer_state,
                {
                    "step": step,
                    "sampler_state": sampler.state,
                    "best": best,
                    "history": history,
                    "config_fingerprint": config.fingerprint,
                },
            )
        if checkpoint_due or step == total_steps:
            save_checkpoint(
                run_dir / "checkpoints" / "latest",
                params,
                optimizer_state,
                {
                    "step": step,
                    "sampler_state": sampler.state,
                    "best": best,
                    "history": history,
                    "config_fingerprint": config.fingerprint,
                },
            )
    previous_summary = {}
    if resume and (run_dir / "summary.json").exists():
        previous_summary = json.loads((run_dir / "summary.json").read_text())
    segment_seconds = time.perf_counter() - started
    steady_state_seconds = sum(step_durations[1:]) if len(step_durations) > 1 else 0.0
    summary = {
        "parameter_count": parameter_count,
        "steps": total_steps,
        "compilation_seconds": previous_summary.get("compilation_seconds", compile_seconds),
        "resume_compilation_seconds": compile_seconds if resume else None,
        "steady_state_training_seconds": previous_summary.get("steady_state_training_seconds", 0.0)
        + steady_state_seconds,
        "training_segment_seconds": segment_seconds,
        "total_training_seconds": previous_summary.get("total_training_seconds", 0.0)
        + segment_seconds,
        "best": best,
        "final_train_loss": history[-1]["loss"] if history else "not evaluated",
        "final_answer_token_accuracy": history[-1]["answer_token_accuracy"]
        if history
        else "not evaluated",
        "evaluations": previous_summary.get("evaluations", {}),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for name, content in (
        (
            "failures.csv",
            "pair_id,a,b,target,prediction,internal_generated_digits,valid_digit_sequence,operand_length_a,operand_length_b,carry_pattern,split\n",
        ),
        ("sample_predictions.txt", "No predictions evaluated yet.\n"),
    ):
        (run_dir / name).write_text(content)
    return summary
