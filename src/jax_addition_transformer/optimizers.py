"""Optax schedules, optimizer variants, and path-based selective decay."""

from __future__ import annotations

from typing import Any

import jax
import optax

from .config import OptimizerConfig


def learning_rate_schedule(config: OptimizerConfig) -> optax.Schedule:
    decay_steps = config.total_steps - 1 if config.end_at_last_update else config.total_steps
    return optax.warmup_cosine_decay_schedule(
        init_value=config.initial_learning_rate,
        peak_value=config.peak_learning_rate,
        warmup_steps=config.warmup_steps,
        decay_steps=decay_steps,
        end_value=config.final_learning_rate,
    )


def _path_parts(path: tuple[Any, ...]) -> tuple[str, ...]:
    return tuple(str(getattr(entry, "key", getattr(entry, "idx", entry))) for entry in path)


def decay_mask(params):
    """Decay attention and FFN matrix kernels, excluding every router kernel."""

    def eligible(path, leaf):
        parts = _path_parts(path)
        return bool(
            getattr(leaf, "ndim", 0) in {2, 3}
            and any("kernel" in part for part in parts)
            and ({"attention", "ffn"} & set(parts))
            and "router" not in parts
        )

    return jax.tree_util.tree_map_with_path(eligible, params)


def make_optimizer(config: OptimizerConfig, params):
    schedule = learning_rate_schedule(config)
    transformations: list[optax.GradientTransformation] = [
        optax.clip_by_global_norm(config.clip_global_norm)
    ]
    if config.name in {"adamw", "adam"}:
        transformations.append(
            optax.scale_by_adam(b1=config.beta1, b2=config.beta2, eps=config.epsilon)
        )
        if config.name == "adamw":
            transformations.append(
                optax.masked(optax.add_decayed_weights(config.weight_decay), decay_mask(params))
            )
    else:
        transformations.append(optax.trace(decay=config.momentum, nesterov=False))
    transformations.append(optax.scale_by_learning_rate(schedule))
    return optax.chain(*transformations), schedule
