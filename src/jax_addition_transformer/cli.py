"""Argparse console entry points."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from flax import nnx

from .checkpointing import restore_checkpoint
from .config import ExperimentConfig
from .data import create_split
from .evaluation import evaluate_ids
from .generation import ask, predict_pair
from .model import AdditionTransformer, assert_parameter_count, parameter_table
from .optimizers import make_optimizer
from .reporting import generate_report
from .training import train


def _config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/exact_10m_t4.json")


def inspect_main() -> None:
    parser = argparse.ArgumentParser(description="Inspect an arithmetic-transformer configuration")
    _config_argument(parser)
    args = parser.parse_args()
    config = ExperimentConfig.load(args.config)
    model = AdditionTransformer(config.model, rngs=nnx.Rngs(params=config.training.seed))
    total = assert_parameter_count(model, 10_000_000 if config.model.is_exact_default else None)
    print("Component                         Parameters")
    print("-------------------------------- ----------")
    for name, count in parameter_table(config.model):
        print(f"{name:32} {count:10,d}")
    print("-------------------------------- ----------")
    print(f"Total trainable parameters       {total:10,d}")


def train_main() -> None:
    parser = argparse.ArgumentParser(description="Train the arithmetic transformer")
    _config_argument(parser)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--smoke-steps", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    summary = train(ExperimentConfig.load(args.config), args.run_dir, args.smoke_steps, args.resume)
    print(json.dumps(summary, indent=2))


def _restore_model(run_dir: Path, checkpoint: str = "best"):
    config = ExperimentConfig.load(run_dir / "config.json")
    model = AdditionTransformer(config.model, rngs=nnx.Rngs(params=config.training.seed))
    graphdef, params = nnx.split(model, nnx.Param)
    optimizer, _ = make_optimizer(config.optimizer, params)
    optimizer_state = optimizer.init(params)
    checkpoint_path = run_dir / "checkpoints" / checkpoint
    if checkpoint == "best" and not checkpoint_path.exists():
        checkpoint_path = run_dir / "checkpoints" / "latest"
    params, _, _ = restore_checkpoint(checkpoint_path, params, optimizer_state, config.fingerprint)
    return nnx.merge(graphdef, params), config


def eval_main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved run")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--exhaustive", action="store_true")
    parser.add_argument("--checkpoint", choices=["best", "latest"], default="best")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    model, config = _restore_model(run_dir, args.checkpoint)
    split = create_split(
        config.training.seed,
        config.training.train_pairs,
        config.training.validation_pairs,
        10**config.task.max_digits,
    )
    ids = (
        np.arange((10**config.task.max_digits) ** 2)
        if args.exhaustive
        else getattr(split, args.split)
    )
    metrics, failures = evaluate_ids(
        model, ids, config.training.evaluation_batch_size, config.task.max_digits
    )
    fieldnames = [
        "pair_id",
        "a",
        "b",
        "target",
        "prediction",
        "internal_generated_digits",
        "valid_digit_sequence",
        "operand_length_a",
        "operand_length_b",
        "carry_pattern",
        "split",
    ]
    with (run_dir / "failures.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        if args.exhaustive:
            split_labels = np.empty((len(ids),), dtype="<U10")
            split_labels[split.train] = "train"
            split_labels[split.validation] = "validation"
            split_labels[split.test] = "test"
        for row in failures:
            label = split_labels[row["pair_id"]] if args.exhaustive else args.split
            writer.writerow({**row, "split": label})
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    key = "exhaustive" if args.exhaustive else args.split
    summary.setdefault("evaluations", {})[key] = metrics
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    examples = [
        (0, 0),
        (1, 9),
        (9, 1),
        (9, 991),
        (99, 1),
        (199, 801),
        (499, 501),
        (500, 500),
        (909, 91),
        (999, 1),
        (999, 999),
    ]
    examples = [
        (a, b)
        for a, b in examples
        if a <= config.task.maximum_operand and b <= config.task.maximum_operand
    ]
    (run_dir / "sample_predictions.txt").write_text(
        "\n".join(
            f"{a:0{config.task.max_digits}d} + {b:0{config.task.max_digits}d} = {predict_pair(model, a, b, config.task)}"
            for a, b in examples
        )
        + "\n"
    )
    print(json.dumps(metrics, indent=2))


def chat_main() -> None:
    parser = argparse.ArgumentParser(description="Interactive model-only arithmetic inference")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", choices=["best", "latest"], default="best")
    args = parser.parse_args()
    model, config = _restore_model(Path(args.run_dir), args.checkpoint)
    print(f"Enter additions using operands 0-{config.task.maximum_operand}; Ctrl-D exits.")
    while True:
        try:
            expression = input("> ")
        except EOFError:
            break
        try:
            print(ask(model, expression, config.task))
        except (TypeError, ValueError) as exc:
            print(f"error: {exc}")


def report_main() -> None:
    parser = argparse.ArgumentParser(description="Generate a report from real artifacts")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    print(generate_report(args.run_dir))
