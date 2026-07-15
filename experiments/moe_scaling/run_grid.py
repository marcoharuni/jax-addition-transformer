"""Resumable runner for the 120-run matched-active MoE scaling grid."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = (
    ROOT
    / "experiments"
    / "moe_scaling"
    / "final_manifest.json"
)

DEFAULT_RUNS_DIR = Path(
    os.environ.get(
        "JAT_MOE_SCALING_RUNS_DIR",
        ROOT
        / "experiments"
        / "moe_scaling"
        / "results"
        / "final",
    )
)

RUNS_DIR = DEFAULT_RUNS_DIR
RESULTS_PATH = RUNS_DIR / "final_results.json"
TRAIN_ENTRY_POINT = (
    "from jax_addition_transformer.cli import train_main; "
    "train_main()"
)


class TrainingProcessError(RuntimeError):
    """Training subprocess failure with its complete streamed output."""

    def __init__(self, return_code: int, command: list[str], output: str):
        self.returncode = return_code
        self.command = command
        self.output = output
        super().__init__(
            f"Training failed with exit code {return_code}.\n"
            f"Command: {shlex.join(command)}\n\n"
            f"Full training output:\n{output}"
        )


def read_json(path: Path) -> Any:
    """Read a JSON file."""

    return json.loads(path.read_text())


def write_json_atomic(path: Path, value: Any) -> None:
    """Write JSON atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(value, indent=2) + "\n"
    )

    temporary.replace(path)


def run_streaming(
    command: list[str],
    compilation_cache: Path | None = None,
) -> None:
    """Run a command while teeing its combined output to the notebook."""

    environment = os.environ.copy()
    source_dir = str(ROOT / "src")
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_dir
        if not existing_pythonpath
        else source_dir + os.pathsep + existing_pythonpath
    )
    environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    environment["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
    if compilation_cache is not None:
        compilation_cache.mkdir(parents=True, exist_ok=True)
        environment["JAX_COMPILATION_CACHE_DIR"] = str(compilation_cache)
        environment["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "0"

    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output: list[str] = []

    if process.stdout is None:
        raise RuntimeError("training subprocess output stream is unavailable")

    for line in process.stdout:
        print(line, end="", flush=True)
        output.append(line)

    return_code = process.wait()

    if return_code != 0:
        raise TrainingProcessError(
            return_code,
            command,
            "".join(output),
        )


def copy_run(source: Path, destination: Path) -> None:
    """Copy a quiescent run via a sibling temporary directory.

    Orbax creates ``*.orbax-checkpoint-tmp`` directories while committing a
    checkpoint.  They are never resumable checkpoints and must not be copied
    to durable storage.
    """

    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".syncing")
    if temporary.exists():
        shutil.rmtree(temporary)
    shutil.copytree(
        source,
        temporary,
        ignore=shutil.ignore_patterns("*.orbax-checkpoint-tmp"),
    )
    if destination.exists():
        shutil.rmtree(destination)
    temporary.replace(destination)


def checkpoint_is_complete(path: Path) -> bool:
    """Return whether ``path`` contains a fully committed Orbax checkpoint."""

    return (
        (path / "metadata.json").is_file()
        and (path / "state").is_dir()
        and not any(path.glob("*.orbax-checkpoint-tmp"))
    )


def run_status(
    run_dir: Path,
    expected_steps: int,
) -> str:
    """Return the current state of one experiment."""

    if (run_dir / "failure.json").exists():
        return "failed"

    summary_path = run_dir / "summary.json"

    if not summary_path.exists():
        if run_dir.exists() and any(run_dir.iterdir()):
            return "incomplete"
        return "missing"

    completed_steps = int(
        read_json(summary_path).get("steps", 0)
    )

    if completed_steps >= expected_steps:
        return "complete"

    return "incomplete"


def collect_results(
    manifest: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Collect run summaries into one JSON file."""

    records: list[dict[str, Any]] = []

    for row in manifest:
        run_dir = RUNS_DIR / row["run_id"]

        record: dict[str, Any] = {
            **row,
            "run_directory": str(run_dir),
            "status": run_status(
                run_dir,
                row["steps"],
            ),
        }

        for filename, key in (
            ("summary.json", "summary"),
            ("environment.json", "environment"),
        ):
            path = run_dir / filename

            if path.exists():
                record[key] = read_json(path)

        failure_path = run_dir / "failure.json"

        if failure_path.exists():
            record["failure"] = read_json(
                failure_path
            )

        records.append(record)

    write_json_atomic(
        RESULTS_PATH,
        records,
    )

    statuses = (
        "complete",
        "incomplete",
        "missing",
        "failed",
    )

    counts = {
        status: sum(
            record["status"] == status
            for record in records
        )
        for status in statuses
    }

    print(
        "Result status: "
        + ", ".join(
            f"{count} {status}"
            for status, count in counts.items()
        )
    )

    return records


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_RUNS_DIR,
    )

    parser.add_argument(
        "--work-root",
        type=Path,
    )

    parser.add_argument(
        "--compilation-cache",
        type=Path,
    )

    parser.add_argument(
        "--seed",
        action="append",
        type=int,
    )

    parser.add_argument(
        "--model",
        action="append",
        help="Select one or more manifest model names.",
    )

    parser.add_argument(
        "--steps",
        action="append",
        type=int,
        help="Select one or more training budgets.",
    )

    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based contiguous shard index for parallel Colab sessions.",
    )

    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="Number of deterministic contiguous shards.",
    )

    parser.add_argument(
        "--limit",
        type=int,
    )

    parser.add_argument(
        "--force-rerun",
        action="store_true",
    )

    parser.add_argument(
        "--rerun-only-missing",
        action="store_true",
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
    )

    parser.add_argument(
        "--keep-checkpoints",
        action="store_true",
    )

    parser.add_argument(
        "--collect-only",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    """Run or inspect the full matched-active MoE scaling grid."""

    global RESULTS_PATH, RUNS_DIR

    args = parse_arguments()
    RUNS_DIR = args.output_root.expanduser().resolve()
    RESULTS_PATH = RUNS_DIR / "final_results.json"

    manifest = read_json(MANIFEST_PATH)

    if len(manifest) != 120:
        raise ValueError(
            "The final MoE scaling manifest must contain "
            f"exactly 120 runs, found {len(manifest)}."
        )

    if args.collect_only:
        collect_results(manifest)
        return

    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must be in [0, shard-count)")

    filtered = [
        row
        for row in manifest
        if (not args.seed or row["seed"] in args.seed)
        and (not args.model or row["model"] in args.model)
        and (not args.steps or row["steps"] in args.steps)
    ]
    shard_start = len(filtered) * args.shard_index // args.shard_count
    shard_end = len(filtered) * (args.shard_index + 1) // args.shard_count
    selected = filtered[shard_start:shard_end]

    if args.rerun_only_missing:
        selected = [
            row
            for row in selected
            if run_status(
                RUNS_DIR / row["run_id"],
                row["steps"],
            )
            in {
                "missing",
                "incomplete",
                "failed",
            }
        ]

    if args.limit is not None:
        selected = selected[: args.limit]

    if not selected:
        print("No runs need execution.")
        collect_results(manifest)
        return

    RUNS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for index, row in enumerate(
        selected,
        start=1,
    ):
        durable_run_dir = RUNS_DIR / row["run_id"]
        work_root = args.work_root.expanduser().resolve() if args.work_root else RUNS_DIR
        run_dir = work_root / row["run_id"]

        status = run_status(
            durable_run_dir,
            row["steps"],
        )

        command = [
            sys.executable,
            "-u",
            "-c",
            TRAIN_ENTRY_POINT,
            "--config",
            str(ROOT / row["config"]),
            "--run-dir",
            str(run_dir),
        ]

        resume_source = durable_run_dir if run_dir != durable_run_dir else run_dir
        latest_checkpoint = (
            resume_source
            / "checkpoints"
            / "latest"
        )

        if (
            checkpoint_is_complete(latest_checkpoint)
            and (resume_source / "config.json").exists()
            and not args.force_rerun
        ):
            command.append("--resume")

        print(
            f"[{index}/{len(selected)}] "
            f"{row['run_id']} ({status})",
            flush=True,
        )

        if args.dry_run:
            print(shlex.join(command))
            continue

        if (
            status == "complete"
            and not args.force_rerun
        ):
            print("Already complete; skipping.")
            continue

        if run_dir != durable_run_dir:
            if run_dir.exists():
                shutil.rmtree(run_dir)
            copy_run(durable_run_dir, run_dir)

        if (
            args.force_rerun
            and run_dir.exists()
        ):
            shutil.rmtree(run_dir)
            if durable_run_dir.exists() and durable_run_dir != run_dir:
                shutil.rmtree(durable_run_dir)

        failure_path = run_dir / "failure.json"
        try:
            run_streaming(
                command,
                compilation_cache=args.compilation_cache,
            )

            if failure_path.exists():
                failure_path.unlink()

            if (
                not args.keep_checkpoints
                and run_status(
                    run_dir,
                    row["steps"],
                )
                == "complete"
            ):
                checkpoint_dir = (
                    run_dir / "checkpoints"
                )

                if checkpoint_dir.exists():
                    shutil.rmtree(
                        checkpoint_dir
                    )

            if run_dir != durable_run_dir:
                copy_run(run_dir, durable_run_dir)

        except Exception as error:
            write_json_atomic(
                failure_path,
                {
                    "run_id": row["run_id"],
                    "exception_type": type(error).__name__,
                    "exception": repr(error),
                    "return_code": getattr(error, "returncode", None),
                    "command": command,
                    "command_text": shlex.join(command),
                    "output": getattr(error, "output", None),
                    "timestamp": (
                        datetime.now(
                            UTC
                        ).isoformat()
                    ),
                },
            )

            if run_dir != durable_run_dir:
                copy_run(run_dir, durable_run_dir)

            collect_results(manifest)

            if not args.continue_on_error:
                raise

        collect_results(manifest)

    collect_results(manifest)


if __name__ == "__main__":
    main()
