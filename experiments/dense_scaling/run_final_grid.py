"""Run and collect the final multi-seed dense scaling grid."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "experiments" / "dense_scaling" / "final_manifest.json"
RUNS_DIR = ROOT / "experiments" / "dense_scaling" / "results" / "final"
RESULTS_PATH = RUNS_DIR / "final_results.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def run_is_complete(run_dir: Path, expected_steps: int) -> bool:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return False

    summary = read_json(summary_path)
    return int(summary.get("steps", -1)) >= expected_steps


def prune_checkpoints(run_dir: Path) -> None:
    checkpoint_dir = run_dir / "checkpoints"
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
        print(f"Pruned checkpoints: {run_dir.name}")


def collect_results(manifest: list[dict[str, Any]]) -> None:
    records: list[dict[str, Any]] = []

    for manifest_row in manifest:
        run_id = str(manifest_row["run_id"])
        run_dir = RUNS_DIR / run_id
        summary_path = run_dir / "summary.json"
        environment_path = run_dir / "environment.json"

        record = dict(manifest_row)
        record["run_directory"] = str(run_dir.relative_to(ROOT))

        if not summary_path.exists():
            record["status"] = "missing"
            records.append(record)
            continue

        summary = read_json(summary_path)
        expected_steps = int(manifest_row["steps"])
        completed_steps = int(summary.get("steps", 0))

        record.update(
            {
                "status": (
                    "complete"
                    if completed_steps >= expected_steps
                    else "partial"
                ),
                "completed_steps": completed_steps,
                "recorded_parameter_count": summary.get("parameter_count"),
                "compilation_seconds": summary.get("compilation_seconds"),
                "steady_state_training_seconds": summary.get(
                    "steady_state_training_seconds"
                ),
                "total_training_seconds": summary.get(
                    "total_training_seconds"
                ),
                "final_train_loss": summary.get("final_train_loss"),
                "final_answer_token_accuracy": summary.get(
                    "final_answer_token_accuracy"
                ),
                "best": summary.get("best"),
            }
        )

        if environment_path.exists():
            environment = read_json(environment_path)
            record["backend"] = environment.get("backend")
            record["devices"] = environment.get("devices")
            record["jax_version"] = environment.get("jax")
            record["git_commit"] = environment.get("git_commit")

        records.append(record)

    write_json_atomic(RESULTS_PATH, records)

    complete = sum(record["status"] == "complete" for record in records)
    partial = sum(record["status"] == "partial" for record in records)
    missing = sum(record["status"] == "missing" for record in records)

    print()
    print(f"Combined results: {RESULTS_PATH.relative_to(ROOT)}")
    print(
        f"Status: {complete} complete, "
        f"{partial} partial, {missing} missing"
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", action="append")
    parser.add_argument("--steps", action="append", type=int)
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--keep-checkpoints",
        action="store_true",
        help="Keep best/latest checkpoints after a completed run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    manifest: list[dict[str, Any]] = json.loads(MANIFEST_PATH.read_text())

    selected = [
        row
        for row in manifest
        if (not args.model or row["model"] in args.model)
        and (not args.steps or row["steps"] in args.steps)
        and (not args.seed or row["seed"] in args.seed)
    ]

    if args.limit is not None:
        selected = selected[: args.limit]

    if not selected:
        raise SystemExit("No configurations matched the requested filters.")

    train_executable = shutil.which("jat-train")
    if train_executable is None:
        raise SystemExit(
            "jat-train was not found. Run with `uv run python "
            "experiments/dense_scaling/run_final_grid.py`."
        )

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Selected {len(selected)} configurations")

    for index, row in enumerate(selected, start=1):
        run_id = str(row["run_id"])
        config_path = ROOT / str(row["config"])
        run_dir = RUNS_DIR / run_id

        command = [
            train_executable,
            "--config",
            str(config_path),
            "--run-dir",
            str(run_dir),
        ]

        print()
        print(f"[{index}/{len(selected)}] {run_id}")

        if args.dry_run:
            print(" ".join(command))
            continue

        if args.force and run_dir.exists():
            shutil.rmtree(run_dir)

        if run_is_complete(run_dir, int(row["steps"])):
            print("Already complete; skipping.")
            if not args.keep_checkpoints:
                prune_checkpoints(run_dir)
            collect_results(manifest)
            continue

        try:
            subprocess.run(command, cwd=ROOT, check=True)
        except subprocess.CalledProcessError:
            collect_results(manifest)
            if args.continue_on_error:
                print(f"Run failed: {run_id}")
                continue
            raise

        if not args.keep_checkpoints:
            prune_checkpoints(run_dir)

        collect_results(manifest)

    collect_results(manifest)


if __name__ == "__main__":
    main()
