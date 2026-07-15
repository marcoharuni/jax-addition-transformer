"""Resumable, fingerprint-validated runner for either 16-run grid."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jax_addition_transformer.config import ExperimentConfig
from jax_addition_transformer.scaling import canonical_fingerprint

from .protocol import (
    FIXED_SEED,
    OUTPUT_ROOTS,
    PROTOCOL_PATH,
    RESULT_SCHEMA_VERSION,
    ROOT,
    RUNS_PER_ARCHITECTURE,
    path_has_suffix,
)


TRAIN_ENTRY_POINT = "from jax_addition_transformer.cli import train_main; train_main()"
REQUIRED_RUN_FILES = (
    "config.json",
    "protocol.json",
    "run_spec.json",
    "split_metadata.json",
    "environment.json",
    "history.jsonl",
    "summary.json",
    "result.json",
    "failures.csv",
    "sample_predictions.txt",
)


class TrainingProcessError(RuntimeError):
    def __init__(self, return_code: int, command: list[str], output: str):
        self.returncode = return_code
        self.command = command
        self.output = output
        super().__init__(
            f"Training failed with exit code {return_code}.\n"
            f"Command: {shlex.join(command)}\n\nFull training output:\n{output}"
        )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def protocol_payload() -> tuple[dict[str, Any], str]:
    stored = read_json(PROTOCOL_PATH)
    claimed = str(stored.pop("protocol_fingerprint"))
    calculated = canonical_fingerprint(stored)
    if claimed != calculated:
        raise ValueError("protocol.json fingerprint is invalid")
    return stored, calculated


def manifest_path(architecture: str) -> Path:
    return ROOT / "experiments" / "dense_moe_scaling" / f"{architecture}_manifest.json"


def checkpoint_is_complete(path: Path) -> bool:
    return (
        (path / "metadata.json").is_file()
        and (path / "state").is_dir()
        and not any(path.rglob("*.orbax-checkpoint-tmp"))
    )


def resumable_checkpoint_step(run_dir: Path, row: dict[str, Any]) -> int:
    checkpoint = run_dir / "checkpoints" / "latest"
    if not checkpoint_is_complete(checkpoint):
        return -1
    try:
        metadata = read_json(checkpoint / "metadata.json")
    except (OSError, ValueError, json.JSONDecodeError):
        return -1
    if metadata.get("config_fingerprint") != row["config_fingerprint"]:
        return -1
    if metadata.get("protocol_fingerprint") != row["protocol_fingerprint"]:
        return -1
    if metadata.get("run_id") != row["run_id"]:
        return -1
    return int(metadata.get("step", -1))


def validate_complete_run(run_dir: Path, row: dict[str, Any]) -> list[str]:
    """Return every reason a purportedly complete run is not reusable."""

    errors = [name for name in REQUIRED_RUN_FILES if not (run_dir / name).is_file()]
    if errors:
        return [f"missing required file: {name}" for name in errors]

    try:
        saved_config = ExperimentConfig.load(run_dir / "config.json")
        if saved_config.fingerprint != row["config_fingerprint"]:
            errors.append("saved config fingerprint mismatch")
        source_config = ExperimentConfig.load(ROOT / row["config"])
        if source_config.fingerprint != row["config_fingerprint"]:
            errors.append("manifest/source config fingerprint mismatch")

        saved_protocol = read_json(run_dir / "protocol.json")
        claimed_protocol = saved_protocol.pop("protocol_fingerprint", None)
        calculated_protocol = canonical_fingerprint(saved_protocol)
        if claimed_protocol != row["protocol_fingerprint"]:
            errors.append("saved protocol fingerprint claim mismatch")
        if calculated_protocol != row["protocol_fingerprint"]:
            errors.append("saved protocol content mismatch")

        if read_json(run_dir / "run_spec.json") != row:
            errors.append("saved run specification mismatch")

        summary = read_json(run_dir / "summary.json")
        result = read_json(run_dir / "result.json")
        environment = read_json(run_dir / "environment.json")
        split = read_json(run_dir / "split_metadata.json")
        if summary.get("schema_version") != "training-summary":
            errors.append("summary schema mismatch")
        if summary.get("run_id") != row["run_id"]:
            errors.append("summary run ID mismatch")
        if summary.get("config_fingerprint") != row["config_fingerprint"]:
            errors.append("summary config fingerprint mismatch")
        if summary.get("protocol_fingerprint") != row["protocol_fingerprint"]:
            errors.append("summary protocol fingerprint mismatch")
        if int(summary.get("steps", -1)) != int(row["horizon_steps"]):
            errors.append("summary horizon mismatch")
        if environment.get("configuration_hash") != row["config_fingerprint"]:
            errors.append("environment config fingerprint mismatch")
        if int(split.get("split_seed", -1)) != int(row["split_seed"]):
            errors.append("split seed mismatch")

        if result.get("schema_version") != RESULT_SCHEMA_VERSION:
            errors.append("result schema mismatch")
        if result.get("status") != "complete" or result.get("run_id") != row["run_id"]:
            errors.append("result identity/status mismatch")
        fingerprints = result.get("fingerprints", {})
        if fingerprints.get("config") != row["config_fingerprint"]:
            errors.append("result config fingerprint mismatch")
        if fingerprints.get("protocol") != row["protocol_fingerprint"]:
            errors.append("result protocol fingerprint mismatch")
        if fingerprints.get("split") != split.get("fingerprint"):
            errors.append("result split fingerprint mismatch")
        if result.get("seeds") != {
            "split": row["split_seed"],
            "initialization": row["initialization_seed"],
            "sampler": row["sampler_seed"],
        }:
            errors.append("result seeds mismatch")
        identity = result.get("identity", {})
        for key, expected in (
            ("architecture", row["architecture"]),
            ("model_id", row["model_id"]),
            ("dense_reference", row["dense_reference"]),
            ("horizon_steps", row["horizon_steps"]),
        ):
            if identity.get(key) != expected:
                errors.append(f"result {key} mismatch")
        parameters = result.get("parameters", {})
        if int(parameters.get("stored", -1)) != int(row["stored_parameters"]):
            errors.append("stored parameter count mismatch")
        if int(parameters.get("active_proxy", -1)) != int(row["active_parameter_proxy"]):
            errors.append("active parameter proxy mismatch")

        if not (run_dir / "history.jsonl").read_text().strip():
            errors.append("empty history")
        for checkpoint_name in ("best", "latest"):
            checkpoint = run_dir / "checkpoints" / checkpoint_name
            if not checkpoint_is_complete(checkpoint):
                errors.append(f"incomplete {checkpoint_name} checkpoint")
                continue
            metadata = read_json(checkpoint / "metadata.json")
            if metadata.get("config_fingerprint") != row["config_fingerprint"]:
                errors.append(f"{checkpoint_name} checkpoint config mismatch")
            if metadata.get("protocol_fingerprint") != row["protocol_fingerprint"]:
                errors.append(f"{checkpoint_name} checkpoint protocol mismatch")
            if metadata.get("run_id") != row["run_id"]:
                errors.append(f"{checkpoint_name} checkpoint run ID mismatch")
            if int(metadata.get("step", -1)) != int(row["horizon_steps"]):
                errors.append(f"{checkpoint_name} checkpoint step mismatch")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        errors.append(f"invalid artifact content: {error}")
    return errors


def run_status(run_dir: Path, row: dict[str, Any]) -> tuple[str, list[str]]:
    if not run_dir.exists() or not any(run_dir.iterdir()):
        return "missing", []
    validation_errors = validate_complete_run(run_dir, row)
    if not validation_errors:
        return "complete", []
    if (run_dir / "failure.json").exists():
        return "failed", validation_errors
    return "incomplete", validation_errors


def copy_run(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".syncing")
    if temporary.exists():
        shutil.rmtree(temporary)
    shutil.copytree(source, temporary, ignore=shutil.ignore_patterns("*.orbax-checkpoint-tmp"))
    if destination.exists():
        shutil.rmtree(destination)
    temporary.replace(destination)


def initialize_output_root(
    output_root: Path, architecture: str, protocol_hash: str, *, dry_run: bool
) -> None:
    expected_root = OUTPUT_ROOTS[architecture]
    if not path_has_suffix(output_root, expected_root):
        raise ValueError(f"output root must end with {expected_root!r}: {output_root}")
    identity_path = output_root / "root_identity.json"
    expected = {
        "protocol_fingerprint": protocol_hash,
        "architecture": architecture,
        "output_root": expected_root,
    }
    if identity_path.exists():
        if read_json(identity_path) != expected:
            raise ValueError(f"refusing incompatible existing output root: {output_root}")
        return
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"refusing non-empty output root without study identity: {output_root}")
    if not dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
        write_json_atomic(identity_path, expected)
        shutil.copy2(PROTOCOL_PATH, output_root / "protocol.json")
        shutil.copy2(manifest_path(architecture), output_root / "manifest.json")


def prepare_run(run_dir: Path, row: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    protocol_destination = run_dir / "protocol.json"
    spec_destination = run_dir / "run_spec.json"
    if protocol_destination.exists() or spec_destination.exists():
        if not protocol_destination.exists() or not spec_destination.exists():
            raise ValueError(f"partial run identity in {run_dir}")
        if read_json(spec_destination) != row:
            raise ValueError(f"run specification collision in {run_dir}")
        saved_protocol = read_json(protocol_destination)
        if saved_protocol.get("protocol_fingerprint") != row["protocol_fingerprint"]:
            raise ValueError(f"protocol collision in {run_dir}")
        return
    if any(run_dir.iterdir()):
        raise ValueError(f"refusing unidentified non-empty run directory: {run_dir}")
    shutil.copy2(PROTOCOL_PATH, protocol_destination)
    write_json_atomic(spec_destination, row)


def run_streaming(
    command: list[str],
    compilation_cache: Path | None,
    sync_callback=None,
    sync_interval_seconds: float = 60.0,
) -> None:
    environment = os.environ.copy()
    source_dir = str(ROOT / "src")
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = source_dir if not existing else source_dir + os.pathsep + existing
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
    if process.stdout is None:
        raise RuntimeError("training subprocess output stream unavailable")
    stop_sync = threading.Event()
    sync_errors: list[Exception] = []

    def sync_periodically() -> None:
        while not stop_sync.wait(sync_interval_seconds):
            try:
                sync_callback()
            except Exception as error:  # pragma: no cover - Drive failures are external.
                sync_errors.append(error)
                print(f"Durable checkpoint sync warning: {error!r}", flush=True)

    sync_thread = None
    if sync_callback is not None:
        sync_thread = threading.Thread(target=sync_periodically, daemon=True)
        sync_thread.start()
    output: list[str] = []
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
            output.append(line)
        return_code = process.wait()
    finally:
        stop_sync.set()
        if sync_thread is not None:
            sync_thread.join()
    if sync_errors:
        print(
            f"Durable sync had {len(sync_errors)} transient failure(s); attempting final sync.",
            flush=True,
        )
    if sync_callback is not None:
        sync_callback()
    if return_code:
        raise TrainingProcessError(return_code, command, "".join(output))


def collect_results(
    output_root: Path, manifest: list[dict[str, Any]], architecture: str
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for row in manifest:
        run_dir = output_root / "runs" / row["run_id"]
        status, errors = run_status(run_dir, row)
        record: dict[str, Any] = {**row, "status": status, "validation_errors": errors}
        if (run_dir / "result.json").is_file():
            record["result"] = read_json(run_dir / "result.json")
        if (run_dir / "failure.json").is_file():
            record["failure"] = read_json(run_dir / "failure.json")
        records.append(record)
    collection = {
        "schema_version": "scaling-collection",
        "architecture": architecture,
        "expected_run_count": RUNS_PER_ARCHITECTURE,
        "records": records,
    }
    write_json_atomic(output_root / "results.json", collection)
    counts = {
        status: sum(record["status"] == status for record in records)
        for status in ("complete", "incomplete", "missing", "failed")
    }
    print("Result status: " + ", ".join(f"{count} {status}" for status, count in counts.items()))
    return collection


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=("dense", "moe"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path)
    parser.add_argument("--compilation-cache", type=Path)
    parser.add_argument("--sync-interval-seconds", type=float, default=60.0)
    parser.add_argument("--model", action="append")
    parser.add_argument("--steps", action="append", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--rerun-only-missing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    output_root = args.output_root.expanduser().resolve()
    work_root = args.work_root.expanduser().resolve() if args.work_root else output_root / "runs"
    _, protocol_hash = protocol_payload()
    manifest: list[dict[str, Any]] = read_json(manifest_path(args.architecture))
    if len(manifest) != RUNS_PER_ARCHITECTURE or len(
        {row["run_id"] for row in manifest}
    ) != RUNS_PER_ARCHITECTURE:
        raise ValueError(
            f"architecture manifest must contain {RUNS_PER_ARCHITECTURE} unique runs"
        )
    if {row["split_seed"] for row in manifest} != {FIXED_SEED}:
        raise ValueError("manifest does not use the fixed seed")
    initialize_output_root(output_root, args.architecture, protocol_hash, dry_run=args.dry_run)

    if args.collect_only:
        if args.dry_run:
            raise ValueError("--collect-only and --dry-run cannot be combined")
        collect_results(output_root, manifest, args.architecture)
        return
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must be in [0, shard-count)")
    if args.sync_interval_seconds <= 0:
        raise ValueError("sync-interval-seconds must be positive")
    filtered = [
        row
        for row in manifest
        if (not args.model or row["model_id"] in args.model)
        and (not args.steps or row["horizon_steps"] in args.steps)
    ]
    start = len(filtered) * args.shard_index // args.shard_count
    end = len(filtered) * (args.shard_index + 1) // args.shard_count
    selected = filtered[start:end]
    if args.rerun_only_missing:
        selected = [
            row
            for row in selected
            if run_status(output_root / "runs" / row["run_id"], row)[0] != "complete"
        ]
    if args.limit is not None:
        selected = selected[: args.limit]
    if not selected:
        print("No runs need execution.")
        if not args.dry_run:
            collect_results(output_root, manifest, args.architecture)
        return

    for index, row in enumerate(selected, start=1):
        durable_run_dir = output_root / "runs" / row["run_id"]
        run_dir = work_root / row["run_id"]
        status, errors = run_status(durable_run_dir, row)
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
        durable_step = resumable_checkpoint_step(durable_run_dir, row)
        work_step = resumable_checkpoint_step(run_dir, row)
        if max(durable_step, work_step) >= 0:
            command.append("--resume")
        print(f"[{index}/{len(selected)}] {row['run_id']} ({status})", flush=True)
        if errors:
            print("Validation: " + "; ".join(errors), flush=True)
        if args.dry_run:
            print(shlex.join(command))
            continue
        if status == "complete":
            print("Already complete; skipping.")
            continue
        if run_dir != durable_run_dir:
            if run_dir.exists() and any(run_dir.iterdir()):
                prepare_run(run_dir, row)
                local_errors = validate_complete_run(run_dir, row)
                if not local_errors:
                    copy_run(run_dir, durable_run_dir)
                    print("Recovered complete local run to durable storage.")
                    collect_results(output_root, manifest, args.architecture)
                    continue
            if durable_step > work_step:
                if run_dir.exists():
                    shutil.rmtree(run_dir)
                copy_run(durable_run_dir, run_dir)
        prepare_run(run_dir, row)
        failure_path = run_dir / "failure.json"
        try:
            sync_callback = (
                (lambda source=run_dir, destination=durable_run_dir: copy_run(source, destination))
                if run_dir != durable_run_dir
                else None
            )
            run_streaming(
                command,
                args.compilation_cache,
                sync_callback,
                args.sync_interval_seconds,
            )
            if failure_path.exists():
                failure_path.unlink()
            final_errors = validate_complete_run(run_dir, row)
            if final_errors:
                raise ValueError(
                    "completed subprocess left invalid artifacts: " + "; ".join(final_errors)
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
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            )
            if run_dir != durable_run_dir:
                copy_run(run_dir, durable_run_dir)
            collect_results(output_root, manifest, args.architecture)
            if not args.continue_on_error:
                raise
        collect_results(output_root, manifest, args.architecture)
    if not args.dry_run:
        collect_results(output_root, manifest, args.architecture)


if __name__ == "__main__":
    main()
