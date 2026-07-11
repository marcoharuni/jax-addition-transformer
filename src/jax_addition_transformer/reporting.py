"""Reports generated only from existing run artifacts."""

from __future__ import annotations

import json
from pathlib import Path


def generate_report(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    environment = json.loads((run_dir / "environment.json").read_text())
    split = json.loads((run_dir / "split_metadata.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    history = [json.loads(line) for line in (run_dir / "history.jsonl").read_text().splitlines()]
    validation_history = [record["validation"] for record in history if "validation" in record]
    final_validation = validation_history[-1] if validation_history else {}
    evaluations = summary.get("evaluations", {})

    def metric(split_name, key):
        return evaluations.get(split_name, {}).get(key, "not evaluated")

    best = summary.get("best") or {}
    lines = [
        "# Experiment report",
        "",
        "This report is generated only from recorded run artifacts.",
        "",
        "## Configuration",
        "",
        "```json",
        json.dumps(config, indent=2),
        "```",
        "",
        "## Recorded results",
        "",
        f"- Parameter count: {summary['parameter_count']:,}",
        f"- Split fingerprint: `{split['fingerprint']}`",
        f"- Compilation time: {summary.get('compilation_seconds', 'not evaluated')}",
        f"- Steady-state training time: {summary.get('steady_state_training_seconds', 'not evaluated')}",
        f"- Total training time: {summary.get('total_training_seconds', 'not evaluated')}",
        f"- Best step: {best.get('step', 'not evaluated')}",
        f"- Final train loss: {summary.get('final_train_loss', 'not evaluated')}",
        f"- Final validation loss: {final_validation.get('teacher_forced_loss', 'not evaluated')}",
        f"- Best validation loss: {best.get('loss', 'not evaluated')}",
        f"- Final answer-token accuracy: {summary.get('final_answer_token_accuracy', 'not evaluated')}",
        f"- Final validation token accuracy: {final_validation.get('teacher_forced_token_accuracy', 'not evaluated')}",
        f"- Validation exact match: {metric('validation', 'greedy_exact_match')}",
        f"- Unseen-test exact match: {metric('test', 'greedy_exact_match')}",
        f"- Exhaustive exact match: {metric('exhaustive', 'greedy_exact_match')}",
        f"- Failure count: {metric('exhaustive', 'failure_count')}",
        "",
        "## Carry-pattern and operand-length tables",
        "",
        "```json",
        json.dumps(evaluations.get("exhaustive", {}).get("slices", "not evaluated"), indent=2),
        "```",
        "",
        "## Environment",
        "",
        "```json",
        json.dumps(environment, indent=2),
        "```",
        "",
        "## Sample predictions",
        "",
        (run_dir / "sample_predictions.txt").read_text(),
        "",
        "Checkpoints: `checkpoints/best` (when validation has run), `checkpoints/latest` (resume state)",
        "",
    ]
    path = run_dir / "report.md"
    path.write_text("\n".join(lines))
    return path
