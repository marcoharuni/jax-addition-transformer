"""Measure cold model construction, generation latency, and peak resident memory."""

from __future__ import annotations

import json
import resource
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_runtime import get_runtime


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main() -> None:
    baseline = peak_rss_mb()
    get_runtime.clear()
    runtime = get_runtime()
    after_startup = peak_rss_mb()
    cases = ((347, 928), (499, 501), (91, 909), (999, 999))
    expected = ("1275", "1000", "1000", "1998")
    details = [runtime.add_with_details(a, b) for a, b in cases]
    if tuple(str(row["answer"]) for row in details) != expected:
        raise AssertionError("measured runtime produced an incorrect model generation")
    result = {
        "startup_seconds": runtime.startup_seconds,
        "compilation_seconds": runtime.model.compilation_seconds,
        "inference_latency_ms": [float(row["latency_seconds"]) * 1000 for row in details],
        "mean_inference_latency_ms": statistics.mean(
            float(row["latency_seconds"]) * 1000 for row in details
        ),
        "baseline_peak_rss_mb": baseline,
        "startup_peak_rss_mb": after_startup,
        "final_peak_rss_mb": peak_rss_mb(),
        "results": {
            f"{a} + {b}": row["answer"] for (a, b), row in zip(cases, details, strict=True)
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
