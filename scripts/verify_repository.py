"""Lightweight publication guardrails."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import nbformat
from flax import nnx

from jax_addition_transformer.config import ExperimentConfig
from jax_addition_transformer.model import AdditionTransformer, assert_parameter_count
from jax_addition_transformer.scaling import canonical_fingerprint

ROOT = Path(__file__).resolve().parents[1]

SOURCE_NAMES = [
    "__init__",
    "config",
    "tokenizer",
    "data",
    "layers",
    "normalization",
    "positional",
    "attention",
    "ffn",
    "moe",
    "model",
    "losses",
    "optimizers",
    "sampling",
    "training",
    "generation",
    "evaluation",
    "checkpointing",
    "reporting",
    "scaling",
    "cli",
]

TEST_NAMES = [
    "config",
    "tokenizer",
    "formatting",
    "carries",
    "splits",
    "sampler",
    "layers",
    "normalization",
    "positions",
    "attention",
    "ffn",
    "moe",
    "moe_config",
    "moe_training",
    "moe_infrastructure",
    "dense_moe_scaling",
    "scaling_worker",
    "model",
    "parameter_count",
    "loss_mask",
    "optimizer_mask",
    "train_step",
    "generation",
    "checkpoint",
    "notebook_sync",
    "resume",
    "evaluation",
]

required = [
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "pyproject.toml",
    "uv.lock",
    ".python-version",
    ".gitignore",
    ".github/workflows/ci.yml",
    "configs/exact_10m_t4.json",
    "docs/design.md",
    "notebooks/01_build_train_exact_10m.ipynb",
    "notebooks/02_dense_scaling_laws.ipynb",
    "notebooks/03_moe_ragged_dot_t4.ipynb",
    "notebooks/04_moe_scaling_t4.ipynb",
    "runs/.gitkeep",
    "scripts/validate_notebook.py",
    "scripts/verify_repository.py",
    "scripts/scaling_worker.py",
    "artifacts/t4-run/history.json",
    "artifacts/t4-run/results.json",
    "artifacts/t4-run/failures.csv",
    "artifacts/t4-run/README.md",
    "artifacts/moe-comparison-t4/README.md",
    "artifacts/moe-comparison-t4/seed-42/config.json",
    "artifacts/moe-comparison-t4/seed-42/history.json",
    "artifacts/moe-comparison-t4/seed-42/results.json",
    "artifacts/moe-comparison-t4/seed-42/failures.csv",
    "artifacts/moe-comparison-t4/seed-42/SHA256SUMS.txt",
    "experiments/dense_moe_scaling/protocol.json",
    "experiments/dense_moe_scaling/manifest.json",
    "experiments/dense_moe_scaling/dense_manifest.json",
    "experiments/dense_moe_scaling/moe_manifest.json",
    "experiments/dense_moe_scaling/generate_configs.py",
    "experiments/dense_moe_scaling/run_grid.py",
    "experiments/dense_moe_scaling/analyze.py",
    "experiments/dense_moe_scaling/compare.py",
    "experiments/dense_moe_scaling/README.md",
]
required += [f"src/jax_addition_transformer/{name}.py" for name in SOURCE_NAMES]
required += [f"tests/test_{name}.py" for name in TEST_NAMES]

missing = [name for name in required if not (ROOT / name).exists()]
if missing:
    raise SystemExit(f"missing required files: {missing}")

obsolete = [
    ROOT / "experiments" / "dense_scaling",
    ROOT / "experiments" / "moe_scaling",
    ROOT / "experiments" / "moe_comparison",
    ROOT / "artifacts" / "dense-scaling-final-t4",
    ROOT / "artifacts" / "dense-scaling-pilot-t4",
]
if existing := [path for path in obsolete if path.exists()]:
    raise SystemExit(f"obsolete historical paths remain: {existing}")

if obsolete_figures := sorted((ROOT / "assets").glob("dense_scaling_*.svg")):
    raise SystemExit(f"obsolete dense scaling figures remain: {obsolete_figures}")

for path in ROOT.rglob("*"):
    if path.resolve() == Path(__file__).resolve() or ".venv" in path.parts:
        continue
    if (
        path.is_file()
        and ".git" not in path.parts
        and path.suffix in {".py", ".md", ".toml", ".json", ".yml", ".cff"}
    ):
        text = path.read_text(errors="ignore").lower()
        if any(marker in text for marker in ("to" + "do", "fix" + "me")):
            raise SystemExit(f"placeholder marker in {path}")
        if "forge" + "-" + "transformer" in text:
            raise SystemExit(f"stale repository name in {path}")

allowed_rasters = {
    ROOT / "assets" / "training_curves.png",
    ROOT / "assets" / "exhaustive_accuracy.png",
}

rasters = {
    path
    for suffix in ("*.png", "*.jpg", "*.jpeg")
    for path in ROOT.rglob(suffix)
    if ".venv" not in path.parts and ".git" not in path.parts
}

unexpected_rasters = rasters - allowed_rasters
missing_rasters = allowed_rasters - rasters

if unexpected_rasters:
    raise SystemExit(f"unexpected raster files: {sorted(unexpected_rasters)}")

if missing_rasters:
    raise SystemExit(f"missing verified result figures: {sorted(missing_rasters)}")

pyproject = (ROOT / "pyproject.toml").read_text()
readme = (ROOT / "README.md").read_text()
for command in ("jat-inspect", "jat-train", "jat-eval", "jat-chat", "jat-report"):
    if command not in pyproject:
        raise SystemExit(f"CLI command missing from project metadata: {command}")

notebook = nbformat.read(ROOT / "notebooks/01_build_train_exact_10m.ipynb", 4)
code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
if "%%writefile" in code:
    raise SystemExit("clean Colab notebook must not contain %%writefile cells")
if "from jax_addition_transformer" in code or "import jax_addition_transformer" in code:
    raise SystemExit("clean Colab notebook must not hide its core model behind package imports")
for marker in ("10_000_000", "train_step", "evaluate_complete_domain"):
    if marker not in code:
        raise SystemExit(
            f"clean Colab notebook is missing required implementation marker: {marker}"
        )

config = ExperimentConfig.load(ROOT / "configs/exact_10m_t4.json")
model = AdditionTransformer(config.model, rngs=nnx.Rngs(params=config.training.seed))
assert_parameter_count(model, 10_000_000)

subprocess.run(
    [sys.executable, str(ROOT / "scripts" / "validate_notebook.py")],
    check=True,
    cwd=ROOT,
)

if "1,000,000 / 1,000,000" not in readme:
    raise SystemExit("README is missing the verified exhaustive result")

moe_result = json.loads((ROOT / "artifacts/moe-comparison-t4/seed-42/results.json").read_text())
if moe_result["overall"]["correct"] != 999_981:
    raise SystemExit("standalone MoE result does not match the verified seed-42 run")
scaling_root = ROOT / "experiments" / "dense_moe_scaling"
scaling_dense = json.loads((scaling_root / "dense_manifest.json").read_text())
scaling_moe = json.loads((scaling_root / "moe_manifest.json").read_text())
scaling_protocol = json.loads((scaling_root / "protocol.json").read_text())
protocol_claim = scaling_protocol.pop("protocol_fingerprint", None)
if protocol_claim != canonical_fingerprint(scaling_protocol):
    raise SystemExit("dense/MoE scaling protocol fingerprint is invalid")
if scaling_protocol["study"] != {
    "architectures": ["dense", "moe"],
    "model_sizes_per_architecture": 4,
    "independent_horizons": [50, 125, 300, 750],
    "runs_per_architecture": 16,
    "fixed_seed": 42,
}:
    raise SystemExit("dense/MoE scaling study dimensions are invalid")
if len(scaling_dense) != 16 or len(scaling_moe) != 16:
    raise SystemExit("dense/MoE scaling manifests must contain 16 dense and 16 MoE runs")
all_scaling_rows = scaling_dense + scaling_moe
if len({row["run_id"] for row in all_scaling_rows}) != 32:
    raise SystemExit("dense/MoE scaling run IDs must be unique")
if any("seed42" in row["run_id"] for row in all_scaling_rows):
    raise SystemExit("canonical dense/MoE scaling run IDs must not contain seed42")
if {row["model_id"] for row in scaling_dense} != {
    "dense_0p16m",
    "dense_0p64m",
    "dense_2p16m",
    "dense_10m",
}:
    raise SystemExit("canonical dense scaling models are invalid")
if {row["model_id"] for row in scaling_moe} != {
    "moe_active_0p16m",
    "moe_active_0p64m",
    "moe_active_2p16m",
    "moe_active_10m",
}:
    raise SystemExit("canonical MoE scaling models are invalid")
if {row["horizon_steps"] for row in all_scaling_rows} != {50, 125, 300, 750}:
    raise SystemExit("dense/MoE scaling horizons are invalid")
if {
    (row["split_seed"], row["initialization_seed"], row["sampler_seed"])
    for row in all_scaling_rows
} != {(42, 42, 42)}:
    raise SystemExit("dense/MoE scaling must bind all three random streams to 42")
if {row["protocol_fingerprint"] for row in all_scaling_rows} != {protocol_claim}:
    raise SystemExit("dense/MoE scaling manifests are not bound to protocol.json")
for architecture, rows in (("dense", scaling_dense), ("moe", scaling_moe)):
    config_paths = list((scaling_root / "configs" / architecture).glob("*.json"))
    if len(config_paths) != 16:
        raise SystemExit(f"dense/MoE scaling {architecture} config count is not 16")
    for row in rows:
        generated = ExperimentConfig.load(ROOT / row["config"])
        if generated.fingerprint != row["config_fingerprint"]:
            raise SystemExit(f"dense/MoE scaling config fingerprint mismatch: {row['run_id']}")

print(
    "Repository verification passed; exact dense model, standalone MoE evidence, "
    "and the 16+16 dense/MoE study are present."
)
