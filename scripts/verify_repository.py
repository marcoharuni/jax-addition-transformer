"""Lightweight publication guardrails."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import nbformat
from flax import nnx

from jax_addition_transformer.config import ExperimentConfig
from jax_addition_transformer.model import AdditionTransformer, assert_parameter_count

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
    "model",
    "losses",
    "optimizers",
    "sampling",
    "training",
    "generation",
    "evaluation",
    "checkpointing",
    "reporting",
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
    "runs/.gitkeep",
    "scripts/validate_notebook.py",
    "scripts/verify_repository.py",
    "artifacts/t4-run/history.json",
    "artifacts/t4-run/results.json",
    "artifacts/t4-run/failures.csv",
    "artifacts/t4-run/README.md",
]
required += [f"src/jax_addition_transformer/{name}.py" for name in SOURCE_NAMES]
required += [f"tests/test_{name}.py" for name in TEST_NAMES]

missing = [name for name in required if not (ROOT / name).exists()]
if missing:
    raise SystemExit(f"missing required files: {missing}")

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
    raise SystemExit(
        "clean Colab notebook must not hide its core model behind package imports"
    )
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

print("Repository verification passed; clean notebook and exact 10M model are present.")
