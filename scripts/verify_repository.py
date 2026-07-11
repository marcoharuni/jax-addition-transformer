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
    "scripts/build_notebook.py",
    "scripts/check_notebook_sync.py",
    "scripts/validate_notebook.py",
    "scripts/verify_repository.py",
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
rasters = [
    path
    for suffix in ("*.png", "*.jpg", "*.jpeg")
    for path in ROOT.rglob(suffix)
    if ".venv" not in path.parts
]
if rasters:
    raise SystemExit("external raster diagrams are forbidden")

pyproject = (ROOT / "pyproject.toml").read_text()
readme = (ROOT / "README.md").read_text()
for command in ("jat-inspect", "jat-train", "jat-eval", "jat-chat", "jat-report"):
    if command not in pyproject or command not in readme:
        raise SystemExit(f"CLI command missing from metadata or README: {command}")

notebook = nbformat.read(ROOT / "notebooks/01_build_train_exact_10m.ipynb", 4)
source_indices = [
    index for index, cell in enumerate(notebook.cells) if cell.metadata.get("canonical_source")
]
project_import_indices = [
    index
    for index, cell in enumerate(notebook.cells)
    if cell.cell_type == "code" and "from jax_addition_transformer" in cell.source
]
if (
    not source_indices
    or not project_import_indices
    or min(project_import_indices) <= max(source_indices)
):
    raise SystemExit("notebook imports project code before displaying every canonical source file")

config = ExperimentConfig.load(ROOT / "configs/exact_10m_t4.json")
model = AdditionTransformer(config.model, rngs=nnx.Rngs(params=config.training.seed))
assert_parameter_count(model, 10_000_000)
for script in ("validate_notebook.py", "check_notebook_sync.py"):
    subprocess.run([sys.executable, str(ROOT / "scripts" / script)], check=True, cwd=ROOT)
print("Repository lightweight verification passed; exact default has 10,000,000 parameters.")
