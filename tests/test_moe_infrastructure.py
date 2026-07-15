import json
import sys
from pathlib import Path

import nbformat

from jax_addition_transformer.config import ExperimentConfig


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.moe_comparison.configs import build_comparison  # noqa: E402
from experiments.moe_comparison.run_comparison import (  # noqa: E402
    checkpoint_is_complete,
    copy_run,
)


def test_manifest_contains_three_matched_runs():
    manifest_path = (
        ROOT
        / "experiments"
        / "moe_comparison"
        / "comparison_manifest.json"
    )

    manifest = json.loads(manifest_path.read_text())
    final_configs = sorted(
        (ROOT / "experiments" / "moe_comparison" / "configs" / "final").glob("*.json")
    )

    assert len(manifest) == 3
    assert [path.stem for path in final_configs] == [
        "moe_10m_s0175_seed123",
        "moe_10m_s0175_seed2026",
        "moe_10m_s0175_seed42",
    ]
    assert len({row["run_id"] for row in manifest}) == 3
    assert {row["model"] for row in manifest} == {"moe_10m"}
    assert {row["steps"] for row in manifest} == {175}
    assert {row["seed"] for row in manifest} == {42, 123, 2026}

    for row in manifest:
        config = ExperimentConfig.load(ROOT / row["config"])

        assert config.model.architecture == "moe"
        assert config.model.num_experts == 4
        assert config.model.top_k == 2
        assert config.model.expert_d_ff == 1240
        assert config.model.ffn_type == "gelu"
        assert config.optimizer.total_steps == 175
        assert config.training.max_steps == 175
        assert config.training.seed == row["seed"]


def test_verified_parameter_counts():
    architecture, manifest = build_comparison(write=False)

    assert len(manifest) == 3
    assert architecture["dense_total_parameters"] == 10_000_000
    assert architecture["moe_total_parameters"] == 17_942_400
    assert architecture["moe_active_parameter_proxy"] == 10_006_400


def test_colab_notebook_is_clean():
    path = ROOT / "notebooks" / "03_moe_ragged_dot_t4.ipynb"

    notebook = nbformat.read(path, 4)
    nbformat.validate(notebook)

    code = "\n".join(
        cell.source
        for cell in notebook.cells
        if cell.cell_type == "code"
    )
    markdown_text = "\n".join(
        cell.source
        for cell in notebook.cells
        if cell.cell_type == "markdown"
    )

    assert len(notebook.cells) == 36
    assert "colab.research.google.com/github/marcoharuni" in markdown_text
    assert "blob/moe-ragged-dot/notebooks/03_moe_ragged_dot_t4.ipynb" in markdown_text
    assert '"jax[cuda12]==0.7.2"' in code
    assert '"flax==0.12.0"' in code
    assert '"optax==0.2.6"' in code
    assert "XLA_PYTHON_CLIENT_PREALLOCATE" in code
    assert "cuda_malloc_async" in code
    assert 'subprocess.run(["nvidia-smi"], check=True)' in code
    assert 'TOKENS = "0123456789 +="' in code
    assert "class RaggedTop2MoE(nnx.Module)" in code
    assert "class AdditionMoETransformer(nnx.Module)" in code
    assert code.count("jax.lax.ragged_dot(") >= 4
    assert "jax.checkpoint(apply_block)" in code
    assert "assert parameter_count == 17_942_400" in code
    assert "assert active_proxy == 10_006_400" in code
    assert "ROUTER_BALANCE_COEFFICIENT" in code
    assert "ROUTER_Z_LOSS_COEFFICIENT" in code
    assert "def train_step(" in code
    assert "def save_checkpoint(" in code
    assert "atomic_write_bytes" in code
    assert "LATEST_CHECKPOINT_PATH.exists()" in code
    assert "Resuming after step" in code
    assert "saved resumable checkpoint" in code
    assert "evaluate_complete_domain" in code
    assert "def predict_pair(" in code
    assert "Validation greedy exact match" in code
    assert "import jax_addition_transformer" not in code
    assert "run_comparison" not in code
    assert "orbax" not in code.lower()

    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code" and not cell.source.startswith("%"):
            compile(cell.source, f"notebook-cell-{index}", "exec")

    assert not any(
        cell.get("outputs")
        for cell in notebook.cells
        if cell.cell_type == "code"
    )


def test_comparison_runner_uses_python_training_entry_point():
    source = (ROOT / "experiments" / "moe_comparison" / "run_comparison.py").read_text()

    assert "sys.executable" in source
    assert "from jax_addition_transformer.cli import train_main" in source
    assert '"-u"' in source
    assert '"--output-root"' in source
    assert 'environment["PYTHONPATH"]' in source
    assert "Full training output:" in source
    assert "capture_output" not in source
    assert "shutil.which" not in source
    assert "sync_callback" not in source


def test_run_copy_ignores_uncommitted_orbax_checkpoint(tmp_path):
    source = tmp_path / "local"
    complete = source / "checkpoints" / "latest"
    complete.mkdir(parents=True)
    (complete / "metadata.json").write_text('{"step": 25}\n')
    (complete / "state").mkdir()
    (complete / "state" / "committed").write_text("ok")
    transient = complete / "state.orbax-checkpoint-tmp"
    transient.mkdir()
    (transient / ".zarray").write_text("changing")

    assert not checkpoint_is_complete(complete)
    transient.rename(source / "checkpoints" / "abandoned.orbax-checkpoint-tmp")
    assert checkpoint_is_complete(complete)

    destination = tmp_path / "durable"
    copy_run(source, destination)

    assert (destination / "checkpoints" / "latest" / "state" / "committed").read_text() == "ok"
    assert not list(destination.rglob("*.orbax-checkpoint-tmp"))


def test_model_does_not_require_nnx_list():
    source = (ROOT / "src" / "jax_addition_transformer" / "model.py").read_text()

    assert "nnx.List(" not in source
    assert "class ModuleSequence(nnx.Module)" in source
