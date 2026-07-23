from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]


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
    assert "blob/main/notebooks/03_moe_ragged_dot_t4.ipynb" in markdown_text
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


def test_model_does_not_require_nnx_list():
    source = (ROOT / "src" / "jax_addition_transformer" / "model.py").read_text()

    assert "nnx.List(" not in source
    assert "class ModuleSequence(nnx.Module)" in source
