from pathlib import Path

import nbformat


def test_notebook_is_clean_and_self_contained():
    root = Path(__file__).resolve().parents[1]
    notebook = nbformat.read(root / "notebooks/01_build_train_exact_10m.ipynb", 4)
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")

    assert "%%writefile" not in code
    assert "from jax_addition_transformer" not in code
    assert "import jax_addition_transformer" not in code
    assert "10_000_000" in code
    assert "evaluate_complete_domain" in code
    assert "train_step" in code
