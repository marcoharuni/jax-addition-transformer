from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
NOTEBOOKS = (
    "01_build_train_exact_10m.ipynb",
    "02_dense_scaling_laws.ipynb",
    "03_moe_ragged_dot_t4.ipynb",
    "04_moe_scaling_t4.ipynb",
)


def test_notebooks_are_valid_clean_and_compilable():
    for notebook_name in NOTEBOOKS:
        notebook = nbformat.read(NOTEBOOK_DIR / notebook_name, 4)
        nbformat.validate(notebook)

        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            assert not cell.get("outputs")
            if cell.source.lstrip().startswith(("%", "!")):
                continue
            compile(cell.source, f"{notebook_name}:cell-{index}", "exec")


def test_exact_notebook_uses_minimal_pinned_colab_setup():
    notebook = nbformat.read(
        NOTEBOOK_DIR / "01_build_train_exact_10m.ipynb",
        4,
    )
    assert notebook.cells[1].source == "## Setup\n"
    assert notebook.cells[2].source.splitlines() == [
        "%pip install -q uv",
        (
            '!uv pip install --system "jax[cuda12]==0.8.1" '
            '"flax==0.12.2" "optax==0.2.6" "numpy==2.3.3"'
        ),
        'print("Restart the Colab runtime once, then run all cells again.")',
    ]

    verification = notebook.cells[3].source
    for expected in (
        "import jaxlib",
        "from flax import nnx",
        'print("JAX:", jax.__version__)',
        'print("JAXlib:", jaxlib.__version__)',
        'print("Flax:", __import__("flax").__version__)',
        'print("Optax:", optax.__version__)',
        'print("NumPy:", np.__version__)',
        'print("JAX backend:", jax.default_backend())',
        'print("Detected device:", jax.devices()[0])',
        'device.platform == "gpu"',
    ):
        assert expected in verification


def test_other_notebooks_keep_their_restored_dependency_cells():
    expected_install_cells = {
        "02_dense_scaling_laws.ipynb": (
            '%pip install -q -U "jax[cuda12]==0.7.2" "flax==0.12.0" '
            '"optax==0.2.6" "numpy==2.3.3" "pandas==2.3.3" '
            '"matplotlib==3.10.7" "scipy==1.16.3"\n'
        ),
        "03_moe_ragged_dot_t4.ipynb": (
            '%pip install -q -U "jax[cuda12]==0.7.2" "flax==0.12.0" '
            '"optax==0.2.6" "numpy==2.3.3" "matplotlib==3.10.7"\n'
        ),
        "04_moe_scaling_t4.ipynb": (
            '%pip install -q -U "jax[cuda12]==0.7.2" "flax==0.12.0" '
            '"optax==0.2.6" "numpy==2.3.3" "pandas==2.3.3" '
            '"matplotlib==3.10.7" "scipy==1.16.3"\n'
        ),
    }
    for notebook_name, expected in expected_install_cells.items():
        notebook = nbformat.read(NOTEBOOK_DIR / notebook_name, 4)
        assert notebook.cells[2].source.rstrip() == expected.rstrip()
        assert "COLAB_RUNTIME_" not in "\n".join(
            cell.source for cell in notebook.cells
        )
