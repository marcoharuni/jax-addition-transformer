from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "colab-runtime.json"
NOTEBOOK_DIR = ROOT / "notebooks"
_SYNC_SPEC = importlib.util.spec_from_file_location(
    "sync_colab_runtime",
    ROOT / "scripts" / "sync_colab_runtime.py",
)
assert _SYNC_SPEC is not None and _SYNC_SPEC.loader is not None
_SYNC_MODULE = importlib.util.module_from_spec(_SYNC_SPEC)
_SYNC_SPEC.loader.exec_module(_SYNC_MODULE)
INSTALL_TAG = _SYNC_MODULE.INSTALL_TAG
SETUP_MARKDOWN = _SYNC_MODULE.SETUP_MARKDOWN
SMOKE_TAG = _SYNC_MODULE.SMOKE_TAG
VALIDATE_TAG = _SYNC_MODULE.VALIDATE_TAG
install_source = _SYNC_MODULE.install_source


def test_colab_runtime_source_of_truth_and_setup_cells_match():
    config = json.loads(CONFIG_PATH.read_text())
    assert config["python"] == "3.12"
    assert config["accelerator"] == "NVIDIA T4"
    assert config["versions"] == {
        "jax": "0.8.1",
        "jaxlib": "0.8.1",
        "jax-cuda12-plugin": "0.8.1",
        "jax-cuda12-pjrt": "0.8.1",
        "flax": "0.12.2",
        "optax": "0.2.6",
        "numpy": "2.3.3",
        "matplotlib": "3.10.7",
        "pandas": "2.3.3",
        "scipy": "1.16.3",
    }

    for notebook_name, requirements in config["notebooks"].items():
        notebook = nbformat.read(NOTEBOOK_DIR / notebook_name, 4)
        nbformat.validate(notebook)
        assert notebook.cells[1].source == SETUP_MARKDOWN

        install_cells = [
            cell
            for cell in notebook.cells
            if INSTALL_TAG in cell.metadata.get("tags", [])
        ]
        validation_cells = [
            cell
            for cell in notebook.cells
            if VALIDATE_TAG in cell.metadata.get("tags", [])
        ]
        assert len(install_cells) == 1
        assert len(validation_cells) == 1
        assert install_cells[0].source == install_source(config, requirements)

        install_tree = ast.parse(install_cells[0].source)
        imported_roots = {
            alias.name.split(".", maxsplit=1)[0]
            for node in ast.walk(install_tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        assert not {"jax", "flax", "optax", "numpy"} & imported_roots
        assert "_RESTART_MARKER.exists()" in install_cells[0].source
        assert "_os.kill(_os.getpid(), _signal.SIGKILL)" in install_cells[0].source
        assert (
            '[_uv, "pip", "check", "--python", str(_ENV_PYTHON)]'
            in install_cells[0].source
        )
        assert '"venv",\n            "--clear"' in install_cells[0].source
        assert "00-jax-addition-colab.pth" in install_cells[0].source
        assert "_sys.path.insert(0, str(_ENV_SITE_PACKAGES))" in install_cells[0].source

        all_code = "\n".join(
            cell.source for cell in notebook.cells if cell.cell_type == "code"
        )
        assert all_code.count('"jax[cuda12]==0.8.1"') == 1
        assert '"jax[cuda12]"' not in all_code
        assert "jax[cuda12]==0.7.2" not in all_code
        assert 'flax==0.12.0' not in all_code


def test_notebooks_compile_and_validate_gpu_without_cpu_fallback():
    config = json.loads(CONFIG_PATH.read_text())
    for notebook_name in config["notebooks"]:
        notebook = nbformat.read(NOTEBOOK_DIR / notebook_name, 4)
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                compile(cell.source, f"{notebook_name}:cell-{index}", "exec")

        validation = next(
            cell.source
            for cell in notebook.cells
            if VALIDATE_TAG in cell.metadata.get("tags", [])
        )
        if notebook_name == "04_moe_scaling_t4.ipynb":
            assert "COLAB_RUNTIME_VALIDATE_GPU_WORKER_V1" in validation
            assert 'jax.default_backend() != "gpu"' in validation
            assert 'os.environ["JAX_PLATFORMS"] = "cpu"' in validation
            assert 'assert jax.default_backend() == "cpu"' in validation
        else:
            assert "COLAB_RUNTIME_VALIDATE_GPU_V1" in validation
            assert '_backend != "gpu"' in validation
            assert '_matrix_product.device.platform != "gpu"' in validation
            assert "nnx.split(_nnx_probe)" in validation


def test_exact_notebook_has_isolated_one_step_smoke_and_required_predictions():
    notebook = nbformat.read(
        NOTEBOOK_DIR / "01_build_train_exact_10m.ipynb",
        4,
    )
    smoke_cells = [
        cell
        for cell in notebook.cells
        if SMOKE_TAG in cell.metadata.get("tags", [])
    ]
    assert len(smoke_cells) == 1
    smoke = smoke_cells[0].source
    assert smoke.count("train_step(") == 1
    assert "parameter_count == 10_000_000" in smoke
    assert "pickle.dumps(" in smoke
    assert "pickle.loads(" in smoke
    assert "nnx.merge(graphdef, _smoke_restored_params)" in smoke
    assert "generate_from_params(" in smoke

    prediction_cell = next(
        cell.source
        for cell in notebook.cells
        if cell.cell_type == "code" and "def predict_pair(a, b):" in cell.source
    )
    for pair in (
        "(0, 0)",
        "(1, 9)",
        "(99, 1)",
        "(123, 456)",
        "(347, 928)",
        "(500, 500)",
        "(999, 1)",
        "(999, 999)",
    ):
        assert pair in prediction_cell


def test_checkpoint_verifier_covers_restore_and_genuine_generation():
    source = (ROOT / "scripts" / "verify_colab_checkpoint.py").read_text()
    assert "pickle.loads(" in source
    assert "nnx.merge(" in source
    assert "parameter_count != 10_000_000" in source
    assert "jax.jit(greedy_generate)" in source
    assert "(347, 928)" in source
    assert "(999, 999)" in source


def test_smoke_verifier_executes_tagged_notebook_code():
    source = (ROOT / "scripts" / "verify_colab_smoke.py").read_text()
    assert '"SMOKE_BATCH_SIZE = 8"' in source
    assert "exec(compile(source" in source
    assert "wrong compatibility environment" in source
    assert "parameter_count" in source
    assert "smoke_finite" in source
