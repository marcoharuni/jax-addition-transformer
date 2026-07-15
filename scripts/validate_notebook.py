from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = (
    ROOT / "notebooks" / "01_build_train_exact_10m.ipynb",
    ROOT / "notebooks" / "02_dense_scaling_laws.ipynb",
    ROOT / "notebooks" / "03_moe_ragged_dot_t4.ipynb",
    ROOT / "notebooks" / "04_moe_scaling_t4.ipynb",
)

for path in NOTEBOOKS:
    notebook = nbformat.read(path, 4)
    nbformat.validate(notebook)
    if any(cell.get("outputs") for cell in notebook.cells if cell.cell_type == "code"):
        raise SystemExit(f"committed notebook contains outputs: {path}")
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code" and not cell.source.startswith("%"):
            compile(cell.source, f"{path.name}:cell-{index}", "exec")
    print(f"Valid notebook: {path}")
