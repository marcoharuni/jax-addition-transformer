from pathlib import Path
import nbformat

path = Path(__file__).resolve().parents[1] / "notebooks/01_build_train_exact_10m.ipynb"
notebook = nbformat.read(path, 4)
nbformat.validate(notebook)
if any(cell.get("outputs") for cell in notebook.cells if cell.cell_type == "code"):
    raise SystemExit("committed notebook contains outputs")
print(f"Valid notebook: {path}")
