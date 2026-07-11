"""Fail when a canonical source cell differs from its source file."""

from pathlib import Path
import nbformat

root = Path(__file__).resolve().parents[1]
path = root / "notebooks/01_build_train_exact_10m.ipynb"
notebook = nbformat.read(path, 4)
seen = set()
for cell in notebook.cells:
    relative = cell.metadata.get("canonical_source")
    if relative:
        seen.add(relative)
        expected = f"%%writefile {relative}\n" + (root / relative).read_text()
        if cell.source != expected:
            raise SystemExit(f"out of sync: {relative}")
canonical = {str(p.relative_to(root)) for p in (root / "src/jax_addition_transformer").glob("*.py")}
if seen != canonical:
    raise SystemExit(
        f"source cell set differs: missing={canonical - seen}, extra={seen - canonical}"
    )
print(f"Notebook synchronized with {len(seen)} source files.")
