import subprocess, sys
from pathlib import Path


def test_notebook_source_sync():
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, root / "scripts/check_notebook_sync.py"], cwd=root, check=True)
