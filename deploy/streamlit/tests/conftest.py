from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STREAMLIT_DIR = ROOT / "deploy" / "streamlit"
MODEL_REPO = ROOT / "deploy" / "huggingface" / "model_repo"

os.environ["MODEL_REPO_DIR"] = str(MODEL_REPO)
sys.path.insert(0, str(STREAMLIT_DIR))
