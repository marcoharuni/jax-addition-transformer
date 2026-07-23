from __future__ import annotations

import sys
from pathlib import Path

DEPLOY = Path(__file__).resolve().parents[1]
MODEL_REPO = DEPLOY / "model_repo"
SPACE = DEPLOY / "space"

sys.path.insert(0, str(MODEL_REPO))
sys.path.insert(0, str(SPACE))
