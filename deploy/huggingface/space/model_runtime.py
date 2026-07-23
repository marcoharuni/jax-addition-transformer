"""Singleton model runtime used by the Space."""

from __future__ import annotations

import os
import threading
from pathlib import Path

from inference import AdditionModel, MODEL_REPO_ID

_MODEL: AdditionModel | None = None
_LOCK = threading.Lock()


def get_model() -> AdditionModel:
    global _MODEL
    if _MODEL is None:
        with _LOCK:
            if _MODEL is None:
                local_repo = os.environ.get("MODEL_REPO_DIR")
                source: str | Path = Path(local_repo) if local_repo else MODEL_REPO_ID
                _MODEL = AdditionModel.from_pretrained(source)
    return _MODEL
