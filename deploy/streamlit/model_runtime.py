"""Cached, public-read-only runtime for Streamlit Community Cloud."""

from __future__ import annotations

import os
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import streamlit as st
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
MODEL_CODE_DIR = ROOT / "deploy" / "huggingface" / "model_repo"
sys.path.insert(0, str(MODEL_CODE_DIR))

from inference import AdditionModel  # noqa: E402

MODEL_REPO_ID = "marcoharuni95/jax-addition-transformer-10m"
MODEL_REVISION = "fbd66de376333c01e9f80e8ccede8eef6a0afb39"


def peak_rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / 1024


@dataclass(frozen=True)
class Runtime:
    model: AdditionModel
    startup_seconds: float
    peak_memory_mb: float
    weights_source: str

    def add_with_details(self, a: int, b: int) -> dict[str, str | float]:
        return self.model.add_with_details(a, b)


@st.cache_resource(show_spinner=False)
def get_runtime() -> Runtime:
    started = time.perf_counter()
    local_repo = os.environ.get("MODEL_REPO_DIR")
    if local_repo:
        weights_path = Path(local_repo) / "model.safetensors"
        source = "local test artifact"
    else:
        weights_path = Path(
            hf_hub_download(
                repo_id=MODEL_REPO_ID,
                filename="model.safetensors",
                revision=MODEL_REVISION,
                token=False,
            )
        )
        source = f"https://huggingface.co/{MODEL_REPO_ID}"
    model = AdditionModel(weights_path)
    return Runtime(
        model=model,
        startup_seconds=time.perf_counter() - started,
        peak_memory_mb=peak_rss_mb(),
        weights_source=source,
    )
