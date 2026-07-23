"""Idempotently publish the prepared model repository and Gradio Space."""

from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import HfApi

MODEL_ID = "marcoharuni95/jax-addition-transformer-10m"
SPACE_ID = "marcoharuni95/jax-addition-transformer"
ROOT = Path(__file__).resolve().parents[3]
DEPLOY = ROOT / "deploy" / "huggingface"
IGNORED = [
    "**/.venv/**",
    "**/__pycache__/**",
    "**/*.pyc",
    "pyproject.toml",
    "uv.lock",
]


def commit_record(info) -> dict[str, str]:
    return {
        "commit_url": str(info.commit_url),
        "commit_oid": str(info.oid),
    }


def main() -> None:
    api = HfApi()
    identity = api.whoami()
    username = identity.get("name") or identity.get("fullname")
    if username != "marcoharuni95":
        raise RuntimeError(f"expected Hugging Face user marcoharuni95, found {username!r}")

    api.create_repo(MODEL_ID, repo_type="model", private=False, exist_ok=True)
    model_commit = api.upload_folder(
        repo_id=MODEL_ID,
        repo_type="model",
        folder_path=DEPLOY / "model_repo",
        commit_message="Publish verified exact-10M JAX addition transformer",
        ignore_patterns=IGNORED,
    )

    api.create_repo(
        SPACE_ID,
        repo_type="space",
        space_sdk="gradio",
        private=False,
        exist_ok=True,
    )
    space_commit = api.upload_folder(
        repo_id=SPACE_ID,
        repo_type="space",
        folder_path=DEPLOY / "space",
        commit_message="Publish JAX addition transformer Gradio Space",
        ignore_patterns=IGNORED,
    )

    result = {
        "user": username,
        "model": commit_record(model_commit),
        "space": commit_record(space_commit),
    }
    output = ROOT / ".local" / "hf-deployment" / "publish-result.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
