# Hugging Face deployment

This directory contains the reproducible deployment source for:

- Model: <https://huggingface.co/marcoharuni95/jax-addition-transformer-10m>
- Space: <https://huggingface.co/spaces/marcoharuni95/jax-addition-transformer>

## Trust boundary and conversion

The v0.1.0 release checkpoint is a trusted Python pickle containing a Flax NNX
`State`, not an Orbax directory. It is read only inside `compat/`, whose
isolated lock pins the notebook-evidenced Flax 0.12.2. The converter verifies
the checkpoint tree against a freshly constructed copy of the exact notebook
architecture, restores 10,000,000 parameters, runs legacy inference, and
writes deterministic inference-only Safetensors.

Normal model and Space startup never read pickle and do not require Flax.

## Reproduce locally

Place the official release ZIP outside Git and extract it under
`.local/hf-deployment/release`, then:

```bash
UV_CACHE_DIR=.local/uv-cache uv sync --project deploy/huggingface/compat
UV_CACHE_DIR=.local/uv-cache uv run --project deploy/huggingface/compat \
  python deploy/huggingface/scripts/convert_checkpoint.py \
  .local/hf-deployment/release/exact_10m_checkpoint.pkl \
  deploy/huggingface/model_repo/model.safetensors
python deploy/huggingface/scripts/prepare_model_repo.py

MODEL_REPO_DIR="$PWD/deploy/huggingface/model_repo" \
UV_CACHE_DIR=.local/uv-cache uv run --project deploy/huggingface/space \
  pytest -q deploy/huggingface/tests
```

The generated weight file is intentionally ignored by the Git repository. It
is published to the model repository by `scripts/publish.py`.
