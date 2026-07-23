# Streamlit Community Cloud deployment

This directory is the free public application deployment for the released
[JAX Addition Transformer model](https://huggingface.co/marcoharuni95/jax-addition-transformer-10m).
It preserves the tested pure-JAX inference implementation under
`deploy/huggingface/model_repo` and downloads the 40 MB Safetensors artifact
with anonymous public read access.

## Community Cloud coordinates

- Repository: `marcoharuni/jax-addition-transformer`
- Branch: `huggingface-deployment`
- Entrypoint: `deploy/streamlit/app.py`
- Python: `3.12`
- Desired subdomain: `jax-addition-transformer`
- Secrets: none

`requirements.txt` is intentionally the only dependency declaration beside
the entrypoint. Streamlit Community Cloud gives that file precedence over the
root development `pyproject.toml` and `uv.lock`. The repository-wide
`.streamlit/config.toml` is at the root because Community Cloud only reads
configuration from that location.

## Runtime behavior

`st.cache_resource` owns one model instance per application process. Its first
construction downloads the pinned public model revision, loads the 54
Safetensors arrays, and JIT-compiles batch-one greedy generation. Subsequent
messages reuse the same model and compiled batch shape.

No Hugging Face token or other secret is required. Inactive free Community
Cloud applications may hibernate and incur the same cold start again when
awakened.

## Local verification

The isolated test environment lives under `tests/` so its `pyproject.toml` and
`uv.lock` cannot override Community Cloud's entrypoint-local
`requirements.txt`.

Run the server from the repository root to match Community Cloud:

```bash
MODEL_REPO_DIR="$PWD/deploy/huggingface/model_repo" \
uv run --project deploy/streamlit/tests \
  streamlit run deploy/streamlit/app.py
```
