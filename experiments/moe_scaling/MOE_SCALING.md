# Final matched-active MoE scaling experiment

This experiment mirrors the completed dense grid using one package-based
training and evaluation protocol.

## Grid

```text
5 model sizes × 8 training budgets × 3 seeds = 120 independent T4 runs
```

Training budgets:

```text
50, 75, 100, 125, 175, 250, 400, 750 steps
```

Seeds:

```text
42, 123, 2026
```

Every MoE has four experts, top-2 routing, no capacity limit, no dropped routes,
two real `jax.lax.ragged_dot` projections, GELU, weighted scatter-add, a
load-balancing coefficient of `1e-2`, and router z-loss coefficient of `1e-3`.

The Colab workflow reproduces the versions recorded by all 120 dense runs:
JAX 0.7.2, Flax 0.12.0, Optax 0.2.6, Orbax 0.11.28, and NumPy 2.0.2.
This deliberate experiment pin takes precedence over the repository's newer
local-development NumPy pin.

## Matching rule

Each MoE preserves the dense reference's layers, width, heads, tokenizer, data
split, sampler, optimizer, schedule, batch size, and validation protocol. Only
the feed-forward sublayer is replaced.

`expert_d_ff = dense_d_ff / top_k` exactly matches selected expert matrix work.
The router remains active, so the active-parameter proxy is slightly above the
dense parameter count. That overhead is explicit rather than hidden.

| MoE name | Dense reference | Stored parameters | Active proxy | Router overhead |
|---|---:|---:|---:|---:|
| `moe_active_0p16m` | 162,176 | 289,664 | 162,688 | 0.316% |
| `moe_active_0p64m` | 643,840 | 1,152,768 | 644,864 | 0.159% |
| `moe_active_2p16m` | 2,164,608 | 3,881,088 | 2,166,912 | 0.106% |
| `moe_active_5p12m` | 5,123,584 | 9,190,912 | 5,127,680 | 0.080% |
| `moe_active_10m` | 10,000,000 | 17,942,400 | 10,006,400 | 0.064% |

## Run in Colab

Use `notebooks/04_moe_scaling_t4.ipynb`. Completed runs are stored on Google
Drive and skipped on later sessions. The runner supports model, step, seed, and
deterministic contiguous shard filters.

The command-line equivalent is:

```bash
uv run python experiments/moe_scaling/configs.py
uv run python experiments/moe_scaling/run_grid.py \
  --output-root /path/to/durable/runs \
  --work-root /fast/local/work \
  --compilation-cache /fast/local/jax-cache \
  --rerun-only-missing \
  --continue-on-error
```

Examples:

```bash
# One smoke run
uv run python experiments/moe_scaling/run_grid.py \
  --model moe_active_0p16m --steps 50 --seed 42 --limit 1

# One of four deterministic parallel shards
uv run python experiments/moe_scaling/run_grid.py \
  --shard-count 4 --shard-index 0 --rerun-only-missing
```

## Final analysis

After all 120 summaries report `complete`:

```bash
uv run python experiments/moe_scaling/analyze_results.py \
  --results experiments/moe_scaling/results/final/final_results.json \
  --artifact artifacts/moe-scaling-final-t4

uv run python experiments/moe_scaling/fit_scaling_law.py
uv run python experiments/moe_scaling/plot_results.py
uv run python experiments/moe_scaling/compare_dense_moe.py
```

The analysis produces:

- 120 run-level rows and 40 three-seed aggregates;
- 50%, 95%, and 99% exact-match transition tables;
- attempted additive Chinchilla-style surfaces using active parameters (primary) and stored parameters (secondary);
- a measured MoE compute-loss frontier;
- paired dense-versus-MoE loss, accuracy, and T4 runtime comparisons;
- combined compute and wall-clock frontiers;
- router entropy and expert-load diagnostics.

## Accounting caveat

`6 × active parameters × sequence tokens` is a consistent architectural proxy,
not a complete MoE FLOP or runtime model. It excludes top-k selection, sorting,
dispatch, scatter-add, and shape-dependent kernel efficiency. For that reason,
the final report treats the measured wall-clock frontier as co-primary with the
proxy-compute frontier.

## Current status

The complete grid, runner, analysis, plots, tests, and Colab workflow are
prepared. The 120 package-based GPU runs must be completed before empirical
MoE scaling conclusions can be committed. The standalone seed-42 result under
`artifacts/moe-comparison-t4` is preserved separately and is not mixed into the
grid.
