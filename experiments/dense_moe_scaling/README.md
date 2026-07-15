# Dense-versus-MoE scaling

This directory defines the replacement scaling infrastructure. It is additive:
the verified historical 120-run dense grid, the prepared historical 120-run MoE
grid, all committed artifacts, and notebooks 01 and 03 remain unchanged.

## Grid

- Fixed split, initialization, and sampler seeds: `42`.
- Dense models: `dense_0p16m`, `dense_0p64m`, `dense_2p16m`, and `dense_10m`.
- MoE models: their four existing matched `moe_active_*` configurations, with
  four experts, top-2 routing, and unchanged expert widths 248, 496, 744, and
  1,240.
- Independent horizons: 50, 125, 300, and 750 optimizer updates.
- Total: 16 fresh dense runs plus 16 fresh MoE runs.

`dense_5p12m` and `moe_active_5p12m` remain valid package model definitions and
remain in the historical workflows; they are excluded only from this canonical
grid.

Every generated configuration binds the common protocol fingerprint and its
own complete configuration fingerprint. Each horizon owns a warmup-cosine
schedule whose last update uses `1e-4`; no shorter horizon is extracted from a
longer run.

## Shared protocol

[`protocol.json`](protocol.json) freezes the data split, hybrid sampler,
13-token vocabulary, 15-token input, four-token answer mask, batch sizes,
precision, initialization, objective formula, AdamW settings, endpoint
validation, rematerialization, checkpoint retention, and exposure/compute
accounting. Dense and MoE execute the same objective expression. The dense
router losses are exact zeros; MoE retains the `0.01` balance and `0.001`
z-loss terms. Validation answer cross-entropy is always reported separately.

Both architectures rematerialize every transformer block. Weight decay applies
to attention and FFN matrix kernels, including expert matrices, but never to a
router kernel.

## Generated inputs

```text
protocol.json
manifest.json
dense_manifest.json
moe_manifest.json
configs/
  dense/       # 16 configurations
  moe/         # 16 configurations
```

Regenerate and verify the checked-in JSON deterministically:

```bash
uv run python -m experiments.dense_moe_scaling.generate_configs
```

## Colab notebooks and package-runner commands

Notebooks 02 and 04 are self-contained Colab workflows. They visibly define the
shared dataset, sampler, model primitives, objective, optimizer schedule, and
evaluation code. Their durable outputs are deliberately paired:

```text
$DRIVE_ROOT/scaling/notebook_02_dense/dense_results.json
$DRIVE_ROOT/scaling/notebook_04_moe/moe_results.json
$DRIVE_ROOT/scaling/dense_moe_comparison/paired_results.json
```

Notebook 04 reads the first path before producing its dense-versus-MoE tables.
Both notebooks use the same seed-42 split and sampler, batch size 2,048,
evaluation batch size 4,000, answer-only objective, FP32 parameters, FP16
compute, optimizer schedule, and four independent horizons. Notebook 04 uses a
fresh GPU worker for every coordinate so process exit releases device memory;
notebook 02 retains its proven in-notebook dense execution and resumable pickle
checkpoints.

The canonical package runner is a separate non-notebook interface. Optional
sharding arguments may be appended to these commands:

```bash
python -m experiments.dense_moe_scaling.run_grid \
  --architecture dense \
  --output-root "$DRIVE_ROOT/scaling/dense" \
  --work-root /content/scaling-work/dense/runs \
  --compilation-cache /content/jax-compilation-cache \
  --rerun-only-missing

python -m experiments.dense_moe_scaling.analyze \
  --architecture dense \
  --output-root "$DRIVE_ROOT/scaling/dense"
```

For MoE and comparison:

```bash
python -m experiments.dense_moe_scaling.run_grid \
  --architecture moe \
  --output-root "$DRIVE_ROOT/scaling/moe" \
  --work-root /content/scaling-work/moe/runs \
  --compilation-cache /content/jax-compilation-cache \
  --rerun-only-missing

python -m experiments.dense_moe_scaling.analyze \
  --architecture moe \
  --output-root "$DRIVE_ROOT/scaling/moe"

python -m experiments.dense_moe_scaling.compare \
  --dense-root "$DRIVE_ROOT/scaling/dense" \
  --moe-root "$DRIVE_ROOT/scaling/moe" \
  --output-root "$DRIVE_ROOT/scaling/comparison"
```

The runner supports `--shard-index`, `--shard-count`, `--model`, `--steps`,
`--limit`, `--dry-run`, `--collect-only`, and `--continue-on-error`. It has no
checkpoint-pruning mode and no destructive force-rerun mode. When local work
storage differs from durable storage, it copies quiescent artifacts to Drive
every 60 seconds (configurable with `--sync-interval-seconds`) and once more
when the subprocess exits. In-progress Orbax temporary directories are never
copied.

## Durable directory structure

```text
$DRIVE_ROOT/scaling/
  dense/
    root_identity.json
    protocol.json
    manifest.json
    results.json
    analysis/{runs.csv,model_horizons.csv,transition_table.csv,summary.json}
    runs/<dense-run-id>/
      {config,protocol,run_spec,split_metadata,environment,summary,result}.json
      history.jsonl
      failures.csv
      sample_predictions.txt
      checkpoints/{best,latest}/{metadata.json,state/}
  moe/
    # identical normalized structure for the 16 MoE runs
  comparison/
    paired_runs.csv
    combined_compute_frontier.csv
    combined_wallclock_frontier.csv
    summary.json
    figures/{compute_frontier,wallclock_frontier}.svg
```

An output root is accepted only when its path suffix and `root_identity.json`
match the requested architecture and protocol fingerprint. A non-empty root
without that identity is rejected. A run is complete only if all required
files, both resumable checkpoints, the exact endpoint step, and protocol,
configuration, split, and run-identity fingerprints validate.

## Single-seed reporting

There is one observation at every architecture/model/horizon coordinate.
Analysis therefore writes `uncertainty: null` and
`uncertainty_status: unavailable_single_seed`. It does not calculate sample
standard deviation and does not substitute zero uncertainty.
