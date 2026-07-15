# Dense-versus-MoE scaling

This directory defines the current scaling study:

- dense models: `dense_0p16m`, `dense_0p64m`, `dense_2p16m`, and `dense_10m`;
- four matched-active MoE models with four experts and top-2 routing;
- independent horizons of 50, 125, 300, and 750 optimizer updates;
- seed 42 for the split, initialization, and sampler;
- 16 dense runs and 16 MoE runs.

Dense sizes are parameter counts. MoE sizes are matched by active-parameter
proxy, with stored parameters reported separately. Both architectures use the
same data, answer-only objective, optimizer schedule, precision, batch size,
rematerialization policy, and endpoint validation.

## Checked-in study definition

```text
protocol.json
manifest.json
dense_manifest.json
moe_manifest.json
configs/
  dense/       # 16 configurations
  moe/         # 16 configurations
```

Regenerate these files deterministically with:

```bash
uv run python -m experiments.dense_moe_scaling.generate_configs
```

## Colab execution

Notebook 02 contains the dense implementation, training, validation, tables,
figures, and scaling analysis. Notebook 04 contains the matched MoE model,
sorted top-2 dispatch, two `jax.lax.ragged_dot` projections per layer, routing
diagnostics, and paired dense-versus-MoE analysis.

Both notebooks save their results to Google Drive and skip compatible completed
runs. Notebook 04 launches one GPU worker per model-horizon coordinate so each
process releases device memory when it exits.
