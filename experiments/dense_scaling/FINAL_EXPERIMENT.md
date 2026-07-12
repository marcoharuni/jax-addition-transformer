# Final dense scaling experiment

The pilot identified a sharp transition into perfect arithmetic accuracy, but it
did not identify stable Chinchilla-style exponents. The final dense experiment
therefore uses a larger, multi-seed rectangular grid.

## Design

- Five model sizes:
  - 162,176 parameters
  - 643,840 parameters
  - 2,164,608 parameters
  - 5,123,584 parameters
  - 10,000,000 parameters
- Eight independent training budgets:
  - 50, 75, 100, 125, 175, 250, 400, 750 steps
- Three seeds:
  - 42, 123, 2026
- Batch size:
  - 2,048 examples
- Total:
  - 120 independently trained models

Each model/data point receives its own learning-rate schedule: linear warmup for
10% of the run, followed by cosine decay to 1e-4 at the final step. This makes
each observation an independently trained endpoint rather than a checkpoint
from a longer schedule.

## Why independent runs

A checkpoint at step 50 from a 750-step cosine schedule is not equivalent to a
model trained with a schedule designed to finish at step 50. The final grid uses
independent runs so that every data budget is evaluated at the end of its own
declared optimization protocol.

## Reproduction

Generate the configurations:

```bash
uv run python experiments/dense_scaling/final_configs.py
```

Inspect the commands without training:

```bash
uv run python experiments/dense_scaling/run_final_grid.py --dry-run
```

Run a single point:

```bash
uv run python experiments/dense_scaling/run_final_grid.py \
  --model dense_0p64m \
  --steps 75 \
  --seed 42
```

Run the complete grid:

```bash
uv run python experiments/dense_scaling/run_final_grid.py \
  --continue-on-error
```

Completed checkpoints are removed by default to keep the experiment directory
small. Pass `--keep-checkpoints` only when trained weights are required.
