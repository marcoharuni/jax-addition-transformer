# Standalone ragged-dot MoE result

This directory preserves the checkpoint-free evidence from the standalone
`notebooks/03_moe_ragged_dot_t4.ipynb` run on a Google Colab NVIDIA T4.

## Seed 42

| Measurement | Result |
|---|---:|
| Stored parameters | 17,942,400 |
| Active-parameter proxy | 10,006,400 |
| Training steps | 175 |
| Training segment | 7.86 minutes |
| Validation exact match | 20,000 / 20,000 |
| Complete-domain exact match | 999,981 / 1,000,000 |
| Unseen-test exact match | 779,984 / 780,000 |
| Failures | 19 |
| Invalid generations | 0 |
| Exhaustive evaluation | 779.29 seconds |

The 19 failures are listed in `seed-42/failures.csv`. They are confined to
three-digit-plus-three-digit examples and concentrated in chained-carry cases.

The large `best_checkpoint.pkl` and `latest_checkpoint.pkl` files are not
committed. They remain external run assets.

## Important protocol distinction

This standalone result is a verified end-to-end demonstration of the top-2
`jax.lax.ragged_dot` model. It is **not silently inserted into the final
package-based 120-run scaling grid**. The standalone notebook uses its own
visible model definition, residual-scaled output initialization, a top-1
Switch-style assignment fraction in its balance term, router-excluding weight
decay, pickle checkpoints, and an exhaustive evaluation flow. The package
runner uses the repository model, an all-top-k balance fraction, the package's
structural FFN decay mask, and Orbax run artifacts.

The final dense-versus-MoE scaling comparison therefore reruns every MoE grid
point, including 10M-active / 175 steps / seed 42, under one consistent package
protocol. This preserves comparability rather than mixing results from two
implementations.
