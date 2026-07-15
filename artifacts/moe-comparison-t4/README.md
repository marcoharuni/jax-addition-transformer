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

## Study relationship

This is the verified standalone result produced by notebook 03. Its 175-step
run is not a coordinate in the current 50, 125, 300, and 750-step scaling
study, so it is reported separately rather than mixed into notebooks 02 and 04.
