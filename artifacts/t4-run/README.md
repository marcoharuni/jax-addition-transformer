# Verified Colab T4 run

This directory contains machine-generated artifacts from the verified default run.

| Measurement | Value |
|---|---:|
| Trainable parameters | 10,000,000 |
| Hardware | NVIDIA T4 (Google Colab) |
| Training steps | 750 |
| JAX compilation + first step | 10.30 s |
| Training time after compilation | 413.23 s (6.89 min) |
| Exhaustive evaluation time | 144.31 s |
| Training pairs | 200,000 |
| Validation pairs | 20,000 |
| Unseen test pairs | 780,000 |
| Complete-domain accuracy | 1,000,000 / 1,000,000 |
| Unseen-test accuracy | 780,000 / 780,000 |
| Failures | 0 |
| Invalid generations | 0 |

The result applies to the fixed domain of ordered additions with operands in `0..999` and the notebook's fixed-width, reversed-answer representation. It is not a claim about four-digit or arbitrary-length addition.

Files:

- `history.json`: measured training and validation history.
- `results.json`: exhaustive evaluation and slice metrics.
- `failures.csv`: failure schema and rows; this run contains only the header because no failures were observed.

The 40 MB checkpoint should be attached to a GitHub Release rather than committed to normal Git history.
