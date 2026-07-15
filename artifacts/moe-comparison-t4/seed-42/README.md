# Seed 42 files

- `config.json`: exact standalone notebook configuration.
- `history.json`: training metrics, compilation time, validation endpoint, and routing loads.
- `results.json`: exhaustive one-million-pair evaluation.
- `failures.csv`: all 19 incorrect additions.
- `SHA256SUMS.txt`: checksums for the four machine-generated evidence files.

The original checkpoint-bearing ZIP is intentionally kept outside Git because
its two pickle checkpoints dominate the archive size.
