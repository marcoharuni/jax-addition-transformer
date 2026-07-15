# Matched 10M Dense-versus-MoE Comparison

## Research question

At approximately equal active parameter count, how does a top-2
Mixture-of-Experts transformer compare with the exact 10,000,000-parameter
dense transformer on fixed-width three-digit addition?

This is a matched architecture comparison at one model size and one training
budget.

## Architectures

| Property | Dense | Matched MoE |
|---|---:|---:|
| Total parameters | 10,000,000 | 17,942,400 |
| Active-parameter proxy | 10,000,000 | 10,006,400 |
| Transformer layers | 5 | 5 |
| Model width | 320 | 320 |
| Attention heads | 5 | 5 |
| Dense FFN width | 2,480 | — |
| Experts per layer | — | 4 |
| Selected experts per token | — | 2 |
| Expert hidden width | — | 1,240 |

The active-parameter proxy is an architectural accounting measure. It is not
a direct measurement of runtime or hardware cost.

## Routing path

```text
Input tokens
    |
    v
Float32 router logits
    |
    v
Top-2 expert selection
    |
    v
Normalize selected probabilities
    |
    v
Duplicate assignments
    |
    v
Group assignments by expert
    |
    v
jax.lax.ragged_dot up projection
    |
    v
GELU
    |
    v
jax.lax.ragged_dot down projection
    |
    v
Weighted scatter-add
    |
    v
Original token order

Every token receives exactly two expert assignments. No assignments are
dropped.

Final experiment

The MoE comparison contains three independently initialized runs:

Run	Steps	Seed
moe_10m_s0175_seed42	175	42
moe_10m_s0175_seed123	175	123
moe_10m_s0175_seed2026	175	2026

The dense and MoE runs use the same task, splits, vocabulary, batch size,
sequence length, loss mask, optimizer family, schedule rule, training budget,
seeds, evaluation protocol, and NVIDIA T4 hardware.

Reported measurements

The final comparison reports:

validation loss;
greedy exact match;
mean and sample standard deviation across three seeds;
compilation time;
steady-state training time;
total training time;
router entropy;
expert assignment fractions;
expert-load coefficient of variation;
load-balancing loss;
router z-loss;
total parameter count;
active-parameter proxy.
Reproducible notebook

Use:

notebooks/03_moe_ragged_dot_t4.ipynb

The notebook follows the clean dense notebook from top to bottom. It installs
the pinned CUDA environment, defines the data and model visibly, checks the
exact stored and active parameter counts, verifies `ragged_dot`, trains one
configured seed, saves atomic best/latest pickle checkpoints, restores the best
model, evaluates the complete domain, and exposes interactive inference.

The default is seed 42. Set `SEED` to 123 or 2026 and run the notebook in a
fresh runtime to create the other independent runs. Each seed has its own
Google Drive directory. The notebook does not hide training behind the
repository comparison runner.

Interpretation

The results apply to this architecture, task, training budget, and hardware
environment. Total parameters, active-parameter proxy, and measured runtime
must be reported separately.
