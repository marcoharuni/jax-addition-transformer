# JAX Addition Transformer

[![Open notebook 01 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marcoharuni/jax-addition-transformer/blob/main/notebooks/01_build_train_exact_10m.ipynb)

A decoder-only transformer written from scratch with JAX primitives and Flax
NNX, trained to add every ordered pair of integers from 0 through 999. The
default dense model has exactly **10,000,000 trainable parameters**.

## Verified exact-10M dense result

The default model was trained on a Google Colab NVIDIA T4 and evaluated with
greedy autoregressive generation over the complete one-million-pair domain.

| Measurement | Result |
|---|---:|
| Parameters | 10,000,000 |
| Training steps | 750 |
| Training time after JAX compilation | 6.89 minutes |
| Validation | 20,000 / 20,000 |
| Unseen test | 780,000 / 780,000 |
| Complete domain | 1,000,000 / 1,000,000 |
| Carry-pattern slices | 100% in all 8 patterns |
| Operand-length slices | 100% in all 9 combinations |
| Invalid generations | 0 |
| Failures | 0 |

The exhaustive evaluation took 144.31 seconds. Machine-generated measurements
are stored in [`artifacts/t4-run`](artifacts/t4-run).

![Training loss, token accuracy, validation exact match, and gradient norm](assets/training_curves.png)

![Accuracy by carry pattern and operand length](assets/exhaustive_accuracy.png)

This result applies only to the fixed `0..999` domain and the representation
described below. It is not a claim about four-digit or arbitrary-length
addition.

## Task and model

Each complete record has 16 characters:

```text
123 + 456 = 9750
```

Operands are zero-padded to three digits. The normal answer `0579` is reversed
to `9750`, so the causal model generates the units digit first and follows the
direction of carry propagation.

The vocabulary contains the ten digits, space, `+`, and `=`. There is no
padding token. Training uses shifted next-token targets, with loss applied only
to the four answer positions.

| Setting | Value |
|---|---:|
| Decoder blocks | 5 |
| Model width | 320 |
| Attention | 5-head MHA |
| Head dimension | 64 |
| FFN width | 2,480 |
| Normalization | Pre-LayerNorm |
| Positions | Learned absolute embeddings |
| Token/output weights | Tied |
| Precision | FP32 parameters, FP16 matrix inputs |

The complete domain contains `1,000 × 1,000 = 1,000,000` ordered pairs:

| Split | Pairs |
|---|---:|
| Training | 200,000 |
| Validation | 20,000 |
| Unseen test | 780,000 |

The deterministic seed-42 split is stratified by operand lengths and carry
pattern. Each training batch combines uniform examples with carry-balanced
examples. Optimization uses AdamW, global-norm clipping at 1.0, and cosine
decay from a peak learning rate of `1e-3` to `1e-4`.

## Verified standalone ragged-dot MoE result

[![Open notebook 03 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marcoharuni/jax-addition-transformer/blob/main/notebooks/03_moe_ragged_dot_t4.ipynb)

The standalone no-drop top-2 Mixture-of-Experts transformer uses exactly two
`jax.lax.ragged_dot` projections per MoE layer. Its 10M-active configuration
has 17,942,400 stored parameters, a 10,006,400 active-parameter proxy, four
experts, and expert width 1,240.

A seed-42 Colab T4 run completed in 7.86 minutes and produced:

| Measurement | Result |
|---|---:|
| Validation exact match | 20,000 / 20,000 |
| Complete domain | 999,981 / 1,000,000 |
| Unseen test | 779,984 / 780,000 |
| Failures | 19 |
| Invalid generations | 0 |
| Exhaustive evaluation | 779.29 seconds |

Verified evidence is stored in
[`artifacts/moe-comparison-t4/seed-42`](artifacts/moe-comparison-t4/seed-42).
This standalone result is separate from the scaling study below.

## Current dense-versus-MoE scaling study

The current study compares four dense models with four matched-active MoE
models at four independently trained horizons:

```text
4 model sizes × 4 horizons (50, 125, 300, 750) × seed 42
```

This produces 16 dense runs and 16 MoE runs: 32 total independent runs. The
dense sizes are parameter counts of approximately 0.16M, 0.64M, 2.16M, and
10M. Each MoE model is matched to its dense reference by active-parameter
proxy, with stored parameters reported separately. Dense and MoE runs share
the same dataset, split, vocabulary, answer-only objective, batch size 2,048,
optimizer schedule, parameter and compute precision, and evaluation procedure.
Notebook 04 preserves top-2 routing and two `jax.lax.ragged_dot` projections per
MoE layer.

**Status: complete.** All 16 dense and 16 matched-active MoE runs finished.
Around the task's algorithmic transition, MoE reached lower validation loss at
comparable estimated training FLOPs, while the current ragged-dot MoE
implementation took substantially longer in measured T4 wall-clock time. At
high exposure, both families approached saturation. This is a single-seed,
task-specific result rather than a universal claim about MoE scaling.

## Notebooks

| Notebook | Purpose | Colab |
|---|---|---|
| [`01_build_train_exact_10m.ipynb`](notebooks/01_build_train_exact_10m.ipynb) | Build, train, and exhaustively evaluate the exact-10M dense model | [Open on `main`](https://colab.research.google.com/github/marcoharuni/jax-addition-transformer/blob/main/notebooks/01_build_train_exact_10m.ipynb) |
| [`02_dense_scaling_laws.ipynb`](notebooks/02_dense_scaling_laws.ipynb) | Run and analyze the current 16-coordinate dense sweep | [Open on `main`](https://colab.research.google.com/github/marcoharuni/jax-addition-transformer/blob/main/notebooks/02_dense_scaling_laws.ipynb) |
| [`03_moe_ragged_dot_t4.ipynb`](notebooks/03_moe_ragged_dot_t4.ipynb) | Reproduce the verified standalone ragged-dot MoE result | [Open on `main`](https://colab.research.google.com/github/marcoharuni/jax-addition-transformer/blob/main/notebooks/03_moe_ragged_dot_t4.ipynb) |
| [`04_moe_scaling_t4.ipynb`](notebooks/04_moe_scaling_t4.ipynb) | Run and analyze the current MoE sweep and compare it with dense results | [Open on `main`](https://colab.research.google.com/github/marcoharuni/jax-addition-transformer/blob/main/notebooks/04_moe_scaling_t4.ipynb) |

Select **Runtime → Change runtime type → T4 GPU**, then use **Runtime → Run
all**.

## Artifacts

Committed verified artifacts:

```text
artifacts/t4-run/                         exact-10M dense result
artifacts/moe-comparison-t4/seed-42/      standalone ragged-dot MoE result
```

## Local development

```bash
uv sync --group dev
uv run pytest -q
uv run ruff check .
uv run jat-inspect --config configs/exact_10m_t4.json
```

## Scope

This is an arithmetic language-model experiment, not a general chatbot. Its
strongest verified claim is exact greedy generation across the complete fixed
three-digit addition domain represented by notebook 01. The completed dense-versus-MoE comparison is a single-seed,
task-specific T4 study and does not establish universal scaling laws.

## License and author

MIT © 2026 Marco Haruni.
