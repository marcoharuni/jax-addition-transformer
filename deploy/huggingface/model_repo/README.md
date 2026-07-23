---
license: mit
library_name: jax
pipeline_tag: text-generation
tags:
  - jax
  - flax
  - arithmetic
  - addition
  - transformer
datasets:
  - synthetic
language:
  - en
model-index:
  - name: JAX Addition Transformer 10M
    results:
      - task:
          type: text-generation
          name: Fixed three-digit addition
        dataset:
          type: synthetic
          name: Complete ordered 0..999 addition domain
        metrics:
          - type: exact_match
            value: 1.0
            name: Complete-domain exact match
---

# JAX Addition Transformer — exact 10M

This is the released dense checkpoint for a decoder-only transformer written
from scratch with JAX primitives and trained to add ordered pairs of integers
from `0` through `999`. The model has exactly **10,000,000 trainable
parameters**.

Try the [live Gradio Space](https://huggingface.co/spaces/marcoharuni95/jax-addition-transformer).

## Verified result

The seed-42 model was trained for 750 steps on a Google Colab NVIDIA T4. The
release was evaluated with genuine greedy autoregressive generation over all
1,000,000 ordered operand pairs.

| Evaluation slice | Correct | Total | Exact match |
|---|---:|---:|---:|
| Training split | 200,000 | 200,000 | 100% |
| Validation split | 20,000 | 20,000 | 100% |
| Unseen test split | 780,000 | 780,000 | 100% |
| Complete domain | 1,000,000 | 1,000,000 | 100% |

All eight carry-pattern slices and all nine operand-length slices reached 100%.
There were zero invalid generations and zero failures. The machine-readable
release evidence is included as `history.json`, `results.json`, and
`failures.csv`.

## Architecture

| Setting | Value |
|---|---:|
| Decoder blocks | 5 |
| Model width | 320 |
| Attention heads | 5 |
| Head dimension | 64 |
| Feed-forward width | 2,480 |
| Context/model input length | 15 |
| Vocabulary | 13 characters |
| Normalization | Pre-LayerNorm |
| Positions | Learned absolute embeddings |
| Output head | Tied token embeddings |
| Parameter / matmul input dtype | float32 / float16 |
| Trainable parameters | 10,000,000 |

The vocabulary is `0123456789 +=`; its exact mapping is recorded in
`tokenizer_config.json`.

## Representation and generation

Operands are zero-padded to three digits. A normal answer is padded to four
digits and reversed internally:

```text
123 + 456 = 9750
```

Here `9750` is `0579` reversed. This lets a causal model generate the units
digit first and then follow the direction of carry propagation. Inference
greedily generates exactly four tokens and rejects any non-digit generation
before reversing the sequence for display.

## Usage

Install the pinned dependencies in `requirements.txt`, then:

```python
from inference import AdditionModel

model = AdditionModel.from_pretrained(
    "marcoharuni95/jax-addition-transformer-10m"
)
print(model.add(347, 928))
# 1275
```

The first call compiles the JAX generation graph for its batch shape.

## Intended use and limitations

This checkpoint is intended for reproducible arithmetic-transformer research,
education, and demonstrations within its fixed domain.

- Both operands must be integers from `0` through `999`.
- Only addition is supported.
- It does not support negative values, decimals, subtraction,
  multiplication, division, four-digit operands, or arbitrary-length
  arithmetic.
- It is not a general conversational language model.
- Natural-language parsing belongs to the Space; the model itself consumes the
  exact fixed-width character prompt.

## Weights and provenance

`model.safetensors` contains 54 deterministic float32 tensors and no optimizer
state. It was converted once from the trusted v0.1.0 release pickle after
restoring its Flax NNX state with the notebook-evidenced Flax 0.12.2
compatibility environment. Normal inference does not use pickle or Flax.
Checksums and conversion metadata are included in
`model.safetensors.metadata.json`.

Source and training notebook:
[marcoharuni/jax-addition-transformer](https://github.com/marcoharuni/jax-addition-transformer)

## Citation

```bibtex
@software{haruni_2026_jax_addition_transformer,
  author = {Marco Haruni},
  title = {JAX Addition Transformer},
  version = {0.1.0},
  year = {2026},
  url = {https://github.com/marcoharuni/jax-addition-transformer}
}
```

## License

MIT © 2026 Marco Haruni.
