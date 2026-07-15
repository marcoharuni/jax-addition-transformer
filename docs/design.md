# Design

## Fixed-width arithmetic language

For task width `d`, the record is `A…A + B…B = R…R`: `d` digits per operand and `d+1` answer digits. Its length is `3d+7`; the model input has length `3d+6`. For `d=3`, `123 + 456 = 9750` encodes normal answer 0579 in reverse. At generation step `k`, the model therefore predicts the sum column at power `10^k` after earlier output digits have represented lower-column carries implicitly.

With logits `z[b,t,v]`, shifted targets `y[b,t]`, and `m[t]=1` only for the final `d+1` positions, the objective is

```text
L = -1 / (B sum_t m[t]) sum_b sum_t m[t] log softmax(z[b,t])[y[b,t]].
```

For the default, `m = [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1]`.

## Decoder and exact parameter count

Token lookup produces `[B,T,320]`; learned positions `[T,320]` are added. Each Pre-LayerNorm block computes `x + Attention(LN(x))`, then `x + FFN(LN(x))`. Five blocks preserve `[B,T,320]`; final normalization and multiplication by the transpose of the `[13,320]` token table produces `[B,T,13]` logits.

Each default attention has four `[320,320]` matrices: 409,600 parameters. Its GELU FFN has `[320,2480]` and `[2480,320]`: 1,587,200. Two LayerNorm scale/bias pairs contribute 1,280, for 1,998,080 per block and 9,990,400 across five. Token positions and final norm add `4,160 + 4,800 + 640`, yielding exactly 10,000,000. The code checks both this algebra and the NNX `Param` tree.

## Normalization and positions

LayerNorm uses `(x-mean(x))/sqrt(mean((x-mean(x))²)+epsilon)` followed by learned scale and bias. RMSNorm omits centering and bias: `x/sqrt(mean(x²)+epsilon)` followed by scale. Both operate in float32.

Learned absolute positions instantiate `[T,D]` parameters. RoPE instead rotates adjacent query/key features by position-dependent angles `p * base^(-2i/head_dim)` and adds no position parameters. These are neutral experiment alternatives; no comparative result is asserted.

## Attention

Default projections form Q, K, V `[B,T,5,64]`. Scores are `QKᵀ/sqrt(64)`, shape `[B,5,T,T]`, in float32. A boolean lower triangle permits key position `s` iff `s <= t`; masked values receive the most negative finite float32 number before softmax. Probabilities sum over visible keys, and multiplying V returns `[B,T,5,64]`.

MHA uses five KV heads. GQA uses a configurable divisor of five and logically repeats each KV group for its query heads. MQA uses one KV head. Divisibility is validated. The normal path does not return attention maps; an inspection path does.

## Feed-forward alternatives

The default is `W_out GELU(W_in x)`. ReLU and squared ReLU replace the activation. SwiGLU computes `W_out(SiLU(W_in x) * W_gate x)`; GEGLU substitutes GELU. Gated forms instantiate their additional matrix only when selected.

## Precision and initialization

Every matrix element is initialized independently as `Normal(0, 0.02²)` by default. Parameters and optimizer state remain float32. On the T4 path, matrix operands are cast to float16 and dot operations request float32 preferred accumulation; normalization, attention scores/softmax, logits, and loss remain float32. CPU tests select float32 computation. Finite loss, gradient, and updated-parameter checks turn numerical failure into an explicit exception.

## Optimization

The schedule linearly warms to the peak and then follows cosine decay. AdamW moments use beta1 0.9, beta2 0.99, and epsilon 1e-8. Global-norm clipping precedes moment transformations. A structural parameter-path mask applies decoupled weight decay only to `kernel` matrices inside attention and FFN modules; normalization, embeddings, learned positions, and bias vectors are excluded.

## Teacher forcing and greedy generation

Teacher forcing supplies correct preceding output digits, so token accuracy measures conditional next-token predictions. Greedy exact match starts with only the 12-token prompt, repeatedly runs the transformer, and inserts argmax tokens into a fixed 15-position buffer. It is stricter: all four digits must be valid and correct. Initial zeros in future buffer slots are not padding; the causal mask makes them invisible until their positions become current.

Evaluation separately measures validation, the 780,000 unseen pairs, and all one million pairs. It groups exact outcomes by operand-length pair, units-to-hundreds carry string, and answer length; checks swapped-input consistency; and records every invalid or incorrect generation.

## Top-2 ragged-dot MoE

The MoE architecture preserves the dense attention stack and replaces each
feed-forward sublayer with four experts. A float32 linear router produces
probabilities over experts for every flattened token. The top two experts are
selected, their probabilities are renormalized, and the resulting assignments
are stably sorted by expert ID. `jax.lax.ragged_dot` then evaluates the grouped
up projection, GELU, and grouped down projection without padding expert groups
to a fixed capacity. Weighted outputs are scatter-added back to the original
token order. Every token receives two routes; no assignment is dropped.

The auxiliary objective is

```text
L_total = L_answer + 1e-2 L_balance + 1e-3 L_z.
```

`L_balance` multiplies the mean router probability and all-top-k assignment
fraction for each expert, then sums and scales by the number of experts.
`L_z` is the mean squared log-normalizer of the float32 router logits. Final
reports retain answer loss, both unweighted auxiliary losses, router entropy,
per-expert assignment fractions, load coefficient of variation, and
maximum-to-mean load ratio.

## Matched-active scaling rule

For a dense FFN width `F` and top-k value `k=2`, each expert uses width `F/k`.
The selected expert matrices therefore match the dense FFN matrix work:

```text
2 dense projections × D × F
=
k selected experts × 2 projections × D × (F/k).
```

All routers are active, so the MoE active-parameter proxy is slightly larger
than the dense parameter count by `layers × D × experts`. Both the exact active
proxy and the larger stored total are reported. The approximate training-compute
axis uses `6 × active parameters × sequence tokens`; routing, sorting, dispatch,
scatter-add, and hardware efficiency are intentionally excluded from that
proxy and captured separately by measured T4 wall-clock time.

## Current dense-versus-MoE scaling study

The study uses four dense models (`dense_0p16m`, `dense_0p64m`,
`dense_2p16m`, and `dense_10m`) and four matched-active MoE models. Each model
is trained independently for 50, 125, 300, and 750 updates, producing 16 dense
and 16 MoE runs. The split, initialization, and sampler use seed 42.

Every horizon starts from a fresh initialization and owns its warmup-cosine
schedule. Dense and MoE use the same dataset, answer-only objective, optimizer,
precision, batch size, and evaluation procedure. Dense router losses are exact
zeros; MoE adds its measured balance and router z-loss terms. Validation answer
cross-entropy is always reported separately.

Both architectures rematerialize every transformer block. Estimated compute is
reported as `6 × active_parameter_proxy × input_tokens`; stored parameters and
measured T4 wall-clock time are reported separately. Notebooks 02 and 04 contain
the complete training and analysis workflows.
