import jax.numpy as jnp, numpy as np
from jax_addition_transformer.losses import answer_mask, masked_cross_entropy


def test_mask_and_ignored_targets():
    mask = answer_mask()
    np.testing.assert_array_equal(mask, [0] * 11 + [1] * 4)
    logits = jnp.zeros((2, 15, 13))
    targets = jnp.zeros((2, 15), dtype=jnp.int32)
    base, _ = masked_cross_entropy(logits, targets)
    ignored = targets.at[:, :11].set(7)
    assert float(masked_cross_entropy(logits, ignored)[0]) == float(base)
    favored = logits.at[:, 11:, 0].set(10)
    assert float(masked_cross_entropy(favored, targets)[0]) < float(base)
