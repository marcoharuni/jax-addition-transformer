import jax.numpy as jnp, numpy as np
from jax_addition_transformer.positional import apply_rope


def test_rope_preserves_pair_norms_and_shape():
    x = jnp.arange(2 * 4 * 3 * 6, dtype=jnp.float32).reshape(2, 4, 3, 6)
    y = apply_rope(x)
    assert y.shape == x.shape
    np.testing.assert_allclose((x * x).sum(-1), (y * y).sum(-1), rtol=2e-5, atol=2e-4)
