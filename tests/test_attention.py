import jax.numpy as jnp, numpy as np, pytest
from flax import nnx
from jax_addition_transformer.attention import CausalAttention, scaled_dot_product_attention
from jax_addition_transformer.config import ModelConfig


@pytest.mark.parametrize(("kind", "kv"), [("mha", 4), ("gqa", 2), ("mqa", 1)])
def test_attention_variants_shapes_probabilities_and_causality(kind, kv):
    c = ModelConfig(
        n_layers=1,
        d_model=32,
        n_heads=4,
        n_kv_heads=kv,
        d_ff=32,
        attention_type=kind,
        position_type="rope",
        compute_dtype="float32",
    )
    layer = CausalAttention(c, rngs=nnx.Rngs(params=0))
    x = jnp.ones((2, 7, 32))
    y, p = layer(x, True)
    assert y.shape == (2, 7, 32) and p.shape == (2, 4, 7, 7)
    np.testing.assert_allclose(p.sum(-1), 1, atol=1e-6)
    assert np.allclose(np.triu(np.asarray(p[0, 0]), 1), 0)
    assert np.isfinite(y).all()


def test_future_values_do_not_change_earlier_attention_output():
    q = jnp.ones((1, 4, 2, 3))
    k = jnp.ones_like(q)
    v = jnp.arange(24, dtype=jnp.float32).reshape(1, 4, 2, 3)
    first, _ = scaled_dot_product_attention(q, k, v)
    changed, _ = scaled_dot_product_attention(q, k, v.at[:, 3].set(999))
    np.testing.assert_allclose(first[:, :3], changed[:, :3])
