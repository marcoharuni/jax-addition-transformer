import jax.numpy as jnp, numpy as np
from jax_addition_transformer.normalization import LayerNorm, RMSNorm


def test_layernorm_reference():
    x = jnp.array([[1.0, 2.0, 4.0]])
    layer = LayerNorm(3)
    ref = (x - x.mean(-1, keepdims=True)) / jnp.sqrt(
        ((x - x.mean(-1, keepdims=True)) ** 2).mean(-1, keepdims=True) + 1e-5
    )
    np.testing.assert_allclose(layer(x), ref, rtol=1e-5)


def test_rmsnorm_reference():
    x = jnp.array([[1.0, 2.0, 4.0]])
    layer = RMSNorm(3)
    ref = x / jnp.sqrt((x * x).mean(-1, keepdims=True) + 1e-5)
    np.testing.assert_allclose(layer(x), ref, rtol=1e-5)
