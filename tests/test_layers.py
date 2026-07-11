import jax.numpy as jnp, numpy as np
from flax import nnx
from jax_addition_transformer.layers import Linear, embedding_lookup


def test_linear_is_direct_matrix_multiplication():
    layer = Linear(2, 3, rngs=nnx.Rngs(params=0))
    x = jnp.array([[1.0, 2.0]])
    np.testing.assert_allclose(layer(x), x @ layer.kernel.value, rtol=1e-6)
    table = jnp.arange(12).reshape(4, 3)
    np.testing.assert_array_equal(
        embedding_lookup(table, jnp.array([2, 0])), np.asarray(table)[[2, 0]]
    )
