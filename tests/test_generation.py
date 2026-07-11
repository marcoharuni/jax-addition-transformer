import jax, jax.numpy as jnp, numpy as np
from jax_addition_transformer.generation import greedy_generate


class PositionModel:
    def __call__(self, x):
        batch, length = x.shape
        desired = jnp.arange(length) % 10
        return jnp.broadcast_to(jax.nn.one_hot(desired, 13), (batch, length, 13))


def test_fixed_buffer_generates_four_digits():
    generated, valid = greedy_generate(PositionModel(), jnp.zeros((3, 12), dtype=jnp.int32))
    assert generated.shape == (3, 4)
    np.testing.assert_array_equal(generated[0], [1, 2, 3, 4])
    assert valid.all()
