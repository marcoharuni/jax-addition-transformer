import jax.numpy as jnp, pytest
from flax import nnx
from jax_addition_transformer.config import ModelConfig
from jax_addition_transformer.ffn import FeedForward


@pytest.mark.parametrize("kind", ["gelu", "relu", "relu2", "swiglu", "geglu"])
def test_ffn_variants(kind):
    c = ModelConfig(
        n_layers=1,
        d_model=20,
        n_heads=2,
        n_kv_heads=2,
        d_ff=32,
        ffn_type=kind,
        compute_dtype="float32",
    )
    assert FeedForward(c, rngs=nnx.Rngs(params=0))(jnp.ones((2, 5, 20))).shape == (2, 5, 20)
