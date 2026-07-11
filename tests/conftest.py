import pytest
from flax import nnx

from jax_addition_transformer.config import ModelConfig
from jax_addition_transformer.model import AdditionTransformer


@pytest.fixture
def tiny_config():
    return ModelConfig(
        n_layers=1,
        d_model=20,
        n_heads=2,
        n_kv_heads=2,
        d_ff=32,
        max_input_length=15,
        compute_dtype="float32",
    )


@pytest.fixture
def tiny_model(tiny_config):
    return AdditionTransformer(tiny_config, rngs=nnx.Rngs(params=0))
