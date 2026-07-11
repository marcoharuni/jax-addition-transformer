from flax import nnx
from jax_addition_transformer.config import ModelConfig
from jax_addition_transformer.model import (
    AdditionTransformer,
    assert_parameter_count,
    count_parameters,
)


def test_exact_default_parameter_tree():
    model = AdditionTransformer(ModelConfig(), rngs=nnx.Rngs(params=42))
    assert count_parameters(model) == 10_000_000
    assert assert_parameter_count(model) == 10_000_000
    assert model.lm_head is None
