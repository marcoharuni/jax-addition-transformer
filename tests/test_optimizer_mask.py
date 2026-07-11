import jax
from flax import nnx
from jax_addition_transformer.optimizers import decay_mask


def test_only_attention_and_ffn_kernels_decay(tiny_model):
    _, params = nnx.split(tiny_model, nnx.Param)
    mask = decay_mask(params)
    values = jax.tree.leaves(mask)
    assert sum(map(bool, values)) == 6
    assert len(values) > 6
