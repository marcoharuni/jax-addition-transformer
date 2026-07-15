import gc

import jax
import pytest
from flax import nnx

from jax_addition_transformer.config import ModelConfig
from jax_addition_transformer.model import (
    AdditionTransformer,
    active_parameter_proxy,
    assert_parameter_count,
)


FINAL_DENSE_MODELS = {
    "dense_0p16m": (2, 64, 1, 496, 162_176, 289_664, 162_688),
    "dense_0p64m": (2, 128, 2, 992, 643_840, 1_152_768, 644_864),
    "dense_2p16m": (3, 192, 3, 1488, 2_164_608, 3_881_088, 2_166_912),
    "dense_10m": (5, 320, 5, 2480, 10_000_000, 17_942_400, 10_006_400),
}


def test_all_matched_parameter_counts_from_real_trees():
    for row in FINAL_DENSE_MODELS.values():
        gc.collect()
        jax.clear_caches()
        layers, width, heads, dense_d_ff, dense_total, moe_total, active_total = row
        dense = ModelConfig(
            n_layers=layers,
            d_model=width,
            n_heads=heads,
            n_kv_heads=heads,
            d_ff=dense_d_ff,
        )
        dense_model = AdditionTransformer(dense, rngs=nnx.Rngs(params=0))
        assert assert_parameter_count(dense_model) == dense_total
        del dense_model
        gc.collect()
        jax.clear_caches()
        moe = ModelConfig.matched_moe(dense)
        assert moe.expert_d_ff == dense_d_ff // 2
        moe_model = AdditionTransformer(moe, rngs=nnx.Rngs(params=0))
        assert assert_parameter_count(moe_model) == moe_total
        assert active_parameter_proxy(moe_model) == active_total
        del moe_model
        gc.collect()
        jax.clear_caches()


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"architecture": "other"}, "architecture"),
        ({"num_experts": 4}, "dense architecture"),
        (
            {"architecture": "moe", "num_experts": 0, "top_k": 1, "expert_d_ff": 8},
            "num_experts",
        ),
        (
            {"architecture": "moe", "num_experts": 2, "top_k": 3, "expert_d_ff": 8},
            "top_k",
        ),
        (
            {"architecture": "moe", "num_experts": 2, "top_k": 1, "expert_d_ff": 0},
            "expert_d_ff",
        ),
        ({"load_balance_loss_coefficient": -1}, "non-negative"),
    ],
)
def test_invalid_moe_configuration_fails_early(changes, message):
    with pytest.raises(ValueError, match=message):
        ModelConfig(**changes)


def test_matching_requires_exact_divisibility_and_dense_source():
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig.matched_moe(ModelConfig(d_ff=17), top_k=2)
    source = ModelConfig.matched_moe(ModelConfig())
    with pytest.raises(ValueError, match="dense source"):
        ModelConfig.matched_moe(source)
