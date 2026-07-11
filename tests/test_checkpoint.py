import jax.numpy as jnp
import numpy as np
from flax import nnx

from jax_addition_transformer.checkpointing import restore_checkpoint, save_checkpoint
from jax_addition_transformer.config import OptimizerConfig
from jax_addition_transformer.optimizers import make_optimizer


def test_checkpoint_round_trip_preserves_predictions(tmp_path, tiny_model):
    graph, params = nnx.split(tiny_model, nnx.Param)
    optimizer, _ = make_optimizer(OptimizerConfig(total_steps=10, warmup_steps=1), params)
    optimizer_state = optimizer.init(params)
    tokens = jnp.arange(30, dtype=jnp.int32).reshape(2, 15) % 13
    before = nnx.merge(graph, params)(tokens)
    save_checkpoint(
        tmp_path / "ckpt",
        params,
        optimizer_state,
        {"config_fingerprint": "abc", "step": 0, "sampler_state": {}, "best": None, "history": []},
    )
    restored, _, metadata = restore_checkpoint(tmp_path / "ckpt", params, optimizer_state, "abc")
    after = nnx.merge(graph, restored)(tokens)
    assert metadata["step"] == 0
    np.testing.assert_array_equal(before, after)
