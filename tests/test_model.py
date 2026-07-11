import jax.numpy as jnp, numpy as np


def test_model_shape_finite_causal_and_tied(tiny_model):
    x = jnp.arange(30, dtype=jnp.int32).reshape(2, 15) % 13
    logits = tiny_model(x)
    assert logits.shape == (2, 15, 13) and np.isfinite(logits).all()
    assert tiny_model.lm_head is None
    changed = x.at[:, 14].set((x[:, 14] + 1) % 13)
    np.testing.assert_allclose(logits[:, :14], tiny_model(changed)[:, :14], atol=1e-5)
    assert not hasattr(tiny_model, "output_embedding")
