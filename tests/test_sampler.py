import copy, numpy as np
from jax_addition_transformer.data import create_split
from jax_addition_transformer.sampling import HybridSampler


def test_sampler_determinism_shape_and_resume():
    ids = create_split().train
    a = HybridSampler.create(ids, 32, 7)
    b = HybridSampler.create(ids, 32, 7)
    np.testing.assert_array_equal(a.sample_ids(), b.sample_ids())
    state = copy.deepcopy(a.state)
    expected = a.sample_ids()
    a.restore_state(state)
    np.testing.assert_array_equal(a.sample_ids(), expected)
    x, y = a.sample_batch()
    assert x.shape == y.shape == (32, 15)
    assert set(a.sample_ids()).issubset(set(ids))
