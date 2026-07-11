import numpy as np
from jax_addition_transformer.data import create_split


def test_default_split_is_exact_disjoint_complete_and_stable():
    split = create_split()
    assert (len(split.train), len(split.validation), len(split.test)) == (200000, 20000, 780000)
    assert split.fingerprint == "922e744e640eb00ad8cf8306d7828484d443d14b65dbffba1464517c6aace1d7"
    combined = np.concatenate((split.train, split.validation, split.test))
    assert len(np.unique(combined)) == 1_000_000
    assert combined.min() == 0 and combined.max() == 999999
