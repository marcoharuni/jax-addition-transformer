import numpy as np
from jax_addition_transformer.data import carry_bits, carry_pattern


def test_units_tens_hundreds_order():
    assert carry_pattern(0, 0) == "000"
    assert carry_pattern(9, 1) == "100"
    assert carry_pattern(99, 1) == "110"
    assert carry_pattern(999, 1) == "111"
    np.testing.assert_array_equal(
        carry_bits(np.array([9, 99]), np.array([1, 1])), [[1, 0, 0], [1, 1, 0]]
    )
