import numpy as np, pytest
from jax_addition_transformer.tokenizer import (
    TOKENS,
    decode,
    decode_internal_answer,
    encode,
    parse_expression,
)


def test_mapping_and_round_trip():
    assert TOKENS == "0123456789 +="
    text = "123 + 456 = 9750"
    np.testing.assert_array_equal(
        encode(text), np.array([1, 2, 3, 10, 11, 10, 4, 5, 6, 10, 12, 10, 9, 7, 5, 0])
    )
    assert decode(encode(text)) == text


def test_internal_and_expression_validation():
    assert decode_internal_answer([9, 7, 5, 0]) == "579"
    assert decode_internal_answer([0, 0, 0, 0]) == "0"
    assert parse_expression("123+456=") == (123, 456)
    with pytest.raises(ValueError):
        parse_expression("1000+1")
    with pytest.raises(ValueError):
        parse_expression("1*2")
