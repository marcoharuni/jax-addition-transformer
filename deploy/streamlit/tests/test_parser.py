import pytest

from input_parser import InputError, parse_addition


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("347 + 928", (347, 928)),
        ("What is 499 plus 501?", (499, 501)),
        ("Please add 91 and 909", (91, 909)),
        ("Add 999 and 999", (999, 999)),
        ("000 + 009", (0, 9)),
    ],
)
def test_valid_inputs(text, expected):
    assert parse_addition(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "12 * 3",
        "12 / 3",
        "12 - 3",
        "-1 + 2",
        "1.5 + 2",
        "1000 + 1",
        "1 + 2 + 3",
        "add 1",
        "",
    ],
)
def test_invalid_inputs(text):
    with pytest.raises(InputError):
        parse_addition(text)
