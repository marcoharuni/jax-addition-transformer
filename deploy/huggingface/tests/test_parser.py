import pytest

from input_parser import InputError, parse_addition


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("347 + 928", (347, 928)),
        ("347+928=", (347, 928)),
        ("347 plus 928", (347, 928)),
        ("What is 347 plus 928?", (347, 928)),
        ("Add 347 and 928", (347, 928)),
        ("Please add 347 to 928", (347, 928)),
        ("000 + 009", (0, 9)),
    ],
)
def test_accepted_addition_phrasings(text, expected):
    assert parse_addition(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "1 * 2",
        "multiply 1 and 2",
        "5 - 2",
        "subtract 2 from 5",
        "8 / 2",
        "1.5 + 2",
        "-1 + 2",
        "1 + -2",
        "1000 + 1",
        "1 + 1000",
        "1 + 2 + 3",
        "add 1",
        "tell me something",
        "",
    ],
)
def test_rejects_unsupported_or_ambiguous_input(text):
    with pytest.raises(InputError):
        parse_addition(text)
