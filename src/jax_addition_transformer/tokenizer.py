"""Stable 13-character vocabulary and arithmetic formatting."""

from __future__ import annotations

import re

import numpy as np

TOKENS = "0123456789 +="
TOKEN_TO_ID = {token: index for index, token in enumerate(TOKENS)}
ID_TO_TOKEN = dict(enumerate(TOKENS))


def encode(text: str) -> np.ndarray:
    try:
        return np.asarray([TOKEN_TO_ID[c] for c in text], dtype=np.int32)
    except KeyError as exc:
        raise ValueError(f"character {exc.args[0]!r} is outside the 13-token vocabulary") from None


def decode(ids: np.ndarray | list[int]) -> str:
    values = np.asarray(ids)
    if values.ndim != 1:
        raise ValueError("token IDs must be one-dimensional")
    if np.any((values < 0) | (values >= len(TOKENS))):
        raise ValueError("token ID outside [0, 12]")
    return "".join(ID_TO_TOKEN[int(i)] for i in values)


def validate_operand(value: int, max_digits: int = 3) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError("operands must be integers")
    value = int(value)
    if not 0 <= value < 10**max_digits:
        raise ValueError(f"operand must be between 0 and {10**max_digits - 1}")
    return value


def format_prompt(a: int, b: int, max_digits: int = 3) -> str:
    a, b = validate_operand(a, max_digits), validate_operand(b, max_digits)
    return f"{a:0{max_digits}d} + {b:0{max_digits}d} = "


def format_training_example(a: int, b: int, max_digits: int = 3) -> str:
    prompt = format_prompt(a, b, max_digits)
    answer = f"{a + b:0{max_digits + 1}d}"[::-1]
    return prompt + answer


def decode_internal_answer(ids: np.ndarray | list[int]) -> str:
    internal = decode(ids)
    if not internal or any(c not in "0123456789" for c in internal):
        raise ValueError("generated answer must contain digits only")
    return internal[::-1].lstrip("0") or "0"


_EXPRESSION = re.compile(r"^\s*(\d+)\s*\+\s*(\d+)\s*(?:=\s*)?$")


def parse_expression(expression: str, max_digits: int = 3) -> tuple[int, int]:
    if not isinstance(expression, str) or (match := _EXPRESSION.fullmatch(expression)) is None:
        raise ValueError("expected an expression such as '123 + 456' or '123+456='")
    return validate_operand(int(match[1]), max_digits), validate_operand(int(match[2]), max_digits)
