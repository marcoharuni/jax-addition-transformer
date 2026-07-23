"""Strict natural-language normalization for the model's fixed addition domain."""

from __future__ import annotations

import re


class InputError(ValueError):
    """A visitor-facing unsupported-input error."""


_NUMBER = r"(\d+)"
_SYMBOLIC = re.compile(rf"^\s*{_NUMBER}\s*\+\s*{_NUMBER}\s*(?:=\s*)?\??\s*$")
_PLUS = re.compile(
    rf"^\s*(?:what\s+is\s+)?{_NUMBER}\s+plus\s+{_NUMBER}\s*\??\s*$",
    re.IGNORECASE,
)
_ADD = re.compile(
    rf"^\s*(?:please\s+)?add\s+{_NUMBER}\s+(?:and|to)\s+{_NUMBER}\s*\??\s*$",
    re.IGNORECASE,
)
_DECIMAL = re.compile(r"\d+\.\d+")
_NEGATIVE = re.compile(r"(?<!\d)-\s*\d+")
_UNSUPPORTED = re.compile(
    r"(?:[*/×÷]|(?:minus|subtract|times|multiply|multiplied|divide|divided)\b)",
    re.IGNORECASE,
)


def parse_addition(text: str) -> tuple[int, int]:
    if not isinstance(text, str) or not text.strip():
        raise InputError("Enter an addition request such as “347 + 928”.")
    if _DECIMAL.search(text):
        raise InputError("Decimals are unsupported. Use two whole-number operands from 0 to 999.")
    if _NEGATIVE.search(text):
        raise InputError("Negative values are unsupported. Both operands must be from 0 to 999.")
    if _UNSUPPORTED.search(text):
        raise InputError("This model supports addition only—not subtraction, multiplication, or division.")

    match = _SYMBOLIC.fullmatch(text) or _PLUS.fullmatch(text) or _ADD.fullmatch(text)
    if match is None:
        numbers = re.findall(r"\d+", text)
        if len(numbers) != 2:
            raise InputError("Provide exactly two integers and a clear addition request.")
        raise InputError(
            "Use “A + B”, “A plus B”, or “Add A and B”. The model is not a general chatbot."
        )

    a, b = (int(value) for value in match.groups())
    if a > 999 or b > 999:
        raise InputError("Each operand must be between 0 and 999 inclusive.")
    return a, b
