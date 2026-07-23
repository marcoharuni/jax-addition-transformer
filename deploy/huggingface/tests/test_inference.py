from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from inference import AdditionModel
from model import load_weights, validate_weights

MODEL_REPO = Path(__file__).resolve().parents[1] / "model_repo"


@pytest.fixture(scope="module")
def addition_model():
    return AdditionModel.from_pretrained(MODEL_REPO)


def test_stable_weights_are_exact(addition_model):
    assert validate_weights(load_weights(MODEL_REPO / "model.safetensors")) == 10_000_000
    assert addition_model.weights_path.name == "model.safetensors"


@pytest.mark.parametrize(
    ("a", "b", "answer", "internal"),
    [
        (0, 0, "0", "0000"),
        (1, 9, "10", "0100"),
        (9, 1, "10", "0100"),
        (10, 90, "100", "0010"),
        (99, 1, "100", "0010"),
        (123, 456, "579", "9750"),
        (347, 928, "1275", "5721"),
        (500, 500, "1000", "0001"),
        (999, 1, "1000", "0001"),
        (999, 999, "1998", "8991"),
    ],
)
def test_genuine_generation(addition_model, a, b, answer, internal):
    details = addition_model.add_with_details(a, b)
    assert details["answer"] == answer
    assert details["internal_digits"] == internal


def test_deterministic_random_sample(addition_model):
    rng = np.random.default_rng(20260723)
    a = rng.integers(0, 1000, size=1000, dtype=np.int32)
    b = rng.integers(0, 1000, size=1000, dtype=np.int32)
    predictions = addition_model.predict_batch(a, b)
    assert predictions == [str(int(left) + int(right)) for left, right in zip(a, b, strict=True)]


@pytest.mark.parametrize("bad", [-1, 1000, 1.5, True])
def test_operand_validation(addition_model, bad):
    with pytest.raises((TypeError, ValueError)):
        addition_model.add(bad, 1)
