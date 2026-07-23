"""Simple tested inference interface for the released addition transformer."""

from __future__ import annotations

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from model import greedy_generate, load_weights

MODEL_REPO_ID = "marcoharuni95/jax-addition-transformer-10m"


def _prompt_tokens(a_values: np.ndarray, b_values: np.ndarray) -> np.ndarray:
    prompts = np.empty((len(a_values), 12), dtype=np.int32)
    prompts[:, 0] = a_values // 100
    prompts[:, 1] = (a_values // 10) % 10
    prompts[:, 2] = a_values % 10
    prompts[:, 3:6] = (10, 11, 10)
    prompts[:, 6] = b_values // 100
    prompts[:, 7] = (b_values // 10) % 10
    prompts[:, 8] = b_values % 10
    prompts[:, 9:12] = (10, 12, 10)
    return prompts


def _validate_operand(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError("each operand must be an integer")
    value = int(value)
    if not 0 <= value <= 999:
        raise ValueError("each operand must be between 0 and 999")
    return value


def _decode_digits(digits: np.ndarray) -> str:
    if digits.shape != (4,) or np.any((digits < 0) | (digits > 9)):
        raise ValueError("model generated an invalid four-token answer")
    internal = "".join(str(int(digit)) for digit in digits)
    return internal[::-1].lstrip("0") or "0"


class AdditionModel:
    """Load once, compile once per batch shape, and generate four answer digits."""

    def __init__(self, weights_path: str | Path):
        self.weights_path = Path(weights_path)
        self.weights = load_weights(self.weights_path)
        self._generate = jax.jit(greedy_generate)
        started = time.perf_counter()
        digits, valid = self._generate(
            self.weights,
            jnp.asarray(_prompt_tokens(np.asarray([0]), np.asarray([0]))),
        )
        jax.block_until_ready(digits)
        if not bool(np.asarray(valid)[0]) or _decode_digits(np.asarray(digits)[0]) != "0":
            raise RuntimeError("model warm-up generation failed")
        self.compilation_seconds = time.perf_counter() - started

    @classmethod
    def from_pretrained(
        cls,
        repo_id_or_path: str | Path = MODEL_REPO_ID,
        *,
        revision: str | None = None,
    ) -> AdditionModel:
        candidate = Path(repo_id_or_path)
        if candidate.is_dir():
            return cls(candidate / "model.safetensors")
        if candidate.is_file():
            return cls(candidate)
        from huggingface_hub import hf_hub_download

        weights_path = hf_hub_download(
            repo_id=str(repo_id_or_path),
            filename="model.safetensors",
            revision=revision,
        )
        return cls(weights_path)

    def predict_batch(
        self, a_values: list[int] | np.ndarray, b_values: list[int] | np.ndarray
    ) -> list[str]:
        a = np.asarray([_validate_operand(value) for value in a_values], dtype=np.int32)
        b = np.asarray([_validate_operand(value) for value in b_values], dtype=np.int32)
        if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape or not len(a):
            raise ValueError("operands must be equally sized non-empty vectors")
        digits, valid = self._generate(self.weights, jnp.asarray(_prompt_tokens(a, b)))
        host_digits, host_valid = np.asarray(digits), np.asarray(valid)
        if not host_valid.all():
            raise RuntimeError("model generated a token outside the digit vocabulary")
        return [_decode_digits(row) for row in host_digits]

    def add_with_details(self, a: int, b: int) -> dict[str, str | float]:
        a, b = _validate_operand(a), _validate_operand(b)
        prompt = f"{a:03d} + {b:03d} = "
        started = time.perf_counter()
        digits, valid = self._generate(
            self.weights,
            jnp.asarray(_prompt_tokens(np.asarray([a]), np.asarray([b]))),
        )
        jax.block_until_ready(digits)
        latency = time.perf_counter() - started
        host_digits = np.asarray(digits)[0]
        if not bool(np.asarray(valid)[0]):
            raise RuntimeError("model generated a token outside the digit vocabulary")
        return {
            "answer": _decode_digits(host_digits),
            "prompt": prompt,
            "internal_digits": "".join(str(int(digit)) for digit in host_digits),
            "latency_seconds": latency,
        }

    def add(self, a: int, b: int) -> str:
        return str(self.add_with_details(a, b)["answer"])
