"""Explicit-state hybrid training sampler."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import pair_ids_to_operands, stratum_codes, tokenize_pairs


@dataclass
class HybridSampler:
    train_ids: np.ndarray
    batch_size: int
    rng: np.random.Generator
    base: int = 1000

    def __post_init__(self) -> None:
        self.train_ids = np.asarray(self.train_ids, dtype=np.int64)
        a, b = pair_ids_to_operands(self.train_ids, self.base)
        codes = stratum_codes(a, b, len(str(self.base - 1)))
        self.strata = [self.train_ids[codes == code] for code in np.unique(codes)]
        if self.batch_size < 2:
            raise ValueError("batch_size must be at least two")

    @classmethod
    def create(
        cls, train_ids: np.ndarray, batch_size: int, seed: int, base: int = 1000
    ) -> HybridSampler:
        return cls(train_ids, batch_size, np.random.default_rng(seed), base)

    @property
    def state(self) -> dict:
        return self.rng.bit_generator.state

    def restore_state(self, state: dict) -> None:
        self.rng.bit_generator.state = state

    def sample_ids(self) -> np.ndarray:
        natural_n = self.batch_size // 2
        balanced_n = self.batch_size - natural_n
        natural = self.rng.choice(self.train_ids, natural_n, replace=True)
        chosen_strata = self.rng.integers(0, len(self.strata), balanced_n)
        balanced = np.fromiter(
            (self.rng.choice(self.strata[i]) for i in chosen_strata),
            dtype=np.int64,
            count=balanced_n,
        )
        result = np.concatenate((natural, balanced))
        self.rng.shuffle(result)
        return result

    def sample_batch(self, max_digits: int = 3) -> tuple[np.ndarray, np.ndarray]:
        sequences = tokenize_pairs(self.sample_ids(), max_digits)
        return sequences[:, :-1], sequences[:, 1:]
