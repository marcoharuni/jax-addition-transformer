"""Vectorized complete-domain data, carries, and deterministic stratified splits."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


def pair_ids_to_operands(pair_ids: np.ndarray, base: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(pair_ids, dtype=np.int64)
    if np.any((ids < 0) | (ids >= base * base)):
        raise ValueError("pair ID outside domain")
    return ids // base, ids % base


def operands_to_pair_ids(a: np.ndarray, b: np.ndarray, base: int = 1000) -> np.ndarray:
    a, b = np.asarray(a), np.asarray(b)
    if np.any((a < 0) | (a >= base) | (b < 0) | (b >= base)):
        raise ValueError("operand outside domain")
    return (base * a + b).astype(np.int64)


def operand_lengths(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if np.any(values < 0):
        raise ValueError("operand lengths require non-negative integers")
    boundaries = 10 ** np.arange(1, 19, dtype=np.int64)
    return (np.searchsorted(boundaries, values, side="right") + 1).astype(np.int8)


def carry_bits(a: np.ndarray | int, b: np.ndarray | int, max_digits: int = 3) -> np.ndarray:
    """Return units-to-most-significant carry bits; column zero is units."""
    a, b = np.asarray(a), np.asarray(b)
    carry = np.zeros(np.broadcast_shapes(a.shape, b.shape), dtype=np.int8)
    result = []
    for power in range(max_digits):
        total = (a // (10**power)) % 10 + (b // (10**power)) % 10 + carry
        carry = (total >= 10).astype(np.int8)
        result.append(carry)
    return np.stack(result, axis=-1)


def carry_pattern(
    a: np.ndarray | int, b: np.ndarray | int, max_digits: int = 3
) -> np.ndarray | str:
    bits = carry_bits(a, b, max_digits)
    if bits.ndim == 1:
        return "".join(str(int(x)) for x in bits)
    weights = 2 ** np.arange(max_digits - 1, -1, -1)
    return (bits * weights).sum(axis=-1).astype(np.int8)


def stratum_codes(a: np.ndarray, b: np.ndarray, max_digits: int = 3) -> np.ndarray:
    carry_states = 2**max_digits
    return (
        ((operand_lengths(a) - 1) * max_digits + (operand_lengths(b) - 1)) * carry_states
        + carry_pattern(a, b, max_digits)
    ).astype(np.int16)


def tokenize_pairs(pair_ids: np.ndarray, max_digits: int = 3) -> np.ndarray:
    """Vectorize AAA + BBB = RRRR without allocating Python strings."""
    base = 10**max_digits
    a, b = pair_ids_to_operands(pair_ids, base)
    answer = a + b
    n = len(a)
    full = np.empty((n, 3 * max_digits + 7), dtype=np.int32)
    powers = 10 ** np.arange(max_digits - 1, -1, -1)
    full[:, :max_digits] = (a[:, None] // powers) % 10
    full[:, max_digits : max_digits + 3] = (10, 11, 10)
    start = max_digits + 3
    full[:, start : start + max_digits] = (b[:, None] // powers) % 10
    full[:, start + max_digits : start + max_digits + 3] = (10, 12, 10)
    answer_powers = 10 ** np.arange(0, max_digits + 1)
    full[:, -max_digits - 1 :] = (answer[:, None] // answer_powers) % 10
    return full


@dataclass(frozen=True)
class DatasetSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    seed: int
    fingerprint: str


def _largest_remainder(counts: np.ndarray, total: int, capacities: np.ndarray) -> np.ndarray:
    ideal = counts.astype(np.float64) * total / counts.sum()
    allocated = np.minimum(np.floor(ideal).astype(np.int64), capacities)
    order = np.lexsort((np.arange(len(counts)), -(ideal - allocated)))
    remaining = total - int(allocated.sum())
    while remaining:
        eligible = order[allocated[order] < capacities[order]]
        take = eligible[:remaining]
        allocated[take] += 1
        remaining -= len(take)
    return allocated


def create_split(
    seed: int = 42, train_size: int = 200_000, validation_size: int = 20_000, base: int = 1000
) -> DatasetSplit:
    ids = np.arange(base * base, dtype=np.int64)
    a, b = pair_ids_to_operands(ids, base)
    max_digits = len(str(base - 1))
    codes = stratum_codes(a, b, max_digits)
    unique, counts = np.unique(codes, return_counts=True)
    train_counts = _largest_remainder(counts, train_size, counts)
    validation_counts = _largest_remainder(counts, validation_size, counts - train_counts)
    rng = np.random.default_rng(seed)
    train_parts, validation_parts, test_parts = [], [], []
    for code, n_train, n_validation in zip(unique, train_counts, validation_counts, strict=True):
        members = ids[codes == code].copy()
        rng.shuffle(members)
        train_parts.append(members[:n_train])
        validation_parts.append(members[n_train : n_train + n_validation])
        test_parts.append(members[n_train + n_validation :])
    train = np.concatenate(train_parts)
    validation = np.concatenate(validation_parts)
    test = np.concatenate(test_parts)
    rng.shuffle(train)
    rng.shuffle(validation)
    rng.shuffle(test)
    digest = hashlib.sha256()
    for name, values in (("train", train), ("validation", validation), ("test", test)):
        digest.update(name.encode())
        digest.update(values.astype("<i8", copy=False).tobytes())
    return DatasetSplit(train, validation, test, seed, digest.hexdigest())
