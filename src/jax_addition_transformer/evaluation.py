"""Teacher-forced and compiled greedy evaluation with exact slice accounting."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from flax import nnx

from .data import carry_pattern, operand_lengths, pair_ids_to_operands, tokenize_pairs
from .generation import greedy_generate
from .losses import masked_cross_entropy


def evaluate_ids(
    model, pair_ids: np.ndarray, batch_size: int = 4000, max_digits: int = 3
) -> tuple[dict, list[dict]]:
    if len(pair_ids) % batch_size:
        raise ValueError("evaluation size must be divisible by batch_size")
    correct_all, failures, predictions = [], [], []
    token_correct, token_total, loss_sum = 0, 0, 0.0
    group_counts: dict[str, dict[str, list[int]]] = {
        "operand_length": {},
        "carry_pattern": {},
        "answer_length": {},
    }
    forward = nnx.jit(lambda current_model, tokens: current_model(tokens))
    generate = nnx.jit(
        lambda current_model, prompts: greedy_generate(current_model, prompts, max_digits + 1)
    )

    def record_group(group: str, key: str, is_correct: bool) -> None:
        counts = group_counts[group].setdefault(key, [0, 0])
        counts[0] += int(is_correct)
        counts[1] += 1

    for start in range(0, len(pair_ids), batch_size):
        ids = pair_ids[start : start + batch_size]
        full = tokenize_pairs(ids, max_digits)
        inputs, targets = jnp.asarray(full[:, :-1]), jnp.asarray(full[:, 1:])
        logits = forward(model, inputs)
        loss, accuracy = masked_cross_entropy(logits, targets)
        loss_sum += float(loss) * batch_size
        token_correct += float(accuracy) * batch_size * (max_digits + 1)
        token_total += batch_size * (max_digits + 1)
        prompt_length = 2 * max_digits + 6
        generated, valid = generate(model, inputs[:, :prompt_length])
        generated, valid = np.asarray(generated), np.asarray(valid)
        expected = full[:, -max_digits - 1 :]
        correct = valid & np.all(generated == expected, axis=1)
        correct_all.extend(correct.tolist())
        predictions.extend(
            [tuple(map(int, row)) if ok else None for row, ok in zip(generated, valid, strict=True)]
        )
        a, b = pair_ids_to_operands(ids, 10**max_digits)
        lengths_a, lengths_b = operand_lengths(a), operand_lengths(b)
        patterns = carry_pattern(a, b, max_digits)
        sums = a + b
        answer_lengths = np.where(sums >= 10**max_digits, max_digits + 1, operand_lengths(sums))
        for i, solved in enumerate(correct):
            record_group("operand_length", f"{lengths_a[i]}x{lengths_b[i]}", bool(solved))
            record_group("carry_pattern", f"{patterns[i]:0{max_digits}b}", bool(solved))
            record_group("answer_length", str(answer_lengths[i]), bool(solved))
        for i in np.flatnonzero(~correct):
            pred = (
                ("".join(map(str, generated[i][::-1])).lstrip("0") or "0")
                if valid[i]
                else "<invalid>"
            )
            failures.append(
                {
                    "pair_id": int(ids[i]),
                    "a": int(a[i]),
                    "b": int(b[i]),
                    "target": str(int(sums[i])),
                    "prediction": pred,
                    "internal_generated_digits": "".join(map(str, generated[i])),
                    "valid_digit_sequence": bool(valid[i]),
                    "operand_length_a": int(lengths_a[i]),
                    "operand_length_b": int(lengths_b[i]),
                    "carry_pattern": f"{patterns[i]:0{max_digits}b}",
                }
            )
    slices = {
        group: {
            key: {"correct": values[0], "total": values[1], "accuracy": values[0] / values[1]}
            for key, values in sorted(rows.items())
        }
        for group, rows in group_counts.items()
    }
    position = {int(pair_id): index for index, pair_id in enumerate(pair_ids)}
    comparable, consistent, base = 0, 0, 10**max_digits
    for index, pair_id in enumerate(pair_ids):
        a, b = divmod(int(pair_id), base)
        swapped = base * b + a
        if swapped in position:
            comparable += 1
            consistent += predictions[index] == predictions[position[swapped]]
    metrics = {
        "teacher_forced_loss": loss_sum / len(pair_ids),
        "teacher_forced_token_accuracy": token_correct / token_total,
        "greedy_exact_match": sum(correct_all) / len(correct_all),
        "failure_count": len(failures),
        "invalid_token_count": sum(not row["valid_digit_sequence"] for row in failures),
        "commutativity_consistency": consistent / comparable if comparable else None,
        "commutativity_comparable_pairs": comparable,
        "slices": slices,
    }
    return metrics, failures
