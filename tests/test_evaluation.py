import numpy as np

from jax_addition_transformer.evaluation import evaluate_ids


def test_evaluation_reports_complete_slices_and_failure_fields(tiny_model):
    metrics, failures = evaluate_ids(tiny_model, np.arange(20), batch_size=10)
    assert {
        "teacher_forced_loss",
        "teacher_forced_token_accuracy",
        "greedy_exact_match",
        "failure_count",
        "invalid_token_count",
        "commutativity_consistency",
        "slices",
    } <= metrics.keys()
    assert {"operand_length", "carry_pattern", "answer_length"} == metrics["slices"].keys()
    if failures:
        assert {
            "pair_id",
            "a",
            "b",
            "target",
            "prediction",
            "internal_generated_digits",
            "valid_digit_sequence",
            "operand_length_a",
            "operand_length_b",
            "carry_pattern",
        } == failures[0].keys()
