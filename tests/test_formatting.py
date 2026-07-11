from jax_addition_transformer.tokenizer import format_prompt, format_training_example


def test_known_formats():
    assert format_prompt(7, 42) == "007 + 042 = "
    assert format_training_example(7, 42) == "007 + 042 = 9400"
    assert format_training_example(99, 1) == "099 + 001 = 0010"
    assert format_training_example(999, 999) == "999 + 999 = 8991"
    assert len(format_training_example(0, 0)) == 16
