import pytest
from jax_addition_transformer.config import ExperimentConfig, ModelConfig, TaskConfig


def test_derived_lengths_and_default_config():
    task = TaskConfig()
    assert (task.full_sequence_length, task.model_input_length, task.answer_digits) == (16, 15, 4)
    ExperimentConfig()


def test_dimension_validation():
    with pytest.raises(ValueError):
        ModelConfig(d_model=321)
    with pytest.raises(ValueError):
        ModelConfig(attention_type="mqa", n_kv_heads=5)
