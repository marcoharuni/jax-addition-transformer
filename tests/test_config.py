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


def test_dense_fingerprint_remains_compatible_with_verified_artifacts():
    config = ExperimentConfig.load(
        "experiments/dense_scaling/configs/final/dense_10m_s0050_seed42.json"
    )
    assert config.fingerprint == "098de4f094856238008c94b746cfdb77f11b51ff73596335b6666758e80edbdc"
