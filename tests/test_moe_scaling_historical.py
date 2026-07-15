"""Regression coverage for the preserved 120-run MoE infrastructure."""

import json
import sys
from pathlib import Path

from jax_addition_transformer.config import ExperimentConfig


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.moe_scaling.configs import (  # noqa: E402
    EXPECTED_ACTIVE_PROXIES,
    EXPECTED_MOE_TOTALS,
    SEEDS,
    STEP_BUDGETS,
    build_grid,
    moe_parameter_counts,
)


def test_historical_final_moe_scaling_manifest_is_complete_and_matched():
    manifest, summary = build_grid(write=False)

    assert len(manifest) == 120
    assert summary["model_count"] == 5
    assert summary["run_count"] == 120
    assert summary["step_budgets"] == list(STEP_BUDGETS)
    assert summary["seeds"] == list(SEEDS)
    assert len({row["run_id"] for row in manifest}) == 120
    assert {row["steps"] for row in manifest} == set(STEP_BUDGETS)
    assert {row["seed"] for row in manifest} == set(SEEDS)

    for architecture in summary["architectures"]:
        name = architecture["model"]
        assert architecture["total_parameter_count"] == EXPECTED_MOE_TOTALS[name]
        assert architecture["active_parameter_proxy"] == EXPECTED_ACTIVE_PROXIES[name]
        assert architecture["active_overhead_parameters"] > 0
        assert architecture["active_overhead_fraction"] < 0.004
        assert architecture["expert_d_ff"] * 2 == architecture["dense_d_ff"]


def test_historical_generated_moe_configs_match_manifest_and_formula():
    manifest = json.loads(
        (ROOT / "experiments" / "moe_scaling" / "final_manifest.json").read_text()
    )
    configs = sorted(
        (ROOT / "experiments" / "moe_scaling" / "configs" / "final").glob("*.json")
    )
    model_configs = sorted(
        (ROOT / "experiments" / "moe_scaling" / "configs" / "final_models").glob(
            "*.json"
        )
    )

    assert len(manifest) == 120
    assert len(configs) == 120
    assert len(model_configs) == 5

    for row in manifest:
        config = ExperimentConfig.load(ROOT / row["config"])
        total, active = moe_parameter_counts(config.as_dict()["model"])
        assert config.model.architecture == "moe"
        assert config.model.num_experts == 4
        assert config.model.top_k == 2
        assert config.model.expert_d_ff * 2 == config.model.d_ff
        assert total == row["total_parameter_count"]
        assert active == row["active_parameter_proxy"]
        assert config.training.seed == row["seed"]
        assert config.training.max_steps == row["steps"]
        assert config.optimizer.total_steps == row["steps"]
        assert config.training.validation_interval == row["steps"]
        assert config.training.checkpoint_interval == row["steps"]
