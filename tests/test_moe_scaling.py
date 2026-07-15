import json
import sys
from pathlib import Path

from jax_addition_transformer.config import ExperimentConfig


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.moe_scaling.compare_dense_moe import combined_frontier  # noqa: E402
from experiments.dense_moe_scaling.generate_configs import (  # noqa: E402
    build_grid as build_v2_grid,
)
from experiments.dense_moe_scaling.protocol import (  # noqa: E402
    HORIZONS as V2_HORIZONS,
)


def test_dense_moe_scaling_workflow_contract_is_complete_and_run_ready():
    _, manifest = build_v2_grid(write=False)
    experiment_root = ROOT / "experiments" / "dense_moe_scaling"
    protocol = json.loads((experiment_root / "protocol.json").read_text())
    documentation = (experiment_root / "README.md").read_text()
    runner = (experiment_root / "run_grid.py").read_text()
    moe_source = (ROOT / "src" / "jax_addition_transformer" / "moe.py").read_text()

    assert len(manifest) == 16
    assert len({row["run_id"] for row in manifest}) == 16
    assert len({row["config_fingerprint"] for row in manifest}) == 16
    assert {row["model_id"] for row in manifest} == {
        "moe_active_0p16m",
        "moe_active_0p64m",
        "moe_active_2p16m",
        "moe_active_10m",
    }
    assert {row["horizon_steps"] for row in manifest} == {50, 125, 300, 750}
    assert {row["horizon_steps"] for row in manifest} == set(V2_HORIZONS)
    assert {
        (row["split_seed"], row["initialization_seed"], row["sampler_seed"])
        for row in manifest
    } == {(42, 42, 42)}

    assert protocol["protocol_version"] == "dense-moe-scaling-v2"
    assert protocol["study"]["runs_per_architecture"] == 16
    assert protocol["study"]["model_sizes_per_architecture"] == 4
    assert protocol["study"]["independent_horizons"] == [50, 125, 300, 750]
    assert protocol["study"]["fixed_seed"] == 42
    assert protocol["output_roots"]["moe"] == "scaling/moe"
    assert protocol["output_roots"]["comparison"] == "scaling/comparison"
    assert "120 independent NVIDIA T4 runs" not in documentation

    for row in manifest:
        assert "seed42" not in row["run_id"]
        assert "v2" not in row["run_id"]
        config = ExperimentConfig.load(ROOT / row["config"])
        assert config.model.architecture == "moe"
        assert config.training.resolved_split_seed == 42
        assert config.training.resolved_initialization_seed == 42
        assert config.training.resolved_sampler_seed == 42
        assert config.training.max_steps == row["horizon_steps"]
        assert config.optimizer.total_steps == row["horizon_steps"]
        assert config.training.checkpoint_interval <= row["horizon_steps"]

    for module in (
        "experiments.dense_moe_scaling.run_grid",
        "experiments.dense_moe_scaling.analyze",
        "experiments.dense_moe_scaling.compare",
    ):
        assert module in documentation
    assert "--rerun-only-missing" in documentation
    assert "resumable" in documentation.lower()
    assert "resumable_checkpoint_step" in runner
    assert "checkpoint_is_complete" in runner
    assert 'command.append("--resume")' in runner
    assert moe_source.count("jax.lax.ragged_dot(") == 2


def test_standalone_seed42_artifacts_are_complete_and_truthful():
    root = ROOT / "artifacts" / "moe-comparison-t4" / "seed-42"
    config = json.loads((root / "config.json").read_text())
    history = json.loads((root / "history.json").read_text())
    results = json.loads((root / "results.json").read_text())
    failures = (root / "failures.csv").read_text().splitlines()

    assert config["seed"] == 42
    assert config["model"]["expert_width"] == 1240
    assert history["final_step"] == 175
    assert results["stored_parameter_count"] == 17_942_400
    assert results["active_parameter_proxy"] == 10_006_400
    assert results["overall"]["correct"] == 999_981
    assert results["overall"]["failures"] == 19
    assert results["overall"]["invalid_generations"] == 0
    assert len(failures) == 20


def test_combined_frontier_can_select_both_architectures():
    dense = [
        {
            "model": "dense_a",
            "parameter_count": "100",
            "steps": "10",
            "estimated_training_flops": "1000",
            "total_training_seconds_mean": "4",
            "validation_loss_mean": "1.0",
            "validation_loss_std": "0.1",
            "greedy_exact_match_mean": "0.2",
        },
        {
            "model": "dense_b",
            "parameter_count": "200",
            "steps": "10",
            "estimated_training_flops": "3000",
            "total_training_seconds_mean": "8",
            "validation_loss_mean": "0.5",
            "validation_loss_std": "0.05",
            "greedy_exact_match_mean": "0.8",
        },
    ]
    moe = [
        {
            "model": "moe_a",
            "active_parameter_proxy": "101",
            "total_parameter_count": "180",
            "steps": "10",
            "estimated_training_flops": "2000",
            "total_training_seconds_mean": "6",
            "validation_loss_mean": "0.7",
            "validation_loss_std": "0.06",
            "greedy_exact_match_mean": "0.6",
        },
        {
            "model": "moe_b",
            "active_parameter_proxy": "201",
            "total_parameter_count": "360",
            "steps": "10",
            "estimated_training_flops": "4000",
            "total_training_seconds_mean": "10",
            "validation_loss_mean": "0.4",
            "validation_loss_std": "0.04",
            "greedy_exact_match_mean": "0.9",
        },
    ]
    frontier = combined_frontier(dense, moe, "estimated_training_flops")
    assert [row["architecture"] for row in frontier] == ["dense", "moe", "dense", "moe"]
