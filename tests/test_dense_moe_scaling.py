import ast
import json
import sys
from pathlib import Path

import jax
import pytest
from flax import nnx

from jax_addition_transformer.config import ExperimentConfig, OptimizerConfig
from jax_addition_transformer.model import AdditionTransformer
from jax_addition_transformer.optimizers import decay_mask, learning_rate_schedule


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.dense_moe_scaling.analyze import analyze  # noqa: E402
from experiments.dense_moe_scaling.compare import compare  # noqa: E402
from experiments.dense_moe_scaling.generate_configs import build_grid  # noqa: E402
from experiments.dense_moe_scaling.protocol import (  # noqa: E402
    HORIZONS,
    PROTOCOL_PATH,
    RUNS_PER_ARCHITECTURE,
)
from experiments.dense_moe_scaling.run_grid import validate_complete_run  # noqa: E402


EXPECTED_EXPERT_WIDTHS = {
    "moe_active_0p16m": 248,
    "moe_active_0p64m": 496,
    "moe_active_2p16m": 744,
    "moe_active_10m": 1240,
}
DENSE_MODEL_IDS = {
    "dense_0p16m",
    "dense_0p64m",
    "dense_2p16m",
    "dense_10m",
}


def notebook_source(name):
    notebook = json.loads((ROOT / "notebooks" / name).read_text())
    source = "\n".join(
        "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        for cell in notebook["cells"]
    )
    return notebook, source


def notebook_python(name):
    notebook, _ = notebook_source(name)
    assignments = {}
    definitions = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if source.startswith("%"):
            continue
        for node in ast.parse(source).body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        assignments[target.id] = node.value
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                definitions[node.name] = node
    return assignments, definitions


def test_scaling_notebooks_use_only_the_fixed_seed_grid():
    for name in ("02_dense_scaling_laws.ipynb", "04_moe_scaling_t4.ipynb"):
        notebook, source = notebook_source(name)
        assert notebook["metadata"]["accelerator"] == "GPU"
        assert notebook["metadata"]["colab"]["gpuType"] == "T4"
        assert not any(cell.get("outputs") for cell in notebook["cells"] if cell["cell_type"] == "code")
        assert "START_FRESH" not in source
        assert "FRONTIER_SEEDS" not in source
        assert "seed123" not in source
        assert "seed2026" not in source
        assert "--evaluate-test" not in source
        assert "archive_if_requested" not in source
        assert "5p12m" not in source.lower()
        assert 'format="svg"' not in source.lower()
        assert '.svg"' not in source.lower()
        assert "HORIZONS = (50, 125, 300, 750)" in source
        assert 'save_figure(figure, "' in source
        assert '.savefig(path, dpi=180' in source


def test_moe_notebook_keeps_visible_implementation_and_isolated_execution():
    _, source = notebook_source("04_moe_scaling_t4.ipynb")
    assert 'os.environ["JAX_PLATFORMS"] = "cpu"' in source
    assert "import jax" in source
    assert "class RaggedTop2MoE" in source
    assert source.count("jax.lax.ragged_dot(") == 2
    assert "def run_moe_experiment" in source
    assert "subprocess.run(command, cwd=REPO_DIR)" in source
    assert 'result.get("run_config") != expected' in source
    assert 'worker_command("moe_active_10m", 50' in source
    assert '"--smoke-test"' in source
    assert "batch size remains 2048" in source.lower()


def test_dense_notebook_contains_the_preserved_seed42_sweep_and_analysis():
    _, source = notebook_source("02_dense_scaling_laws.ipynb")
    assert {name for name in DENSE_MODEL_IDS if name in source} == DENSE_MODEL_IDS
    assert "class DenseFeedForward" in source
    assert "class DenseAdditionTransformer" in source
    assert "def run_dense_experiment" in source
    assert "dense_results.append(run_dense_experiment(model_id, config, horizon))" in source
    assert 'DENSE_ROOT = DRIVE_ROOT / "scaling" / "notebook_02_dense"' in source
    assert 'DENSE_RESULTS_PATH = DENSE_ROOT / "dense_results.json"' in source
    assert "len(dense_results) == 16" in source
    assert "Dense validation loss" in source
    assert "Dense exact match" in source
    assert "Empirical compute frontier" in source
    assert "log_space_r_squared" in source


def test_dense_and_moe_notebooks_share_the_same_data_and_training_protocol():
    dense_assignments, dense_definitions = notebook_python("02_dense_scaling_laws.ipynb")
    moe_assignments, moe_definitions = notebook_python("04_moe_scaling_t4.ipynb")
    shared_constants = (
        "SEED",
        "VOCAB_SIZE",
        "MAX_SEQUENCE_LENGTH",
        "MODEL_INPUT_LENGTH",
        "PROMPT_LENGTH",
        "ANSWER_DIGITS",
        "TRAIN_SIZE",
        "VALIDATION_SIZE",
        "TEST_SIZE",
        "BATCH_SIZE",
        "EVAL_BATCH_SIZE",
        "HORIZONS",
        "PEAK_LR",
        "FINAL_LR",
        "WEIGHT_DECAY",
        "GRAD_CLIP",
        "PARAM_DTYPE",
        "COMPUTE_DTYPE",
        "TOKENS",
        "ANSWER_MASK",
    )
    shared_definitions = (
        "pair_ids_to_operands",
        "operand_lengths",
        "carry_codes",
        "stratum_codes",
        "make_sequences",
        "largest_remainder",
        "build_split",
        "HybridBatcher",
        "normal_parameter",
        "matrix_multiply",
        "Linear",
        "LayerNorm",
        "gelu",
        "CausalMHA",
        "ModuleSequence",
        "answer_loss",
        "greedy_generate",
        "make_learning_rate",
        "path_parts",
        "all_finite",
    )
    for name in shared_constants:
        assert ast.dump(dense_assignments[name], include_attributes=False) == ast.dump(
            moe_assignments[name], include_attributes=False
        ), name
    for name in shared_definitions:
        assert ast.dump(dense_definitions[name], include_attributes=False) == ast.dump(
            moe_definitions[name], include_attributes=False
        ), name

    _, dense_source = notebook_source("02_dense_scaling_laws.ipynb")
    _, moe_source = notebook_source("04_moe_scaling_t4.ipynb")
    assert 'DENSE_ROOT = DRIVE_ROOT / "scaling" / "notebook_02_dense"' in dense_source
    assert (
        'DENSE_RESULTS_PATH = DRIVE_ROOT / "scaling" / "notebook_02_dense" '
        '/ "dense_results.json"'
    ) in moe_source


def test_moe_notebook_contains_routing_analysis_and_result_gated_comparison():
    _, source = notebook_source("04_moe_scaling_t4.ipynb")
    assert {name for name in EXPECTED_EXPERT_WIDTHS if name in source} == set(
        EXPECTED_EXPERT_WIDTHS
    )
    assert "router_entropy" in source
    assert "DENSE_RESULTS_PATH.exists()" in source
    assert "len(dense_payload.get(\"results\", [])) == 16" in source
    assert "len(pairs) == 16" in source
    assert "active_parameters" in source
    assert "stored_parameters" in source
    assert "paired_results.json" in source
    assert "Fresh-process moe_active_10m_h0050 smoke test" in source
    assert "Conclusions from completed dense and MoE runs" in source


def test_canonical_grid_is_exactly_sixteen_plus_sixteen_and_fixed_seed():
    dense, moe = build_grid(write=False)

    assert len(dense) == len(moe) == RUNS_PER_ARCHITECTURE == 16
    assert len({row["run_id"] for row in dense + moe}) == 32
    assert {row["horizon_steps"] for row in dense + moe} == set(HORIZONS)
    assert {row["model_id"] for row in dense} == {
        "dense_0p16m",
        "dense_0p64m",
        "dense_2p16m",
        "dense_10m",
    }
    assert {row["model_id"] for row in moe} == set(EXPECTED_EXPERT_WIDTHS)
    assert {
        (row["split_seed"], row["initialization_seed"], row["sampler_seed"]) for row in dense + moe
    } == {(42, 42, 42)}
    assert len({row["protocol_fingerprint"] for row in dense + moe}) == 1

    for row in dense + moe:
        assert "seed42" not in row["run_id"]
        assert "v2" not in row["run_id"]
        config = ExperimentConfig.load(ROOT / row["config"])
        assert config.fingerprint == row["config_fingerprint"]
        assert config.training.max_steps == row["horizon_steps"]
        assert config.optimizer.total_steps == row["horizon_steps"]
        assert config.optimizer.warmup_steps == max(1, row["horizon_steps"] // 10)
        assert config.optimizer.end_at_last_update
        assert config.training.validation_interval == row["horizon_steps"]
        assert config.training.checkpoint_interval == min(50, row["horizon_steps"])
        assert config.model.uses_rematerialization
        assert config.model.load_balance_loss_coefficient == pytest.approx(0.01)
        assert config.model.router_z_loss_coefficient == pytest.approx(0.001)
        assert row["examples_seen"] == row["horizon_steps"] * 2048
        assert row["input_tokens_15_seen"] == row["examples_seen"] * 15
        assert row["supervised_tokens_4_seen"] == row["examples_seen"] * 4

    for row in moe:
        config = ExperimentConfig.load(ROOT / row["config"])
        assert config.model.expert_d_ff == EXPECTED_EXPERT_WIDTHS[row["model_id"]]


def test_canonical_generated_files_match_pure_generator():
    expected_dense, expected_moe = build_grid(write=False)
    experiment_root = ROOT / "experiments" / "dense_moe_scaling"
    assert json.loads((experiment_root / "dense_manifest.json").read_text()) == expected_dense
    assert json.loads((experiment_root / "moe_manifest.json").read_text()) == expected_moe
    assert len(list((experiment_root / "configs" / "dense").glob("*.json"))) == 16
    assert len(list((experiment_root / "configs" / "moe").glob("*.json"))) == 16


def test_v2_schedule_reaches_final_rate_on_last_update():
    config = OptimizerConfig(
        warmup_steps=5,
        total_steps=50,
        final_learning_rate=1e-4,
        end_at_last_update=True,
    )
    schedule = learning_rate_schedule(config)
    assert float(schedule(0)) == pytest.approx(0.0)
    assert float(schedule(49)) == pytest.approx(1e-4)


def test_router_kernel_is_excluded_from_weight_decay():
    dense = ExperimentConfig.load(
        ROOT
        / "experiments"
        / "dense_moe_scaling"
        / "configs"
        / "moe"
        / "moe_active_0p16m_h0050.json"
    )
    model = AdditionTransformer(dense.model, rngs=nnx.Rngs(params=42))
    _, params = nnx.split(model, nnx.Param)
    mask = decay_mask(params)
    path_values = [
        ("/".join(str(getattr(part, "key", getattr(part, "idx", part))) for part in path), value)
        for path, value in jax.tree_util.tree_flatten_with_path(mask)[0]
    ]
    router_values = [bool(value) for path, value in path_values if "router" in path]
    expert_values = [
        bool(value)
        for path, value in path_values
        if "ffn" in path and ("up_kernel" in path or "down_kernel" in path)
    ]
    assert router_values and not any(router_values)
    assert expert_values and all(expert_values)


def test_completion_validation_requires_matching_fingerprints_and_checkpoints(tmp_path):
    row = json.loads(
        (ROOT / "experiments" / "dense_moe_scaling" / "dense_manifest.json").read_text()
    )[0]
    run_dir = tmp_path / row["run_id"]
    run_dir.mkdir()
    config = ExperimentConfig.load(ROOT / row["config"])
    config.save(run_dir / "config.json")
    (run_dir / "protocol.json").write_text(PROTOCOL_PATH.read_text())
    (run_dir / "run_spec.json").write_text(json.dumps(row, indent=2) + "\n")
    split_fingerprint = "a" * 64
    (run_dir / "split_metadata.json").write_text(
        json.dumps({"split_seed": 42, "fingerprint": split_fingerprint}) + "\n"
    )
    environment = {"configuration_hash": row["config_fingerprint"], "backend": "gpu"}
    (run_dir / "environment.json").write_text(json.dumps(environment) + "\n")
    (run_dir / "history.jsonl").write_text('{"step": 50}\n')
    summary = {
        "schema_version": "training-summary-v2",
        "run_id": row["run_id"],
        "config_fingerprint": row["config_fingerprint"],
        "protocol_fingerprint": row["protocol_fingerprint"],
        "steps": row["horizon_steps"],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary) + "\n")
    result = {
        "schema_version": "scaling-result-v2",
        "status": "complete",
        "run_id": row["run_id"],
        "identity": {
            "architecture": row["architecture"],
            "model_id": row["model_id"],
            "dense_reference": row["dense_reference"],
            "horizon_steps": row["horizon_steps"],
        },
        "fingerprints": {
            "config": row["config_fingerprint"],
            "protocol": row["protocol_fingerprint"],
            "split": split_fingerprint,
        },
        "seeds": {"split": 42, "initialization": 42, "sampler": 42},
        "parameters": {
            "stored": row["stored_parameters"],
            "active_proxy": row["active_parameter_proxy"],
        },
    }
    (run_dir / "result.json").write_text(json.dumps(result) + "\n")
    (run_dir / "failures.csv").write_text("pair_id\n")
    (run_dir / "sample_predictions.txt").write_text("not evaluated\n")
    for name in ("best", "latest"):
        checkpoint = run_dir / "checkpoints" / name
        (checkpoint / "state").mkdir(parents=True)
        (checkpoint / "metadata.json").write_text(
            json.dumps(
                {
                    "config_fingerprint": row["config_fingerprint"],
                    "protocol_fingerprint": row["protocol_fingerprint"],
                    "run_id": row["run_id"],
                    "step": row["horizon_steps"],
                }
            )
            + "\n"
        )

    assert validate_complete_run(run_dir, row) == []
    result["fingerprints"]["protocol"] = "b" * 64
    (run_dir / "result.json").write_text(json.dumps(result) + "\n")
    assert "result protocol fingerprint mismatch" in validate_complete_run(run_dir, row)


def test_single_seed_analysis_has_no_standard_deviation_path():
    sources = "\n".join(
        (ROOT / "experiments" / "dense_moe_scaling" / name).read_text()
        for name in ("analyze.py", "compare.py")
    )
    assert "statistics.stdev" not in sources
    assert "sample_std" not in sources
    assert "unavailable_single_seed" in sources


def synthetic_collection(rows):
    records = []
    for index, row in enumerate(rows):
        answer_loss = 1.0 / (index + 2)
        result = {
            "schema_version": "scaling-result-v2",
            "status": "complete",
            "run_id": row["run_id"],
            "identity": {
                "protocol_version": "dense-moe-scaling-v2",
                "architecture": row["architecture"],
                "model_id": row["model_id"],
                "dense_reference": row["dense_reference"],
                "horizon_steps": row["horizon_steps"],
            },
            "fingerprints": {
                "protocol": row["protocol_fingerprint"],
                "config": row["config_fingerprint"],
                "split": "a" * 64,
            },
            "seeds": {"split": 42, "initialization": 42, "sampler": 42},
            "exposure": {
                "examples_seen": row["examples_seen"],
                "input_tokens_15_seen": row["input_tokens_15_seen"],
                "supervised_tokens_4_seen": row["supervised_tokens_4_seen"],
                "training_pool_size": 200_000,
                "training_pool_repetition_ratio": row["training_pool_repetition_ratio"],
            },
            "parameters": {
                "stored": row["stored_parameters"],
                "active_proxy": row["active_parameter_proxy"],
            },
            "compute": {"estimated_training_flops": row["estimated_training_flops"]},
            "timing_seconds": {
                "compilation": 1.0,
                "training_steps": float(index + 1),
                "validation": 2.0,
                "steady_state_training": float(index + 0.5),
                "run_segment_wallclock": float(index + 4),
            },
            "losses": {
                "validation_answer_cross_entropy": answer_loss,
                "final_training_answer_cross_entropy": answer_loss + 0.1,
                "final_load_balancing_loss": 0.0,
                "final_router_z_loss": 0.0,
                "final_weighted_load_balancing_loss": 0.0,
                "final_weighted_router_z_loss": 0.0,
                "final_total_optimized_loss": answer_loss + 0.1,
            },
            "accuracy": {
                "validation_greedy_exact_match": min(0.99, index / RUNS_PER_ARCHITECTURE),
                "final_training_answer_token_accuracy": min(
                    0.99, index / RUNS_PER_ARCHITECTURE
                ),
            },
            "router": {
                "num_experts": 4 if row["architecture"] == "moe" else None,
                "top_k": 2 if row["architecture"] == "moe" else None,
                "expert_d_ff": 1 if row["architecture"] == "moe" else None,
                "load_balance_loss_coefficient": 0.01,
                "router_z_loss_coefficient": 0.001,
                "metrics": [],
            },
            "environment": {"backend": "gpu", "git_commit": "test"},
        }
        records.append({**row, "status": "complete", "validation_errors": [], "result": result})
    return {
        "schema_version": "scaling-collection-v2",
        "architecture": rows[0]["architecture"],
        "expected_run_count": RUNS_PER_ARCHITECTURE,
        "records": records,
    }


def test_analyzer_and_comparison_accept_normalized_single_seed_schema(tmp_path):
    dense_rows, moe_rows = build_grid(write=False)
    dense_path = tmp_path / "dense.json"
    moe_path = tmp_path / "moe.json"
    dense_path.write_text(json.dumps(synthetic_collection(dense_rows)) + "\n")
    moe_path.write_text(json.dumps(synthetic_collection(moe_rows)) + "\n")

    dense_summary = analyze(dense_path, tmp_path / "dense-analysis", "dense")
    moe_summary = analyze(moe_path, tmp_path / "moe-analysis", "moe")
    comparison = compare(
        dense_path,
        moe_path,
        tmp_path / "scaling" / "comparison",
    )

    assert dense_summary["run_count"] == moe_summary["run_count"] == 16
    assert dense_summary["uncertainty"] is None
    assert not dense_summary["sample_standard_deviation_calculated"]
    assert comparison["paired_run_count"] == 16
    assert comparison["uncertainty"] is None
    assert (tmp_path / "scaling" / "comparison" / "paired_runs.csv").is_file()
