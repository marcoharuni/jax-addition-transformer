import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts" / "scaling_worker.py"


def test_worker_sets_memory_environment_before_importing_jax():
    source = WORKER.read_text()
    import_position = source.index("import jax\n")
    for assignment in (
        'os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"',
        'os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"',
        'os.environ["JAX_COMPILATION_CACHE_DIR"] = "/content/jax_compilation_cache/scaling"',
        'os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "0"',
        'os.environ.pop("JAX_PLATFORMS", None)',
    ):
        assert source.index(assignment) < import_position
    assert "XLA_PYTHON_CLIENT_MEM_FRACTION" not in source


def test_worker_cli_and_safety_contract_are_visible():
    source = WORKER.read_text()
    tree = ast.parse(source)
    assert tree is not None
    for argument in ("--family", "--model-id", "--horizon", "--seed", "--output-dir"):
        assert f'parser.add_argument("{argument}"' in source
    assert "BATCH_SIZE = 2048" in source
    assert "EVAL_BATCH_SIZE = 4000" in source
    assert "capture_oom_diagnostics" in source
    assert "nvidia_smi_oom.txt" in source
    assert "jax_device_memory.prof" in source
    assert 'choices=(SEED,)' in source
    assert "--evaluate-test" not in source
    assert "latest_checkpoint.pkl" in source
    assert "best_checkpoint.pkl" in source
    assert 'checkpoint.get("run_config") != run_config' in source
    assert "checkpoint_parameter_count" in source
    assert "refusing to mix scientific protocols" in source
    assert "preserve_completed_dense" in source
    assert '"numpy": "2.3.3"' in source
    assert "runtime versions do not match notebook 04" in source


def notebook_python_nodes():
    notebook = json.loads((ROOT / "notebooks" / "04_moe_scaling_t4.ipynb").read_text())
    nodes = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if source.startswith("%"):
            continue
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                nodes[node.name] = node
    return nodes


def top_level_assignments(tree):
    return {
        target.id: node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }


def test_worker_constants_and_model_grid_match_notebook_04():
    worker_assignments = top_level_assignments(ast.parse(WORKER.read_text()))
    notebook = json.loads((ROOT / "notebooks" / "04_moe_scaling_t4.ipynb").read_text())
    notebook_assignments = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if source.startswith("%"):
            continue
        notebook_assignments.update(top_level_assignments(ast.parse(source)))
    exact_constants = (
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
        "N_EXPERTS",
        "TOP_K",
        "ROUTER_BALANCE_COEFFICIENT",
        "ROUTER_Z_LOSS_COEFFICIENT",
        "MOE_CONFIGS",
    )
    for name in exact_constants:
        assert ast.dump(worker_assignments[name], include_attributes=False) == ast.dump(
            notebook_assignments[name], include_attributes=False
        ), name


def test_worker_scientific_core_matches_notebook_04_ast():
    worker_tree = ast.parse(WORKER.read_text())
    worker_nodes = {
        node.name: node
        for node in worker_tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    notebook_nodes = notebook_python_nodes()
    exact_nodes = (
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
        "RaggedTop2MoE",
        "MoETransformerBlock",
        "MoEAdditionTransformer",
        "moe_parameter_formulas",
        "answer_loss",
        "greedy_generate",
        "make_learning_rate",
        "path_parts",
        "all_finite",
    )
    for name in exact_nodes:
        assert ast.dump(worker_nodes[name], include_attributes=False) == ast.dump(
            notebook_nodes[name], include_attributes=False
        ), name


def test_worker_has_exact_notebook_04_routing_optimizer_and_rematerialization():
    source = WORKER.read_text()
    assert source.count("jax.lax.ragged_dot(") == 2
    assert "x, diagnostics = jax.checkpoint(apply_block)(x)" in source
    assert "top_logits, top_expert_ids = jax.lax.top_k(router_logits, TOP_K)" in source
    assert "top_gate_weights = jax.nn.softmax(top_logits, axis=-1)" in source
    assert "balance_loss = N_EXPERTS * jnp.sum(top1_fraction * router_importance)" in source
    assert "residual_scale = 0.02 / math.sqrt(2 * config[\"n_layers\"])" in source
    assert "optax.scale_by_adam(b1=0.9, b2=0.99, eps=1e-8)" in source
    assert "decay_steps=horizon - 1" in source
    assert "PARAM_DTYPE = jnp.float32" in source
    assert "COMPUTE_DTYPE = jnp.float16" in source


def test_worker_uses_legacy_result_and_checkpoint_compatibility_fields():
    source = WORKER.read_text()
    for field in (
        '"run_config": run_config',
        '"step": step',
        '"params": jax.device_get(params)',
        '"optimizer_state": jax.device_get(optimizer_state)',
        '"batcher_rng_state": batcher.rng.bit_generator.state',
        '"history": history',
        '"optimizer_seconds": optimizer_seconds',
        '"compilation_seconds": compilation_seconds',
    ):
        assert field in source
    assert 'completed.get("run_config") == run_config' in source
    assert 'checkpoint.get("run_config") != run_config' in source
    assert 'batcher.rng.bit_generator.state = checkpoint["batcher_rng_state"]' in source


@pytest.mark.parametrize(
    "arguments",
    [
        ["--family", "dense", "--model-id", "bad", "--horizon", "50", "--seed", "42"],
        [
            "--family",
            "moe",
            "--model-id",
            "moe_active_0p16m",
            "--horizon",
            "51",
            "--seed",
            "42",
        ],
    ],
)
def test_worker_rejects_invalid_coordinates_without_starting_training(tmp_path, arguments):
    environment = os.environ.copy()
    environment["JAX_PLATFORMS"] = "cpu"
    completed = subprocess.run(
        [sys.executable, str(WORKER), *arguments, "--output-dir", str(tmp_path)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 2
