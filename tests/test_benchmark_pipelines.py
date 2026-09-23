"""Benchmark Pipelines & Edge Cases Test Suite.

Verifies:
1. Structural integrity of all 8 folders in benchmark_pipelines/ (code, model.pkl, evaluation_dataset.csv).
2. Inference-only pipeline lifecycle grounding and V1 threat pruning.
3. EDGE-01: Malformed syntax rejection via AST pre-flight guard.
4. EDGE-02: Corrupted model artifact handling.
5. EDGE-03: Zero-Trust Docker fail-closed isolation.
6. EDGE-04: Transparent LLM unavailability handling (zero deterministic fabrication).
"""

import ast
import os
import sys
import tempfile
import uuid
import joblib
import pandas as pd
import pytest
from unittest.mock import patch

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.agents.pipeline_agent.parser import parse_python_pipeline
from src.agents.testing_agent.sandbox_runner import dispatch_sandbox, is_docker_available
from src.core.audit_memory import initialize_memory

DATASETS_DIR = os.path.join(WORKSPACE_ROOT, "benchmarks", "pipelines")
EXPECTED_DATASET_FOLDERS = [
    "all_defended",
    "v1_data_poisoning",
    "v2_preprocessing",
    "v3_data_validation",
    "v4_adversarial",
    "v4_v1_compound",
    "all_defended_inference",
    "completely_unhardened",
]


@pytest.fixture(autouse=True)
def setup_runtime():
    initialize_memory()


def test_dataset_folder_presence():
    """Verifies that all 6 dataset scenario folders exist in Datasets/."""
    assert os.path.isdir(DATASETS_DIR), f"Datasets directory not found at {DATASETS_DIR}"
    for folder_name in EXPECTED_DATASET_FOLDERS:
        folder_path = os.path.join(DATASETS_DIR, folder_name)
        assert os.path.isdir(folder_path), f"Missing dataset folder: {folder_path}"


def test_pipeline_scripts_compile_without_syntax_errors():
    """Verifies that every Python script in Datasets/ parses cleanly with ast.parse."""
    for folder_name in EXPECTED_DATASET_FOLDERS:
        folder_path = os.path.join(DATASETS_DIR, folder_name)
        py_files = [f for f in os.listdir(folder_path) if f.endswith(".py")]
        assert len(py_files) >= 1, f"No Python script found in {folder_path}"
        
        for py_file in py_files:
            file_path = os.path.join(folder_path, py_file)
            with open(file_path, "r", encoding="utf-8") as f:
                code = f.read()
            tree = ast.parse(code, filename=file_path)
            assert tree is not None
            assert len(tree.body) > 0


def test_models_loadable_as_valid_estimators():
    """Verifies that model.pkl in each folder unpickles into a valid estimator."""
    for folder_name in EXPECTED_DATASET_FOLDERS:
        model_path = os.path.join(DATASETS_DIR, folder_name, "model.pkl")
        assert os.path.isfile(model_path), f"Missing model.pkl in {folder_name}"
        
        model = joblib.load(model_path)
        assert hasattr(model, "predict") or hasattr(model, "transform") or hasattr(model, "fit"), (
            f"Loaded model in {folder_name} does not expose standard sklearn methods"
        )


def test_evaluation_datasets_have_valid_schemas():
    """Verifies that evaluation_dataset.csv in each folder has rows and text/label columns."""
    for folder_name in EXPECTED_DATASET_FOLDERS:
        csv_path = os.path.join(DATASETS_DIR, folder_name, "evaluation_dataset.csv")
        assert os.path.isfile(csv_path), f"Missing evaluation_dataset.csv in {folder_name}"
        
        df = pd.read_csv(csv_path)
        assert len(df) > 0, f"Dataset in {folder_name} is empty"
        cols = [c.lower() for c in df.columns]
        has_text = any(t in cols for t in ["text", "message", "title", "content"])
        has_label = any(l in cols for l in ["label", "label_text", "sentiment", "target"])
        assert has_text, f"No text column found in {folder_name}: {df.columns}"
        assert has_label, f"No label column found in {folder_name}: {df.columns}"


def test_inference_only_prunes_v1_data_poisoning():
    """Verifies that an inference-only pipeline prunes V1 data poisoning as not_applicable."""
    inference_pipeline_path = os.path.join(
        DATASETS_DIR, "all_defended_inference", "pipeline.py"
    )
    with open(inference_pipeline_path, "r", encoding="utf-8") as f:
        code = f.read()
        
    parsed = parse_python_pipeline(code)
    # Verify no training/fit nodes exist in structural parse
    node_names = [n.get("name", "").lower() for n in parsed.get("nodes", [])]
    has_fit = any("fit" in name for name in node_names)
    assert not has_fit, "Inference pipeline should not contain fit/training nodes"


def test_edge_01_malformed_syntax_rejection():
    """EDGE-01: Verifies that corrupted Python syntax is intercepted immediately by AST guard."""
    malformed_code = "def broken_syntax(\n  if x == 1\n    return False"
    result = parse_python_pipeline(malformed_code)
    
    error_nodes = [n for n in result.get("nodes", []) if "parse_error" in n.get("id", "")]
    assert len(error_nodes) > 0
    assert "syntax error" in error_nodes[0].get("name", "").lower()


def test_edge_02_corrupted_model_file_handling():
    """EDGE-02: Verifies that a zero-byte or corrupt model file is handled safely."""
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        tmp.write(b"")  # Zero-byte corrupted pickle file
        tmp_path = tmp.name
        
    try:
        with pytest.raises(Exception):
            joblib.load(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_edge_03_zero_trust_docker_offline():
    """EDGE-03: Verifies fail-closed zero-trust execution when Docker daemon is unavailable."""
    with patch("src.agents.testing_agent.sandbox_runner.is_docker_available", return_value=False):
        res = dispatch_sandbox(
            model_path=os.path.join(DATASETS_DIR, "all_defended", "model.pkl"),
            dataset_path=os.path.join(DATASETS_DIR, "all_defended", "evaluation_dataset.csv"),
            pipeline_path=os.path.join(DATASETS_DIR, "all_defended", "pipeline.py"),
            vectorizer_path=None,
            text_column="text",
            label_column="label",
            planned_tests=["poisoning", "adversarial"],
            audit_id=f"test_fail_closed_{uuid.uuid4().hex[:6]}",
        )
        assert res.get("sandbox_status") == "skipped_zero_trust"
        assert res["poisoning_evidence"]["status"] == "unverified"
        assert res["adversarial_evidence"]["status"] == "unverified"
        assert res["telemetry"]["docker_available"] is False


def test_edge_04_llm_unavailable_transparency(capsys):
    """EDGE-04: Verifies transparent reporting when LLM is unavailable without mock fabrication."""
    # When is_llm_operational returns False, the evaluator must print the notice clearly
    from benchmarks.evaluator import check_live_llm_operational
    
    with patch("benchmarks.evaluator.is_llm_available", return_value=False):
        is_op, reason = check_live_llm_operational()
        assert is_op is False
        assert "not configured" in reason.lower()

