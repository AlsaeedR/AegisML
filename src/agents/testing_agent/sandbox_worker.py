import os
import sys
import json
import time
from typing import Any, Dict, List

from .loader import (
    load_trained_model,
    load_dataset,
    load_vectorizer,
    resolve_vectorizer_from_agent1,
)
from .poisoning_test import run_poisoning_test
from .adversarial_test import run_adversarial_test
from .preprocess_test import run_preprocess_checks
from .validation_test import run_validation_checks


def execute_worker():
    """
    In-container entrypoint for the AegisML sandbox worker.
    Reads /workspace/input/strategy.json, runs dynamic tests,
    and writes /workspace/output/test_results.json.
    """
    input_file = os.getenv("AEGISML_INPUT_JSON", "/workspace/input/strategy.json")
    output_file = os.getenv("AEGISML_OUTPUT_JSON", "/workspace/output/test_results.json")

    start_time = time.time()
    execution_log: List[str] = []
    
    if not os.path.exists(input_file):
        error_payload = {
            "status": "error",
            "error": f"Input configuration file not found at: {input_file}",
            "execution_log": ["Failed: strategy input file missing"],
            "telemetry": {"duration_seconds": 0},
        }
        _write_output(output_file, error_payload)
        sys.exit(1)

    with open(input_file, "r", encoding="utf-8") as f:
        strategy = json.load(f)

    model_path = strategy.get("model_path", "/workspace/data/model.pkl")
    dataset_path = strategy.get("dataset_path", "/workspace/data/dataset.csv")
    pipeline_path = strategy.get("pipeline_path", "/workspace/data/pipeline.py")
    vectorizer_path = strategy.get("vectorizer_path")
    text_column = strategy.get("text_column", "Comment")
    label_column = strategy.get("label_column", "Topic")
    agent_1_results = strategy.get("agent_1_results")
    planned_tests = strategy.get("planned_tests", ["V1_poisoning", "V4_adversarial", "V2_preprocessing", "V3_validation"])
    
    # Attack configs
    adv_config = strategy.get("adversarial_config", {})
    poison_config = strategy.get("poisoning_config", {})

    execution_log.append(f"Starting sandbox worker inside container. Target tests: {planned_tests}")

    results: Dict[str, Any] = {
        "status": "success",
        "planned_tests": planned_tests,
    }

    try:
        # Load artifacts safely inside container
        execution_log.append(f"Loading model from {model_path} and dataset from {dataset_path}")
        model = load_trained_model(model_path)
        X_text, y_true = load_dataset(dataset_path, text_column, label_column)

        if not vectorizer_path:
            base_dir = os.path.dirname(dataset_path) or "/workspace/data"
            vectorizer_path = resolve_vectorizer_from_agent1(agent_1_results, base_dir=base_dir)

        vectorizer = load_vectorizer(vectorizer_path)
        execution_log.append("Artifacts loaded successfully.")

        test_state = {
            "model": model,
            "vectorizer": vectorizer,
            "X_text": X_text,
            "y_true": y_true,
            "pipeline_path": pipeline_path if os.path.exists(pipeline_path) else None,
            "adversarial_config": adv_config,
            "poisoning_config": poison_config,
        }

        # V1: Data Poisoning
        if "V1_poisoning" in planned_tests:
            execution_log.append("Executing V1_poisoning test...")
            poison_res = run_poisoning_test(test_state)
            results["poisoning_evidence"] = poison_res.get("poisoning_evidence", {})
            execution_log.append(f"V1 complete: {results['poisoning_evidence'].get('status')}")

        # V4: Adversarial Robustness
        if "V4_adversarial" in planned_tests:
            execution_log.append("Executing V4_adversarial attack test...")
            adv_res = run_adversarial_test(test_state)
            results["adversarial_evidence"] = adv_res.get("adversarial_evidence", {})
            execution_log.append(f"V4 complete: {results['adversarial_evidence'].get('status')}")

        # V2: Preprocessing Attack Surface
        if "V2_preprocessing" in planned_tests:
            execution_log.append("Executing V2_preprocessing test...")
            prep_res = run_preprocess_checks(test_state)
            results["preprocessing_evidence"] = prep_res.get("preprocessing_evidence", {})
            execution_log.append(f"V2 complete: {results['preprocessing_evidence'].get('status')}")

        # V3: Data Validation Weaknesses
        if "V3_validation" in planned_tests:
            execution_log.append("Executing V3_validation test...")
            val_res = run_validation_checks(test_state)
            results["validation_evidence"] = val_res.get("validation_evidence", {})
            execution_log.append(f"V3 complete: {results['validation_evidence'].get('status')}")

    except Exception as e:
        results["status"] = "error"
        results["error"] = str(e)
        execution_log.append(f"Error during test execution: {str(e)}")

    duration = round(time.time() - start_time, 2)
    results["execution_log"] = execution_log
    results["telemetry"] = {
        "duration_seconds": duration,
        "container": True,
    }

    _write_output(output_file, results)


def _write_output(output_path: str, data: Dict[str, Any]):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


if __name__ == "__main__":
    execute_worker()

