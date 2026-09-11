import os
import sys
import json
import uuid
import shutil
import tempfile
import subprocess
from typing import Any, Dict, List, Optional


def is_docker_available() -> bool:
    """
    Check if the Docker daemon is accessible and responding.
    """
    try:
        res = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=4,
        )
        return res.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        return False


def is_insecure_local_allowed() -> bool:
    """
    Check if developer has explicitly opted into dangerous in-process execution.
    Defaults to False to enforce Zero-Trust Fail-Closed policy.
    """
    return os.getenv("AEGISML_ALLOW_INSECURE_LOCAL_TESTING", "false").lower() in ("true", "1", "yes")


def dispatch_sandbox(
    model_path: str,
    dataset_path: str,
    pipeline_path: str,
    vectorizer_path: Optional[str],
    text_column: str,
    label_column: str,
    planned_tests: List[str],
    strategy_config: Optional[Dict[str, Any]] = None,
    agent_1_results: Optional[Dict[str, Any]] = None,
    timeout_seconds: int = 180,
) -> Dict[str, Any]:
    """
    Host-side orchestrator that dispatches dynamic security tests into an isolated
    Docker sandbox with Zero-Trust Fail-Closed enforcement.
    """
    docker_ready = is_docker_available()
    allow_insecure = is_insecure_local_allowed()

    if not docker_ready and not allow_insecure:
        # STRICT ZERO-TRUST FAIL-CLOSED POLICY
        print(
            "\n[AegisML Security Gate] Docker sandbox is unavailable. "
            "Zero-Trust policy active: untrusted model/code will NOT be executed on host. "
            "Marking dynamic tests as skipped."
        )
        return _build_fail_closed_skip_response(
            planned_tests=planned_tests,
            reason=(
                "Zero-Trust sandbox policy active: Docker daemon is unavailable. "
                "Untrusted model deserialization and pipeline execution were halted to "
                "protect the host environment from potential RCE or resource exhaustion."
            )
        )

    if docker_ready:
        return _run_in_docker(
            model_path=model_path,
            dataset_path=dataset_path,
            pipeline_path=pipeline_path,
            vectorizer_path=vectorizer_path,
            text_column=text_column,
            label_column=label_column,
            planned_tests=planned_tests,
            strategy_config=strategy_config,
            agent_1_results=agent_1_results,
            timeout_seconds=timeout_seconds,
        )

    # Insecure local execution explicitly allowed by developer
    print(
        "\n[SECURITY WARNING] AEGISML_ALLOW_INSECURE_LOCAL_TESTING=true is set. "
        "Executing untrusted artifacts directly in host process!"
    )
    return _run_insecure_local(
        model_path=model_path,
        dataset_path=dataset_path,
        pipeline_path=pipeline_path,
        vectorizer_path=vectorizer_path,
        text_column=text_column,
        label_column=label_column,
        planned_tests=planned_tests,
        strategy_config=strategy_config,
        agent_1_results=agent_1_results,
    )


def _run_in_docker(
    model_path: str,
    dataset_path: str,
    pipeline_path: str,
    vectorizer_path: Optional[str],
    text_column: str,
    label_column: str,
    planned_tests: List[str],
    strategy_config: Optional[Dict[str, Any]],
    agent_1_results: Optional[Dict[str, Any]],
    timeout_seconds: int,
) -> Dict[str, Any]:
    """
    Executes the sandbox worker inside an air-gapped Docker container.
    """
    container_name = f"aegisml-sandbox-{uuid.uuid4().hex[:8]}"
    image_name = os.getenv("AEGISML_DOCKER_IMAGE", "aegisml-sandbox:latest")

    data_dir = os.path.abspath(os.path.dirname(dataset_path))
    temp_dir = tempfile.mkdtemp(prefix="aegisml_sandbox_")
    input_dir = os.path.join(temp_dir, "input")
    output_dir = os.path.join(temp_dir, "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    strategy_payload = {
        "model_path": f"/workspace/data/{os.path.basename(model_path)}",
        "dataset_path": f"/workspace/data/{os.path.basename(dataset_path)}",
        "pipeline_path": f"/workspace/data/{os.path.basename(pipeline_path)}",
        "vectorizer_path": (
            f"/workspace/data/{os.path.basename(vectorizer_path)}"
            if vectorizer_path else None
        ),
        "text_column": text_column,
        "label_column": label_column,
        "planned_tests": planned_tests,
        "agent_1_results": agent_1_results,
        **(strategy_config or {}),
    }

    input_json_path = os.path.join(input_dir, "strategy.json")
    with open(input_json_path, "w", encoding="utf-8") as f:
        json.dump(strategy_payload, f, indent=2)

    docker_cmd = [
        "docker", "run", "--rm",
        "--name", container_name,
        "--network", "none",
        "--memory", "4g",
        "--cpus", "2.0",
        "--pids-limit", "128",
        "-v", f"{data_dir}:/workspace/data:ro",
        "-v", f"{input_dir}:/workspace/input:ro",
        "-v", f"{output_dir}:/workspace/output:rw",
        image_name,
    ]

    try:
        proc = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        
        output_json_path = os.path.join(output_dir, "test_results.json")
        if os.path.exists(output_json_path):
            with open(output_json_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            results["sandbox_status"] = "executed"
            results.setdefault("telemetry", {})["exit_code"] = proc.returncode
            results["telemetry"]["container_id"] = container_name
            return results
        else:
            return {
                "sandbox_status": "error",
                "error": f"Container finished without producing output. Exit code: {proc.returncode}. Stderr: {proc.stderr}",
                "telemetry": {"exit_code": proc.returncode, "stderr": proc.stderr},
            }

    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "kill", container_name], capture_output=True)
        return {
            "sandbox_status": "error",
            "error": f"Sandbox container exceeded execution watchdog timeout ({timeout_seconds}s).",
            "telemetry": {"timeout": True},
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _build_fail_closed_skip_response(
    planned_tests: List[str],
    reason: str,
) -> Dict[str, Any]:
    """
    Builds structured skipped evidence payloads adhering to Zero-Trust invariant.
    """
    def _make_skip(vid: str, name: str) -> Dict[str, Any]:
        return {
            "vulnerability_id": vid,
            "vulnerability_name": name,
            "status": "unverified",
            "severity": None,
            "evidence": {
                "status": "skipped",
                "reason": reason,
            },
            "summary": "Dynamic test skipped: Docker sandbox isolation required by Zero-Trust policy.",
        }

    return {
        "sandbox_status": "skipped_zero_trust",
        "planned_tests": planned_tests,
        "poisoning_evidence": _make_skip("V1", "Data Poisoning"),
        "preprocessing_evidence": _make_skip("V2", "Preprocessing Attack Surface"),
        "validation_evidence": _make_skip("V3", "Data Validation Weaknesses"),
        "adversarial_evidence": _make_skip("V4", "Adversarial Robustness"),
        "execution_log": [
            "Zero-Trust Policy Gate: Docker daemon offline.",
            "Dynamic testing aborted on host to prevent untrusted code execution.",
            "Passed unverified status to Agent 3 for static-only reporting.",
        ],
        "telemetry": {
            "execution_mode": "fail_closed_skip",
            "docker_available": False,
        },
    }


def _run_insecure_local(
    model_path: str,
    dataset_path: str,
    pipeline_path: str,
    vectorizer_path: Optional[str],
    text_column: str,
    label_column: str,
    planned_tests: List[str],
    strategy_config: Optional[Dict[str, Any]],
    agent_1_results: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Runs tests locally in-process. Only reachable when AEGISML_ALLOW_INSECURE_LOCAL_TESTING=true.
    """
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

    results: Dict[str, Any] = {
        "sandbox_status": "executed_insecure_local",
        "planned_tests": planned_tests,
        "execution_log": ["Executed in-process via developer override."],
    }

    model = load_trained_model(model_path)
    X_text, y_true = load_dataset(dataset_path, text_column, label_column)
    if not vectorizer_path:
        base_dir = os.path.dirname(dataset_path) or "data"
        vectorizer_path = resolve_vectorizer_from_agent1(agent_1_results, base_dir=base_dir)
    vectorizer = load_vectorizer(vectorizer_path)

    adv_cfg = (strategy_config or {}).get("adversarial_config", {})

    if "V1_poisoning" in planned_tests:
        p_res = run_poisoning_test(model, X_text, y_true, vectorizer=vectorizer)
        results["poisoning_evidence"] = p_res.get("poisoning_evidence", {})

    if "V4_adversarial" in planned_tests:
        sample_size = adv_cfg.get("sample_size", 50)
        max_rel_budget = adv_cfg.get("max_relative_perturbation_budget", 0.5)
        a_res = run_adversarial_test(
            model=model,
            X_text=X_text,
            y_true=y_true,
            vectorizer=vectorizer,
            n_samples=sample_size,
            max_relative_perturbation_budget=max_rel_budget,
        )
        results["adversarial_evidence"] = a_res.get("adversarial_evidence", {})

    if "V2_preprocessing" in planned_tests:
        prep_state = {
            "model": model,
            "vectorizer": vectorizer,
            "pipeline_path": pipeline_path if os.path.exists(pipeline_path) else None,
        }
        prep_res = run_preprocess_checks(prep_state)
        results["preprocessing_evidence"] = prep_res.get("preprocessing_evidence", {})

    if "V3_validation" in planned_tests:
        val_res = run_validation_checks(model, X_text, y_true, vectorizer=vectorizer)
        results["validation_evidence"] = val_res.get("validation_evidence", {})

    return results

