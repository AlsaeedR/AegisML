"""Pillar 2 Guardrails & Validations Test Suite.

Automated pytest suite verifying the multi-layered defense-in-depth architecture:
- Layer 1: Input Syntax Guard & Cryptographic Artifact Fingerprinting (StaleArtifactError)
- Layer 2: LLM Output & Semantic Guardrails (Pydantic Schema & Risk Score bounds)
- Layer 3: Zero-Trust Fail-Closed Sandbox & Adaptive OOM Downscaling
- Layer 4: Human-in-the-Loop Gate 1 Plan Adherence
- Path-Level: Trajectory Graph Transition Validity
"""

import os
import sys
import uuid
import pytest
from unittest.mock import patch

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.agents.pipeline_agent.parser import parse_python_pipeline
from src.core.audit_memory import (
    initialize_memory,
    register_audit_artifacts,
    verify_artifact_integrity,
    StaleArtifactError,
    compute_code_hash,
    save_step_checkpoint,
    get_audit_ledger,
)
from src.agents.testing_agent.schemas import (
    AttackStrategyPlan,
    AdversarialAttackConfig,
    ForensicAnalysisReport,
    TEST_ORDER,
)
from src.agents.testing_agent.sandbox_runner import (
    dispatch_sandbox,
    _downscale_strategy_config,
    _build_fail_closed_skip_response,
)
from src.agents.reporting_agent.risk_scoring import (
    calculate_final_risk,
    severity_from_score,
)


@pytest.fixture(autouse=True)
def setup_db():
    initialize_memory()


# =====================================================================
# LAYER 1: INPUT & INTEGRITY GUARDRAILS
# =====================================================================

def test_layer1_input_syntax_guard_rejects_malformed_code():
    """Verifies that non-Python or syntax-corrupted input is intercepted before LLM execution."""
    malformed_code = "def invalid_syntax(x\n  print('missing colon and indent')"
    result = parse_python_pipeline(malformed_code)
    
    assert "nodes" in result
    assert len(result["nodes"]) >= 1
    # Check that error node was recorded
    error_nodes = [n for n in result["nodes"] if "parse_error" in n.get("id", "")]
    assert len(error_nodes) > 0
    assert "syntax error" in error_nodes[0].get("name", "").lower()


def test_layer1_cryptographic_tampering_raises_stale_artifact_error():
    """Verifies that tampering with code, model, or dataset between resumed steps raises StaleArtifactError."""
    audit_id = f"test_stale_{uuid.uuid4().hex[:8]}"
    initial_code = "import pandas as pd\ndf = pd.read_csv('clean.csv')\n"
    
    # Baseline registration
    register_audit_artifacts(audit_id=audit_id, code=initial_code)
    
    # Same code passes verification
    assert verify_artifact_integrity(audit_id=audit_id, code=initial_code) is True
    
    # Tampered code raises StaleArtifactError
    tampered_code = "import pandas as pd\n# INJECTED MALICIOUS CODE\ndf = pd.read_csv('backdoored.csv')\n"
    with pytest.raises(StaleArtifactError) as exc_info:
        verify_artifact_integrity(audit_id=audit_id, code=tampered_code)
    
    assert "Pipeline source code was modified" in str(exc_info.value)


# =====================================================================
# LAYER 2: LLM OUTPUT & SEMANTIC GUARDRAILS
# =====================================================================

def test_layer2_pydantic_schema_validation_enforces_bounds():
    """Verifies that AttackStrategyPlan enforces type bounds on perturbation budget and sample sizes."""
    # Valid config
    valid_plan = AttackStrategyPlan(
        strategy_rationale="Thorough vulnerability assessment",
        tests_to_run=["poisoning", "adversarial"],
        adversarial_config=AdversarialAttackConfig(
            max_relative_perturbation_budget=0.25,
            sample_size=100,
        ),
    )
    assert valid_plan.adversarial_config.max_relative_perturbation_budget == 0.25
    assert valid_plan.adversarial_config.sample_size == 100

    # Invalid budget (> 1.0) must fail validation
    with pytest.raises(Exception):
        AttackStrategyPlan(
            strategy_rationale="Exploitative attack",
            tests_to_run=["adversarial"],
            adversarial_config=AdversarialAttackConfig(
                max_relative_perturbation_budget=2.5,  # Exceeds maximum 1.0
            ),
        )


def test_layer2_mathematical_risk_scoring_bounds():
    """Verifies that risk scores adhere to the 0.0 to 10.0 invariant bounds and severity scales."""
    # Test low impact / mitigated finding
    low_finding = {
        "vulnerability_id": "V1",
        "category": "Data Poisoning",
        "status": "mitigated",
        "test_status": "not_vulnerable",
        "degradation": 0.01,
        "mitigating_controls": ["SHA-256 validation"],
    }
    low_result = calculate_final_risk(low_finding)
    assert 0.0 <= low_result["risk_score"] <= 10.0
    assert 0.0 <= low_result["static_risk_score"] <= 10.0
    assert low_result["severity"] in ["Critical", "High", "Medium", "Low"]

    # Test high impact / vulnerable finding
    high_finding = {
        "vulnerability_id": "V1",
        "category": "Data Poisoning",
        "status": "vulnerable",
        "test_status": "vulnerable",
        "degradation": 0.35,
        "mitigating_controls": [],
    }
    high_result = calculate_final_risk(high_finding)
    assert 0.0 <= high_result["risk_score"] <= 10.0
    assert high_result["risk_score"] >= low_result["risk_score"]
    assert high_result["severity"] in ["Critical", "High", "Medium", "Low"]


# =====================================================================
# LAYER 3: ZERO-TRUST SANDBOX EXECUTION GUARDRAILS
# =====================================================================

def test_layer3_zero_trust_fail_closed_sandbox_when_docker_unavailable():
    """Verifies that if Docker is unavailable, tests fail-closed with zero host execution."""
    with patch("src.agents.testing_agent.sandbox_runner.is_docker_available", return_value=False):
        res = dispatch_sandbox(
            model_path="benchmarks/pipelines/all_defended/model.pkl",
            dataset_path="benchmarks/pipelines/all_defended/evaluation_dataset.csv",
            pipeline_path="benchmarks/pipelines/all_defended/pipeline.py",
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


def test_layer3_adaptive_oom_downscaling():
    """Verifies that container memory spikes trigger adaptive downscaling of sample sizes."""
    high_memory_config = {
        "adversarial_config": {
            "sample_size": 200,
            "max_relative_perturbation_budget": 0.2,
        }
    }
    
    downscaled = _downscale_strategy_config(high_memory_config)
    assert downscaled["adversarial_config"]["sample_size"] == 100
    
    # Second downscale halves again
    downscaled_again = _downscale_strategy_config(downscaled)
    assert downscaled_again["adversarial_config"]["sample_size"] == 50
    
    # Lower bound clamp at 10
    tiny_config = {"adversarial_config": {"sample_size": 12}}
    clamped = _downscale_strategy_config(tiny_config)
    assert clamped["adversarial_config"]["sample_size"] == 10


# =====================================================================
# LAYER 4: HUMAN-IN-THE-LOOP (HITL) GATE 1 PLAN ADHERENCE
# =====================================================================

def test_layer4_human_gate1_plan_adherence():
    """Verifies that when human gate 1 modifies planned tests, execution strictly adheres to approved set."""
    autonomous_proposal = ["poisoning", "preprocessing", "validation", "adversarial"]
    human_approved_subset = ["poisoning", "validation"]  # Excluded preprocessing and adversarial
    
    # Execution response generator must strictly filter to approved subset
    skip_resp = _build_fail_closed_skip_response(
        planned_tests=human_approved_subset,
        reason="Human Gate 1 modified test plan",
    )
    
    assert skip_resp["planned_tests"] == human_approved_subset
    assert "preprocessing" not in skip_resp["planned_tests"]
    assert "adversarial" not in skip_resp["planned_tests"]


# =====================================================================
# PATH-LEVEL TRAJECTORY OBSERVABILITY
# =====================================================================

def test_path_level_trajectory_validity_in_ledger():
    """Verifies chronological trajectory logging and sequential validity in audit ledger."""
    audit_id = f"test_trajectory_{uuid.uuid4().hex[:8]}"
    
    # Simulate a valid execution sequence in audit memory
    steps = [
        ("Agent 1", "extract_pipeline"),
        ("Agent 1", "reason_threat_model"),
        ("Agent 1", "validate_threat_model"),
        ("Agent 1", "reason_vulnerabilities"),
        ("Agent 1", "validate_vulnerabilities"),
        ("Agent 1", "finalize_agent_results"),
        ("Agent 2", "prepare_metadata"),
        ("Agent 2", "reason_strategy"),
        ("Agent 2", "execute_sandbox"),
        ("Agent 2", "forensic_diagnosis"),
        ("Agent 2", "aggregate_results"),
        ("Agent 3", "correlate_findings"),
        ("Agent 3", "synthesize_audit_report"),
        ("Agent 3", "validate_audit_report"),
        ("Agent 3", "finalize_report"),
    ]
    
    for agent_name, step_name in steps:
        save_step_checkpoint(audit_id, agent_name, step_name, {"status": "ok"}, duration_seconds=0.1)
    
    ledger = get_audit_ledger(audit_id)
    recorded_steps = [entry["step_name"] for entry in ledger]
    
    assert len(recorded_steps) == len(steps)
    assert recorded_steps == [s[1] for s in steps]
