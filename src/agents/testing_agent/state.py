from typing import TypedDict, Any, Dict, List, Optional


class TestingAgentState(TypedDict, total=False):
    """
    Shared state for Agent 2 (Vulnerability Testing Agent).
    Maintains input configurations, artifacts loaded into memory, dynamic testing plans
    derived from Agent 1's findings, empirical test evidence, and hypothesis verifications.
    """
    # Upstream data received from Agent 1
    agent_1_results: Optional[Dict[str, Any]]

    # Artifact references and dataset configuration
    model_path: str
    dataset_path: str
    vectorizer_path: Optional[str]
    text_column: str
    label_column: str
    test_targets: Optional[List[str]]

    # In-memory models and data
    model: Any
    vectorizer: Optional[Any]
    X_text: List[str]
    y_true: List[Any]

    # Dynamic execution planning and reasoning log
    planned_tests: List[str]
    execution_plan_log: List[str]

    # Empirical test evidence for MVP vulnerabilities
    poisoning_evidence: Dict[str, Any]
    adversarial_evidence: Dict[str, Any]
    preprocessing_evidence: Dict[str, Any]
    validation_evidence: Dict[str, Any]

    # Final aggregated results and cross-verification against Agent 1 hypotheses
    hypothesis_verifications: List[Dict[str, Any]]
    structured_test_results: Dict[str, Any]
    status: str
