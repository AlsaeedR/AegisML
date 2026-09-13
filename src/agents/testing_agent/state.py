from typing import TypedDict, Any, Dict, List, Optional

class TestingAgentState(TypedDict, total=False):
    agent_1_results: Optional[Dict[str, Any]]
    audit_id: Optional[str]
    model_path: str
    dataset_path: str
    pipeline_path: str
    vectorizer_path: Optional[str]
    text_column: str
    label_column: str
    test_targets: Optional[List[str]]
    dataset_profile: Optional[Dict[str, Any]]
    attack_strategy_plan: Optional[Dict[str, Any]]
    planned_tests: List[str]
    execution_plan_log: List[str]
    model: Any
    vectorizer: Optional[Any]
    X_text: List[str]
    y_true: List[Any]
    sandbox_status: str
    sandbox_telemetry: Optional[Dict[str, Any]]
    poisoning_evidence: Dict[str, Any]
    adversarial_evidence: Dict[str, Any]
    preprocessing_evidence: Dict[str, Any]
    validation_evidence: Dict[str, Any]
    forensic_analysis: Optional[Dict[str, Any]]
    hypothesis_verifications: List[Dict[str, Any]]
    structured_test_results: Dict[str, Any]
    status: str
