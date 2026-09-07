from typing import TypedDict, Any, Dict, List, Optional

class TestingAgentState(TypedDict, total=False):
    model_path: str
    dataset_path: str
    text_column: str
    label_column: str
    test_targets: Optional[List[str]]
    model: Any
    X_text: List[str]
    y_true: List[Any]
    poisoning_evidence: Dict[str, Any]
    adversarial_evidence: Dict[str, Any]
    preprocessing_evidence: Dict[str, Any]
    validation_evidence: Dict[str, Any]
    structured_test_results: Dict[str, Any]
