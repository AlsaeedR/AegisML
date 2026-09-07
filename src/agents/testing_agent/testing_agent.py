from typing import Optional, List, Dict, Any
from .graph import build_testing_agent_graph
from .state import TestingAgentState


def run_testing_agent(
    agent_1_results: Optional[Dict[str, Any]] = None,
    model_path: str = "data/model.pkl",
    dataset_path: str = "data/dataset.csv",
    text_column: str = "text",
    label_column: str = "label",
    vectorizer_path: Optional[str] = None,
    test_targets: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Executes Agent 2 (Vulnerability Testing Agent).

    Ingests Agent 1 findings (pipeline graph, threat model, and vulnerability findings),
    autonomously plans and dispatches empirical security tests, and cross-verifies
    empirical measurements against upstream hypotheses.

    Parameters:
        agent_1_results: Output dictionary from Agent 1 (run_pipeline_agent).
        model_path: Path to serialized trained model artifact.
        dataset_path: Path to evaluation dataset CSV.
        text_column: Name of the feature text column.
        label_column: Name of the target label column.
        vectorizer_path: Optional path to a separately saved vectorizer artifact.
        test_targets: Optional manual override of vulnerability test targets.

    Returns:
        State dictionary containing structured test results and hypothesis verifications.
    """
    app = build_testing_agent_graph()

    initial_state: TestingAgentState = {
        "agent_1_results": agent_1_results,
        "model_path": model_path,
        "dataset_path": dataset_path,
        "text_column": text_column,
        "label_column": label_column,
        "vectorizer_path": vectorizer_path,
        "test_targets": test_targets,
        "status": "initialized",
    }

    return app.invoke(initial_state)


if __name__ == "__main__":
    result = run_testing_agent(
        model_path="data/model.pkl",
        dataset_path="data/dataset.csv",
        text_column="text",
        label_column="label",
    )
    print(result["structured_test_results"])
