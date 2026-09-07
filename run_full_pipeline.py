import json

from src.agents.pipeline_agent.pipeline_agent import run_pipeline_agent
from src.agents.testing_agent.testing_agent import run_testing_agent


def run_full_pipeline(
    target_python_file: str,
    model_path: str,
    dataset_path: str,
    text_column: str = "text",
    label_column: str = "label",
):
    # --- Step 1: Agent 1, first pass (identify test targets) ---
    with open(target_python_file, "r", encoding="utf-8") as f:
        code = f.read()

    print("Running Agent 1 (first pass: pipeline graph + threat model + test targets)...")
    agent1_first_pass = run_pipeline_agent(code)
    test_targets = agent1_first_pass["vulnerability_findings"]

    # --- Step 2: Agent 2 (run real tests, produce evidence) ---
    print("Running Agent 2 (poisoning + adversarial robustness tests)...")
    agent2_result = run_testing_agent(
        model_path=model_path,
        dataset_path=dataset_path,
        text_column=text_column,
        label_column=label_column,
        test_targets=test_targets,
    )
    test_results = agent2_result["structured_test_results"]

    # --- Step 3: Agent 1, second pass (incorporate real test evidence) ---
    print("Running Agent 1 (second pass: incorporating Agent 2 evidence)...")
    agent1_final = run_pipeline_agent(code, testing_agent_results=test_results)

    return {
        "pipeline_graph": agent1_final["pipeline_graph"],
        "threat_model": agent1_final["threat_model"],
        "vulnerability_findings": agent1_final["vulnerability_findings"],
        "agent2_test_results": test_results,
    }


if __name__ == "__main__":
    result = run_full_pipeline(
        target_python_file="data/21011088.py",
        model_path="data/model.pkl",
        dataset_path="data/dataset.csv",
        text_column="text",
        label_column="label",
    )

    print("\n=== FINAL COMBINED RESULT ===")
    print(json.dumps(result, indent=2, default=str))
