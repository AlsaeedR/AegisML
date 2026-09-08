import json
import os
from src.agents.pipeline_agent.pipeline_agent import run_pipeline_agent
from src.agents.testing_agent.testing_agent import run_testing_agent

# ---------------------------------------------------------
# Configuration: Target Pipeline and Evaluation Artifacts
# ---------------------------------------------------------
target_pipeline_path = os.path.join("data", "21011088.py")
model_path = os.path.join("data", "model.pkl")
dataset_path = os.path.join("data", "dataset.csv")

print(f"Reading target pipeline from: {target_pipeline_path}")
with open(target_pipeline_path, "r", encoding="utf-8") as f:
    python_code = f.read()

# ---------------------------------------------------------
# Phase 1: Execute Agent 1 (Pipeline & Threat Modeling)
# ---------------------------------------------------------
print("\nExecuting Agent 1 (Pipeline & Threat Modeling Agent)...")
agent_1_result = run_pipeline_agent(python_code)

print("\n=== [AGENT 1] PIPELINE GRAPH ===")
print(json.dumps(agent_1_result.get("pipeline_graph"), indent=2))

print("\n=== [AGENT 1] THREAT MODEL (NIST AI 100-2e2025) ===")
print(json.dumps(agent_1_result.get("threat_model"), indent=2))

print("\n=== [AGENT 1] VULNERABILITY FINDINGS & RECOMMENDATIONS ===")
print(json.dumps(agent_1_result.get("vulnerability_findings"), indent=2))

# ---------------------------------------------------------
# Phase 2: Execute Agent 2 (Vulnerability Testing Agent)
# Guided by Agent 1's threat model and vulnerability findings
# ---------------------------------------------------------
if os.path.exists(model_path) and os.path.exists(dataset_path):
    print("\nExecuting Agent 2 (Vulnerability Testing Agent)...")
    agent_2_result = run_testing_agent(
        agent_1_results=agent_1_result,
        model_path=model_path,
        dataset_path=dataset_path,
        pipeline_path=target_pipeline_path,
        text_column="text",
        label_column="label",
    )

    print("\n=== [AGENT 2] DYNAMIC TEST RESULTS ===")
    print(json.dumps(agent_2_result.get("structured_test_results"), indent=2))
else:
    print(
        f"\nSkipping Agent 2 dynamic testing: evaluation artifacts not found "
        f"('{model_path}' or '{dataset_path}')."
    )
