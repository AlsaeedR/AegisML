import json
import os
from src.agents.pipeline_agent.pipeline_agent import run_pipeline_agent

# Target ML pipeline to audit (can point to any Python pipeline file)
target_path = os.path.join("data", "21011088.py")

print(f"Reading target pipeline from: {target_path}")
with open(target_path, "r", encoding="utf-8") as f:
    python_code = f.read()

# Execute Agent 1: Pipeline & Threat Modeling Agent
print("Executing Agent 1 (Pipeline & Threat Modeling Agent)...")
result = run_pipeline_agent(python_code)

print("\n=== PIPELINE GRAPH ===")
print(json.dumps(result.get("pipeline_graph"), indent=2))

print("\n=== THREAT MODEL (NIST AI 100-2e2025) ===")
print(json.dumps(result.get("threat_model"), indent=2))

print("\n=== VULNERABILITY FINDINGS & RECOMMENDATIONS ===")
print(json.dumps(result.get("vulnerability_findings"), indent=2))
