from src.agents.pipeline_agent.pipeline_agent import run_pipeline_agent

# Load the Python ML pipeline file
with open("data/21011088.py", "r", encoding="utf-8") as f:
    python_code = f.read()

# Run the agent
result = run_pipeline_agent(python_code)

print(result["pipeline_graph"])
print(result["threat_model"])
print(result["vulnerability_findings"])
