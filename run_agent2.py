from src.agents.testing_agent.testing_agent import run_testing_agent
import json

result = run_testing_agent(
    model_path="data/model.pkl",
    dataset_path="data/dataset.csv",
    text_column="text",
    label_column="label",
)

print(json.dumps(result["structured_test_results"], indent=2, default=str))
