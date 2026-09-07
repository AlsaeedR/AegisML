from typing import Optional, List
from .graph import build_testing_agent_graph
from .state import TestingAgentState

def run_testing_agent(model_path: str, dataset_path: str, text_column: str, label_column: str, test_targets: Optional[List[str]]=None):
    app = build_testing_agent_graph()
    initial_state: TestingAgentState = {'model_path': model_path, 'dataset_path': dataset_path, 'text_column': text_column, 'label_column': label_column, 'test_targets': test_targets}
    return app.invoke(initial_state)
if __name__ == '__main__':
    result = run_testing_agent(model_path='data/model.pkl', dataset_path='data/dataset.csv', text_column='text', label_column='label')
    print(result['structured_test_results'])
