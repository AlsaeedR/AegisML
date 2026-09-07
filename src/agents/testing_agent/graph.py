from langgraph.graph import StateGraph, END
from .state import TestingAgentState
from .loader import load_trained_model, load_dataset
from .poisoning_test import run_poisoning_test
from .adversarial_test import run_adversarial_test
from .target_matching import extract_target_ids

def node_load_artifacts(state: TestingAgentState):
    model = load_trained_model(state['model_path'])
    X_text, y_true = load_dataset(state['dataset_path'], state['text_column'], state['label_column'])
    return {'model': model, 'X_text': X_text, 'y_true': y_true}

def node_poisoning_test(state: TestingAgentState):
    target_ids = extract_target_ids(state.get('test_targets'))
    # target_ids is None -> Agent 1 gave no targets (or Agent 2 is running
    # standalone) -> run every implemented test, same as before.
    if target_ids is not None and 'V1' not in target_ids:
        return {
            'poisoning_evidence': {
                'vulnerability_id': 'V1',
                'vulnerability_name': 'Data Poisoning',
                'status': 'not_applicable',
                'severity': None,
                'evidence': {
                    'reason': "Agent 1 did not flag Data Poisoning as a "
                    "test target for this pipeline, so this test was "
                    "skipped.",
                },
            }
        }
    return run_poisoning_test(state)

def node_adversarial_test(state: TestingAgentState):
    target_ids = extract_target_ids(state.get('test_targets'))
    if target_ids is not None and 'V4' not in target_ids:
        return {
            'adversarial_evidence': {
                'vulnerability_id': 'V4',
                'vulnerability_name': 'Adversarial Robustness',
                'status': 'not_applicable',
                'severity': None,
                'evidence': {
                    'reason': "Agent 1 did not flag Adversarial Robustness "
                    "as a test target for this pipeline, so this test was "
                    "skipped.",
                },
            }
        }
    return run_adversarial_test(state)

def node_aggregate_results(state: TestingAgentState):
    results = []
    if 'poisoning_evidence' in state:
        results.append(state['poisoning_evidence'])
    if 'adversarial_evidence' in state:
        results.append(state['adversarial_evidence'])
    if 'preprocessing_evidence' not in state:
        results.append({'vulnerability_id': 'V2', 'vulnerability_name': 'Preprocessing Attack Surface', 'status': 'not_tested', 'severity': None, 'evidence': {'reason': 'Test not implemented yet.'}})
    if 'validation_evidence' not in state:
        results.append({'vulnerability_id': 'V3', 'vulnerability_name': 'Data Validation Weaknesses', 'status': 'not_tested', 'severity': None, 'evidence': {'reason': 'Test not implemented yet.'}})
    return {'structured_test_results': {'results': results}}

def build_testing_agent_graph():
    graph = StateGraph(TestingAgentState)
    graph.add_node('load_artifacts', node_load_artifacts)
    graph.add_node('poisoning_test', node_poisoning_test)
    graph.add_node('adversarial_test', node_adversarial_test)
    graph.add_node('aggregate_results', node_aggregate_results)
    graph.set_entry_point('load_artifacts')
    graph.add_edge('load_artifacts', 'poisoning_test')
    graph.add_edge('poisoning_test', 'adversarial_test')
    graph.add_edge('adversarial_test', 'aggregate_results')
    graph.add_edge('aggregate_results', END)
    return graph.compile()
