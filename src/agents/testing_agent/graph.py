import os
from typing import Dict, Any, List, Literal
from langgraph.graph import StateGraph, END

from .state import TestingAgentState
from .loader import (
    load_trained_model,
    load_dataset,
    load_vectorizer,
    resolve_vectorizer_from_agent1,
)
from .poisoning_test import run_poisoning_test
from .adversarial_test import run_adversarial_test
from .preprocess_test import run_preprocess_checks
from .validation_test import run_validation_checks

# Define TEST_ORDER here so it doesn't crash
TEST_ORDER = ["V1_poisoning", "V4_adversarial", "V2_preprocessing", "V3_validation"]

def node_load_artifacts(state: TestingAgentState) -> Dict[str, Any]:
    model = load_trained_model(state["model_path"])
    X_text, y_true = load_dataset(
        state["dataset_path"], state["text_column"], state["label_column"]
    )

    # Load the pipeline source code for V2 static scanning
    pipeline_path = state.get("pipeline_path")
    code = None
    if pipeline_path and os.path.exists(pipeline_path):
        with open(pipeline_path, "r", encoding="utf-8") as f:
            code = f.read()

    vectorizer_path = state.get("vectorizer_path")
    if not vectorizer_path:
        base_dir = os.path.dirname(state["dataset_path"]) or "data"
        vectorizer_path = resolve_vectorizer_from_agent1(state.get("agent_1_results"), base_dir=base_dir)

    vectorizer = load_vectorizer(vectorizer_path)

    return {
        "model": model,
        "vectorizer": vectorizer,
        "vectorizer_path": vectorizer_path,
        "X_text": X_text,
        "y_true": y_true,
        "code": code,
    }

def node_plan_tests(state: TestingAgentState) -> Dict[str, Any]:
    agent_1 = state.get("agent_1_results")
    explicit_targets = state.get("test_targets")
    planned_tests: List[str] = []
    plan_log: List[str] = []

    if explicit_targets:
        if any(t.upper() in ["V1", "POISONING", "DATA_POISONING"] for t in explicit_targets):
            planned_tests.append("V1_poisoning")
        if any(t.upper() in ["V2", "PREPROCESSING"] for t in explicit_targets):
            planned_tests.append("V2_preprocessing")
        if any(t.upper() in ["V3", "VALIDATION"] for t in explicit_targets):
            planned_tests.append("V3_validation")
        if any(t.upper() in ["V4", "ADVERSARIAL", "ADVERSARIAL_ROBUSTNESS"] for t in explicit_targets):
            planned_tests.append("V4_adversarial")
        plan_log.append(f"Planned tests from explicit targets: {planned_tests}")
    elif agent_1:
        vulns = agent_1.get("vulnerability_findings", {}).get("vulnerabilities", [])
        flagged_categories = {v.get("vulnerability_id"): v.get("category") for v in vulns}

        if "V1" in flagged_categories or any("poison" in str(c).lower() for c in flagged_categories.values()):
            planned_tests.append("V1_poisoning")
            plan_log.append("Scheduled V1_poisoning based on Agent 1 data poisoning finding.")
        if "V2" in flagged_categories or any("preprocess" in str(c).lower() for c in flagged_categories.values()):
            planned_tests.append("V2_preprocessing")
            plan_log.append("Scheduled V2_preprocessing based on Agent 1 preprocessing finding.")
        if "V3" in flagged_categories or any("validation" in str(c).lower() for c in flagged_categories.values()):
            planned_tests.append("V3_validation")
            plan_log.append("Scheduled V3_validation based on Agent 1 data validation finding.")
        if "V4" in flagged_categories or any("adversarial" in str(c).lower() for c in flagged_categories.values()):
            planned_tests.append("V4_adversarial")
            plan_log.append("Scheduled V4_adversarial based on Agent 1 adversarial robustness finding.")
    else:
        # Fixed this line so it doesn't crash!
        planned_tests = list(TEST_ORDER)
        plan_log.append("No upstream Agent 1 findings provided. Scheduling full default test suite.")

    return {
        "planned_tests": planned_tests,
        "execution_plan_log": plan_log,
        "status": "tests_planned",
    }

def route_after_planning(state: TestingAgentState) -> Literal["poisoning_test", "adversarial_test", "preprocess_test", "validation_test", "aggregate_results"]:
    planned = state.get("planned_tests", [])
    if "V1_poisoning" in planned:
        return "poisoning_test"
    if "V4_adversarial" in planned:
        return "adversarial_test"
    if "V2_preprocessing" in planned:
        return "preprocess_test"
    if "V3_validation" in planned:
        return "validation_test"
    return "aggregate_results"

def node_poisoning_test(state: TestingAgentState) -> Dict[str, Any]:
    return run_poisoning_test(state)

def route_after_poisoning(state: TestingAgentState) -> Literal["adversarial_test", "preprocess_test", "validation_test", "aggregate_results"]:
    planned = state.get("planned_tests", [])
    if "V4_adversarial" in planned:
        return "adversarial_test"
    if "V2_preprocessing" in planned:
        return "preprocess_test"
    if "V3_validation" in planned:
        return "validation_test"
    return "aggregate_results"

def node_adversarial_test(state: TestingAgentState) -> Dict[str, Any]:
    return run_adversarial_test(state)

# Added this routing function so it doesn't loop back to V1!
def route_after_adversarial(state: TestingAgentState) -> Literal["preprocess_test", "validation_test", "aggregate_results"]:
    planned = state.get("planned_tests", [])
    if "V2_preprocessing" in planned:
        return "preprocess_test"
    if "V3_validation" in planned:
        return "validation_test"
    return "aggregate_results"

def node_preprocess_test(state: TestingAgentState) -> Dict[str, Any]:
    return run_preprocess_checks(state)

# Added this routing function so it doesn't loop back to V1!
def route_after_preprocess(state: TestingAgentState) -> Literal["validation_test", "aggregate_results"]:
    planned = state.get("planned_tests", [])
    if "V3_validation" in planned:
        return "validation_test"
    return "aggregate_results"

def node_validation_test(state: TestingAgentState) -> Dict[str, Any]:
    return run_validation_checks(state)

def node_aggregate_results(state: TestingAgentState) -> Dict[str, Any]:
    """
    Aggregates all empirical evidence and performs hypothesis verification
    by correlating test measurements against Agent 1 qualitative findings.
    """
    results: List[Dict[str, Any]] = []
    verifications: List[Dict[str, Any]] = []
    agent_1 = state.get("agent_1_results")

    # 1. Process Poisoning Results (V1)
    if "poisoning_evidence" in state:
        poisoning = state["poisoning_evidence"]
        results.append(poisoning)
        if agent_1:
            drop = poisoning.get("evidence", {}).get("generic_test", {}).get("accuracy_drop", 0.0)
            verified = drop > 0.05
            verifications.append({
                "vulnerability_id": "V1",
                "category": "Data Poisoning",
                "hypothesis_confirmed": verified,
                "empirical_drop": drop,
                "assessment": (
                    f"Empirical test confirms Agent 1 hypothesis: model exhibited {drop * 100:.1f}% "
                    "accuracy drop under simulated label flipping."
                    if verified
                    else "Empirical test did not observe significant accuracy degradation."
                ),
            })

    # 2. Process Adversarial Results (V4)
    if "adversarial_evidence" in state:
        adversarial = state["adversarial_evidence"]
        results.append(adversarial)
        if agent_1:
            asr = adversarial.get("evidence", {}).get("attack_success_rate_within_budget", 0.0)
            verified = asr >= 0.3 if asr is not None else False
            verifications.append({
                "vulnerability_id": "V4",
                "category": "Adversarial Robustness",
                "hypothesis_confirmed": verified,
                "attack_success_rate": asr,
                "assessment": (
                    f"Empirical test confirms Agent 1 hypothesis: evasion attacks succeeded with "
                    f"{asr * 100:.1f}% attack success rate within the perturbation budget."
                    if verified
                    else "Attacks did not achieve high success rates within realistic perturbation limits."
                ),
            })

    # 3. Process Preprocessing Results (V2)
    if "preprocessing_evidence" in state:
        results.append(state["preprocessing_evidence"])

    # 4. Process Validation Results (V3)
    if "validation_evidence" in state:
        results.append(state["validation_evidence"])

    # 5. Preserved placeholders (only if V2/V3 didn't run)
    if not state.get("preprocessing_evidence"):
        results.append({
            "vulnerability_id": "V2",
            "vulnerability_name": "Preprocessing Attack Surface",
            "status": "not_tested",
            "severity": None,
            "evidence": {"reason": "Test module pending implementation by assigned team member."},
        })

    if not state.get("validation_evidence"):
        results.append({
            "vulnerability_id": "V3",
            "vulnerability_name": "Data Validation Weaknesses",
            "status": "not_tested",
            "severity": None,
            "evidence": {"reason": "Test module pending implementation by assigned team member."},
        })

    return {
        "structured_test_results": {
            "results": results,
            "hypothesis_verifications": verifications,
            "execution_log": state.get("execution_plan_log", []),
        },
        "hypothesis_verifications": verifications,
        "status": "completed",
    }


def build_testing_agent_graph():
    graph = StateGraph(TestingAgentState)

    graph.add_node("load_artifacts", node_load_artifacts)
    graph.add_node("plan_tests", node_plan_tests)
    graph.add_node("poisoning_test", node_poisoning_test)
    graph.add_node("adversarial_test", node_adversarial_test)
    graph.add_node("preprocess_test", node_preprocess_test)
    graph.add_node("validation_test", node_validation_test)
    graph.add_node("aggregate_results", node_aggregate_results)

    graph.set_entry_point("load_artifacts")
    graph.add_edge("load_artifacts", "plan_tests")

    graph.add_conditional_edges(
        "plan_tests", route_after_planning,
        {"poisoning_test": "poisoning_test", "adversarial_test": "adversarial_test", "preprocess_test": "preprocess_test", "validation_test": "validation_test", "aggregate_results": "aggregate_results"}
    )

    graph.add_conditional_edges(
        "poisoning_test", route_after_poisoning,
        {"adversarial_test": "adversarial_test", "preprocess_test": "preprocess_test", "validation_test": "validation_test", "aggregate_results": "aggregate_results"}
    )

    # FIXED: Use route_after_adversarial here, not route_after_planning!
    graph.add_conditional_edges(
        "adversarial_test", route_after_adversarial,
        {"preprocess_test": "preprocess_test", "validation_test": "validation_test", "aggregate_results": "aggregate_results"}
    )

    # FIXED: Use route_after_preprocess here, not route_after_planning!
    graph.add_conditional_edges(
        "preprocess_test", route_after_preprocess,
        {"validation_test": "validation_test", "aggregate_results": "aggregate_results"}
    )

    graph.add_edge("validation_test", "aggregate_results")
    graph.add_edge("aggregate_results", END)

    return graph.compile()