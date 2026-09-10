import importlib.util
import os
from typing import Any, Dict, List, Literal

from langgraph.graph import END, StateGraph

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


TEST_ORDER = [
    "V1_poisoning",
    "V2_preprocessing",
    "V3_validation",
    "V4_adversarial",
]


def node_load_artifacts(
    state: TestingAgentState,
) -> Dict[str, Any]:
    model = load_trained_model(
        state["model_path"]
    )

    X_text, y_true = load_dataset(
        state["dataset_path"],
        state["text_column"],
        state["label_column"],
    )

    vectorizer_path = state.get(
        "vectorizer_path"
    )

    if not vectorizer_path:
        base_dir = (
            os.path.dirname(
                state["dataset_path"]
            )
            or "data"
        )

        vectorizer_path = (
            resolve_vectorizer_from_agent1(
                state.get(
                    "agent_1_results"
                ),
                base_dir=base_dir,
            )
        )

    vectorizer = load_vectorizer(
        vectorizer_path
    )

    return {
        "model": model,
        "vectorizer": vectorizer,
        "vectorizer_path": vectorizer_path,
        "X_text": X_text,
        "y_true": y_true,
    }


def node_plan_tests(
    state: TestingAgentState,
) -> Dict[str, Any]:
    agent_1 = state.get(
        "agent_1_results"
    )

    explicit_targets = state.get(
        "test_targets"
    )

    planned_tests: List[str] = []
    plan_log: List[str] = []

    if explicit_targets:

        normalized_targets = {
            str(target).upper()
            for target in explicit_targets
        }

        if normalized_targets.intersection(
            {
                "V1",
                "POISONING",
                "DATA_POISONING",
            }
        ):
            planned_tests.append(
                "V1_poisoning"
            )

        if normalized_targets.intersection(
            {
                "V2",
                "PREPROCESSING",
            }
        ):
            planned_tests.append(
                "V2_preprocessing"
            )

        if normalized_targets.intersection(
            {
                "V3",
                "VALIDATION",
            }
        ):
            planned_tests.append(
                "V3_validation"
            )

        if normalized_targets.intersection(
            {
                "V4",
                "ADVERSARIAL",
                "ADVERSARIAL_ROBUSTNESS",
            }
        ):
            planned_tests.append(
                "V4_adversarial"
            )

        planned_tests = [
            test
            for test in TEST_ORDER
            if test in planned_tests
        ]

        plan_log.append(
            "Planned tests from explicit targets: "
            f"{planned_tests}"
        )

    elif agent_1:

        vulnerabilities = (
            agent_1.get(
                "vulnerability_findings",
                {},
            ).get(
                "vulnerabilities",
                [],
            )
        )

        flagged_categories = {
            vulnerability.get(
                "vulnerability_id"
            ): vulnerability.get(
                "category"
            )
            for vulnerability
            in vulnerabilities
        }

        if (
            "V1" in flagged_categories
            or any(
                "poison"
                in str(category).lower()
                for category
                in flagged_categories.values()
            )
        ):
            planned_tests.append(
                "V1_poisoning"
            )
            plan_log.append(
                "Scheduled V1_poisoning based on "
                "Agent 1 data poisoning finding."
            )

        if (
            "V2" in flagged_categories
            or any(
                "preprocess"
                in str(category).lower()
                for category
                in flagged_categories.values()
            )
        ):
            planned_tests.append(
                "V2_preprocessing"
            )
            plan_log.append(
                "Scheduled V2_preprocessing based on "
                "Agent 1 preprocessing finding."
            )

        if (
            "V3" in flagged_categories
            or any(
                "validation"
                in str(category).lower()
                for category
                in flagged_categories.values()
            )
        ):
            planned_tests.append(
                "V3_validation"
            )
            plan_log.append(
                "Scheduled V3_validation based on "
                "Agent 1 data validation finding."
            )

        if (
            "V4" in flagged_categories
            or any(
                "adversarial"
                in str(category).lower()
                for category
                in flagged_categories.values()
            )
        ):
            planned_tests.append(
                "V4_adversarial"
            )
            plan_log.append(
                "Scheduled V4_adversarial based on "
                "Agent 1 adversarial robustness finding."
            )

        planned_tests = [
            test
            for test in TEST_ORDER
            if test in planned_tests
        ]

    else:
        planned_tests = list(
            TEST_ORDER
        )

        plan_log.append(
            "No upstream Agent 1 findings provided. "
            "Scheduling full default test suite."
        )

    return {
        "planned_tests": planned_tests,
        "execution_plan_log": plan_log,
        "status": "tests_planned",
    }


def route_after_planning(
    state: TestingAgentState,
) -> Literal[
    "poisoning_test",
    "preprocess_test",
    "validation_test",
    "adversarial_test",
    "aggregate_results",
]:
    planned = state.get(
        "planned_tests",
        []
    )

    if "V1_poisoning" in planned:
        return "poisoning_test"

    if "V2_preprocessing" in planned:
        return "preprocess_test"

    if "V3_validation" in planned:
        return "validation_test"

    if "V4_adversarial" in planned:
        return "adversarial_test"

    return "aggregate_results"


def node_poisoning_test(
    state: TestingAgentState,
) -> Dict[str, Any]:
    return run_poisoning_test(
        state
    )


def route_after_poisoning(
    state: TestingAgentState,
) -> Literal[
    "preprocess_test",
    "validation_test",
    "adversarial_test",
    "aggregate_results",
]:
    planned = state.get(
        "planned_tests",
        []
    )

    if "V2_preprocessing" in planned:
        return "preprocess_test"

    if "V3_validation" in planned:
        return "validation_test"

    if "V4_adversarial" in planned:
        return "adversarial_test"

    return "aggregate_results"



PREPROCESSOR_NAMES = (
    "preprocess_text",
    "preprocess",
    "clean_text",
    "clean",
    "normalize_text",
    "normalize",
)


def _resolve_pipeline_source_path(
    state: TestingAgentState,
) -> str | None:
    """
    Resolve the uploaded pipeline/source Python file
    from common state keys.

    Different integration layers may use slightly
    different names, so this helper keeps the testing
    agent tolerant without changing the existing API.
    """

    candidate_keys = (
        "pipeline_path",
        "pipeline_file_path",
        "source_code_path",
        "source_path",
        "code_path",
    )

    for key in candidate_keys:
        value = state.get(key)

        if (
            isinstance(value, str)
            and value.strip()
            and os.path.isfile(value)
        ):
            return value

    agent_1 = state.get(
        "agent_1_results"
    )

    if isinstance(agent_1, dict):
        for key in candidate_keys:
            value = agent_1.get(key)

            if (
                isinstance(value, str)
                and value.strip()
                and os.path.isfile(value)
            ):
                return value

    return None


def _load_external_text_preprocessor(
    pipeline_path: str,
):
    """
    Load a recognized external text-preprocessing
    function from the uploaded pipeline/source file.

    This is used only for the V2 dynamic test so the
    test can follow projects whose inference path is:

        raw text -> clean_text/preprocess_text
                 -> serialized model.predict()

    The existing serialized model is not modified.
    """

    module_name = (
        "_aegisml_dynamic_pipeline_"
        + str(
            abs(
                hash(
                    os.path.abspath(
                        pipeline_path
                    )
                )
            )
        )
    )

    try:
        spec = (
            importlib.util.spec_from_file_location(
                module_name,
                pipeline_path,
            )
        )

        if (
            spec is None
            or spec.loader is None
        ):
            return (
                None,
                None,
                "Could not create a module spec "
                "for the uploaded pipeline file.",
            )

        module = (
            importlib.util.module_from_spec(
                spec
            )
        )

        spec.loader.exec_module(
            module
        )

    except Exception as exc:
        return (
            None,
            None,
            (
                "External preprocessing discovery "
                f"failed: {exc}"
            ),
        )

    for name in PREPROCESSOR_NAMES:
        candidate = getattr(
            module,
            name,
            None,
        )

        if callable(candidate):
            return (
                candidate,
                name,
                None,
            )

    return (
        None,
        None,
        (
            "No recognized external text "
            "preprocessing function was found."
        ),
    )


def node_preprocess_test(
    state: TestingAgentState,
) -> Dict[str, Any]:
    """
    Run V2 using the serialized model preprocessing
    path and, when available, the project's external
    text preprocessing function.

    The callable is added only to a temporary copy of
    the state used for this node, so no callable needs
    to be persisted in LangGraph state.
    """

    preprocess_state = dict(
        state
    )

    existing_preprocessor = any(
        callable(
            preprocess_state.get(key)
        )
        for key in (
            "preprocess_function",
            "preprocessing_function",
            "text_preprocessor",
            "preprocessor",
        )
    )

    pipeline_path = None
    discovery_status = (
        "not_requested"
        if existing_preprocessor
        else "not_found"
    )
    discovery_error = None

    if not existing_preprocessor:
        pipeline_path = (
            _resolve_pipeline_source_path(
                state
            )
        )

        if pipeline_path:
            (
                external_preprocessor,
                external_preprocessor_name,
                discovery_error,
            ) = (
                _load_external_text_preprocessor(
                    pipeline_path
                )
            )

            if callable(
                external_preprocessor
            ):
                preprocess_state[
                    "preprocess_function"
                ] = external_preprocessor

                discovery_status = "loaded"
            else:
                discovery_status = (
                    "not_loaded"
                )

    result = run_preprocess_checks(
        preprocess_state
    )

    preprocessing_evidence = (
        result.get(
            "preprocessing_evidence"
        )
    )

    if isinstance(
        preprocessing_evidence,
        dict,
    ):
        evidence = (
            preprocessing_evidence.get(
                "evidence"
            )
        )

        if isinstance(
            evidence,
            dict,
        ):
            evidence[
                "external_preprocessor_discovery"
            ] = {
                "status": discovery_status,
                "pipeline_path_available": (
                    pipeline_path is not None
                ),
                "error": discovery_error,
            }

    return result


def route_after_preprocess(
    state: TestingAgentState,
) -> Literal[
    "validation_test",
    "adversarial_test",
    "aggregate_results",
]:
    planned = state.get(
        "planned_tests",
        []
    )

    if "V3_validation" in planned:
        return "validation_test"

    if "V4_adversarial" in planned:
        return "adversarial_test"

    return "aggregate_results"


def node_validation_test(
    state: TestingAgentState,
) -> Dict[str, Any]:
    return run_validation_checks(
        state
    )


def route_after_validation(
    state: TestingAgentState,
) -> Literal[
    "adversarial_test",
    "aggregate_results",
]:
    planned = state.get(
        "planned_tests",
        []
    )

    if "V4_adversarial" in planned:
        return "adversarial_test"

    return "aggregate_results"


def node_adversarial_test(
    state: TestingAgentState,
) -> Dict[str, Any]:
    return run_adversarial_test(
        state
    )


def _verification_from_status(
    vulnerability_id: str,
    category: str,
    result: Dict[str, Any],
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Build a hypothesis-verification record from the
    dynamic test result without re-deriving the test's
    own vulnerability thresholds in the graph layer.
    """

    status = str(
        result.get(
            "status",
            "not_tested",
        )
    ).lower()

    if status == "vulnerable":
        hypothesis_confirmed = True
        assessment = (
            "Dynamic testing produced empirical "
            "evidence consistent with the Agent 1 "
            "security hypothesis."
        )

    elif status == "not_vulnerable":
        hypothesis_confirmed = False
        assessment = (
            "Dynamic testing did not confirm the "
            "Agent 1 security hypothesis under the "
            "evaluated conditions."
        )

    elif status == "not_applicable":
        hypothesis_confirmed = None
        assessment = (
            "The dynamic test was not applicable "
            "to the evaluated pipeline shape."
        )

    else:
        hypothesis_confirmed = None
        assessment = (
            "Dynamic testing did not provide enough "
            "evidence to verify or reject the "
            "Agent 1 security hypothesis."
        )

    verification = {
        "vulnerability_id": vulnerability_id,
        "category": category,
        "test_status": status,
        "hypothesis_confirmed": (
            hypothesis_confirmed
        ),
        "assessment": assessment,
    }

    if extra:
        verification.update(
            extra
        )

    return verification


def _not_tested_result(
    vulnerability_id: str,
    vulnerability_name: str,
) -> Dict[str, Any]:
    return {
        "vulnerability_id": vulnerability_id,
        "vulnerability_name": vulnerability_name,
        "status": "not_tested",
        "severity": None,
        "evidence": {
            "reason": (
                "This test was not scheduled or "
                "did not produce dynamic evidence."
            )
        },
    }


def node_aggregate_results(
    state: TestingAgentState,
) -> Dict[str, Any]:
    """
    Aggregate Agent 2 empirical evidence and, when
    Agent 1 findings are available, record whether
    each dynamic test confirms the corresponding
    qualitative security hypothesis.

    The graph does not recalculate test-specific
    vulnerability thresholds. Each test module owns
    its status and severity decision.
    """

    results: List[Dict[str, Any]] = []
    verifications: List[Dict[str, Any]] = []

    agent_1 = state.get(
        "agent_1_results"
    )

    # V1 - Data Poisoning
    poisoning = state.get(
        "poisoning_evidence"
    )

    if poisoning:
        results.append(
            poisoning
        )

        if agent_1:
            generic = (
                poisoning.get(
                    "evidence",
                    {},
                ).get(
                    "generic_test",
                    {},
                )
            )

            max_drop = generic.get(
                "max_accuracy_drop"
            )

            verifications.append(
                _verification_from_status(
                    "V1",
                    "Data Poisoning",
                    poisoning,
                    {
                        "max_accuracy_drop": (
                            max_drop
                        ),
                        "worst_flip_fraction": (
                            generic.get(
                                "worst_flip_fraction"
                            )
                        ),
                    },
                )
            )

    # V2 - Preprocessing Attack Surface
    preprocessing = state.get(
        "preprocessing_evidence"
    )

    if preprocessing:
        results.append(
            preprocessing
        )

        if agent_1:
            dynamic = (
                preprocessing.get(
                    "evidence",
                    {}
                )
            )

            verifications.append(
                _verification_from_status(
                    "V2",
                    "Preprocessing Attack Surface",
                    preprocessing,
                    {
                        "failure_rate": dynamic.get(
                            "failure_rate"
                        ),
                        "failed_test_cases": dynamic.get(
                            "failed_test_cases"
                        ),
                        "semantic_issue_count": dynamic.get(
                            "semantic_issue_count"
                        ),
                        "semantic_issue_rate": dynamic.get(
                            "semantic_issue_rate"
                        ),
                        "semantic_issue_cases": dynamic.get(
                            "semantic_issue_cases"
                        ),
                    },
                )
            )

    # V3 - Data Validation Weaknesses
    validation = state.get(
        "validation_evidence"
    )

    if validation:
        results.append(
            validation
        )

        if agent_1:
            validation_evidence = (
                validation.get(
                    "evidence",
                    {},
                )
            )

            verifications.append(
                _verification_from_status(
                    "V3",
                    "Data Validation Weaknesses",
                    validation,
                    {
                        "accuracy_drop": (
                            validation_evidence.get(
                                "accuracy_drop"
                            )
                        ),
                    },
                )
            )

    # V4 - Adversarial Robustness
    adversarial = state.get(
        "adversarial_evidence"
    )

    if adversarial:
        results.append(
            adversarial
        )

        if agent_1:
            asr = (
                adversarial.get(
                    "evidence",
                    {},
                ).get(
                    "attack_success_rate_within_budget"
                )
            )

            verifications.append(
                _verification_from_status(
                    "V4",
                    "Adversarial Robustness",
                    adversarial,
                    {
                        "attack_success_rate": asr,
                    },
                )
            )

    present_ids = {
        result.get(
            "vulnerability_id"
        )
        for result in results
    }

    default_results = [
        (
            "V1",
            "Data Poisoning",
        ),
        (
            "V2",
            "Preprocessing Attack Surface",
        ),
        (
            "V3",
            "Data Validation Weaknesses",
        ),
        (
            "V4",
            "Adversarial Robustness",
        ),
    ]

    for (
        vulnerability_id,
        vulnerability_name,
    ) in default_results:

        if vulnerability_id not in present_ids:
            results.append(
                _not_tested_result(
                    vulnerability_id,
                    vulnerability_name,
                )
            )

    result_order = {
        "V1": 1,
        "V2": 2,
        "V3": 3,
        "V4": 4,
    }

    results.sort(
        key=lambda result: result_order.get(
            result.get(
                "vulnerability_id"
            ),
            99,
        )
    )

    return {
        "structured_test_results": {
            "results": results,
            "hypothesis_verifications": (
                verifications
            ),
            "execution_log": state.get(
                "execution_plan_log",
                [],
            ),
        },
        "hypothesis_verifications": (
            verifications
        ),
        "status": "completed",
    }


def build_testing_agent_graph():
    graph = StateGraph(
        TestingAgentState
    )

    graph.add_node(
        "load_artifacts",
        node_load_artifacts,
    )

    graph.add_node(
        "plan_tests",
        node_plan_tests,
    )

    graph.add_node(
        "poisoning_test",
        node_poisoning_test,
    )

    graph.add_node(
        "preprocess_test",
        node_preprocess_test,
    )

    graph.add_node(
        "validation_test",
        node_validation_test,
    )

    graph.add_node(
        "adversarial_test",
        node_adversarial_test,
    )

    graph.add_node(
        "aggregate_results",
        node_aggregate_results,
    )

    graph.set_entry_point(
        "load_artifacts"
    )

    graph.add_edge(
        "load_artifacts",
        "plan_tests",
    )

    graph.add_conditional_edges(
        "plan_tests",
        route_after_planning,
        {
            "poisoning_test": "poisoning_test",
            "preprocess_test": "preprocess_test",
            "validation_test": "validation_test",
            "adversarial_test": "adversarial_test",
            "aggregate_results": "aggregate_results",
        },
    )

    graph.add_conditional_edges(
        "poisoning_test",
        route_after_poisoning,
        {
            "preprocess_test": "preprocess_test",
            "validation_test": "validation_test",
            "adversarial_test": "adversarial_test",
            "aggregate_results": "aggregate_results",
        },
    )

    graph.add_conditional_edges(
        "preprocess_test",
        route_after_preprocess,
        {
            "validation_test": "validation_test",
            "adversarial_test": "adversarial_test",
            "aggregate_results": "aggregate_results",
        },
    )

    graph.add_conditional_edges(
        "validation_test",
        route_after_validation,
        {
            "adversarial_test": "adversarial_test",
            "aggregate_results": "aggregate_results",
        },
    )

    graph.add_edge(
        "adversarial_test",
        "aggregate_results",
    )

    graph.add_edge(
        "aggregate_results",
        END,
    )

    return graph.compile()
