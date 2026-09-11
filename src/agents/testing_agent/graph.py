import os
from typing import Any, Dict, List, Optional
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate

from src.core.llm import get_llm, is_llm_available
from .state import TestingAgentState
from .schemas import (
    AttackStrategyPlan,
    AdversarialAttackConfig,
    PoisoningAttackConfig,
    PreprocessingTestConfig,
    ValidationTestConfig,
    ForensicAnalysisReport,
    ForensicFinding,
)
from .tools import (
    inspect_dataset_profile,
    calculate_perturbation_budget,
    resolve_threat_surface,
    COGNITIVE_PLANNING_TOOLS,
)
from .sandbox_runner import dispatch_sandbox


TEST_ORDER = [
    "V1_poisoning",
    "V4_adversarial",
    "V2_preprocessing",
    "V3_validation",
]


def node_prepare_metadata(state: TestingAgentState) -> Dict[str, Any]:
    """
    Host-side metadata extraction and safety validation.
    Performs safe inspection of dataset properties without deserializing untrusted models.
    """
    dataset_path = state.get("dataset_path", "")
    text_column = state.get("text_column")
    label_column = state.get("label_column")

    profile = inspect_dataset_profile.invoke({
        "dataset_path": dataset_path,
        "text_column": text_column,
        "label_column": label_column,
    })

    log = list(state.get("execution_plan_log") or [])
    log.append(f"Host-side safe metadata inspection completed for: {os.path.basename(dataset_path)}")

    return {
        "dataset_profile": profile,
        "execution_plan_log": log,
    }


def node_reason_strategy(state: TestingAgentState) -> Dict[str, Any]:
    """
    Cognitive Pre-Attack Reasoning Node.
    Uses LLM bound to LangChain tools to inspect threat surface and formulate
    mathematically calibrated attack parameters (Bounded Agency Pattern).
    """
    log = list(state.get("execution_plan_log") or [])
    agent_1 = state.get("agent_1_results") or {}
    dataset_profile = state.get("dataset_profile") or {}
    explicit_targets = state.get("test_targets")

    # Safe fallback plan if LLM is unavailable or fails
    fallback_plan = _get_default_strategy_plan(explicit_targets, agent_1, dataset_profile)

    if not is_llm_available():
        log.append("LLM credentials unavailable. Applied deterministic attack strategy plan.")
        return {
            "attack_strategy_plan": fallback_plan.model_dump(),
            "planned_tests": fallback_plan.selected_tests,
            "execution_plan_log": log,
        }

    try:
        llm = get_llm(temperature=0.0)
        structured_llm = llm.with_structured_output(AttackStrategyPlan)

        threat_model = agent_1.get("threat_model", {})
        vuln_findings = agent_1.get("vulnerability_findings", {}).get("vulnerabilities", [])

        system_prompt = (
            "You are the AegisML Lead Penetration Testing Strategist (Agent 2).\n"
            "Your objective is to inspect the static threat model from Agent 1 and the target dataset "
            "profile to formulate a mathematically grounded, highly targeted dynamic testing strategy.\n"
            "Calibrate perturbation budgets according to feature representation and dimensionality.\n"
            "Target tests available: V1_poisoning, V4_adversarial, V2_preprocessing, V3_validation."
        )

        user_prompt = (
            f"Dataset Profile:\n{dataset_profile}\n\n"
            f"Agent 1 Threat Model:\n{threat_model}\n\n"
            f"Agent 1 Static Findings:\n{vuln_findings}\n\n"
            f"Explicit Targets Override: {explicit_targets}\n\n"
            "Formulate the optimal attack strategy plan."
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", user_prompt),
        ])

        chain = prompt | structured_llm
        strategy_plan: AttackStrategyPlan = chain.invoke({})

        # Ensure canonical ordering of selected tests
        ordered_tests = [t for t in TEST_ORDER if t in strategy_plan.selected_tests]
        if not ordered_tests:
            ordered_tests = list(TEST_ORDER)
        strategy_plan.selected_tests = ordered_tests

        log.append(f"Cognitive strategy formulated. Planned tests: {strategy_plan.selected_tests}")
        log.append(f"Strategy rationale: {strategy_plan.planning_rationale}")

        return {
            "attack_strategy_plan": strategy_plan.model_dump(),
            "planned_tests": strategy_plan.selected_tests,
            "execution_plan_log": log,
        }

    except Exception as e:
        log.append(f"LLM strategy formulation encountered error ({str(e)}). Used deterministic fallback.")
        return {
            "attack_strategy_plan": fallback_plan.model_dump(),
            "planned_tests": fallback_plan.selected_tests,
            "execution_plan_log": log,
        }


def node_execute_sandbox(state: TestingAgentState) -> Dict[str, Any]:
    """
    Containerized Sandbox Dispatcher.
    Executes attacks inside an air-gapped Docker container with Fail-Closed Zero-Trust enforcement.
    """
    log = list(state.get("execution_plan_log") or [])
    planned_tests = state.get("planned_tests", TEST_ORDER)
    strategy_config = state.get("attack_strategy_plan", {})

    log.append(f"Dispatching dynamic tests to sandbox orchestrator: {planned_tests}")

    resolved_text_col = state.get("text_column") or (state.get("dataset_profile") or {}).get("text_column") or "text"
    resolved_label_col = state.get("label_column") or (state.get("dataset_profile") or {}).get("label_column") or "label"

    sandbox_result = dispatch_sandbox(
        model_path=state.get("model_path", "data/model.pkl"),
        dataset_path=state.get("dataset_path", "data/dataset.csv"),
        pipeline_path=state.get("pipeline_path") or "data/pipeline.py",
        vectorizer_path=state.get("vectorizer_path"),
        text_column=resolved_text_col,
        label_column=resolved_label_col,
        planned_tests=planned_tests,
        strategy_config=strategy_config,
        agent_1_results=state.get("agent_1_results"),
    )

    log.extend(sandbox_result.get("execution_log", []))

    return {
        "sandbox_status": sandbox_result.get("sandbox_status", "unknown"),
        "sandbox_telemetry": sandbox_result.get("telemetry", {}),
        "poisoning_evidence": sandbox_result.get("poisoning_evidence", {}),
        "adversarial_evidence": sandbox_result.get("adversarial_evidence", {}),
        "preprocessing_evidence": sandbox_result.get("preprocessing_evidence", {}),
        "validation_evidence": sandbox_result.get("validation_evidence", {}),
        "execution_plan_log": log,
    }


def node_forensic_diagnosis(state: TestingAgentState) -> Dict[str, Any]:
    """
    Cognitive Post-Attack Forensic Diagnosis Node.
    Synthesizes empirical attack telemetry, identifies mathematical root causes,
    and cross-verifies against Agent 1's qualitative hypotheses.
    """
    log = list(state.get("execution_plan_log") or [])
    sandbox_status = state.get("sandbox_status", "")

    if sandbox_status == "skipped_zero_trust":
        log.append("Dynamic tests skipped under Zero-Trust policy. Forensic analysis deferred.")
        return {
            "forensic_analysis": {
                "overall_forensic_summary": (
                    "Empirical penetration testing was halted on the host because the Docker sandbox "
                    "was offline. Zero-Trust policy prevented untrusted artifact execution. "
                    "All findings remain based on Agent 1's static threat model."
                ),
                "findings": [],
            },
            "execution_plan_log": log,
        }

    # Collect available empirical evidence
    evidence_bundle = {
        "V1_poisoning": state.get("poisoning_evidence"),
        "V4_adversarial": state.get("adversarial_evidence"),
        "V2_preprocessing": state.get("preprocessing_evidence"),
        "V3_validation": state.get("validation_evidence"),
    }

    if not is_llm_available():
        fallback_forensics = _get_default_forensic_report(evidence_bundle)
        return {
            "forensic_analysis": fallback_forensics.model_dump(),
            "execution_plan_log": log,
        }

    try:
        llm = get_llm(temperature=0.1)
        structured_llm = llm.with_structured_output(ForensicAnalysisReport)

        system_prompt = (
            "You are the AegisML Lead ML Forensic Security Diagnostician (Agent 2).\n"
            "Analyze the empirical DAST test results and diagnose mathematical root causes.\n"
            "Confirm or refute whether the empirical evidence validates Agent 1's static hypotheses."
        )

        user_prompt = (
            f"Empirical Test Telemetry:\n{evidence_bundle}\n\n"
            f"Agent 1 Static Findings:\n{state.get('agent_1_results', {}).get('vulnerability_findings')}\n\n"
            "Provide forensic findings and an overall forensic synthesis."
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", user_prompt),
        ])

        chain = prompt | structured_llm
        report: ForensicAnalysisReport = chain.invoke({})
        log.append("Cognitive forensic diagnosis completed successfully.")

        return {
            "forensic_analysis": report.model_dump(),
            "execution_plan_log": log,
        }
    except Exception as e:
        log.append(f"LLM forensic diagnosis encountered error ({str(e)}). Used deterministic fallback.")
        fallback = _get_default_forensic_report(evidence_bundle)
        return {
            "forensic_analysis": fallback.model_dump(),
            "execution_plan_log": log,
        }


def node_aggregate_results(state: TestingAgentState) -> Dict[str, Any]:
    """
    Aggregates all Agent 2 outputs into the structured payload expected by Agent 3.
    """
    results: List[Dict[str, Any]] = []
    verifications: List[Dict[str, Any]] = []
    agent_1 = state.get("agent_1_results")

    # V1
    p_ev = state.get("poisoning_evidence")
    if p_ev:
        results.append(p_ev)
        if agent_1:
            drop = p_ev.get("evidence", {}).get("accuracy_drop") or p_ev.get("evidence", {}).get("generic_test", {}).get("max_accuracy_drop")
            verifications.append(_verification_from_status("V1", "Data Poisoning", p_ev, {"accuracy_drop": drop}))

    # V2
    prep_ev = state.get("preprocessing_evidence")
    if prep_ev:
        results.append(prep_ev)
        if agent_1:
            verifications.append(_verification_from_status("V2", "Preprocessing Attack Surface", prep_ev, prep_ev.get("evidence", {})))

    # V3
    val_ev = state.get("validation_evidence")
    if val_ev:
        results.append(val_ev)
        if agent_1:
            verifications.append(_verification_from_status("V3", "Data Validation Weaknesses", val_ev, {"accuracy_drop": val_ev.get("evidence", {}).get("accuracy_drop")}))

    # V4
    adv_ev = state.get("adversarial_evidence")
    if adv_ev:
        results.append(adv_ev)
        if agent_1:
            asr = adv_ev.get("evidence", {}).get("attack_success_rate_within_budget")
            verifications.append(_verification_from_status("V4", "Adversarial Robustness", adv_ev, {"attack_success_rate": asr}))

    # Fill in any missing default tests as not_tested
    present_ids = {r.get("vulnerability_id") for r in results}
    default_tests = [
        ("V1", "Data Poisoning"),
        ("V2", "Preprocessing Attack Surface"),
        ("V3", "Data Validation Weaknesses"),
        ("V4", "Adversarial Robustness"),
    ]
    for vid, vname in default_tests:
        if vid not in present_ids:
            results.append({
                "vulnerability_id": vid,
                "vulnerability_name": vname,
                "status": "not_tested",
                "severity": None,
                "evidence": {"reason": "Test was not scheduled in strategy."},
            })

    result_order = {"V1": 1, "V4": 2, "V2": 3, "V3": 4}
    results.sort(key=lambda r: result_order.get(r.get("vulnerability_id", ""), 99))

    return {
        "structured_test_results": {
            "results": results,
            "hypothesis_verifications": verifications,
            "attack_strategy_plan": state.get("attack_strategy_plan"),
            "forensic_analysis": state.get("forensic_analysis"),
            "sandbox_status": state.get("sandbox_status"),
            "sandbox_telemetry": state.get("sandbox_telemetry"),
            "execution_log": state.get("execution_plan_log") or [],
        },
        "hypothesis_verifications": verifications,
        "status": "completed",
    }


def _verification_from_status(
    vulnerability_id: str,
    vulnerability_name: str,
    evidence_dict: Dict[str, Any],
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    test_status = evidence_dict.get("status", "unverified")
    dynamic_severity = evidence_dict.get("severity")

    if test_status == "vulnerable":
        correlation = "Confirmed Risk"
        rationale = f"Empirical testing confirmed susceptibility ({dynamic_severity or 'detected'})."
    elif test_status == "not_vulnerable":
        correlation = "False Positive (Mitigated)"
        rationale = "Empirical tests showed the model resisted attack conditions."
    elif test_status == "not_applicable":
        correlation = "Not Applicable"
        rationale = "Test was not applicable to the target pipeline structure."
    else:
        correlation = "Unverified"
        rationale = "Empirical testing was inconclusive or skipped under Zero-Trust policy."

    return {
        "vulnerability_id": vulnerability_id,
        "category": vulnerability_name,
        "correlation_status": correlation,
        "test_status": test_status,
        "dynamic_severity": dynamic_severity,
        "correlation_rationale": rationale,
        "metrics": metrics,
    }


def _get_default_strategy_plan(
    explicit_targets: Optional[List[str]],
    agent_1_results: Dict[str, Any],
    dataset_profile: Dict[str, Any],
) -> AttackStrategyPlan:
    avg_words = dataset_profile.get("text_stats", {}).get("avg_word_count", 100)
    budget_info = calculate_perturbation_budget.invoke({
        "avg_word_count": avg_words,
        "high_sparsity": True,
    })

    adv_cfg = AdversarialAttackConfig(
        sample_size=budget_info.get("recommended_sample_size", 50),
        max_relative_perturbation_budget=budget_info.get("recommended_max_relative_budget", 0.4),
        max_iter=budget_info.get("recommended_max_iter", 45),
        rationale=budget_info.get("rationale", "Standard budget for text classification."),
    )

    planned = ["V1_poisoning", "V4_adversarial", "V2_preprocessing", "V3_validation"]
    if explicit_targets:
        planned = [t for t in planned if any(x.lower() in t.lower() for x in explicit_targets)]
        if not planned:
            planned = list(TEST_ORDER)

    return AttackStrategyPlan(
        selected_tests=planned,
        adversarial_config=adv_cfg,
        planning_rationale="Deterministic rule-based baseline strategy.",
    )


def _get_default_forensic_report(evidence_bundle: Dict[str, Any]) -> ForensicAnalysisReport:
    findings = []
    for test_key, ev in evidence_bundle.items():
        if not ev:
            continue
        status = ev.get("status", "unverified")
        vid = ev.get("vulnerability_id", test_key[:2])
        vname = ev.get("vulnerability_name", test_key)
        findings.append(ForensicFinding(
            vulnerability_id=vid,
            category=vname,
            hypothesis_confirmation="Confirmed Risk" if status == "vulnerable" else "False Positive (Mitigated)" if status == "not_vulnerable" else "Unverified",
            root_cause_diagnosis=ev.get("summary", "Automated empirical metric diagnosis."),
            empirical_metric_summary=str(ev.get("evidence", {})),
            recommended_focus_area="Remediation guidance deferred to Agent 3.",
        ))

    return ForensicAnalysisReport(
        findings=findings,
        overall_forensic_summary="Empirical test results evaluated across dynamic testing modules.",
    )


def build_testing_agent_graph():
    graph = StateGraph(TestingAgentState)

    graph.add_node("prepare_metadata", node_prepare_metadata)
    graph.add_node("reason_strategy", node_reason_strategy)
    graph.add_node("execute_sandbox", node_execute_sandbox)
    graph.add_node("forensic_diagnosis", node_forensic_diagnosis)
    graph.add_node("aggregate_results", node_aggregate_results)

    graph.set_entry_point("prepare_metadata")
    graph.add_edge("prepare_metadata", "reason_strategy")
    graph.add_edge("reason_strategy", "execute_sandbox")
    graph.add_edge("execute_sandbox", "forensic_diagnosis")
    graph.add_edge("forensic_diagnosis", "aggregate_results")
    graph.add_edge("aggregate_results", END)

    return graph.compile()
