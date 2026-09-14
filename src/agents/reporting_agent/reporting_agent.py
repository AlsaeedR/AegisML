import json
from typing import Any, Dict, List, Literal, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from src.core.llm import get_llm, is_llm_available
from .schemas import ReportingAgentState
from .tools import (
    score_finding_tool,
    compile_audit_report_tool,
    validate_audit_report_schema,
    validate_audit_report_semantics,
    make_reporting_tools,
)


def node_correlate_findings(state: ReportingAgentState) -> Dict[str, Any]:
    """Combines Agent 1 static findings with Agent 2 dynamic test telemetry."""
    agent_1 = state.get("agent_1_results") or {}
    agent_2 = state.get("agent_2_results") or {}

    static_findings = agent_1.get("vulnerability_findings", {}).get("vulnerabilities", [])
    dynamic_results = agent_2.get("structured_test_results", {}).get("results", [])

    dynamic_by_id = {
        result.get("vulnerability_id"): result
        for result in dynamic_results
        if result.get("vulnerability_id")
    }

    correlated_findings: List[Dict[str, Any]] = []

    for static_finding in static_findings:
        vulnerability_id = static_finding.get("vulnerability_id")
        dynamic_result = dynamic_by_id.get(
            vulnerability_id,
            {
                "vulnerability_id": vulnerability_id,
                "status": "not_tested",
                "severity": None,
                "evidence": {"reason": "No matching Agent 2 result was available."},
            },
        )

        correlated_finding = {
            "vulnerability_id": vulnerability_id,
            "category": static_finding.get("category"),
            "affected_components": static_finding.get("affected_components", []),
            "description": static_finding.get("description", ""),
            "status": static_finding.get("status", "vulnerable"),
            "mitigating_controls": static_finding.get("mitigating_controls", []),
            "recommendations": [],
            "test_status": dynamic_result.get("status", "not_tested"),
            "dynamic_severity": dynamic_result.get("severity"),
            "evidence": dynamic_result.get("evidence", {}),
        }

        final_risk = score_finding_tool(correlated_finding)
        correlated_finding.update(final_risk)
        correlated_findings.append(correlated_finding)

    log = list(state.get("execution_log", []))
    log.append(
        f"Correlated {len(correlated_findings)} findings with empirical telemetry "
        "and applied deterministic NIST AI 100-2e2025 risk scoring."
    )

    return {
        "correlated_findings": correlated_findings,
        "retry_count": 0,
        "max_retries": 3,
        "validation_errors": None,
        "execution_log": log,
        "status": "risk_scoring_completed",
    }


def node_synthesize_audit_report(state: ReportingAgentState) -> Dict[str, Any]:
    """Autonomous Synthesis Node — Active Tool-Calling ReAct Loop."""
    log = list(state.get("execution_log", []))
    correlated_findings = state.get("correlated_findings", [])
    agent_1 = state.get("agent_1_results") or {}
    agent_2 = state.get("agent_2_results") or {}
    validation_errors = state.get("validation_errors")

    tool_context: str = ""

    if is_llm_available():
        reporting_tools = make_reporting_tools(
            agent_1_results=agent_1,
            agent_2_results=agent_2,
            correlated_findings=correlated_findings,
        )
        tool_map = {t.name: t for t in reporting_tools}
        llm_with_tools = get_llm(temperature=0.0).bind_tools(reporting_tools)

        error_feedback = (
            f"\nATTENTION PRIOR VALIDATION ERRORS TO RESOLVE:\n{validation_errors}\n"
            if validation_errors else ""
        )

        system_msg = SystemMessage(content=(
            "You are the AegisML Lead Security Auditor & Governance Agent (Agent 3).\n"
            "You have tools to investigate pipeline components, empirical telemetry, and NIST baselines:\n"
            "- bound_query_pipeline_architecture(component_id): checks AST node details.\n"
            "- bound_query_empirical_telemetry(vulnerability_id): probes empirical attack telemetry.\n"
            "- verify_nist_compliance(vulnerability_id): retrieves NIST AI 100-2e2025 controls.\n\n"
            "Use these tools to ground your executive synthesis and evidence-informed recommendations. "
            "When finished probing, stop calling tools."
        ))

        human_msg = HumanMessage(content=(
            f"Correlated Findings Count: {len(correlated_findings)}\n"
            f"Findings Overview: {[f.get('vulnerability_id') for f in correlated_findings]}\n"
            f"{error_feedback}"
            "Query additional architecture or empirical telemetry as needed before report synthesis."
        ))

        messages: List[Any] = [system_msg, human_msg]

        for _ in range(2):
            response = llm_with_tools.invoke(messages)
            messages.append(response)

            if not response.tool_calls:
                break

            tool_findings: List[str] = []
            for tc in response.tool_calls:
                t_name = tc["name"]
                t_args = tc.get("args", {})
                t_id = tc["id"]

                if t_name in tool_map:
                    result = tool_map[t_name].invoke(t_args)
                    log.append(f"Reporting agent queried tool '{t_name}' with {t_args}.")
                    tool_findings.append(f"[{t_name}({t_args})]: {json.dumps(result, indent=2)}")
                else:
                    result = {"error": f"Unknown tool: {t_name}"}

                messages.append(ToolMessage(content=str(result), tool_call_id=t_id))

            if tool_findings:
                tool_context += "\n".join(tool_findings) + "\n"

    raw_report = compile_audit_report_tool(
        findings=correlated_findings,
        tool_context=tool_context or None,
        validation_errors=validation_errors,
    )

    log.append("Synthesized draft security audit report.")
    return {
        "final_report": raw_report,
        "execution_log": log,
        "status": "report_synthesized",
    }


def node_validate_audit_report(state: ReportingAgentState) -> Dict[str, Any]:
    """Validation node: runs syntactic schema verification and semantic quality checks."""
    raw_report = state.get("final_report", {})
    correlated_findings = state.get("correlated_findings", [])

    is_valid, errors, validated_report = validate_audit_report_schema(raw_report)
    if not is_valid or validated_report is None:
        current_retries = state.get("retry_count", 0) + 1
        return {
            "validation_errors": errors,
            "retry_count": current_retries,
            "status": "audit_report_validation_failed",
        }

    report_dict = validated_report.model_dump()
    sem_valid, sem_errors = validate_audit_report_semantics(report_dict, correlated_findings)
    if not sem_valid:
        current_retries = state.get("retry_count", 0) + 1
        return {
            "validation_errors": sem_errors,
            "retry_count": current_retries,
            "status": "audit_report_validation_failed",
        }

    return {
        "final_report": report_dict,
        "overall_risk": report_dict.get("overall_risk"),
        "validation_errors": None,
        "retry_count": 0,
        "status": "audit_report_validated",
    }


def route_after_report_validation(
    state: ReportingAgentState,
) -> Literal["synthesize_audit_report", "finalize_report"]:
    """Conditional router: loops back on failure or advances to finalization."""
    has_errors = state.get("validation_errors") is not None
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)

    if has_errors and retry_count < max_retries:
        return "synthesize_audit_report"

    return "finalize_report"


def node_finalize_report(state: ReportingAgentState) -> Dict[str, Any]:
    """Final node: seals the validated report and overall risk assessment."""
    report = state.get("final_report", {})
    log = list(state.get("execution_log", []))
    log.append("Completed and sealed final AegisML security audit report.")

    return {
        "overall_risk": report.get("overall_risk"),
        "final_report": report,
        "validation_errors": None,
        "execution_log": log,
        "status": "completed",
    }


def build_reporting_agent_graph():
    """Assembles and compiles the StateGraph for Agent 3."""
    graph = StateGraph(ReportingAgentState)

    graph.add_node("correlate_findings", node_correlate_findings)
    graph.add_node("synthesize_audit_report", node_synthesize_audit_report)
    graph.add_node("validate_audit_report", node_validate_audit_report)
    graph.add_node("finalize_report", node_finalize_report)

    graph.set_entry_point("correlate_findings")
    graph.add_edge("correlate_findings", "synthesize_audit_report")
    graph.add_edge("synthesize_audit_report", "validate_audit_report")

    graph.add_conditional_edges(
        "validate_audit_report",
        route_after_report_validation,
        {
            "synthesize_audit_report": "synthesize_audit_report",
            "finalize_report": "finalize_report",
        },
    )

    graph.add_edge("finalize_report", END)

    return graph.compile()


def run_reporting_agent(
    agent_1_results: Dict[str, Any],
    agent_2_results: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute the Reporting Agent."""
    app = build_reporting_agent_graph()

    initial_state: ReportingAgentState = {
        "agent_1_results": agent_1_results,
        "agent_2_results": agent_2_results,
        "correlated_findings": [],
        "execution_log": [],
        "status": "initialized",
    }

    return app.invoke(initial_state)