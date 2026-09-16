import json
import time
from typing import Any, Dict, List, Literal, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from src.core.llm import get_llm, is_llm_available
from src.core.audit_memory import get_step_checkpoint, save_step_checkpoint
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
    audit_id = state.get("audit_id")
    t0 = time.time()
    agent_1 = state.get("agent_1_results") or {}
    agent_2 = state.get("agent_2_results") or {}

    vuln_obj = agent_1.get("vulnerability_findings") or {}
    if isinstance(vuln_obj, list):
        static_findings = vuln_obj
    elif isinstance(vuln_obj, dict):
        static_findings = vuln_obj.get("vulnerabilities", [])
    else:
        static_findings = []

    dynamic_results = agent_2.get("structured_test_results", {}).get("results", [])

    dynamic_by_id: Dict[str, Any] = {}
    for result in dynamic_results:
        vid = str(result.get("vulnerability_id", "")).strip().upper()
        if vid:
            dynamic_by_id[vid] = result
            if len(vid) >= 2:
                dynamic_by_id[vid[:2]] = result

    # Also check direct evidence fields on agent_2 (e.g. poisoning_evidence, etc.)
    evidence_field_map = {
        "V1": "poisoning_evidence",
        "V2": "preprocessing_evidence",
        "V3": "validation_evidence",
        "V4": "adversarial_evidence",
    }
    for short_vid, field_name in evidence_field_map.items():
        if short_vid not in dynamic_by_id:
            direct_ev = agent_2.get(field_name)
            if isinstance(direct_ev, dict) and direct_ev.get("status"):
                dynamic_by_id[short_vid] = direct_ev

    if audit_id:
        cached = get_step_checkpoint(audit_id, "correlate_findings")
        if cached is not None:
            cached_findings = cached.get("correlated_findings", [])
            cached_has_dynamic = any(
                f.get("test_status") in ("vulnerable", "not_vulnerable")
                for f in cached_findings
            )
            current_has_dynamic = any(
                r.get("status") in ("vulnerable", "not_vulnerable")
                for r in dynamic_by_id.values()
            )
            if cached_has_dynamic == current_has_dynamic:
                return cached

    pipeline_graph = agent_1.get("pipeline_graph")
    threat_model = agent_1.get("threat_model")

    correlated_findings: List[Dict[str, Any]] = []

    for static_finding in static_findings:
        vulnerability_id = static_finding.get("vulnerability_id")
        short_id = str(vulnerability_id)[:2].upper() if vulnerability_id else ""
        dynamic_result = dynamic_by_id.get(
            vulnerability_id,
            dynamic_by_id.get(
                short_id,
                {
                    "vulnerability_id": vulnerability_id,
                    "status": "not_tested",
                    "severity": None,
                    "evidence": {"reason": "No matching Agent 2 result was available."},
                },
            ),
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

        final_risk = score_finding_tool(
            correlated_finding,
            pipeline_graph=pipeline_graph,
            threat_model=threat_model,
        )
        correlated_finding.update(final_risk)
        correlated_findings.append(correlated_finding)

    log = list(state.get("execution_log", []))
    log.append(
        f"Correlated {len(correlated_findings)} findings with empirical telemetry "
        "and applied deterministic NIST AI 100-2e2025 risk scoring."
    )

    res = {
        "correlated_findings": correlated_findings,
        "retry_count": 0,
        "max_retries": 3,
        "validation_errors": None,
        "execution_log": log,
        "status": "risk_scoring_completed",
    }
    if audit_id:
        save_step_checkpoint(audit_id, "Agent 3", "correlate_findings", res, duration_seconds=round(time.time() - t0, 3))
    return res


def node_synthesize_audit_report(state: ReportingAgentState) -> Dict[str, Any]:
    """Autonomous Synthesis Node — Active Tool-Calling ReAct Loop."""
    audit_id = state.get("audit_id")
    if audit_id and not state.get("validation_errors"):
        cached = get_step_checkpoint(audit_id, "synthesize_audit_report")
        if cached is not None:
            cached_has_dynamic = any(
                f.get("test_status") in ("vulnerable", "not_vulnerable")
                for f in cached.get("final_report", {}).get("findings", [])
            )
            current_has_dynamic = any(
                f.get("test_status") in ("vulnerable", "not_vulnerable")
                for f in state.get("correlated_findings", [])
            )
            if cached_has_dynamic == current_has_dynamic:
                return cached

    t0 = time.time()
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
    res = {
        "final_report": raw_report,
        "execution_log": log,
        "status": "report_synthesized",
    }
    if audit_id:
        save_step_checkpoint(audit_id, "Agent 3", "synthesize_audit_report", res, duration_seconds=round(time.time() - t0, 3))
    return res


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

    res = {
        "final_report": report_dict,
        "overall_risk": report_dict.get("overall_risk"),
        "validation_errors": None,
        "retry_count": 0,
        "status": "audit_report_validated",
    }
    audit_id = state.get("audit_id")
    if audit_id:
        save_step_checkpoint(audit_id, "Agent 3", "validate_audit_report", res)
    return res


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

    res = {
        "overall_risk": report.get("overall_risk"),
        "final_report": report,
        "validation_errors": None,
        "execution_log": log,
        "status": "completed",
    }
    audit_id = state.get("audit_id")
    if audit_id:
        save_step_checkpoint(audit_id, "Agent 3", "finalize_report", res)
    return res


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
    audit_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute the Reporting Agent."""
    app = build_reporting_agent_graph()

    initial_state: ReportingAgentState = {
        "agent_1_results": agent_1_results,
        "agent_2_results": agent_2_results,
        "correlated_findings": [],
        "execution_log": [],
        "status": "initialized",
        "audit_id": audit_id,
    }

    return app.invoke(initial_state)