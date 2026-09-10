from typing import Any, Dict, List

from langgraph.graph import StateGraph, END

from .state import ReportingAgentState
from .report_builder import build_audit_report
from .risk_scoring import calculate_final_risk


def node_correlate_findings(
    state: ReportingAgentState
) -> Dict[str, Any]:
    """
    Combine Agent 1 static findings with Agent 2
    dynamic security test results using
    vulnerability IDs, then calculate the
    final evidence-informed risk.
    """

    agent_1 = state.get(
        "agent_1_results"
    ) or {}

    agent_2 = state.get(
        "agent_2_results"
    ) or {}

    static_findings = (
        agent_1
        .get(
            "vulnerability_findings",
            {},
        )
        .get(
            "vulnerabilities",
            [],
        )
    )

    dynamic_results = (
        agent_2
        .get(
            "structured_test_results",
            {},
        )
        .get(
            "results",
            [],
        )
    )

    dynamic_by_id = {
        result.get(
            "vulnerability_id"
        ): result
        for result in dynamic_results
        if result.get(
            "vulnerability_id"
        )
    }

    correlated_findings: List[
        Dict[str, Any]
    ] = []

    for static_finding in static_findings:
        vulnerability_id = (
            static_finding.get(
                "vulnerability_id"
            )
        )

        dynamic_result = (
            dynamic_by_id.get(
                vulnerability_id,
                {
                    "vulnerability_id": (
                        vulnerability_id
                    ),
                    "status": "not_tested",
                    "severity": None,
                    "evidence": {
                        "reason": (
                            "No matching Agent 2 "
                            "result was available."
                        )
                    },
                },
            )
        )

        correlated_finding = {
            "vulnerability_id": (
                vulnerability_id
            ),
            "category": (
                static_finding.get(
                    "category"
                )
            ),

            "affected_components": (
                static_finding.get(
                    "affected_components",
                    [],
                )
            ),

            "description": (
                static_finding.get(
                    "description",
                    "",
                )
            ),
            "recommendations": (
                static_finding.get(
                    "recommendations",
                    [],
                )
            ),

            # Agent 2 dynamic evidence.
            "test_status": (
                dynamic_result.get(
                    "status",
                    "not_tested",
                )
            ),
            "dynamic_severity": (
                dynamic_result.get(
                    "severity"
                )
            ),
            "evidence": (
                dynamic_result.get(
                    "evidence",
                    {},
                )
            ),
        }

        final_risk = (
            calculate_final_risk(
                correlated_finding
            )
        )

        correlated_finding.update(
            final_risk
        )

        correlated_findings.append(
            correlated_finding
        )

    log = list(
        state.get(
            "execution_log",
            [],
        )
    )

    log.append(
        (
            f"Correlated "
            f"{len(correlated_findings)} "
            "Agent 1 findings with Agent 2 "
            "results and calculated final "
            "evidence-informed risk scores."
        )
    )

    return {
        "correlated_findings": (
            correlated_findings
        ),
        "execution_log": log,
        "status": (
            "risk_scoring_completed"
        ),
    }


def node_build_report(
    state: ReportingAgentState
) -> Dict[str, Any]:
    """
    Generate the final structured
    security audit report.
    """

    report = build_audit_report(
        state.get(
            "correlated_findings",
            [],
        )
    )

    log = list(
        state.get(
            "execution_log",
            [],
        )
    )

    log.append(
        (
            "Generated final AegisML "
            "security audit report."
        )
    )

    return {
        "overall_risk": report.get(
            "overall_risk"
        ),
        "final_report": report,
        "execution_log": log,
        "status": "completed",
    }


def build_reporting_agent_graph():
    """
    Assemble and compile the
    Reporting Agent workflow.
    """

    graph = StateGraph(
        ReportingAgentState
    )

    graph.add_node(
        "correlate_findings",
        node_correlate_findings,
    )

    graph.add_node(
        "build_report",
        node_build_report,
    )

    graph.set_entry_point(
        "correlate_findings"
    )

    graph.add_edge(
        "correlate_findings",
        "build_report",
    )

    graph.add_edge(
        "build_report",
        END,
    )

    return graph.compile()