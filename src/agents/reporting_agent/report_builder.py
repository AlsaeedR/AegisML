import json
from typing import Any, Dict, List

from langchain_core.prompts import ChatPromptTemplate

from .llm import get_llm
from .schemas import AuditReport


def build_audit_report(
    findings: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Build the final structured audit report from
    Agent 1 risk findings and Agent 2 test evidence.
    """

    overall_risk = _build_overall_risk(
        findings
    )

    recommendations = _collect_recommendations(
        findings
    )

    executive_summary = _build_executive_summary(
        findings,
        overall_risk,
    )

    report = {
        "executive_summary": executive_summary,
        "overall_risk": overall_risk,
        "findings": findings,
        "recommendations": recommendations,
    }

    validated_report = AuditReport(
        **report
    )

    return validated_report.model_dump()


def _build_overall_risk(
    findings: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Summarize the risk values already produced
    by the Pipeline Agent.
    """

    if not findings:
        return {
            "overall_risk_score": 0.0,
            "overall_severity": "Low",
            "total_findings": 0,
            "critical_findings": 0,
            "high_findings": 0,
            "medium_findings": 0,
            "low_findings": 0,
        }

    highest_finding = max(
        findings,
        key=lambda finding: float(
            finding.get(
                "risk_score",
                0.0,
            )
        ),
    )

    overall_score = float(
        highest_finding.get(
            "risk_score",
            0.0,
        )
    )

    overall_severity = highest_finding.get(
        "severity",
        "Low",
    )

    severities = [
        str(
            finding.get(
                "severity",
                "Low",
            )
        ).capitalize()
        for finding in findings
    ]

    return {
        "overall_risk_score": overall_score,
        "overall_severity": overall_severity,
        "total_findings": len(findings),
        "critical_findings": severities.count(
            "Critical"
        ),
        "high_findings": severities.count(
            "High"
        ),
        "medium_findings": severities.count(
            "Medium"
        ),
        "low_findings": severities.count(
            "Low"
        ),
    }


def _collect_recommendations(
    findings: List[Dict[str, Any]]
) -> List[str]:
    """
    Collect and deduplicate remediation recommendations.
    """

    recommendations: List[str] = []

    for finding in findings:
        for recommendation in finding.get(
            "recommendations",
            [],
        ):
            if recommendation not in recommendations:
                recommendations.append(
                    recommendation
                )

    return recommendations


def _build_executive_summary(
    findings: List[Dict[str, Any]],
    overall_risk: Dict[str, Any],
) -> str:
    """
    Use the LLM to generate an executive summary
    grounded in Agent 1 and Agent 2 results.
    """

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are the Reporting Agent in the "
                "AegisML security auditing system. "
                "Write one concise professional executive "
                "summary for a security dashboard. "
                "Use only the supplied overall risk data "
                "and correlated security findings. "
                "Keep the summary between 60 and 90 words "
                "and write it as a single paragraph. "
                "Start directly with the assessment. "
                "Do not include a title, heading, label, "
                "Markdown formatting, bullet points, "
                "or phrases such as 'Executive Summary'. "
                "Mention the overall risk level and score, "
                "highlight the most important dynamically "
                "confirmed vulnerability, and briefly "
                "distinguish confirmed dynamic vulnerabilities "
                "from findings that were not confirmed by testing. "
                "For adversarial testing, use "
                "attack_success_rate_within_budget when referring "
                "to attack success rate. "
                "When presenting rates or proportions from evidence, "
                "convert decimal values such as 0.94 into percentages "
                "such as 94% for readability. "
                "Do not calculate, modify, or reinterpret "
                "risk scores, severity levels, test statuses, "
                "or evidence. "
                "Do not invent findings or recommendations. "
                "Return only the final summary paragraph."
            ),
            (
                "user",
                "OVERALL RISK:\n{overall_risk}\n\n"
                "CORRELATED SECURITY FINDINGS:\n"
                "{findings}\n\n"
                "Write the executive summary."
            ),
        ]
    )

    chain = prompt | llm

    response = chain.invoke(
        {
            "overall_risk": json.dumps(
                overall_risk,
                indent=2,
            ),
            "findings": json.dumps(
                findings,
                indent=2,
            ),
        }
    )

    return response.content.strip()