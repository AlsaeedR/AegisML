from typing import Any, Dict, List

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
        overall_risk
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
    overall_risk: Dict[str, Any]
) -> str:
    """
    Generate a concise executive summary.
    """

    return (
        "AegisML completed static threat analysis and "
        "dynamic security testing of the target machine "
        "learning pipeline. "
        f"The highest identified risk score is "
        f"{overall_risk['overall_risk_score']}/10, "
        f"with an overall "
        f"{overall_risk['overall_severity']} risk level. "
        f"The assessment includes "
        f"{overall_risk['critical_findings']} Critical, "
        f"{overall_risk['high_findings']} High, "
        f"{overall_risk['medium_findings']} Medium, and "
        f"{overall_risk['low_findings']} Low findings."
    )