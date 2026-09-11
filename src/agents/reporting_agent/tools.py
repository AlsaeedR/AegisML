from typing import Any, Dict, List, Optional, Tuple
from pydantic import ValidationError

from .risk_scoring import calculate_final_risk
from .report_builder import build_audit_report
from .schemas import AuditReport


def score_finding_tool(
    finding: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Tool wrapping the NIST AI 100-2 evidence-informed risk calculation
    for a single correlated vulnerability finding.
    """
    return calculate_final_risk(finding)


def compile_audit_report_tool(
    findings: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Tool wrapping report compilation, LLM executive summary synthesis,
    and evidence-informed remediation recommendation generation.
    """
    return build_audit_report(findings)


def validate_audit_report_schema(
    raw_data: Any
) -> Tuple[bool, Optional[str], Optional[AuditReport]]:
    """
    Tool wrapping Pydantic validation for the final structured AuditReport.
    Returns a success flag, an error diagnostic string if invalid, and the validated model.
    """
    try:
        validated = AuditReport.model_validate(raw_data)
        return True, None, validated
    except ValidationError as e:
        error_lines = []
        for err in e.errors():
            loc = " -> ".join(str(p) for p in err.get("loc", []))
            msg = err.get("msg", "Invalid value")
            error_lines.append(f"Field '{loc}': {msg}")
        return False, "\n".join(error_lines), None

