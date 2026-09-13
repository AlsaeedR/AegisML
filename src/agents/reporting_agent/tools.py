from typing import Any, Dict, List, Optional, Tuple
from pydantic import ValidationError
from langchain_core.tools import tool

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
    findings: List[Dict[str, Any]],
    tool_context: Optional[str] = None,
    validation_errors: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Tool wrapping report compilation, LLM executive summary synthesis,
    and evidence-informed remediation recommendation generation.
    """
    return build_audit_report(
        findings=findings,
        tool_context=tool_context,
        validation_errors=validation_errors,
    )


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


def validate_audit_report_semantics(
    report: Dict[str, Any],
    correlated_findings: List[Dict[str, Any]],
) -> Tuple[bool, Optional[str]]:
    """
    Semantic validation for the audit report.
    Verifies that:
    - All correlated findings are accounted for in the report findings.
    - Active findings (status != 'not_applicable') have actionable, non-empty recommendations.
    - The executive summary is substantive (> 50 characters).
    - Strategic overarching recommendations are present and non-empty.
    """
    errors: List[str] = []

    # 1. Executive summary quality check
    summary = str(report.get("executive_summary", "")).strip()
    if len(summary) < 50:
        errors.append("Executive summary is too brief or empty; requires comprehensive synthesis.")

    # 2. Overall strategic recommendations check
    overall_recs = report.get("recommendations", [])
    if not overall_recs or len(overall_recs) == 0:
        errors.append("Report lacks overarching strategic recommendations.")
    else:
        for idx, rec in enumerate(overall_recs):
            if not isinstance(rec, str) or len(rec.strip()) < 10:
                errors.append(f"Overall recommendation #{idx+1} is trivial or too brief: '{rec}'.")

    # 3. Finding-level recommendations check
    findings = report.get("findings", [])
    present_vids = {f.get("vulnerability_id") for f in findings if f.get("vulnerability_id")}
    required_vids = {f.get("vulnerability_id") for f in correlated_findings if f.get("vulnerability_id")}

    missing_vids = required_vids - present_vids
    if missing_vids:
        errors.append(f"Report findings omit required vulnerability IDs: {missing_vids}.")

    for f in findings:
        vid = f.get("vulnerability_id", "Unknown")
        corr_stat = str(f.get("correlation_status", "")).lower()
        recs = f.get("recommendations", [])

        # Non-applicable findings can have empty recommendations; active findings must have guidance
        if corr_stat != "not applicable":
            if not recs or len(recs) == 0:
                errors.append(f"Finding '{vid}' has no actionable remediation recommendations.")
            elif any(not isinstance(r, str) or len(r.strip()) < 8 for r in recs):
                errors.append(f"Finding '{vid}' contains incomplete or generic recommendation strings.")

    if errors:
        return False, "\n".join(errors)

    return True, None


@tool
def query_pipeline_architecture(
    component_id: Optional[str] = None,
    pipeline_graph: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Inspect the extracted pipeline architecture, components, and trust boundaries discovered by Agent 1.
    Use this to ground remediation guidance in actual pipeline steps.

    Args:
        component_id: Optional filter for a specific component ID (e.g. 'data_ingestion_1').
        pipeline_graph: Graph structure from Agent 1.
    """
    graph = pipeline_graph or {}
    nodes = graph.get("nodes", [])

    if component_id:
        target = component_id.strip().lower()
        matched = [
            n for n in nodes
            if target in n.get("id", "").lower() or target in n.get("description", "").lower()
        ]
        return {
            "component_id": component_id,
            "matched_nodes": matched,
            "found": len(matched) > 0,
        }

    return {
        "total_nodes": len(nodes),
        "nodes": nodes,
        "entry_points": [n for n in nodes if n.get("type") == "data_ingestion"],
    }


@tool
def query_empirical_telemetry(
    vulnerability_id: str,
    structured_test_results: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Query detailed dynamic test measurements, degradation numbers, and execution logs
    produced by Agent 2 for a specific vulnerability (V1, V2, V3, V4).

    Args:
        vulnerability_id: Vulnerability ID to inspect ('V1', 'V2', 'V3', or 'V4').
        structured_test_results: Empirical results from Agent 2.
    """
    results_obj = structured_test_results or {}
    test_results = results_obj.get("results", [])

    v_target = vulnerability_id.strip().upper()
    matched = next(
        (r for r in test_results if r.get("vulnerability_id", "").strip().upper() == v_target),
        None
    )

    if not matched:
        return {
            "error": f"No empirical telemetry found for {vulnerability_id}.",
            "available_vulnerability_ids": [r.get("vulnerability_id") for r in test_results],
        }

    return {
        "vulnerability_id": v_target,
        "vulnerability_name": matched.get("vulnerability_name"),
        "status": matched.get("status"),
        "severity": matched.get("severity"),
        "evidence": matched.get("evidence", {}),
    }


@tool
def verify_nist_compliance(
    vulnerability_id: str,
) -> Dict[str, Any]:
    """
    Retrieve authoritative NIST AI 100-2e2025 taxonomy mapping, lifecycle stages,
    and standard defensive controls for an MVP vulnerability category.

    Args:
        vulnerability_id: One of 'V1', 'V2', 'V3', 'V4'.
    """
    taxonomy = {
        "V1": {
            "category": "Data Poisoning",
            "lifecycle_stage": "Data Ingestion & Model Training",
            "nist_threat_type": "Integrity / Availability Poisoning",
            "recommended_controls": [
                "Cryptographic dataset checksums and signatures (SHA-256)",
                "Statistical outlier filtering and poison sample detection",
                "Strict origin verification for ingestion feeds",
            ],
        },
        "V2": {
            "category": "Preprocessing Attack Surface",
            "lifecycle_stage": "Data Preprocessing",
            "nist_threat_type": "Evasion & Denial-of-Service / ReDoS",
            "recommended_controls": [
                "Input length clamping and maximum token bounds",
                "Regex execution timeout guards",
                "Character allowlisting and defensive exception handling",
            ],
        },
        "V3": {
            "category": "Data Validation Weaknesses",
            "lifecycle_stage": "Data Ingestion & Validation",
            "nist_threat_type": "Schema Poisoning & Integrity Corruption",
            "recommended_controls": [
                "Strict runtime schema assertions (Pydantic / Great Expectations)",
                "Automated null checking and type validation gates",
                "Record deduplication before model consumption",
            ],
        },
        "V4": {
            "category": "Adversarial Robustness",
            "lifecycle_stage": "Model Training & Inference",
            "nist_threat_type": "Adversarial Evasion (HopSkipJump / Boundary Attacks)",
            "recommended_controls": [
                "Adversarial retraining on perturbed samples",
                "Input feature quantization and noise smoothing",
                "Confidence thresholding and defensive ensemble voting",
            ],
        },
    }

    vid = vulnerability_id.strip().upper()
    return taxonomy.get(vid, {
        "error": f"Vulnerability ID '{vulnerability_id}' not found in NIST taxonomy.",
        "valid_ids": ["V1", "V2", "V3", "V4"],
    })


REPORTING_AGENT_TOOLS = [
    query_pipeline_architecture,
    query_empirical_telemetry,
    verify_nist_compliance,
]


def make_reporting_tools(
    agent_1_results: Optional[Dict[str, Any]],
    agent_2_results: Optional[Dict[str, Any]],
    correlated_findings: List[Dict[str, Any]],
) -> List[Any]:
    """
    Factory creating reporting exploration tools pre-bound via closure to
    the current execution state artifacts.
    """
    agent_1 = agent_1_results or {}
    agent_2 = agent_2_results or {}

    pipeline_graph = agent_1.get("pipeline_graph", {})
    structured_test_results = agent_2.get("structured_test_results", {})

    @tool
    def bound_query_pipeline_architecture(
        component_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Inspect the target pipeline components and data ingestion boundaries discovered by Agent 1.
        """
        return query_pipeline_architecture.invoke({
            "component_id": component_id,
            "pipeline_graph": pipeline_graph,
        })

    @tool
    def bound_query_empirical_telemetry(
        vulnerability_id: str,
    ) -> Dict[str, Any]:
        """
        Query detailed dynamic test measurements, degradation numbers, and execution logs
        produced by Agent 2 for a specific vulnerability (V1, V2, V3, V4).
        """
        return query_empirical_telemetry.invoke({
            "vulnerability_id": vulnerability_id,
            "structured_test_results": structured_test_results,
        })

    return [
        bound_query_pipeline_architecture,
        bound_query_empirical_telemetry,
        verify_nist_compliance,
    ]
