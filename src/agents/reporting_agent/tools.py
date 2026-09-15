import json
from typing import Any, Dict, List, Optional, Tuple
from pydantic import ValidationError
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate

from src.core.llm import get_llm
from .risk_scoring import calculate_final_risk
from .schemas import AuditReport


# ---------------------------------------------------------------------------
# Report Construction Logic (formerly report_builder.py)
# ---------------------------------------------------------------------------

def _build_overall_risk(findings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build the overall final risk summary using evidence-informed risk scores."""
    if not findings:
        return {
            "overall_risk_score": 0.0,
            "overall_severity": "Low",
            "total_findings": 0,
            "confirmed_findings": 0,
            "false_positive_findings": 0,
            "hidden_risk_findings": 0,
            "unverified_findings": 0,
            "not_applicable_findings": 0,
            "mitigated_findings": 0,
            "critical_findings": 0,
            "high_findings": 0,
            "medium_findings": 0,
            "low_findings": 0,
        }

    correlation_statuses = [
        str(f.get("correlation_status", "Unverified")).lower()
        for f in findings
    ]

    applicable_findings = [
        f for f in findings
        if str(f.get("correlation_status", "Unverified")).lower() != "not applicable"
    ]

    if applicable_findings:
        highest_finding = max(
            applicable_findings,
            key=lambda f: float(f.get("risk_score", f.get("final_risk_score", 0.0))),
        )
        overall_score = float(highest_finding.get("risk_score", highest_finding.get("final_risk_score", 0.0)))
        overall_severity = str(highest_finding.get("severity", highest_finding.get("final_severity", "Low")))
    else:
        overall_score = 0.0
        overall_severity = "Not Applicable"

    final_severities = [
        str(f.get("severity", f.get("final_severity", "Low"))).capitalize()
        for f in applicable_findings
    ]

    confirmed_findings = sum(
        1 for f in findings
        if str(f.get("test_status", "")).lower() == "vulnerable"
    )

    return {
        "overall_risk_score": overall_score,
        "overall_severity": overall_severity,
        "total_findings": len(findings),
        "confirmed_findings": confirmed_findings,
        "false_positive_findings": correlation_statuses.count("false positive"),
        "hidden_risk_findings": correlation_statuses.count("hidden risk"),
        "unverified_findings": correlation_statuses.count("unverified"),
        "not_applicable_findings": correlation_statuses.count("not applicable"),
        "mitigated_findings": (
            correlation_statuses.count("defended / mitigated")
            + correlation_statuses.count("mitigated")
        ),
        "critical_findings": final_severities.count("Critical"),
        "high_findings": final_severities.count("High"),
        "medium_findings": final_severities.count("Medium"),
        "low_findings": final_severities.count("Low"),
    }


def _get_fallback_recommendations(
    findings: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Deterministic NIST AI 100-2 grounded recommendations fallback."""
    updated_findings: List[Dict[str, Any]] = []
    overall_recs: List[str] = []

    fallback_map: Dict[str, Dict[str, List[str]]] = {
        "V1": {
            "default": [
                "Implement SHA-256 cryptographic checksums on training data ingestion pipelines to verify dataset integrity.",
                "Deploy statistical anomaly detection and outlier filtering to identify poisoned sample distributions.",
            ],
            "confirmed": [
                "Deploy robust sanitization filters on training inputs to detect adversarial label flips before fitting.",
                "Enforce strict dataset provenance verification and immutable versioning for training data feeds.",
            ],
        },
        "V2": {
            "false_positive": [
                "Current pipeline survived malformed and adversarial text inputs without failure. Maintain existing input bounds.",
                "Introduce automated fuzzing regression tests into CI/CD to prevent future preprocessing regressions.",
            ],
            "default": [
                "Implement strict input length caps and character allowlists to mitigate buffer and injection attack surfaces.",
                "Add defensive exception handling around tokenization and vectorization steps to prevent pipeline crashes.",
            ],
        },
        "V3": {
            "confirmed": [
                "Enforce automated schema validation and deduplication prior to model training to prevent accuracy degradation.",
                "Integrate missing value assertion checks and drop empty or invalid records before fitting pipeline estimators.",
            ],
            "default": [
                "Deploy Great Expectations or Pydantic schema validation on training datasets to catch schema corruption.",
                "Implement automated null-checking and record deduplication rules across all data loading steps.",
            ],
        },
        "V4": {
            "confirmed": [
                "Incorporate adversarial training with bounded perturbation samples (e.g. via ART) to harden decision boundaries.",
                "Implement prediction confidence threshold gating and input vector clipping to reject evasion attempts.",
            ],
            "default": [
                "Evaluate model robustness against evasion perturbations using bounded norm constraints.",
                "Deploy input feature transformation and sanitization to reduce susceptibility to gradient-based attacks.",
            ],
        },
    }

    for finding in findings:
        f = dict(finding)
        vid = str(f.get("vulnerability_id", ""))
        status = str(f.get("correlation_status", "")).lower()
        recs: List[str] = []

        if vid in fallback_map:
            if "false positive" in status and "false_positive" in fallback_map[vid]:
                recs = fallback_map[vid]["false_positive"]
            elif "confirmed" in status and "confirmed" in fallback_map[vid]:
                recs = fallback_map[vid]["confirmed"]
            else:
                recs = fallback_map[vid].get("default", [])
        else:
            components = f.get("affected_components", [])
            comp_str = ", ".join(components) if components else "pipeline stages"
            recs = [
                f"Implement defense-in-depth security controls around affected components: {comp_str}.",
                "Monitor runtime inference for anomalous prediction confidence distributions.",
            ]

        f["recommendations"] = recs
        updated_findings.append(f)
        for recommendation in recs:
            if recommendation not in overall_recs:
                overall_recs.append(recommendation)

    return updated_findings, overall_recs


def _generate_evidence_informed_recommendations(
    findings: List[Dict[str, Any]],
    overall_risk: Dict[str, Any],
    tool_context: Optional[str] = None,
    validation_errors: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Synthesizes empirical evidence and final risk scores into actionable remediation recommendations."""
    try:
        llm = get_llm(temperature=0.2)
        tool_section = f"\nADDITIONAL ARCHITECTURAL & TELEMETRY TOOL FINDINGS:\n{tool_context}\n" if tool_context else ""
        error_section = f"\nATTENTION PRIOR VALIDATION ERRORS TO RESOLVE:\n{validation_errors}\n" if validation_errors else ""

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are the Remediation and Risk Governance Agent in the AegisML security auditing system.\n"
                "Your objective is to generate actionable, evidence-informed security remediation recommendations "
                "based on empirical dynamic verification results and final risk scores.\n\n"
                "GUIDELINES:\n"
                "1. Finding-Level Recommendations: For each vulnerability finding (keyed by vulnerability_id):\n"
                "   - If 'Confirmed Risk', prescribe concrete code-level defenses tailored to the affected code and attack vector.\n"
                "   - If 'False Positive (Mitigated)', recognize existing pipeline resilience and recommend regression testing.\n"
                "   - If 'Data Validation Weaknesses' is confirmed, prescribe schema assertion (Pydantic / Great Expectations) and null handling.\n"
                "   - If 'Adversarial Robustness' is confirmed, prescribe adversarial training, input clipping, or confidence gating.\n"
                "2. Overall Recommendations: Provide 3-5 prioritized, high-impact executive recommendations ordering "
                "immediate critical fixes first, followed by medium-term pipeline hardening.\n\n"
                "Output strictly valid JSON with this schema:\n"
                "{\n"
                '  "finding_recommendations": {\n'
                '    "V1": ["string", "string"],\n'
                '    "V2": ["string", "string"],\n'
                '    "V3": ["string", "string"],\n'
                '    "V4": ["string", "string"]\n'
                "  },\n"
                '  "overall_recommendations": ["string", "string", "string"]\n'
                "}\n"
                "Do not include markdown code block ticks or explanation outside the JSON."
            ),
            (
                "user",
                "OVERALL FINAL RISK SUMMARY:\n{overall_risk}\n\n"
                "CORRELATED FINDINGS WITH EMPIRICAL EVIDENCE:\n{findings}\n"
                "{tool_section}"
                "{error_section}\n"
                "Generate evidence-informed remediation recommendations as JSON:"
            ),
        ])

        chain = prompt | llm
        response = chain.invoke({
            "overall_risk": json.dumps(overall_risk, indent=2),
            "findings": json.dumps(findings, indent=2),
            "tool_section": tool_section,
            "error_section": error_section,
        })

        content = str(response.content).strip()
        if content.startswith("```"):
            lines = content.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines).strip()

        data = json.loads(content)
        finding_recs = data.get("finding_recommendations", {})
        overall_recs = data.get("overall_recommendations", [])

        updated_findings: List[Dict[str, Any]] = []
        for finding in findings:
            f = dict(finding)
            vid = str(f.get("vulnerability_id", ""))
            if vid in finding_recs and finding_recs[vid]:
                f["recommendations"] = finding_recs[vid]
            else:
                f["recommendations"] = finding.get("recommendations", [])
            updated_findings.append(f)

        if not overall_recs:
            for f in updated_findings:
                for rec in f.get("recommendations", []):
                    if rec not in overall_recs:
                        overall_recs.append(rec)

        return updated_findings, overall_recs

    except Exception:
        return _get_fallback_recommendations(findings)


def _build_executive_summary(
    findings: List[Dict[str, Any]],
    overall_risk: Dict[str, Any],
    validation_errors: Optional[str] = None,
) -> str:
    """Generate a concise executive summary grounded in the final evidence-informed Agent 3 results."""
    try:
        llm = get_llm()
        error_section = f"\nATTENTION PRIOR VALIDATION ERRORS TO RESOLVE:\n{validation_errors}\n" if validation_errors else ""

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are the Risk Scoring and Reporting Agent in the AegisML security auditing system. "
                "Write one concise professional executive summary for a security dashboard. "
                "Use only the supplied overall final risk assessment and correlated security findings. "
                "Keep the summary between 20 and 40 words and write it as a single paragraph. "
                "Start directly with the assessment. Clearly mention the overall final risk level and score. "
                "Highlight the most important dynamically confirmed finding when one exists. "
                "Return only the final summary paragraph."
            ),
            (
                "user",
                "OVERALL FINAL RISK:\n{overall_risk}\n\n"
                "CORRELATED SECURITY FINDINGS:\n{findings}\n"
                "{error_section}\n"
                "Write the executive summary."
            ),
        ])

        chain = prompt | llm
        response = chain.invoke({
            "overall_risk": json.dumps(overall_risk, indent=2),
            "findings": json.dumps(findings, indent=2),
            "error_section": error_section,
        })
        return response.content.strip()
    except Exception:
        score = overall_risk.get("overall_risk_score", 0.0)
        sev = overall_risk.get("overall_severity", "Low")
        return f"AegisML security assessment completed with an overall risk score of {score:.1f}/10 ({sev}). Dynamic testing confirmed vulnerability findings requiring code-level remediation."


def build_audit_report(
    findings: List[Dict[str, Any]],
    tool_context: Optional[str] = None,
    validation_errors: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the final structured AegisML audit report from correlated findings."""
    overall_risk = _build_overall_risk(findings)
    applicable_for_recommendations = [
        f for f in findings
        if str(f.get("correlation_status", "")).lower() != "not applicable"
    ]

    recommended_findings, recommendations = _generate_evidence_informed_recommendations(
        applicable_for_recommendations,
        overall_risk,
        tool_context=tool_context,
        validation_errors=validation_errors,
    )

    recommendations_by_id = {
        str(f.get("vulnerability_id", "")): f.get("recommendations", [])
        for f in recommended_findings
    }

    final_findings = [
        {
            **f,
            "recommendations": (
                []
                if str(f.get("correlation_status", "")).lower() == "not applicable"
                else recommendations_by_id.get(
                    str(f.get("vulnerability_id", "")),
                    f.get("recommendations", []),
                )
            ),
        }
        for f in findings
    ]

    executive_summary = _build_executive_summary(
        final_findings,
        overall_risk,
        validation_errors=validation_errors,
    )

    report = {
        "executive_summary": executive_summary,
        "overall_risk": overall_risk,
        "findings": final_findings,
        "recommendations": recommendations,
    }

    validated_report = AuditReport(**report)
    return validated_report.model_dump()


# ---------------------------------------------------------------------------
# Governance & Risk Scoring ReAct Tools
# ---------------------------------------------------------------------------

def score_finding_tool(
    finding: Dict[str, Any],
    pipeline_graph: Optional[Dict[str, Any]] = None,
    threat_model: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Tool wrapping the NIST AI 100-2 evidence-informed dynamic risk calculation for a finding."""
    return calculate_final_risk(finding, pipeline_graph=pipeline_graph, threat_model=threat_model)


def compile_audit_report_tool(
    findings: List[Dict[str, Any]],
    tool_context: Optional[str] = None,
    validation_errors: Optional[str] = None,
) -> Dict[str, Any]:
    """Tool wrapping report compilation and evidence-informed recommendation generation."""
    return build_audit_report(
        findings=findings,
        tool_context=tool_context,
        validation_errors=validation_errors,
    )


def validate_audit_report_schema(raw_data: Any) -> Tuple[bool, Optional[str], Optional[AuditReport]]:
    """Pydantic validation for the final structured AuditReport."""
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
    """Semantic validation for the audit report."""
    errors: List[str] = []

    summary = str(report.get("executive_summary", "")).strip()
    if len(summary) < 50:
        errors.append("Executive summary is too brief or empty; requires comprehensive synthesis.")

    overall_recs = report.get("recommendations", [])
    if not overall_recs:
        errors.append("Report lacks overarching strategic recommendations.")
    else:
        for idx, rec in enumerate(overall_recs):
            if not isinstance(rec, str) or len(rec.strip()) < 10:
                errors.append(f"Overall recommendation #{idx+1} is trivial or too brief: '{rec}'.")

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
        if corr_stat != "not applicable":
            if not recs:
                errors.append(f"Finding '{vid}' has no actionable remediation recommendations.")

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
    """Produces LangChain @tool functions for Agent 3 investigation."""
    agent_1 = agent_1_results or {}
    agent_2 = agent_2_results or {}
    pipeline_graph = agent_1.get("pipeline_graph", {})
    structured_test_results = agent_2.get("structured_test_results", {})

    @tool
    def bound_query_pipeline_architecture(
        component_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Inspect the target pipeline components and data ingestion boundaries discovered by Agent 1."""
        return query_pipeline_architecture.invoke({
            "component_id": component_id,
            "pipeline_graph": pipeline_graph,
        })

    @tool
    def bound_query_empirical_telemetry(
        vulnerability_id: str,
    ) -> Dict[str, Any]:
        """Query detailed dynamic test measurements, degradation numbers, and execution logs produced by Agent 2."""
        return query_empirical_telemetry.invoke({
            "vulnerability_id": vulnerability_id,
            "structured_test_results": structured_test_results,
        })

    return [
        bound_query_pipeline_architecture,
        bound_query_empirical_telemetry,
        verify_nist_compliance,
    ]

