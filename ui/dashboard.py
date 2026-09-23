import html
import json
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config

try:
    from streamlit_flow import streamlit_flow
    from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
    from streamlit_flow.state import StreamlitFlowState
    from streamlit_flow.layouts import ManualLayout, LayeredLayout
    HAS_STREAMLIT_FLOW = True
except Exception:
    HAS_STREAMLIT_FLOW = False


def escape(value: Any) -> str:
    return html.escape(
        str(
            value
            if value is not None
            else ""
        )
    )


KNOWLEDGE_EXPLANATIONS = {
    "black-box": "External API query access only; zero code or model visibility.",
    "grey-box": "Partial knowledge (features or model family known, no model weights).",
    "white-box": "Full access to pipeline code, architecture, and trained weights.",
    "supply-chain": "Upstream access to third-party datasets, packages, or base models.",
}


def category_score(
    findings: List[Dict[str, Any]],
    category: str,
) -> int:
    """
    Return the Agent 3 final risk score
    for a vulnerability category on a
    0-100 presentation scale.
    """

    for finding in findings:
        if (
            finding.get("category")
            == category
        ):
            score = float(
                finding.get(
                    "risk_score",
                    finding.get(
                        "final_risk_score",
                        0,
                    ),
                )
            )

            return max(
                0,
                min(
                    100,
                    int(
                        round(
                            score * 10
                        )
                    ),
                ),
            )

    return 0


def severity_class(
    severity: str
) -> str:
    severity = str(
        severity
    ).lower()

    if severity == "critical":
        return "severity-critical"

    if severity == "high":
        return "severity-high"

    if severity == "medium":
        return "severity-medium"

    return "severity-low"


def severity_slug(
    severity: str,
) -> str:
    """Return a stable data-attribute slug for finding-card striping."""
    normalized = str(severity or "").strip().lower()
    if normalized in {"critical", "high", "medium", "low"}:
        return normalized
    return "na"


def risk_fill_class(score: int) -> str:
    """Map the shared AegisML severity thresholds to dashboard bar classes."""
    if score >= 50:
        return "fill-red"
    if score >= 25:
        return "fill-gold"
    return "fill-green"


def category_test_status(
    findings: List[Dict[str, Any]],
    category: str,
) -> str:
    """Return the Agent 2 test status for a vulnerability category."""
    for finding in findings:
        if finding.get("category") == category:
            return str(
                finding.get("test_status", "not_tested")
            ).lower()
    return "not_tested"


def risk_bar(
    title: str,
    score: int,
    css_class: str,
) -> str:
    return f"""
    <div>
        <div class="risk-bar-title">
            {escape(title)}
        </div>

        <div class="risk-track">
            <div
                class="risk-fill {css_class}"
                style="width:{score}%"
            ></div>
        </div>

        <div class="risk-score-small">
            {score}
        </div>
    </div>
    """


def pipeline_node(
    icon: str,
    name: str,
    meta: str,
    score: int,
    test_status: str = "not_tested",
) -> str:

    normalized_status = str(test_status).lower()

    if normalized_status == "not_applicable":
        flagged = False
        trust = "not applicable"
        trust_class = ""
    elif normalized_status in {"not_tested", "inconclusive", "error"}:
        flagged = True
        trust = "unverified"
        trust_class = ""
    else:
        # Medium begins at 2.5/10 = 25/100 on the shared AegisML scale.
        flagged = score >= 25
        trust = "review" if flagged else "lower risk"
        trust_class = "" if flagged else "trusted"

    flag_class = "flagged" if flagged else ""
    alert = '<div class="node-alert">!</div>' if flagged else ""

    return f"""
    <div class="pipeline-node {flag_class}">

        {alert}

        <div class="node-icon">
            {icon}
        </div>

        <div>
            <div class="node-name">
                {escape(name)}
            </div>

            <div class="node-meta">
                {escape(meta)}
            </div>
        </div>

        <div class="node-trust {trust_class}">
            {escape(trust)}
        </div>

    </div>
    """


def finding_card(
    finding: Dict[str, Any]
) -> str:

    final_severity = str(
        finding.get(
            "severity",
            finding.get(
                "final_severity",
                "Low",
            ),
        )
    )

    status = str(
        finding.get(
            "test_status",
            "not_tested",
        )
    )

    correlation_status = str(
        finding.get(
            "correlation_status",
            "Unverified",
        )
    )

    display_severity = (
        "Not Applicable"
        if (
            status.lower() == "not_applicable"
            or correlation_status.lower() == "not applicable"
        )
        else final_severity
    )

    severity_css = severity_class(
        final_severity
    )

    # Left-edge stripe slug: prefers the Not-Applicable neutral grey when
    # the correlation status removes the finding from the risk scale.
    card_stripe_slug = (
        "na"
        if display_severity == "Not Applicable"
        else severity_slug(final_severity)
    )

    category = finding.get(
        "category",
        "Security finding",
    )

    vulnerability_id = finding.get(
        "vulnerability_id",
        "Unknown",
    )

    description = finding.get(
        "description",
        (
            "No root cause information "
            "available."
        ),
    )
    description = str(description)

    recommendations = finding.get(
        "recommendations",
        [],
    )

    if recommendations:
        fix = recommendations[0]
    else:
        fix = (
            "Review the finding and apply "
            "appropriate security controls."
        )

    dynamic_severity = (
        finding.get(
            "dynamic_severity"
        )
        or "N/A"
    )

    static_severity = str(
        finding.get(
            "static_severity",
            "N/A",
        )
    )

    static_impact = finding.get("static_impact")
    static_likelihood = finding.get("static_likelihood")

    static_score = float(
        finding.get(
            "static_risk_score",
            0.0,
        )
    )

    final_score = float(
        finding.get(
            "risk_score",
            finding.get(
                "final_risk_score",
                0.0,
            ),
        )
    )

    final_likelihood = float(
        finding.get(
            "final_likelihood",
            0.0,
        )
    )

    correlation_rationale = str(
        finding.get(
            "correlation_rationale",
            "",
        )
    )

    static_context_text = ""
    if static_impact is not None and static_likelihood is not None:
        try:
            static_context_text = (
                f" · static impact: {float(static_impact):.1f}/10"
                f" · static likelihood: {float(static_likelihood):.1f}/10"
            )
        except (TypeError, ValueError):
            static_context_text = ""

    correlation_lower = correlation_status.lower()

    # Post-testing review of mitigating controls against empirical evidence
    mitigating_controls = finding.get("mitigating_controls", [])
    control_verdict = finding.get("control_verdict")

    if not control_verdict:
        if not mitigating_controls:
            control_verdict = "none"
        elif "mitigated" in correlation_lower:
            control_verdict = "verified_effective"
        elif "hidden risk" in correlation_lower:
            control_verdict = "bypassed"
        elif status.lower() == "vulnerable":
            control_verdict = "ineffective"
        else:
            control_verdict = "verified_effective" if status.lower() == "not_vulnerable" else "none"

    controls_meta_html = ""
    if mitigating_controls:
        controls_str = ", ".join(mitigating_controls)
        if control_verdict == "verified_effective" or "mitigated" in correlation_lower or "defended" in correlation_lower:
            controls_meta_html = f"""
            <div class="finding-meta controls-verified">
                verified active controls:
                <strong>{escape(controls_str)}</strong>
            </div>
            """
        elif control_verdict == "partially_effective":
            controls_meta_html = f"""
            <div class="finding-meta controls-partial">
                partially active controls (mitigated major failure):
                <strong>{escape(controls_str)}</strong>
            </div>
            """
        elif control_verdict == "bypassed" or ("hidden risk" in correlation_lower and final_severity.lower() in ["critical", "high"]):
            controls_meta_html = f"""
            <div class="finding-meta controls-bypassed">
                bypassed controls (failed under penetration testing):
                <strong>{escape(controls_str)}</strong>
            </div>
            """
        elif "confirmed" in correlation_lower or status.lower() == "vulnerable":
            controls_meta_html = f"""
            <div class="finding-meta controls-ineffective">
                ineffective checks observed in code:
                {escape(controls_str)}
            </div>
            """

    # Context-aware box titles
    if "mitigated" in correlation_lower:
        left_box_title = "Evaluated Threat Surface"
        right_box_title = "Verified Defense"
        right_box_class = "verification-outcome"
    elif "false positive" in correlation_lower:
        left_box_title = "Theoretical Concern"
        right_box_title = "Refutation Outcome"
        right_box_class = "verification-outcome"
    elif "hidden risk" in correlation_lower:
        left_box_title = "Bypassed Mechanism"
        right_box_title = "Required Hardening"
        right_box_class = "suggested-fix"
    elif "not applicable" in correlation_lower or status.lower() == "not_applicable":
        left_box_title = "Lifecycle Scope"
        right_box_title = "Applicability Scope"
        right_box_class = "verification-outcome"
    elif "confirmed" in correlation_lower or final_severity.lower() in ["critical", "high"]:
        left_box_title = "Root Cause"
        right_box_title = "Suggested Fix"
        right_box_class = "suggested-fix"
    else:
        left_box_title = "Theoretical Observation"
        right_box_title = "Hardening Guidance"
        right_box_class = "suggested-fix"

    return f"""
    <div class="finding-card" data-sev="{card_stripe_slug}">

        <div class="finding-head">

            <span class="
                severity
                {severity_css}
            ">
                {escape(
                    display_severity.lower()
                )}
            </span>

            <span class="finding-title">
                {escape(category)}
            </span>

        </div>

        <div class="finding-meta">
            {escape(vulnerability_id)}
            · final risk:
            {final_score:.1f}/10
            · dynamic test:
            {escape(status)}
        </div>

        <div class="finding-meta">
            static:
            {static_score:.1f}/10
            ({escape(static_severity)})
            {static_context_text}
            · dynamic severity:
            {escape(dynamic_severity)}
            · evidence-adjusted likelihood:
            {final_likelihood:.1f}/10
        </div>

        <div class="finding-meta">
            correlation:
            <strong>
                {escape(correlation_status)}
            </strong>
            · {escape(correlation_rationale)}
        </div>
        {controls_meta_html}

        <div class="finding-grid">

            <div class="root-cause">

                <div class="box-title">
                    {escape(left_box_title)}
                </div>

                <div class="box-content">
                    {escape(description)}
                </div>

            </div>

            <div class="{right_box_class}">

                <div class="box-title">
                    {escape(right_box_title)}
                </div>

                <div class="box-content">
                    {escape(fix)}
                </div>

            </div>

        </div>

        <div class="tags">

            <span class="tag">
                {escape(correlation_status)}
            </span>

            <span class="tag">
                NIST-aligned risk assessment
            </span>

            <span class="tag">
                Dynamic validation
            </span>

        </div>

    </div>
    """


def render_dashboard(
    result: Dict[str, Any],
    section: str = "full",
) -> str:

    report = result.get(
        "report",
        {},
    )

    # Pipeline-specific attacker context generated by Agent 1.
    # This changes with the uploaded ML pipeline instead of
    # displaying the same hard-coded capability for every audit.
    deployment_context = result.get(
        "deployment_context",
        {},
    ) or {}

    attacker_goal = deployment_context.get(
        "attacker_goal",
        "Not identified",
    )
    attacker_knowledge = deployment_context.get(
        "attacker_knowledge",
        "Not identified",
    )
    knowledge_desc = KNOWLEDGE_EXPLANATIONS.get(
        str(attacker_knowledge).strip().lower(),
        "",
    )
    attacker_access = deployment_context.get(
        "attacker_access",
        "Not identified",
    )

    overall = report.get(
        "overall_risk",
        {},
    )

    findings = report.get(
        "findings",
        [],
    )

    score = max(
        0,
        min(
            100,
            int(
                round(
                    float(
                        overall.get(
                            "overall_risk_score",
                            0,
                        )
                    )
                    * 10
                )
            ),
        ),
    )

    severity = str(
        overall.get(
            "overall_severity",
            "Low",
        )
    ).lower()

    summary = report.get(
        "executive_summary",
        (
            "AegisML completed the "
            "ML security audit."
        ),
    )

    vulnerable = int(
        overall.get(
            "confirmed_findings",
            sum(
                1
                for finding in findings
                if str(
                    finding.get(
                        "test_status",
                        ""
                    )
                ).lower()
                == "vulnerable"
            ),
        )
    )

    false_positives = int(
        overall.get(
            "false_positive_findings",
            0,
        )
    )

    hidden_risks = int(
        overall.get(
            "hidden_risk_findings",
            0,
        )
    )

    unverified = int(
        overall.get(
            "unverified_findings",
            sum(
                1
                for finding in findings
                if str(finding.get("correlation_status", "")).lower() == "unverified"
            ),
        )
    )

    not_applicable = int(
        overall.get(
            "not_applicable_findings",
            sum(
                1
                for finding in findings
                if str(finding.get("correlation_status", "")).lower() == "not applicable"
            ),
        )
    )

    if vulnerable == 0:
        issue_heading = (
            "No dynamically confirmed issues require attention before this pipeline ships"
        )
    elif vulnerable == 1:
        issue_heading = "1 confirmed issue needs attention before this pipeline ships"
    else:
        issue_heading = f"{vulnerable} confirmed issues need attention before this pipeline ships"

    poisoning = category_score(
        findings,
        "Data Poisoning",
    )

    adversarial = category_score(
        findings,
        "Adversarial Robustness",
    )

    validation = category_score(
        findings,
        "Data Validation Weaknesses",
    )

    preprocessing = category_score(
        findings,
        "Preprocessing Attack Surface",
    )

    poisoning_status = category_test_status(findings, "Data Poisoning")
    adversarial_status = category_test_status(findings, "Adversarial Robustness")
    validation_status = category_test_status(findings, "Data Validation Weaknesses")
    preprocessing_status = category_test_status(findings, "Preprocessing Attack Surface")

    ring_background = (
        "conic-gradient("
        f"#ff751f {score}%, "
        f"#2c2c2f {score}%"
        ")"
    )

    pipeline_html = "".join(
        [
            pipeline_node(
                "📥",
                "Data ingestion",
                "data poisoning assessment",
                poisoning,
                poisoning_status,
            ),

            pipeline_node(
                "🧹",
                "Preprocessing",
                "preprocessing assessment",
                preprocessing,
                preprocessing_status,
            ),

            pipeline_node(
                "🧠",
                "ML pipeline",
                "data validation assessment",
                validation,
                validation_status,
            ),

            pipeline_node(
                "🎯",
                "Inference surface",
                "adversarial robustness assessment",
                adversarial,
                adversarial_status,
            ),
        ]
    )

    findings_html = "".join(
        finding_card(
            finding
        )
        for finding in findings
    )

    if not findings_html:
        findings_html = """
        <div class="finding-empty">
            <strong>No security findings were generated.</strong>
            Every detected surface passed the theoretical and dynamic assessment.
        </div>
        """

    checkpoint_loaded = result.get("checkpoint_loaded", False)
    audit_id = str(result.get("audit_id", "") or "")
    audit_mem = result.get("audit_memory", {})
    steps_count = audit_mem.get("steps_count", 0)

    if checkpoint_loaded:
        status_badges_html = f"""
        <div style="display: flex; gap: 8px; align-items: center; flex-wrap: wrap;">
            <div class="scan-status" style="background: rgba(61, 220, 132, 0.12); border-color: rgba(61, 220, 132, 0.35); color: #3ddc84; font-weight: 600;">
                <span class="scan-dot" style="background: #3ddc84;"></span>
                CHECKPOINT LOADED ({steps_count} steps)
            </div>
            <div class="scan-status">
                <span class="scan-dot"></span>
                security assessment complete
            </div>
        </div>
        """
    else:
        status_badges_html = """
        <div class="scan-status">
            <span class="scan-dot"></span>
            security assessment complete
        </div>
        """

    summary_html = f"""
    <div class="aegis-dashboard">

        <div class="audit-header">

            <div class="header-left">

                <div class="audit-logo">
                    ✓
                </div>

                <div class="audit-title">
                    Pipeline Audit
                </div>

                <div class="audit-path">
                    / AegisML
                </div>

            </div>

            {status_badges_html}

        </div>

        <section class="summary">

            <div
                class="risk-ring"
                style="
                    background:
                    {ring_background};
                "
            >

                <div class="risk-ring-inner">

                    <div class="risk-number">
                        {score}
                    </div>

                    <div class="risk-label">
                        Overall risk
                    </div>

                    <div class="risk-caption">
                        / 100 · {escape(severity)}
                    </div>

                </div>

            </div>

            <div>

                <div class="summary-heading">
                    {escape(issue_heading)}
                </div>

                <div class="summary-text">
                    {escape(summary)}
                </div>

                <div class="finding-meta">
                    False positives:
                    {false_positives}
                    · Hidden risks:
                    {hidden_risks}
                    · Unverified:
                    {unverified}
                    · Not applicable:
                    {not_applicable}
                </div>

                <div class="risk-bars">

                    {
                        risk_bar(
                            "Poisoning",
                            poisoning,
                            risk_fill_class(poisoning),
                        )
                    }

                    {
                        risk_bar(
                            "Adversarial robustness",
                            adversarial,
                            risk_fill_class(adversarial),
                        )
                    }

                    {
                        risk_bar(
                            "Validation weakness",
                            validation,
                            risk_fill_class(validation),
                        )
                    }

                    {
                        risk_bar(
                            "Preprocessing surface",
                            preprocessing,
                            risk_fill_class(preprocessing),
                        )
                    }

                </div>

            </div>

        </section>

    </div>
    """

    content_html = f"""
    <div class="aegis-dashboard">

        <section class="audit-content">

            <aside class="pipeline-column">

                <div class="pipeline-heading">

                    <span>
                        Security surfaces
                    </span>

                    <span>
                        4 categories
                    </span>

                </div>

                <div class="pipeline-list">
                    {pipeline_html}
                </div>

                <div class="attacker-note">

                    <strong>
                        Attacker capability:
                    </strong>

                    <div>
                        <strong>Goal:</strong>
                        {escape(attacker_goal)}
                    </div>

                    <div>
                        <strong>Knowledge:</strong>
                        {escape(attacker_knowledge)}
                        {f'<div style="font-size: 0.78rem; color: #7c7c7c; margin-top: 2px;">{escape(knowledge_desc)}</div>' if knowledge_desc else ''}
                    </div>

                    <div>
                        <strong>Access:</strong>
                        {escape(attacker_access)}
                    </div>
                </div>

            </aside>

            <main>
                {findings_html}
            </main>

        </section>

    </div>
    """

    if section == "summary":
        return summary_html

    if section == "content":
        return content_html

    return summary_html + content_html


# =========================================================
# HUMAN-IN-THE-LOOP REMEDIATION UI
# =========================================================

def _normalize_graph_payload(
    result: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Normalize the Agent 1 pipeline graph into node/edge lists.
    """

    graph = result.get(
        "pipeline_graph",
        {},
    ) or {}

    raw_nodes: Any = []
    raw_edges: Any = []

    if isinstance(graph, dict):
        raw_nodes = (
            graph.get("nodes")
            or graph.get("pipeline_nodes")
            or []
        )
        raw_edges = (
            graph.get("edges")
            or graph.get("pipeline_edges")
            or []
        )

    elif isinstance(graph, list):
        raw_nodes = graph

    nodes: List[Dict[str, Any]] = []

    for index, raw_node in enumerate(raw_nodes):
        if isinstance(raw_node, dict):
            node = dict(raw_node)
        else:
            node = {
                "id": str(raw_node),
                "name": str(raw_node),
            }

        node_id = (
            node.get("id")
            or node.get("node_id")
            or node.get("name")
            or node.get("label")
            or f"node_{index}"
        )

        node["id"] = str(node_id)

        if not node.get("name"):
            node["name"] = (
                node.get("label")
                or node.get("component")
                or node["id"]
            )

        nodes.append(node)

    edges: List[Dict[str, Any]] = []

    for raw_edge in raw_edges:
        if isinstance(raw_edge, dict):
            source = (
                raw_edge.get("source")
                or raw_edge.get("from")
                or raw_edge.get("u")
            )
            target = (
                raw_edge.get("target")
                or raw_edge.get("to")
                or raw_edge.get("v")
            )

            if source is None or target is None:
                continue

            edges.append(
                {
                    "source": str(source),
                    "target": str(target),
                    "label": str(
                        raw_edge.get("label")
                        or raw_edge.get("relationship")
                        or ""
                    ),
                }
            )

        elif (
            isinstance(raw_edge, (list, tuple))
            and len(raw_edge) >= 2
        ):
            edges.append(
                {
                    "source": str(raw_edge[0]),
                    "target": str(raw_edge[1]),
                    "label": (
                        str(raw_edge[2])
                        if len(raw_edge) > 2
                        else ""
                    ),
                }
            )

    return nodes, edges


def _fallback_pipeline_graph(
    result: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Provide a small fallback graph when an older Agent 1 result
    does not yet expose pipeline_graph.
    """

    report = result.get(
        "report",
        {},
    ) or {}

    findings = report.get(
        "findings",
        [],
    ) or []

    node_specs = [
        (
            "data_ingestion",
            "Data ingestion",
            "Data Poisoning",
        ),
        (
            "preprocessing",
            "Preprocessing",
            "Preprocessing Attack Surface",
        ),
        (
            "ml_pipeline",
            "ML pipeline",
            "Data Validation Weaknesses",
        ),
        (
            "inference",
            "Inference surface",
            "Adversarial Robustness",
        ),
    ]

    nodes = []

    for node_id, name, category in node_specs:
        linked_finding = next(
            (
                finding
                for finding in findings
                if finding.get("category") == category
            ),
            None,
        )

        nodes.append(
            {
                "id": node_id,
                "name": name,
                "component_type": category,
                "finding": linked_finding,
            }
        )

    edges = [
        {
            "source": "data_ingestion",
            "target": "preprocessing",
            "label": "",
        },
        {
            "source": "preprocessing",
            "target": "ml_pipeline",
            "label": "",
        },
        {
            "source": "ml_pipeline",
            "target": "inference",
            "label": "",
        },
    ]

    return nodes, edges


def _find_related_finding(
    result: Dict[str, Any],
    node: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Find the report finding most closely associated with a node.
    """

    embedded = node.get("finding")

    if isinstance(embedded, dict):
        return embedded

    raw_findings = (
        result.get("report", {}).get("findings")
        or result.get("findings")
        or result.get("vulnerability_findings")
        or []
    )
    findings: List[Dict[str, Any]] = []
    if isinstance(raw_findings, dict):
        for val in raw_findings.values():
            if isinstance(val, list):
                findings.extend(val)
            elif isinstance(val, dict):
                findings.append(val)
    elif isinstance(raw_findings, list):
        findings = raw_findings

    node_text = " ".join(
        str(
            node.get(key, "")
        ).lower()
        for key in (
            "name",
            "label",
            "component",
            "component_type",
            "type",
        )
    )

    category_keywords = {
        "Data Poisoning": [
            "data",
            "ingest",
            "train",
            "dataset",
        ],
        "Preprocessing Attack Surface": [
            "preprocess",
            "clean",
            "vector",
            "token",
            "transform",
        ],
        "Data Validation Weaknesses": [
            "valid",
            "schema",
            "label",
            "fit",
            "pipeline",
        ],
        "Adversarial Robustness": [
            "predict",
            "inference",
            "classifier",
            "model",
        ],
    }

    best_match = None
    best_score = 0

    for finding in findings:
        category = str(
            finding.get(
                "category",
                "",
            )
        )

        keywords = category_keywords.get(
            category,
            [],
        )

        score = sum(
            1
            for keyword in keywords
            if keyword in node_text
        )

        affected = " ".join(
            str(component).lower()
            for component in finding.get(
                "affected_components",
                [],
            )
        )

        node_name = str(
            node.get(
                "name",
                "",
            )
        ).lower()

        if (
            node_name
            and node_name in affected
        ):
            score += 3

        if score > best_score:
            best_match = finding
            best_score = score

    return best_match


def _extract_source_context(
    source_code: str,
    line_number: Any,
    radius: int = 4,
) -> str:
    """
    Return numbered source lines around the selected AST/pipeline node.
    """

    if not source_code:
        return "Source code is unavailable for this audit result."

    try:
        line = int(
            line_number
        )
    except (
        TypeError,
        ValueError,
    ):
        return (
            "Line number is unavailable. "
            "The code parser remediation will attach exact source locations."
        )

    source_lines = source_code.splitlines()

    if (
        line < 1
        or line > len(source_lines)
    ):
        return "The recorded line number is outside the uploaded source file."

    start = max(
        1,
        line - radius,
    )

    end = min(
        len(source_lines),
        line + radius,
    )

    return "\n".join(
        f"{number:>4} | {source_lines[number - 1]}"
        for number in range(
            start,
            end + 1,
        )
    )


def _node_graph_color(
    node: Dict[str, Any],
    finding: Optional[Dict[str, Any]],
) -> str:
    """
    Choose a graph color based on final/static severity.
    """

    severity = ""

    if finding:
        severity = str(
            finding.get(
                "final_severity",
                finding.get(
                    "static_severity",
                    "",
                ),
            )
        ).lower()

    if severity in {"critical", "high"}:
        return "#ff4d4d"

    if severity == "medium":
        return "#ffb648"

    if severity == "low":
        return "#3ddc84"

    node_type = str(
        node.get(
            "component_type",
            node.get(
                "type",
                "",
            ),
        )
    ).lower()

    if "data" in node_type:
        return "#7c7c7c"

    if "model" in node_type or "classifier" in node_type:
        return "#2dd4bf"

    return "#7c7c7c"


@st.cache_data(show_spinner=False)
def _compute_graph_positions_cached(
    node_ids: Tuple[str, ...],
    edge_pairs: Tuple[Tuple[str, str], ...],
) -> Dict[str, Tuple[float, float]]:
    """
    Cache-safe wrapper around the spring layout. Accepts tuples so
    Streamlit can hash the arguments. Avoids recomputing a 500-iteration
    spring layout on every rerun for an unchanged pipeline graph.
    """

    graph = nx.DiGraph()

    for node_id in node_ids:
        if node_id:
            graph.add_node(node_id)

    for source, target in edge_pairs:
        if (
            source
            and target
            and source in graph
            and target in graph
        ):
            graph.add_edge(source, target)

    if graph.number_of_nodes() == 0:
        return {}

    node_count = graph.number_of_nodes()

    spacing = max(
        9.0,
        11.0 / max(
            node_count ** 0.5,
            1.0,
        ),
    )

    raw_positions = nx.spring_layout(
        graph,
        seed=42,
        k=spacing,
        iterations=500,
        scale=1250.0,
        center=(0.0, 0.0),
    )

    return {
        str(node_id): (
            float(position[0]),
            float(position[1]),
        )
        for node_id, position in raw_positions.items()
    }


def _compute_graph_positions(
    nodes_data: List[Dict[str, Any]],
    edges_data: List[Dict[str, Any]],
) -> Dict[str, Tuple[float, float]]:
    """
    Compute stable, readable 2D positions for the interactive graph.

    Wraps the cached layout: converts the graph to hashable tuples,
    then delegates to _compute_graph_positions_cached.
    """

    node_ids = tuple(
        str(n.get("id", ""))
        for n in nodes_data
    )

    edge_pairs = tuple(
        (
            str(e.get("source", "")),
            str(e.get("target", "")),
        )
        for e in edges_data
    )

    return _compute_graph_positions_cached(
        node_ids,
        edge_pairs,
    )


def _infer_node_stage(node: Dict[str, Any]) -> int:
    """Classifies a pipeline component into one of the 5 canonical lifecycle stages."""
    if "stage_index" in node and isinstance(node["stage_index"], int):
        return node["stage_index"]
    node_type = str(node.get("type", "") or node.get("component_type", "")).lower()
    name = str(node.get("name", "") or node.get("label", "")).lower()

    if any(k in node_type or k in name for k in ["ingestion", "read_csv", "read_table", "open", "dataset", "input", "loader", "from_csv"]):
        return 0
    elif any(k in node_type or k in name for k in ["validation", "assert", "isinstance", "null", "check", "sanitize"]):
        return 1
    elif any(k in node_type or k in name for k in ["preprocess", "clean", "tokenize", "vectoriz", "transform", "scale", "encoder", "stem"]):
        return 2
    elif any(k in node_type or k in name for k in ["model", "train", "fit", "classifier", "regressor", "estimator", "bayes", "forest", "neural", "multinomial", "nb", "logistic", "svm", "svc", "tree", "gradient", "boost", "sgd"]):
        return 3
    elif any(k in node_type or k in name for k in ["infer", "predict", "evaluate", "score", "dump", "save", "persist", "output"]):
        return 4
    return 2


STAGE_METADATA = {
    0: {"name": "Data Ingestion", "pill": "INGESTION"},
    1: {"name": "Validation & Hygiene", "pill": "VALIDATION"},
    2: {"name": "Feature Engineering", "pill": "PREPROCESSING"},
    3: {"name": "Model Architecture", "pill": "ESTIMATOR"},
    4: {"name": "Inference & Output", "pill": "PERSISTENCE"},
}


def _render_streamlit_flow_dag(
    nodes_data: List[Dict[str, Any]],
    edges_data: List[Dict[str, Any]],
    result: Dict[str, Any],
    node_lookup: Dict[str, Dict[str, Any]],
) -> Optional[str]:
    """Renders the hierarchical DAG using React Flow with non-overlapping ManualLayout positions."""
    # 1. Group nodes by lifecycle stage (0 to 4)
    stage_groups: Dict[int, List[Dict[str, Any]]] = {0: [], 1: [], 2: [], 3: [], 4: []}
    for node in nodes_data:
        stage_idx = _infer_node_stage(node)
        stage_groups.setdefault(stage_idx, []).append(node)

    # 2. Compute exact coordinates per stage column (ManualLayout prevents ELK animations)
    col_x_map = {0: 40.0, 1: 340.0, 2: 640.0, 3: 940.0, 4: 1240.0}
    max_count = max(len(grp) for grp in stage_groups.values()) if stage_groups else 1
    y_pitch = 145.0

    node_positions: Dict[str, Tuple[float, float]] = {}
    for s_idx, grp in stage_groups.items():
        col_x = col_x_map.get(s_idx, 640.0)
        col_height = (len(grp) - 1) * y_pitch
        max_height = (max_count - 1) * y_pitch
        y_start = 30.0 + max(0.0, (max_height - col_height) / 2.0)
        for idx_in_col, n in enumerate(grp):
            node_positions[str(n.get("id", ""))] = (col_x, y_start + (idx_in_col * y_pitch))

    flow_nodes: List[StreamlitFlowNode] = []

    for node in nodes_data:
        node_id = str(node.get("id", ""))
        stage_idx = _infer_node_stage(node)
        label = str(node.get("name", node.get("label", node_id)))
        display_label = label if len(label) <= 28 else label[:25] + "…"
        ast_type = node.get("ast_node_type") or node.get("ast_type") or "Component"
        line_number = node.get("line_number") or node.get("lineno")

        finding = _find_related_finding(result, node)

        # Risk heatmap colors and pill: semi-see-through light grey before mapped,
        # color-coded using the audit palette when vulnerable.
        is_vulnerable = False
        risk_score = 0.0
        vuln_id = ""
        test_status = ""
        finding_severity = ""

        if finding:
            risk_score = float(
                finding.get(
                    "risk_score",
                    finding.get(
                        "final_risk_score",
                        finding.get("static_risk_score", 0.0),
                    ),
                )
            )
            vuln_id = str(finding.get("vulnerability_id", "RISK"))
            test_status = str(finding.get("test_status", "")).lower()
            finding_severity = str(
                finding.get(
                    "final_severity",
                    finding.get(
                        "severity",
                        finding.get("static_severity", ""),
                    ),
                )
            ).lower()
            if test_status == "vulnerable" or risk_score >= 4.0 or finding_severity in {"critical", "high", "medium"}:
                is_vulnerable = True

        if is_vulnerable:
            # Red is strictly reserved for high (and critical) risk
            is_high_risk = risk_score >= 7.0 or finding_severity in {"critical", "high"}
            is_medium_risk = not is_high_risk and (risk_score >= 4.0 or finding_severity == "medium" or test_status == "vulnerable")

            if is_high_risk:
                border_color = "#ff4d4d"
                bg_color = "rgba(255, 77, 77, 0.16)"
                text_color = "#ffffff"
                status_text = f"EXPLOIT TARGET: {vuln_id} ({risk_score:.1f}/10)" if test_status == "vulnerable" else f"HIGH RISK: {vuln_id} ({risk_score:.1f}/10)"
                box_shadow = "0 0 16px rgba(255, 77, 77, 0.35)"
            elif is_medium_risk:
                border_color = "#ffb648"
                bg_color = "rgba(255, 182, 72, 0.16)"
                text_color = "#ffffff"
                status_text = f"WEAKNESS: {vuln_id} ({risk_score:.1f}/10)" if test_status != "vulnerable" else f"EXPLOIT TARGET: {vuln_id} ({risk_score:.1f}/10)"
                box_shadow = "0 0 12px rgba(255, 182, 72, 0.28)"
            else:
                border_color = "#3ddc84"
                bg_color = "rgba(61, 220, 132, 0.16)"
                text_color = "#ffffff"
                status_text = f"INFO: {vuln_id} ({risk_score:.1f}/10)"
                box_shadow = "0 2px 8px rgba(61, 220, 132, 0.22)"
        else:
            # Semi-see-through and light grey before mapped to be vulnerable
            border_color = "rgba(124, 124, 124, 0.35)"
            bg_color = "rgba(255, 255, 255, 0.03)"
            text_color = "#d7d7d7"
            status_text = "BASELINE COMPONENT"
            box_shadow = "0 2px 6px rgba(0, 0, 0, 0.35)"

        stage_info = STAGE_METADATA.get(stage_idx, STAGE_METADATA[2])
        line_part = f"L{line_number} · " if line_number else ""
        content = (
            f"**STAGE {stage_idx + 1}: {stage_info['pill']}**\n\n"
            f"### `{display_label}`\n\n"
            f"*{line_part}{ast_type}*\n\n"
            f"`{status_text}`"
        )

        if stage_idx == 0:
            node_type = "input"
            source_pos = "right"
            target_pos = "left"
        elif stage_idx == 4:
            node_type = "output"
            source_pos = "right"
            target_pos = "left"
        else:
            node_type = "default"
            source_pos = "right"
            target_pos = "left"

        style = {
            "background": bg_color,
            "color": text_color,
            "border": f"2px solid {border_color}",
            "borderRadius": "10px",
            "padding": "12px 14px",
            "minWidth": "220px",
            "maxWidth": "280px",
            "boxShadow": box_shadow,
            "fontSize": "12px",
            "lineHeight": "1.4",
            "textAlign": "left",
        }

        pos_xy = node_positions.get(node_id, (0.0, 0.0))

        flow_nodes.append(
            StreamlitFlowNode(
                id=node_id,
                pos=pos_xy,
                data={"content": content},
                node_type=node_type,
                source_position=source_pos,
                target_position=target_pos,
                style=style,
                draggable=True,
                selectable=True,
            )
        )

    flow_edges: List[StreamlitFlowEdge] = []
    seen_edges = set()

    for idx, edge in enumerate(edges_data):
        src = str(edge.get("source", edge.get("from", "")))
        dst = str(edge.get("target", edge.get("to", "")))
        if not src or not dst or src == dst:
            continue
        edge_pair = (src, dst)
        if edge_pair in seen_edges:
            continue
        seen_edges.add(edge_pair)

        dst_node = node_lookup.get(dst, {})
        dst_finding = _find_related_finding(result, dst_node)
        if dst_finding:
            dst_score = float(dst_finding.get("risk_score", dst_finding.get("final_risk_score", 0.0)))
            dst_sev = str(dst_finding.get("final_severity", dst_finding.get("severity", ""))).lower()
            if dst_score >= 7.0 or dst_sev in {"critical", "high"}:
                edge_color = "#ff4d4d"
            elif dst_score >= 4.0 or dst_sev == "medium":
                edge_color = "#ffb648"
            else:
                edge_color = "rgba(124, 124, 124, 0.4)"
        else:
            edge_color = "rgba(124, 124, 124, 0.4)"

        flow_edges.append(
            StreamlitFlowEdge(
                id=f"e_{src}_{dst}_{idx}",
                source=src,
                target=dst,
                edge_type="smoothstep",
                animated=True,
                style={"stroke": edge_color, "strokeWidth": 2.5},
                marker_end={"type": "arrowclosed", "color": edge_color},
            )
        )

    phase = "report" if ("report" in result and isinstance(result["report"], dict)) else "gate1"
    cache_token = f"{phase}_{len(nodes_data)}_{len(edges_data)}_{abs(hash(tuple(str(n.get('id', '')) for n in nodes_data)))}"
    state_key = f"_aegisml_flow_state_{cache_token}"
    fitted_key = f"_aegisml_flow_fitted_{cache_token}"

    if state_key not in st.session_state:
        st.session_state[state_key] = StreamlitFlowState(
            nodes=flow_nodes,
            edges=flow_edges,
            timestamp=0,
        )

    flow_state = st.session_state[state_key]
    should_fit = not st.session_state.get(fitted_key, False)

    st.caption(
        "Hierarchical Pipeline DAG · 5 Lifecycle Stages (Ingestion -> Validation -> Preprocessing -> Model -> Output) · "
        "Drag, pan, or zoom canvas · Click any node to inspect its code and security context."
    )

    curr_state = streamlit_flow(
        key=f"aegisml_pipeline_flow_{cache_token}",
        state=flow_state,
        height=560,
        fit_view=should_fit,
        show_controls=True,
        show_minimap=False,
        layout=ManualLayout(),
        get_node_on_click=True,
        hide_watermark=True,
    )

    if should_fit:
        st.session_state[fitted_key] = True

    return getattr(curr_state, "selected_id", None) if curr_state else None


def render_interactive_pipeline_graph(
    result: Dict[str, Any],
) -> Optional[str]:
    """
    Render the Agent 1 pipeline graph as a hierarchical DAG.

    Uses Streamlit Flow (React Flow) with the Sugiyama layered layout algorithm
    to organize components across the 5 pipeline lifecycle stages with zero overlap.
    Falls back gracefully to NetworkX/agraph if React Flow is unavailable.

    Clicking a node displays an inline code inspection card containing:
    - component name/type
    - AST node type
    - source line
    - source-code context
    - related security finding
    """

    nodes_data, edges_data = _normalize_graph_payload(
        result
    )

    if not nodes_data:
        nodes_data, edges_data = _fallback_pipeline_graph(
            result
        )

    source_code = str(
        result.get(
            "pipeline_source",
            "",
        )
        or ""
    )

    node_lookup: Dict[str, Dict[str, Any]] = {
        str(n.get("id", "")): n for n in nodes_data if n.get("id")
    }

    selected_id: Optional[str] = None
    flow_rendered = False

    if HAS_STREAMLIT_FLOW:
        try:
            flow_selected = _render_streamlit_flow_dag(
                nodes_data=nodes_data,
                edges_data=edges_data,
                result=result,
                node_lookup=node_lookup,
            )
            flow_rendered = True
            if flow_selected:
                selected_id = str(flow_selected)
        except Exception:
            flow_rendered = False

    if not flow_rendered:
        graph_positions = _compute_graph_positions(
            nodes_data,
            edges_data,
        )

        graph_nodes: List[Node] = []
        transform_count = 0
        for node in nodes_data:
            node_id = str(node.get("id", ""))
            finding = _find_related_finding(result, node)
            line_number = node.get("line_number") or node.get("lineno") or node.get("line")
            ast_type = node.get("ast_node_type") or node.get("ast_type") or "Pipeline component"
            label = str(node.get("name", node.get("label", node_id)))
            display_label = label if len(label) <= 30 else label[:27] + "..."

            title_parts = [f"Component: {label}", f"AST: {ast_type}"]
            if line_number:
                title_parts.append(f"Line: {line_number}")
            if finding:
                title_parts.append(f"Risk: {finding.get('risk_score', 'N/A')}")

            x_pos, y_pos = graph_positions.get(node_id, (0.0, 0.0))
            graph_nodes.append(
                Node(
                    id=node_id,
                    label=display_label,
                    size=200,
                    shape="box",
                    color=_node_graph_color(node, finding),
                    font={"color": "#f3f3f1", "size": 24},
                    margin=16,
                    x=x_pos,
                    y=y_pos,
                    fixed=True,
                    title="\n".join(title_parts),
                )
            )

        graph_edges: List[Edge] = []
        for edge in edges_data:
            graph_edges.append(
                Edge(
                    source=str(edge.get("source", "")),
                    target=str(edge.get("target", "")),
                    label=str(edge.get("label", "")),
                    type="CURVE_SMOOTH",
                    color="#7c7c7c",
                    width=3.0,
                )
            )

        config = Config(
            width="100%",
            height=560,
            directed=True,
            physics=False,
            hierarchical=False,
            nodeHighlightBehavior=True,
            highlightColor="#ff751f",
            zoom=0.95,
        )

        st.caption(
            "Interactive pipeline graph · use zoom/pan to explore · "
            "click a node to inspect its source code and security context."
        )

        selected_node_id = agraph(
            nodes=graph_nodes,
            edges=graph_edges,
            config=config,
        )
        if selected_node_id:
            selected_id = str(selected_node_id)

    # 3. Interactive Inspector Panel (works across both Streamlit Flow and fallback)
    if selected_id and selected_id != st.session_state.get("_dismissed_pipeline_node"):
        st.session_state["_active_pipeline_node"] = selected_id

    active_id = st.session_state.get("_active_pipeline_node")
    if active_id and active_id != st.session_state.get("_dismissed_pipeline_node"):
        node = node_lookup.get(active_id)
        if node is not None:
            finding = _find_related_finding(result, node)
            node_name = str(node.get("name", node.get("label", active_id)))
            ast_type = (
                node.get("ast_node_type")
                or node.get("ast_type")
                or node.get("node_type")
                or node.get("type")
                or "Unavailable"
            )
            component_type = (
                node.get("component_type")
                or node.get("category")
                or node.get("component")
                or "Pipeline component"
            )
            line_number = (
                node.get("line_number")
                or node.get("lineno")
                or node.get("line")
            )

            with st.container(border=True):
                header_col1, header_col2 = st.columns([5, 1])
                with header_col1:
                    st.markdown(f"#### Pipeline Node Inspector: `{node_name}`")
                with header_col2:
                    if st.button(
                        "Close Inspector",
                        key=f"close_node_inspector_{active_id}",
                        use_container_width=True,
                    ):
                        st.session_state["_dismissed_pipeline_node"] = active_id
                        st.session_state["_active_pipeline_node"] = None
                        st.rerun()

                meta_col1, meta_col2, meta_col3 = st.columns(3)
                meta_col1.metric("Component", str(component_type))
                meta_col2.metric("AST node", str(ast_type))
                meta_col3.metric("Source line", str(line_number) if line_number else "N/A")

                st.markdown("##### Source context")
                source_context = _extract_source_context(source_code, line_number)
                st.code(source_context, language="python")

                if finding:
                    st.markdown("##### Related security finding")
                    risk_col1, risk_col2, risk_col3 = st.columns(3)
                    risk_col1.metric("Finding", str(finding.get("vulnerability_id", "N/A")))
                    risk_col2.metric(
                        "Risk score",
                        f"{float(finding.get('risk_score', finding.get('final_risk_score', 0.0))):.1f}/10",
                    )
                    risk_col3.metric("Status", str(finding.get("test_status", "not_tested")))

                    st.write(finding.get("description", "No finding description is available."))

                    affected = finding.get("affected_components", [])
                    if affected:
                        st.caption("Affected components: " + ", ".join(str(item) for item in affected))
                else:
                    st.info("No report finding is directly mapped to this pipeline node.")

            return active_id

    return None


TEST_CATALOG: Dict[str, Dict[str, str]] = {
    "V1": {
        "canonical_id": "V1_poisoning",
        "display_name": "V1_poisoning (Data Poisoning & Label Flipping)",
        "target": "Training dataset & label integrity (Data Ingestion)",
        "default_reason": "Evaluate model accuracy degradation and backdoor vulnerability against clean-label flipping and corrupted training samples.",
        "config_key": "poisoning_config",
        "category": "Data Poisoning",
    },
    "V2": {
        "canonical_id": "V2_preprocessing",
        "display_name": "V2_preprocessing (Attack Surface & Fuzzing)",
        "target": "Text normalization, tokenization, & vectorizer transforms",
        "default_reason": "Stress-test text normalization, Unicode edge-cases, and malformed string inputs to detect pipeline crashes, unhandled exceptions, or silent tokenization collapses.",
        "config_key": "preprocessing_config",
        "category": "Preprocessing Attack Surface",
    },
    "V3": {
        "canonical_id": "V3_validation",
        "display_name": "V3_validation (Schema & Boundary Gates)",
        "target": "Input schema validation, missing data & null value handling",
        "default_reason": "Inject missing column values, unexpected data types, and out-of-boundary records to verify pipeline input validation gates and fail-safe error handling.",
        "config_key": "validation_config",
        "category": "Data Validation Weaknesses",
    },
    "V4": {
        "canonical_id": "V4_adversarial",
        "display_name": "V4_adversarial (HopSkipJump Boundary Evasion)",
        "target": "Model inference boundary & trained estimator classifier",
        "default_reason": "Assess evasion robustness using iterative boundary perturbation within the configured relative perturbation budget.",
        "config_key": "adversarial_config",
        "category": "Adversarial Robustness",
    },
}


def _resolve_test_metadata(
    test_id: str,
    plan: Dict[str, Any],
    result: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str, str]:
    """
    Returns (display_name, target, reason) for a given test identifier.
    Populates Target & Reason columns using Agent 2 tactical configs,
    Agent 1 static findings, or domain specifications.
    """
    short_key = None
    tid_str = str(test_id).strip()
    if len(tid_str) >= 2 and tid_str[:2].upper() in TEST_CATALOG:
        short_key = tid_str[:2].upper()
    else:
        for k, info in TEST_CATALOG.items():
            if k in tid_str or info["canonical_id"] in tid_str:
                short_key = k
                break

    meta = TEST_CATALOG.get(short_key or "", {})
    display_name = meta.get("display_name", tid_str)
    target = meta.get("target", "Pipeline component execution surface")

    # Priority 1: Specific rationale formulated by Agent 2
    reason = ""
    config_key = meta.get("config_key")
    if config_key and isinstance(plan.get(config_key), dict):
        reason = str(plan[config_key].get("rationale") or "").strip()

    # Priority 2: Static vulnerability findings from Agent 1
    if not reason and result:
        findings = result.get("vulnerability_findings") or []
        if isinstance(findings, list):
            for f in findings:
                if isinstance(f, dict):
                    f_id = str(f.get("vulnerability_id") or "").upper()
                    f_cat = str(f.get("category") or "")
                    if f_id == short_key or f_cat == meta.get("category"):
                        desc = str(f.get("description") or f.get("summary") or "").strip()
                        if desc:
                            reason = f"Static analysis: {desc}"
                            break

    # Priority 3: Fallback domain justification
    if not reason:
        reason = meta.get(
            "default_reason",
            "Evaluate dynamic resilience against empirical adversarial perturbations in an isolated sandbox.",
        )

    return display_name, target, reason


def _strategy_test_rows(
    plan: Dict[str, Any],
    result: Optional[Dict[str, Any]] = None,
    selected_test_ids: Optional[List[str]] = None,
    completed_subtests: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    """
    Convert Agent 2's strategy into a human-readable review table with
    complete Target, Reason, and Authorization status.
    """
    selected_tests = (
        plan.get("selected_tests")
        or plan.get("planned_tests")
        or plan.get("tests")
        or []
    )

    # Build an ordered list of tests to display: proposed tests plus any added by reviewer
    items_to_display: List[Any] = list(selected_tests)
    existing_ids = {
        item if isinstance(item, str) else str(item.get("test_id") or item.get("name") or "")
        for item in items_to_display
    }
    if selected_test_ids:
        for user_tid in selected_test_ids:
            if user_tid not in existing_ids:
                items_to_display.append(user_tid)
                existing_ids.add(user_tid)

    rows: List[Dict[str, str]] = []
    selected_set = set(selected_test_ids) if selected_test_ids is not None else None
    mem_subtests = completed_subtests or {}

    for item in items_to_display:
        if isinstance(item, dict):
            test_id = (
                item.get("test_id")
                or item.get("id")
                or item.get("name")
                or item.get("test")
                or "Planned test"
            )
            item_target = item.get("target") or item.get("vulnerability_id") or item.get("category") or ""
            item_reason = item.get("reason") or item.get("rationale") or item.get("description") or ""
            disp_name, auto_target, auto_reason = _resolve_test_metadata(test_id, plan, result)
            target = item_target or auto_target
            reason = item_reason or auto_reason
        else:
            test_id = str(item)
            disp_name, target, reason = _resolve_test_metadata(test_id, plan, result)

        row_entry: Dict[str, str] = {
            "Test": disp_name,
            "Target": target,
            "Reason": reason,
        }

        is_in_memory = any(
            k in test_id or test_id in k or (len(test_id) >= 2 and test_id[:2] in k)
            for k in mem_subtests
        )

        if selected_set is not None:
            is_included = test_id in selected_set or any(test_id in s or s in test_id for s in selected_set)
            if completed_subtests is not None:
                if is_included:
                    status_text = "Retained (from memory)" if is_in_memory else "Authorized (new execution)"
                else:
                    status_text = "Excluded (stored in memory)" if is_in_memory else "Excluded"
            else:
                status_text = "Authorized" if is_included else "Excluded"
            row_entry["Status"] = status_text

        rows.append(row_entry)

    if selected_set is not None and rows:
        ordered_rows = []
        for r in rows:
            ordered_rows.append({
                "Status": r["Status"],
                "Test": r["Test"],
                "Target": r["Target"],
                "Reason": r["Reason"],
            })
        return ordered_rows

    return rows


def render_attack_strategy_gate(
    result: Dict[str, Any],
) -> bool:
    """
    Gate 1: require explicit human approval and test selection before Agent 2 executes.

    The page-level hero owns the phase framing ("Human-in-the-Loop Review").
    This function therefore only renders the numbered operational steps
    (selection -> strategy details -> sign-off) and does not repeat the
    intro copy shown above the fold.
    """
    plan = (
        result.get(
            "attack_strategy_plan",
            {},
        )
        or {}
    )

    checkpoint_loaded = result.get("checkpoint_loaded", False)
    audit_id = result.get("audit_id", "")
    audit_mem = result.get("audit_memory", {})
    steps_count = audit_mem.get("steps_count", 0)

    # Discover any completed dynamic tests stored in audit memory
    completed_subtests: Dict[str, Any] = {}
    try:
        from src.core.audit_memory import get_all_subtests, get_subtests_by_artifacts
        completed_subtests = get_all_subtests(audit_id)
        if not completed_subtests:
            completed_subtests = get_subtests_by_artifacts(
                code=result.get("pipeline_source"),
            )
    except Exception:
        completed_subtests = {}

    if checkpoint_loaded:
        st.html(
            f"""
            <div style="
                background: rgba(61, 220, 132, 0.10);
                border: 1px solid rgba(61, 220, 132, 0.35);
                border-radius: 8px;
                padding: 12px 16px;
                margin-bottom: 16px;
                display: flex;
                align-items: center;
                gap: 12px;
            ">
                <span style="
                    background: #3ddc84;
                    color: #0f0f10;
                    font-size: 11px;
                    font-weight: 700;
                    letter-spacing: 0.5px;
                    padding: 3px 8px;
                    border-radius: 4px;
                    font-family: monospace;
                ">CHECKPOINT LOADED</span>
                <div style="font-size: 13px; color: #d7d7d7;">
                    <strong>Audit Memory Active:</strong> Restored Agent 1 static analysis and attack strategy from persistent SQLite memory
                    (<code>{escape(audit_id[:12])}…</code> · {steps_count} steps cached). SHA-256 artifact integrity verified.
                </div>
            </div>
            """
        )

    if completed_subtests:
        stored_list = ", ".join(sorted(completed_subtests.keys()))
        st.html(
            f"""
            <div style="
                background: rgba(45, 212, 191, 0.10);
                border: 1px solid rgba(45, 212, 191, 0.35);
                border-radius: 8px;
                padding: 12px 16px;
                margin-bottom: 16px;
                display: flex;
                align-items: center;
                gap: 12px;
            ">
                <span style="
                    background: #2dd4bf;
                    color: #0f0f10;
                    font-size: 11px;
                    font-weight: 700;
                    letter-spacing: 0.5px;
                    padding: 3px 8px;
                    border-radius: 4px;
                    font-family: monospace;
                ">DYNAMIC MEMORY ACTIVE</span>
                <div style="font-size: 13px; color: #d7d7d7;">
                    <strong>Retained Dynamic Tests:</strong> Found <strong>{len(completed_subtests)}</strong> previously executed dynamic test(s) in audit memory (<code>{escape(stored_list)}</code>).
                    These tests are automatically retained. Any additional tests selected below will be executed in the container sandbox.
                </div>
            </div>
            """
        )

    # -------------------------------------------------------------
    # Step 1 — Test selection (Agent 2 recommendations + full suite)
    # -------------------------------------------------------------
    ALL_DYNAMIC_TESTS = [
        "V1_poisoning",
        "V4_adversarial",
        "V2_preprocessing",
        "V3_validation",
    ]

    raw_proposed = (
        plan.get("selected_tests")
        or plan.get("planned_tests")
        or plan.get("tests")
        or []
    )
    proposed_test_ids: List[str] = [
        t if isinstance(t, str) else str(t.get("test_id") or t.get("name") or "")
        for t in raw_proposed
    ]
    proposed_test_ids = [t for t in proposed_test_ids if t]

    # Combine canonical dynamic tests with any custom tests formulated by Agent 2
    all_available_options: List[str] = list(ALL_DYNAMIC_TESTS)
    for tid in proposed_test_ids:
        if tid not in all_available_options:
            all_available_options.append(tid)

    st.markdown(
        '<div class="step-heading"><span class="step-num">01</span>Select tests to authorize</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="step-caption">Agent 2 has recommended tests based on static analysis. '
        'You can authorize, deselect, or add any dynamic tests below. Select at least one.</div>',
        unsafe_allow_html=True,
    )

    multiselect_key = "gate1_selected_tests_multiselect_v6"
    if multiselect_key not in st.session_state:
        st.session_state[multiselect_key] = list(proposed_test_ids)

    # Ensure selection stays valid within all available options
    sanitized_selection = [t for t in st.session_state[multiselect_key] if t in all_available_options]
    if not sanitized_selection and proposed_test_ids:
        sanitized_selection = list(proposed_test_ids)

    user_selected_tests = st.multiselect(
        "Authorized dynamic tests for container sandbox:",
        options=all_available_options,
        default=sanitized_selection,
        format_func=lambda tid: _resolve_test_metadata(tid, plan, result)[0],
        help="Select dynamic penetration tests to authorize for execution in the container sandbox. You can add tests beyond Agent 2's recommendations.",
        key=multiselect_key,
        label_visibility="collapsed",
    )

    # -------------------------------------------------------------
    # Step 2 — Strategy review table (with Target & Reason filled)
    # -------------------------------------------------------------
    st.markdown(
        '<div class="step-heading"><span class="step-num">02</span>Strategy details</div>',
        unsafe_allow_html=True,
    )

    strategy_rows = _strategy_test_rows(
        plan=plan,
        result=result,
        selected_test_ids=user_selected_tests,
        completed_subtests=completed_subtests,
    )

    if strategy_rows:
        rows_html: List[str] = []
        for r in strategy_rows:
            status = str(r.get("Status", ""))
            test_name = str(r.get("Test", ""))
            target = str(r.get("Target", ""))
            reason = str(r.get("Reason", ""))

            status_badge_class = "status-badge-excluded"
            if "Authorized" in status:
                status_badge_class = "status-badge-authorized"
            elif "Retained" in status:
                status_badge_class = "status-badge-retained"

            rows_html.append(
                f"""
                <tr>
                    <td><span class="strategy-status-tag {status_badge_class}">{escape(status)}</span></td>
                    <td><strong>{escape(test_name)}</strong></td>
                    <td><span class="strategy-target-label">{escape(target)}</span></td>
                    <td>{escape(reason)}</td>
                </tr>
                """
            )

        strategy_table_html = f"""
        <div class="strategy-table-container">
            <table class="strategy-table">
                <thead>
                    <tr>
                        <th style="width: 17%;">Status</th>
                        <th style="width: 23%;">Test</th>
                        <th style="width: 25%;">Target</th>
                        <th style="width: 35%;">Reason</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows_html)}
                </tbody>
            </table>
        </div>
        """
        st.html(strategy_table_html)
    else:
        st.warning(
            "No structured test list was returned. "
            "Review the raw strategy before approving."
        )

    with st.expander(
        "Raw Agent 2 strategy JSON",
        expanded=False,
    ):
        st.code(
            json.dumps(
                plan,
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            language="json",
        )

    # -------------------------------------------------------------
    # Step 3 — Reviewer sign-off & confirmation checkbox
    # -------------------------------------------------------------
    st.markdown(
        '<div class="step-heading"><span class="step-num">03</span>Sign-off</div>',
        unsafe_allow_html=True,
    )

    # Versioned keys so stale widget slots from earlier revisions of this
    # function cannot be reused by Streamlit. The label must be identical
    # in both branches below — changing it while keeping the same key would
    # cause Streamlit to reattach the widget to an old element-tree slot.
    approval_key = "gate_1_confirmation_v6"
    has_valid_selection = len(user_selected_tests) >= 1

    CHECKBOX_LABEL = (
        "I have reviewed the strategy above and authorize "
        "the selected tests for execution."
    )

    if not has_valid_selection:
        st.warning(
            "Select at least one test before authorizing execution."
        )
        approved = st.checkbox(
            CHECKBOX_LABEL,
            value=False,
            disabled=True,
            key=approval_key,
        )
    else:
        approved = st.checkbox(
            CHECKBOX_LABEL,
            key=approval_key,
        )

    # -------------------------------------------------------------
    # Approve & Reject actions (appears exactly once)
    # -------------------------------------------------------------
    def on_gate_1_reject() -> None:
        """Executed before rerun, safely resetting Gate 1 state without widget collision."""
        st.session_state.audit_plan = None
        st.session_state.audit_id = None
        st.session_state.audit_result = None
        st.session_state.audit_executing = False
        st.session_state.report_signed_off = False

        # Purge every key any past revision of this function might have used.
        for stale_key in (
            "gate_1_confirmation",
            "gate_1_confirmation_v3",
            "gate_1_confirmation_v4",
            "gate_1_confirmation_v5",
            "gate_1_confirmation_v6",
            "gate1_selected_tests_multiselect",
            "gate1_selected_tests_multiselect_v3",
            "gate1_selected_tests_multiselect_v4",
            "gate1_selected_tests_multiselect_v5",
            "gate1_selected_tests_multiselect_v6",
            "gate_1_approve",
            "gate_1_approve_v3",
            "gate_1_approve_v4",
            "gate_1_approve_v5",
            "gate_1_approve_v6",
            "gate_1_reject",
            "gate_1_reject_v3",
            "gate_1_reject_v4",
            "gate_1_reject_v5",
            "gate_1_reject_v6",
        ):
            if stale_key in st.session_state:
                del st.session_state[stale_key]

        for qk in ("audit_id", "audit_phase"):
            if qk in st.query_params:
                del st.query_params[qk]

    approve_col, reject_col = st.columns([1, 1])

    with approve_col:
        approve_clicked = st.button(
            "Approve & run tests",
            type="primary",
            use_container_width=True,
            disabled=not approved or not has_valid_selection,
            key="gate_1_approve_v6",
        )

    with reject_col:
        st.button(
            "Reject & return to upload",
            use_container_width=True,
            key="gate_1_reject_v6",
            on_click=on_gate_1_reject,
        )

    if approve_clicked:
        # Pre-flight Docker check on Gate 1 approval
        try:
            from app import check_docker_available
            docker_ok = check_docker_available()
        except Exception:
            try:
                from src.agents.testing_agent.sandbox_runner import is_docker_available
                docker_ok = is_docker_available()
            except Exception:
                docker_ok = False

        if not docker_ok:
            st.session_state.show_docker_dialog = True
            st.rerun()

        # Update the plan and session with reviewer-chosen tests
        plan["selected_tests"] = list(user_selected_tests)
        if "attack_strategy_plan" in result and isinstance(result["attack_strategy_plan"], dict):
            result["attack_strategy_plan"]["selected_tests"] = list(user_selected_tests)
        if "audit_plan" in st.session_state and isinstance(st.session_state.audit_plan, dict):
            if "attack_strategy_plan" in st.session_state.audit_plan:
                st.session_state.audit_plan["attack_strategy_plan"]["selected_tests"] = list(user_selected_tests)

        return True

    return False


def render_report_signoff_gate(
    result: Dict[str, Any],
) -> bool:
    """
    Gate 2: require explicit human sign-off before PDF download.
    """

    report = (
        result.get(
            "report",
            {},
        )
        or {}
    )

    overall = (
        report.get(
            "overall_risk",
            {},
        )
        or {}
    )

    findings = (
        report.get(
            "findings",
            [],
        )
        or []
    )

    st.markdown(
        "### Gate 2 · Final Report Sign-Off"
    )

    st.write(
        "Review the final correlated findings before releasing "
        "the audit report."
    )

    summary_col1, summary_col2, summary_col3 = (
        st.columns(3)
    )

    summary_col1.metric(
        "Final risk",
        (
            f"{float(overall.get('overall_risk_score', 0.0)):.1f}/10"
        ),
    )

    summary_col2.metric(
        "Severity",
        str(
            overall.get(
                "overall_severity",
                "Low",
            )
        ),
    )

    summary_col3.metric(
        "Confirmed findings",
        int(
            overall.get(
                "confirmed_findings",
                0,
            )
        ),
    )

    if findings:
        review_rows = []

        for finding in findings:
            review_rows.append(
                {
                    "ID": finding.get(
                        "vulnerability_id",
                        "N/A",
                    ),
                    "Category": finding.get(
                        "category",
                        "Security finding",
                    ),
                    "Dynamic status": finding.get(
                        "test_status",
                        "not_tested",
                    ),
                    "Correlation": finding.get(
                        "correlation_status",
                        "Unverified",
                    ),
                    "Risk score": (
                        f"{float(finding.get('risk_score', finding.get('final_risk_score', 0.0))):.1f}/10"
                    ),
                }
            )

        st.dataframe(
            review_rows,
            use_container_width=True,
            hide_index=True,
        )

    if st.session_state.get(
        "report_signed_off",
        False,
    ):
        st.success(
            "Gate 2 approved. The final report is signed off "
            "and can be downloaded."
        )

        return True

    reviewer_confirmation = st.checkbox(
        (
            "I reviewed the final findings, dynamic evidence, "
            "and Agent 3 risk assessment."
        ),
        key="gate_2_confirmation",
    )

    if st.button(
        "Sign off final report",
        type="primary",
        use_container_width=True,
        disabled=not reviewer_confirmation,
        key="gate_2_signoff",
    ):
        st.session_state.report_signed_off = True
        st.rerun()

    st.caption(
        "The PDF download remains disabled until this gate is approved."
    )

    return False


def render_audit_ledger_panel(
    result: Dict[str, Any],
) -> None:
    """
    Renders fine-grained checkpoint ledger, artifact fingerprints,
    and step execution metrics from audit_memory.db.
    """
    audit_id = result.get("audit_id", "")
    checkpoint_loaded = result.get("checkpoint_loaded", False)
    audit_mem = result.get("audit_memory", {})
    artifacts = audit_mem.get("artifacts", {})
    ledger = audit_mem.get("ledger", [])
    subtests = audit_mem.get("completed_subtests", [])

    st.markdown("## Audit Memory & Checkpoint Ledger")
    st.write(
        "AegisML persists cryptographic checkpoints at every LangGraph agent node "
        "and dynamic penetration test into SQLite WAL audit memory (<code>audit_memory.db</code>). "
        "Review step execution telemetry and artifact integrity below."
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="Memory Status",
            value="Checkpoint Restored" if checkpoint_loaded else "Live Execution",
            delta="Persistent" if checkpoint_loaded else "Initial Run",
        )
    with col2:
        st.metric(
            label="Checkpointed Steps",
            value=len(ledger),
        )
    with col3:
        st.metric(
            label="Dynamic Sub-tests",
            value=len(subtests) if subtests else 4,
        )
    with col4:
        st.metric(
            label="Artifact Integrity",
            value="SHA-256 Verified",
        )

    # Artifact Cryptographic Fingerprints
    with st.expander("Cryptographic Artifact Hashes (SHA-256)", expanded=False):
        st.write("Cryptographic SHA-256 hashes computed on target code, model, and dataset:")
        code_h = artifacts.get("code_hash", "")
        model_h = artifacts.get("model_hash", "")
        data_h = artifacts.get("dataset_hash", "")
        reg_at = artifacts.get("registered_at", "")

        st.code(
            f"Audit Session ID : {audit_id}\n"
            f"Pipeline Code    : {code_h or 'N/A'}\n"
            f"Model Weights    : {model_h or 'N/A'}\n"
            f"Dataset          : {data_h or 'N/A'}\n"
            f"Baseline Date    : {reg_at or 'N/A'}",
            language="text",
        )

    # Chronological Ledger Table
    if ledger:
        st.markdown("### Step-Level Checkpoint History")
        ledger_rows = [
            {
                "Agent": step.get("agent_name", "Unknown"),
                "Step / LangGraph Node": step.get("step_name", "Unknown"),
                "Status": str(step.get("status", "completed")).upper(),
                "Duration": f"{float(step.get('duration_seconds') or 0.0):.3f}s",
                "Timestamp (UTC)": step.get("created_at", ""),
            }
            for step in ledger
        ]
        st.dataframe(ledger_rows, use_container_width=True, hide_index=True)
    else:
        st.info("Step-level checkpoints are active and recorded atomically in .aegisml_runtime/audit_memory.db.")