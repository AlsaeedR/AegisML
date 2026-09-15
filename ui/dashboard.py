import html
import json
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config


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
            <div class="finding-meta" style="color: #15803d;">
                verified active controls:
                <strong>{escape(controls_str)}</strong>
            </div>
            """
        elif control_verdict == "partially_effective":
            controls_meta_html = f"""
            <div class="finding-meta" style="color: #0284c7;">
                partially active controls (mitigated major failure):
                <strong>{escape(controls_str)}</strong>
            </div>
            """
        elif control_verdict == "bypassed" or ("hidden risk" in correlation_lower and final_severity.lower() in ["critical", "high"]):
            controls_meta_html = f"""
            <div class="finding-meta" style="color: #c2410c;">
                bypassed controls (failed under penetration testing):
                <strong>{escape(controls_str)}</strong>
            </div>
            """
        elif "confirmed" in correlation_lower or status.lower() == "vulnerable":
            controls_meta_html = f"""
            <div class="finding-meta" style="color: #64748b;">
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
    <div class="finding-card">

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
        f"#a27d31 {score}%, "
        f"#dddeda {score}%"
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
                "🌐",
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
        <div class="finding-card">
            No security findings were generated.
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

            <div class="scan-status">
                <span class="scan-dot"></span>
                security assessment complete
            </div>

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
                        {f'<div style="font-size: 0.78rem; color: #64748b; margin-top: 2px;">{escape(knowledge_desc)}</div>' if knowledge_desc else ''}
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

    findings = (
        result.get("report", {})
        .get("findings", [])
    ) or []

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

    if severity == "critical":
        return "#b91c1c"

    if severity == "high":
        return "#c2410c"

    if severity == "medium":
        return "#a27d31"

    if severity == "low":
        return "#4f8060"

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
        return "#536b78"

    if "model" in node_type or "classifier" in node_type:
        return "#65558f"

    return "#667064"



def _compute_graph_positions(
    nodes_data: List[Dict[str, Any]],
    edges_data: List[Dict[str, Any]],
) -> Dict[str, Tuple[float, float]]:
    """
    Compute stable, readable 2D positions for the interactive graph.

    The previous hierarchical layout compressed a mostly sequential AST graph
    into a narrow vertical column. NetworkX spring_layout spreads connected
    nodes across the available canvas while keeping the same result stable
    between reruns.
    """

    graph = nx.DiGraph()

    for node in nodes_data:
        node_id = str(
            node.get(
                "id",
                "",
            )
        )

        if node_id:
            graph.add_node(
                node_id
            )

    for edge in edges_data:
        source = str(
            edge.get(
                "source",
                "",
            )
        )

        target = str(
            edge.get(
                "target",
                "",
            )
        )

        if (
            source
            and target
            and source in graph
            and target in graph
        ):
            graph.add_edge(
                source,
                target,
            )

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


def render_interactive_pipeline_graph(
    result: Dict[str, Any],
) -> Optional[str]:
    """
    Render the Agent 1 pipeline graph as an interactive NetworkX-backed graph.

    The graph uses deterministic 2D NetworkX positions instead of the narrow
    hierarchical column that previously made the nodes look like a single line.

    Clicking a node opens a code-inspection dialog containing:
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

    graph_positions = _compute_graph_positions(
        nodes_data,
        edges_data,
    )

    graph_nodes: List[Node] = []
    node_lookup: Dict[str, Dict[str, Any]] = {}

    transform_count = 0
    for node in nodes_data:
        node_id = str(
            node.get(
                "id",
                "",
            )
        )

        node_lookup[node_id] = node

        finding = _find_related_finding(
            result,
            node,
        )

        line_number = (
            node.get("line_number")
            or node.get("lineno")
            or node.get("line")
        )

        ast_type = (
            node.get("ast_node_type")
            or node.get("ast_type")
            or node.get("node_type")
            or node.get("type")
            or "Pipeline component"
        )

        label = str(
            node.get(
                "name",
                node.get(
                    "label",
                    node_id,
                ),
            )
        )

        display_label = (
            label
            if len(label) <= 30
            else label[:27] + "..."
        )

        title_parts = [
            f"Component: {label}",
            f"AST: {ast_type}",
        ]

        if line_number:
            title_parts.append(
                f"Line: {line_number}"
            )

        if finding:
            title_parts.append(
                "Risk: "
                f"{finding.get('risk_score', finding.get('final_risk_score', finding.get('static_risk_score', 'N/A')))}"
            )

        x_position, y_position = graph_positions.get(
            node_id,
            (
                0.0,
                0.0,
            ),
        )
        if label == "transform":
            transform_count += 1
            if transform_count == 1:
                x_position -= 120
                y_position += 80

            elif transform_count == 2:
                x_position += 140
                y_position -= 70
        
        graph_nodes.append(
            Node(
                id=node_id,
                label=display_label,
                size=250,
                shape="box",
                color=_node_graph_color(
                    node,
                    finding,
                ),
                font={
                    "color": "#ffffff",
                    "size": 42,
                },
                margin=24,
                widthConstraint={
                    "minimum": 210,
                    "maximum": 360,
                },
                 x=x_position,
                y=y_position,
                fixed=True,
                title="\n".join(
                    title_parts
                ),
            )
        )

    graph_edges: List[Edge] = []

    for edge in edges_data:
        graph_edges.append(
            Edge(
                source=str(
                    edge.get(
                        "source",
                        "",
                    )
                ),
                target=str(
                    edge.get(
                        "target",
                        "",
                    )
                ),
                label=str(
                    edge.get(
                        "label",
                        "",
                    )
                ),
                type="CURVE_SMOOTH",
                color="#d6a7a2",
                width=3.0,
            )
        )

    config = Config(
        width="100%",
        height=620,
        directed=True,
        physics=False,
        hierarchical=False,
        nodeHighlightBehavior=True,
        highlightColor="#f0c36d",
        collapsible=False,
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
        selected_id = str(
            selected_node_id
        )

        node = node_lookup.get(
            selected_id
        )

        if node is not None:
            finding = _find_related_finding(
                result,
                node,
            )

            @st.dialog(
                "Pipeline node inspection",
                width="large",
            )
            def show_node_inspection() -> None:
                node_name = str(
                    node.get(
                        "name",
                        node.get(
                            "label",
                            selected_id,
                        ),
                    )
                )

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

                st.markdown(
                    f"### {node_name}"
                )

                meta_col1, meta_col2, meta_col3 = (
                    st.columns(3)
                )

                meta_col1.metric(
                    "Component",
                    str(
                        component_type
                    ),
                )

                meta_col2.metric(
                    "AST node",
                    str(
                        ast_type
                    ),
                )

                meta_col3.metric(
                    "Source line",
                    (
                        str(
                            line_number
                        )
                        if line_number
                        else "N/A"
                    ),
                )

                st.markdown(
                    "#### Source context"
                )

                source_context = _extract_source_context(
                    source_code,
                    line_number,
                )

                st.code(
                    source_context,
                    language="python",
                )

                if finding:
                    st.markdown(
                        "#### Related security finding"
                    )

                    risk_col1, risk_col2, risk_col3 = (
                        st.columns(3)
                    )

                    risk_col1.metric(
                        "Finding",
                        str(
                            finding.get(
                                "vulnerability_id",
                                "N/A",
                            )
                        ),
                    )

                    risk_col2.metric(
                        "Risk score",
                        (
                            f"{float(finding.get('risk_score', finding.get('final_risk_score', 0.0))):.1f}/10"
                        ),
                    )

                    risk_col3.metric(
                        "Status",
                        str(
                            finding.get(
                                "test_status",
                                "not_tested",
                            )
                        ),
                    )

                    st.write(
                        finding.get(
                            "description",
                            "No finding description is available.",
                        )
                    )

                    affected = finding.get(
                        "affected_components",
                        [],
                    )

                    if affected:
                        st.caption(
                            "Affected components: "
                            + ", ".join(
                                str(item)
                                for item in affected
                            )
                        )

                else:
                    st.info(
                        "No report finding is directly mapped "
                        "to this pipeline node."
                    )

            show_node_inspection()

        return selected_id

    return None

def _strategy_test_rows(
    plan: Dict[str, Any],
) -> List[Dict[str, str]]:
    """
    Convert Agent 2's strategy into a human-readable review table.
    """

    selected_tests = (
        plan.get("selected_tests")
        or plan.get("planned_tests")
        or plan.get("tests")
        or []
    )

    rows: List[Dict[str, str]] = []

    for item in selected_tests:
        if isinstance(item, dict):
            test_id = (
                item.get("test_id")
                or item.get("id")
                or item.get("name")
                or item.get("test")
                or "Planned test"
            )

            target = (
                item.get("vulnerability_id")
                or item.get("category")
                or item.get("target")
                or ""
            )

            reason = (
                item.get("reason")
                or item.get("rationale")
                or item.get("justification")
                or item.get("description")
                or ""
            )

            rows.append(
                {
                    "Test": str(test_id),
                    "Target": str(target),
                    "Reason": str(reason),
                }
            )

        else:
            rows.append(
                {
                    "Test": str(item),
                    "Target": "",
                    "Reason": "",
                }
            )

    return rows


def render_attack_strategy_gate(
    result: Dict[str, Any],
) -> bool:
    """
    Gate 1: require explicit human approval before Agent 2 executes.
    """

    plan = (
        result.get(
            "attack_strategy_plan",
            {},
        )
        or {}
    )

    st.markdown(
        "## Gate 1 · Attack Strategy Review"
    )

    st.write(
        "Agent 2 has prepared the dynamic testing strategy, "
        "but **no dynamic attack has executed yet**. "
        "Review the proposed tests before authorizing execution."
    )

    strategy_rows = _strategy_test_rows(
        plan
    )

    if strategy_rows:
        st.dataframe(
            strategy_rows,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning(
            "No structured test list was returned. "
            "Review the raw strategy before approving."
        )

    with st.expander(
        "Raw Agent 2 strategy",
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

    approval_key = (
        "gate_1_confirmation"
    )

    approved = st.checkbox(
        (
            "I reviewed the proposed attack strategy "
            "and authorize Agent 2 to execute these dynamic tests."
        ),
        key=approval_key,
    )

    approve_col, reject_col = st.columns(
        [1, 1]
    )

    with approve_col:
        approve_clicked = st.button(
            "Approve strategy & run tests",
            type="primary",
            use_container_width=True,
            disabled=not approved,
            key="gate_1_approve",
        )

    with reject_col:
        if st.button(
            "Reject & return to upload",
            use_container_width=True,
            key="gate_1_reject",
        ):
            st.session_state.audit_plan = None
            st.session_state.audit_id = None
            st.session_state.audit_result = None
            st.session_state.report_signed_off = False
            st.session_state[
                approval_key
            ] = False
            st.rerun()

    return bool(
        approve_clicked
    )


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
