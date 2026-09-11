import html
from typing import Any, Dict, List


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
                    "final_risk_score",
                    0,
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
            "final_severity",
            "Low",
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
            "final_risk_score",
            0.0,
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
        if control_verdict == "verified_effective" or "mitigated" in correlation_lower:
            controls_meta_html = f"""
            <div class="finding-meta" style="color: #15803d;">
                verified active controls:
                <strong>{escape(controls_str)}</strong>
            </div>
            """
        elif control_verdict == "bypassed" or "hidden risk" in correlation_lower:
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