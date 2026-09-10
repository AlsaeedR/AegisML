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
) -> str:

    flagged = score >= 40

    flag_class = (
        "flagged"
        if flagged
        else ""
    )

    alert = (
        '<div class="node-alert">!</div>'
        if flagged
        else ""
    )

    trust = (
        "review"
        if flagged
        else "lower risk"
    )

    trust_class = (
        ""
        if flagged
        else "trusted"
    )

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
    if len(description) > 220:
        description = description[:217].rsplit(" ", 1)[0] + "..."

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

    status = str(
        finding.get(
            "test_status",
            "not_tested",
        )
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

    correlation_status = str(
        finding.get(
            "correlation_status",
            "Unverified",
        )
    )

    correlation_rationale = str(
        finding.get(
            "correlation_rationale",
            "",
        )
    )

    is_false_positive = "false positive" in correlation_status.lower() or (
        final_severity.lower() == "low" and status == "not_vulnerable"
    )
    is_confirmed = "confirmed" in correlation_status.lower() or final_severity.lower() in ["critical", "high"]

    if is_false_positive:
        left_box_title = "Theoretical concern"
        right_box_title = "Verification outcome"
        right_box_class = "verification-outcome"
    elif is_confirmed:
        left_box_title = "Root cause"
        right_box_title = "Suggested fix"
        right_box_class = "suggested-fix"
    else:
        left_box_title = "Theoretical observation"
        right_box_title = "Hardening guidance"
        right_box_class = "suggested-fix"

    return f"""
    <div class="finding-card">

        <div class="finding-head">

            <span class="
                severity
                {severity_css}
            ">
                {escape(
                    final_severity.lower()
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

    issue_word = (
        "confirmed issue needs"
        if vulnerable == 1
        else "confirmed issues need"
    )

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
            ),

            pipeline_node(
                "🧹",
                "Preprocessing",
                "preprocessing assessment",
                preprocessing,
            ),

            pipeline_node(
                "🧠",
                "ML pipeline",
                "data validation assessment",
                validation,
            ),

            pipeline_node(
                "🌐",
                "Inference surface",
                "adversarial robustness assessment",
                adversarial,
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
                    {vulnerable}
                    {issue_word}
                    attention before this
                    pipeline ships
                </div>

                <div class="summary-text">
                    {escape(summary)}
                </div>

                <div class="finding-meta">
                    False positives:
                    {false_positives}
                    · Hidden risks:
                    {hidden_risks}
                </div>

                <div class="risk-bars">

                    {
                        risk_bar(
                            "Poisoning",
                            poisoning,
                            "fill-red",
                        )
                    }

                    {
                        risk_bar(
                            "Adversarial robustness",
                            adversarial,
                            "fill-gold",
                        )
                    }

                    {
                        risk_bar(
                            "Validation weakness",
                            validation,
                            "fill-gold",
                        )
                    }

                    {
                        risk_bar(
                            "Preprocessing surface",
                            preprocessing,
                            "fill-green",
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
