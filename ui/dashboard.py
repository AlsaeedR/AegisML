import html
from typing import Any, Dict, List


def escape(value: Any) -> str:
    return html.escape(
        str(value if value is not None else "")
    )


def category_score(
    findings: List[Dict[str, Any]],
    category: str,
) -> int:
    for finding in findings:
        if finding.get("category") == category:
            score = float(
                finding.get("risk_score", 0)
            )

            return int(score * 10)

    return 0


def severity_class(severity: str) -> str:
    severity = str(severity).lower()

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

    flagged = score >= 50

    flag_class = "flagged" if flagged else ""

    alert = (
        '<div class="node-alert">!</div>'
        if flagged
        else ""
    )

    trust = "partial" if flagged else "trusted"

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
            {trust}
        </div>

    </div>
    """


def finding_card(
    finding: Dict[str, Any]
) -> str:

    severity = str(
        finding.get(
            "severity",
            "Low",
        )
    )

    severity_css = severity_class(
        severity
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
        "No root cause information available.",
    )

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

    status = finding.get(
        "test_status",
        "not_tested",
    )

    dynamic_severity = finding.get(
        "dynamic_severity",
        "N/A",
    )

    return f"""
    <div class="finding-card">

        <div class="finding-head">

            <span class="
                severity
                {severity_css}
            ">
                {escape(severity.lower())}
            </span>

            <span class="finding-title">
                {escape(category)}
            </span>

        </div>

        <div class="finding-meta">
            {escape(vulnerability_id)}
            · dynamic test: {escape(status)}
            · dynamic severity:
            {escape(dynamic_severity)}
        </div>

        <div class="finding-grid">

            <div class="root-cause">

                <div class="box-title">
                    Root cause
                </div>

                <div class="box-content">
                    {escape(description)}
                </div>

            </div>

            <div class="suggested-fix">

                <div class="box-title">
                    Suggested fix
                </div>

                <div class="box-content">
                    {escape(fix)}
                </div>

            </div>

        </div>

        <div class="tags">

            <span class="tag">
                OWASP ML Security
            </span>

            <span class="tag">
                NIST AI RMF
            </span>

        </div>

    </div>
    """


def render_dashboard(
    result: Dict[str, Any]
) -> str:

    report = result.get(
        "report",
        {},
    )

    overall = report.get(
        "overall_risk",
        {},
    )

    findings = report.get(
        "findings",
        [],
    )

    score = int(
        float(
            overall.get(
                "overall_risk_score",
                0,
            )
        )
        * 10
    )

    severity = str(
        overall.get(
            "overall_severity",
            "Low",
        )
    ).lower()

    summary = report.get(
        "executive_summary",
        "AegisML completed the ML security audit.",
    )

    vulnerable = sum(
        1
        for finding in findings
        if finding.get(
            "test_status"
        ) == "vulnerable"
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
                "dataset input",
                poisoning,
            ),

            pipeline_node(
                "🧹",
                "Preprocessing",
                "pipeline transforms",
                preprocessing,
            ),

            pipeline_node(
                "🧠",
                "ML classifier",
                "trained model",
                validation,
            ),

            pipeline_node(
                "🌐",
                "Inference surface",
                "adversarial testing",
                adversarial,
            ),
        ]
    )

    findings_html = "".join(
        finding_card(finding)
        for finding in findings
    )

    if not findings_html:
        findings_html = """
        <div class="finding-card">
            No security findings were generated.
        </div>
        """

    return f"""
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
                        / 100 ·
                        {escape(severity)}
                        · static
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


        <div class="audit-tabs">

            <div class="audit-tab">
                Overview
            </div>

            <div class="audit-tab active">
                Pipeline & findings
            </div>

            <div class="audit-tab">
                Governance mapping
            </div>

            <div class="audit-tab">
                Full report
            </div>

        </div>


        <section class="audit-content">

            <aside class="pipeline-column">

                <div class="pipeline-heading">

                    <span>
                        Pipeline graph
                    </span>

                    <span>
                        4 nodes
                    </span>

                </div>

                <div class="pipeline-list">
                    {pipeline_html}
                </div>


                <div class="attacker-note">

                    <strong>
                        Attacker capability:
                    </strong>

                    can influence pipeline input data
                    and probe the model through
                    adversarial inputs. Flagged nodes
                    represent stages associated with
                    security findings.

                </div>

            </aside>


            <main>
                {findings_html}
            </main>

        </section>

    </div>
    """