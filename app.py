import io
import json
import re
from pathlib import Path
from typing import Any, Dict

import requests
import streamlit as st

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Preformatted,
)

from ui.dashboard import render_dashboard


API_URL = "http://127.0.0.1:8000"


def clean_executive_summary(summary: Any) -> str:
    """Keep the executive summary concise without repeating the V4 percentage."""
    text = str(summary or "")
    return re.sub(
        r", where the model demonstrated a \d+(?:\.\d+)?% attack success rate "
        r"within budget, indicating high susceptibility to adversarial attacks",
        ", where dynamic testing demonstrated high susceptibility to adversarial attacks",
        text,
        flags=re.IGNORECASE,
    )


# ---------------------------------------------------------
# PDF report generation
# ---------------------------------------------------------

def build_pdf_report(
    report: Dict[str, Any]
) -> bytes:
    """
    Generate the final AegisML security audit report
    as a real PDF document.
    """

    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="AegisML Security Audit Report",
        author="AegisML",
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "AegisTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        alignment=TA_CENTER,
        spaceAfter=18,
    )

    heading_style = ParagraphStyle(
        "AegisHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        spaceBefore=8,
        spaceAfter=8,
    )

    subheading_style = ParagraphStyle(
        "AegisSubheading",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        spaceBefore=6,
        spaceAfter=5,
    )

    body_style = ParagraphStyle(
        "AegisBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=14,
        spaceAfter=6,
    )

    small_style = ParagraphStyle(
        "AegisSmall",
        parent=body_style,
        fontSize=8.5,
        leading=12,
    )

    story = []

    # -----------------------------------------------------
    # Title
    # -----------------------------------------------------

    story.append(
        Paragraph(
            "AegisML Security Audit Report",
            title_style,
        )
    )

    story.append(
        Paragraph(
            "Multi-Agent ML Security Assessment",
            ParagraphStyle(
                "Subtitle",
                parent=body_style,
                alignment=TA_CENTER,
                fontSize=10,
                spaceAfter=18,
            ),
        )
    )

    overall = report.get(
        "overall_risk",
        {},
    )

    findings = report.get(
        "findings",
        [],
    )

    # -----------------------------------------------------
    # Overall Risk
    # -----------------------------------------------------

    story.append(
        Paragraph(
            "Overall Risk Assessment",
            heading_style,
        )
    )

    overall_score = float(
        overall.get(
            "overall_risk_score",
            0.0,
        )
    )

    overall_severity = str(
        overall.get(
            "overall_severity",
            "Low",
        )
    )

    overall_data = [
        [
            "Final Risk Score",
            f"{overall_score:.1f}/10",
        ],
        [
            "Final Severity",
            overall_severity,
        ],
        [
            "Total Findings",
            str(
                overall.get(
                    "total_findings",
                    len(findings),
                )
            ),
        ],
        [
            "Confirmed Findings",
            str(
                overall.get(
                    "confirmed_findings",
                    0,
                )
            ),
        ],
        [
            "False Positives",
            str(
                overall.get(
                    "false_positive_findings",
                    0,
                )
            ),
        ],
        [
            "Hidden Risks",
            str(
                overall.get(
                    "hidden_risk_findings",
                    0,
                )
            ),
        ],
        [
            "Unverified Findings",
            str(
                overall.get(
                    "unverified_findings",
                    0,
                )
            ),
        ],
    ]

    overall_table = Table(
        overall_data,
        colWidths=[
            65 * mm,
            85 * mm,
        ],
    )

    overall_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (0, -1),
                    colors.HexColor("#eeeeea"),
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (0, -1),
                    "Helvetica-Bold",
                ),
                (
                    "FONTNAME",
                    (1, 0),
                    (1, -1),
                    "Helvetica",
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    9,
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.HexColor("#cccccc"),
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
            ]
        )
    )

    story.append(overall_table)
    story.append(Spacer(1, 12))

    # -----------------------------------------------------
    # Executive Summary
    # -----------------------------------------------------

    story.append(
        Paragraph(
            "Executive Summary",
            heading_style,
        )
    )

    executive_summary = clean_executive_summary(
        report.get(
            "executive_summary",
            "No executive summary was generated.",
        )
    )

    story.append(
        Paragraph(
            str(executive_summary),
            body_style,
        )
    )

    story.append(Spacer(1, 8))

    # -----------------------------------------------------
    # Findings
    # -----------------------------------------------------

    story.append(
        Paragraph(
            "Security Findings",
            heading_style,
        )
    )

    if not findings:
        story.append(
            Paragraph(
                "No security findings were generated.",
                body_style,
            )
        )

    for index, finding in enumerate(
        findings,
        start=1,
    ):

        vulnerability_id = str(
            finding.get(
                "vulnerability_id",
                "Unknown",
            )
        )

        category = str(
            finding.get(
                "category",
                "Security Finding",
            )
        )

        story.append(
            Paragraph(
                f"{index}. {vulnerability_id} - {category}",
                subheading_style,
            )
        )

        static_score = float(
            finding.get(
                "static_risk_score",
                0.0,
            )
        )

        static_severity = str(
            finding.get(
                "static_severity",
                "N/A",
            )
        )

        test_status = str(
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

        correlation_status = str(
            finding.get(
                "correlation_status",
                "Unverified",
            )
        )

        final_score = float(
            finding.get(
                "final_risk_score",
                0.0,
            )
        )

        final_severity = str(
            finding.get(
                "final_severity",
                "Low",
            )
        )

        impact = float(
            finding.get(
                "impact",
                0.0,
            )
        )

        final_likelihood = float(
            finding.get(
                "final_likelihood",
                0.0,
            )
        )

        assessment_data = [
            [
                Paragraph(
                    "<b>Assessment</b>",
                    small_style,
                ),
                Paragraph(
                    "<b>Result</b>",
                    small_style,
                ),
            ],
            [
                "Agent 1 Theoretical Risk",
                (
                    f"{static_score:.1f}/10 "
                    f"({static_severity})"
                ),
            ],
            [
                "Agent 2 Empirical Status",
                test_status,
            ],
            [
                "Agent 2 Dynamic Severity",
                str(dynamic_severity),
            ],
            [
                "Agent 3 Correlation",
                correlation_status,
            ],
            [
                "Impact",
                f"{impact:.1f}/10",
            ],
            [
                "Evidence-Adjusted Likelihood",
                f"{final_likelihood:.1f}/10",
            ],
            [
                "Agent 3 Final Risk",
                (
                    f"{final_score:.1f}/10 "
                    f"({final_severity})"
                ),
            ],
        ]

        assessment_table = Table(
            assessment_data,
            colWidths=[
                70 * mm,
                80 * mm,
            ],
            repeatRows=1,
        )

        assessment_table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.HexColor("#30342c"),
                    ),
                    (
                        "TEXTCOLOR",
                        (0, 0),
                        (-1, 0),
                        colors.white,
                    ),
                    (
                        "BACKGROUND",
                        (0, 1),
                        (0, -1),
                        colors.HexColor("#f1f1ed"),
                    ),
                    (
                        "FONTNAME",
                        (0, 1),
                        (0, -1),
                        "Helvetica-Bold",
                    ),
                    (
                        "FONTSIZE",
                        (0, 0),
                        (-1, -1),
                        8.5,
                    ),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.4,
                        colors.HexColor("#cccccc"),
                    ),
                    (
                        "VALIGN",
                        (0, 0),
                        (-1, -1),
                        "TOP",
                    ),
                    (
                        "LEFTPADDING",
                        (0, 0),
                        (-1, -1),
                        6,
                    ),
                    (
                        "RIGHTPADDING",
                        (0, 0),
                        (-1, -1),
                        6,
                    ),
                    (
                        "TOPPADDING",
                        (0, 0),
                        (-1, -1),
                        5,
                    ),
                    (
                        "BOTTOMPADDING",
                        (0, 0),
                        (-1, -1),
                        5,
                    ),
                ]
            )
        )

        story.append(assessment_table)
        story.append(Spacer(1, 7))

        description = str(
            finding.get(
                "description",
                "",
            )
        )

        if description:
            story.append(
                Paragraph(
                    "<b>Description</b>",
                    small_style,
                )
            )

            story.append(
                Paragraph(
                    description,
                    body_style,
                )
            )

        correlation_rationale = str(
            finding.get(
                "correlation_rationale",
                "",
            )
        )

        if correlation_rationale:
            story.append(
                Paragraph(
                    "<b>Correlation Analysis</b>",
                    small_style,
                )
            )

            story.append(
                Paragraph(
                    correlation_rationale,
                    body_style,
                )
            )

        risk_rationale = str(
            finding.get(
                "risk_rationale",
                "",
            )
        )

        if risk_rationale:
            story.append(
                Paragraph(
                    "<b>Final Risk Rationale</b>",
                    small_style,
                )
            )

            story.append(
                Paragraph(
                    risk_rationale,
                    body_style,
                )
            )

        evidence = finding.get(
            "evidence",
            {},
        )

        if evidence:
            story.append(
                Paragraph(
                    "<b>Dynamic Evidence</b>",
                    small_style,
                )
            )

            evidence_style = ParagraphStyle(
                "EvidenceText",
                parent=small_style,
                fontName="Courier",
                fontSize=7.5,
                leading=10,
                leftIndent=6,
                rightIndent=6,
                spaceAfter=7,
            )

            evidence_text = json.dumps(
                evidence,
                indent=2,
                ensure_ascii=True,
                default=str,
            )

            story.append(
                Preformatted(
                    evidence_text,
                    evidence_style,
                    maxLineLength=90,
                )
            )

        affected_components = finding.get(
            "affected_components",
            [],
        )

        if affected_components:
            components_text = ", ".join(
                str(component)
                for component
                in affected_components
            )

            story.append(
                Paragraph(
                    "<b>Affected Components</b>",
                    small_style,
                )
            )

            story.append(
                Paragraph(
                    components_text,
                    body_style,
                )
            )

        recommendations = finding.get(
            "recommendations",
            [],
        )

        if recommendations:
            story.append(
                Paragraph(
                    "<b>Recommendations</b>",
                    small_style,
                )
            )

            for recommendation in recommendations:
                story.append(
                    Paragraph(
                        f"• {recommendation}",
                        body_style,
                    )
                )

        if index < len(findings):
            story.append(Spacer(1, 10))

    # -----------------------------------------------------
    # Overall Recommendations
    # -----------------------------------------------------

    overall_recommendations = report.get(
        "recommendations",
        [],
    )

    if overall_recommendations:
        story.append(PageBreak())

        story.append(
            Paragraph(
                "Overall Recommendations",
                heading_style,
            )
        )

        for recommendation in overall_recommendations:
            story.append(
                Paragraph(
                    f"• {recommendation}",
                    body_style,
                )
            )

    # -----------------------------------------------------
    # Assessment methodology
    # -----------------------------------------------------

    story.append(Spacer(1, 12))

    story.append(
        Paragraph(
            "Assessment Methodology",
            heading_style,
        )
    )

    story.append(
        Paragraph(
            (
                "Agent 1 provides the theoretical/static "
                "threat context. Agent 2 performs empirical "
                "security testing. Agent 3 correlates both "
                "sources, identifies discrepancies such as "
                "false positives and hidden risks, and "
                "calculates the final evidence-informed "
                "risk assessment."
            ),
            body_style,
        )
    )

    document.build(story)

    pdf_bytes = buffer.getvalue()
    buffer.close()

    return pdf_bytes


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="AegisML",
    page_icon="icon.png",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------
# Global Streamlit styling
# ---------------------------------------------------------

st.markdown(
    """
    <style>

    .stApp {
        background: #f4f4f0;
    }

    .block-container {
        max-width: 1420px;
        padding-top: 0.5rem;
        padding-left: 3rem;
        padding-right: 3rem;
        padding-bottom: 3rem;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    div[data-testid="stFileUploader"] {
        background: transparent;
    }

    .stButton > button {
        min-height: 45px;
        border-radius: 8px;
        font-weight: 600;
    }

    .stButton > button[kind="primary"] {
        border: none;
        background: #30342c;
        color: #fdfdfc;
        font-size: 15px;
        font-weight: 650;
        letter-spacing: 0.3px;
        min-height: 48px;
        box-shadow: 0 2px 6px rgba(48, 52, 44, 0.12);
        transition: all 0.2s ease;
    }

    .stButton > button[kind="primary"]:hover {
        background: #1f221c;
        color: #ffffff;
        box-shadow: 0 4px 14px rgba(48, 52, 44, 0.2);
    }

    .stDownloadButton > button {
        min-height: 44px;
        border-radius: 8px;
        background: #30342c;
        color: white;
        border: none;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Load dashboard stylesheet
# ---------------------------------------------------------

css_path = (
    Path(__file__).parent
    / "ui"
    / "styles.css"
)

css = css_path.read_text(
    encoding="utf-8"
)

st.html(
    f"<style>{css}</style>"
)


# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------

if "audit_result" not in st.session_state:
    st.session_state.audit_result = None


# =========================================================
# UPLOAD SCREEN
# =========================================================

if st.session_state.audit_result is None:

    # -----------------------------------------------------
    # Upload screen header
    # -----------------------------------------------------

    st.html(
        """
        <div class="upload-hero">
            <div class="upload-hero-left">
                <div class="upload-brand-row">
                    <div class="upload-logo">A</div>
                    <h1 class="upload-hero-title">AegisML Pipeline Audit</h1>
                </div>
                <p class="upload-hero-subtitle">
                    Upload machine learning assets to run static threat analysis,
                    dynamic security testing, and automated security reporting.
                </p>
            </div>
            <div class="upload-status-badge">
                <span class="upload-status-dot"></span>
                STANDBY &middot; READY TO AUDIT
            </div>
        </div>
        """
    )

    # -----------------------------------------------------
    # File uploads
    # -----------------------------------------------------

    col1, col2, col3 = st.columns(
        3,
        gap="medium",
    )

    with col1:
        with st.container(border=True):
            st.html(
                """
                <div class="asset-card-head">
                    <div class="asset-card-title-wrap">
                        <div class="asset-icon-box" style="font-family: monospace; font-size: 15px;">{ }</div>
                        <div class="asset-title">Pipeline Source</div>
                    </div>
                    <span class="asset-target-tag tag-sast">AGENT 1 &middot; SAST</span>
                </div>
                <div class="asset-desc">
                    Python ML pipeline source script defining component hierarchy, data transformations, and model definitions.
                </div>
                """
            )
            pipeline_file = st.file_uploader(
                "Pipeline source",
                type=["py"],
                help="Python ML pipeline source file.",
                label_visibility="collapsed",
            )

    with col2:
        with st.container(border=True):
            st.html(
                """
                <div class="asset-card-head">
                    <div class="asset-card-title-wrap">
                        <div class="asset-icon-box" style="font-family: monospace; font-size: 12px; font-weight: 700;">PKL</div>
                        <div class="asset-title">Trained Model</div>
                    </div>
                    <span class="asset-target-tag tag-dast">AGENT 2 &middot; DAST</span>
                </div>
                <div class="asset-desc">
                    Pickle serialized model artifact executed within an isolated environment for dynamic adversarial attacks.
                </div>
                """
            )
            model_file = st.file_uploader(
                "Trained model",
                type=["pkl"],
                help="Pickle model generated by the ML pipeline.",
                label_visibility="collapsed",
            )

    with col3:
        with st.container(border=True):
            st.html(
                """
                <div class="asset-card-head">
                    <div class="asset-card-title-wrap">
                        <div class="asset-icon-box" style="font-family: monospace; font-size: 12px; font-weight: 700;">CSV</div>
                        <div class="asset-title">Evaluation Dataset</div>
                    </div>
                    <span class="asset-target-tag tag-data">TEST FIXTURE</span>
                </div>
                <div class="asset-desc">
                    Evaluation dataset in tabular CSV format containing samples used to benchmark model accuracy and evasion.
                </div>
                """
            )
            dataset_file = st.file_uploader(
                "Dataset",
                type=["csv"],
                help="CSV dataset used by the model.",
                label_visibility="collapsed",
            )

    st.write("")

    # -----------------------------------------------------
    # Dataset settings (Collapsible Advanced Configuration)
    # -----------------------------------------------------

    with st.expander("Advanced Configuration · Schema Mapping", expanded=False):
        st.caption(
            "Specify dataset column names if your CSV does not use the default "
            "'text' and 'label' headers."
        )
        setting1, setting2 = st.columns(2)

        with setting1:
            text_column = st.text_input(
                "Feature / Text Column",
                value="text",
                help="CSV column containing model input samples.",
            )

        with setting2:
            label_column = st.text_input(
                "Ground Truth / Label Column",
                value="label",
                help="CSV column containing class labels.",
            )

    st.write("")

    # -----------------------------------------------------
    # Run audit
    # -----------------------------------------------------

    if st.button(
        "Run security audit  →",
        type="primary",
        use_container_width=True,
    ):

        if not all(
            [
                pipeline_file,
                model_file,
                dataset_file,
            ]
        ):

            st.warning(
                "Please upload the pipeline, model, "
                "and dataset."
            )

        else:

            files = {
                "pipeline_file": (
                    pipeline_file.name,
                    pipeline_file.getvalue(),
                    "text/x-python",
                ),

                "model_file": (
                    model_file.name,
                    model_file.getvalue(),
                    "application/octet-stream",
                ),

                "dataset_file": (
                    dataset_file.name,
                    dataset_file.getvalue(),
                    "text/csv",
                ),
            }

            data = {
                "text_column": text_column,
                "label_column": label_column,
            }

            try:

                with st.spinner(
                    "AegisML is analysing the ML pipeline..."
                ):

                    response = requests.post(
                        f"{API_URL}/audit",
                        files=files,
                        data=data,
                        timeout=600,
                    )

                if response.status_code != 200:

                    try:

                        detail = response.json().get(
                            "detail",
                            "Audit failed.",
                        )

                    except Exception:

                        detail = response.text

                    st.error(detail)

                else:

                    st.session_state.audit_result = (
                        response.json()
                    )

                    st.rerun()

            except requests.exceptions.ConnectionError:

                st.error(
                    "Could not connect to FastAPI. "
                    "Make sure the API server is running."
                )

            except requests.exceptions.Timeout:

                st.error(
                    "The audit exceeded the request timeout."
                )

            except requests.exceptions.RequestException as exc:

                st.error(
                    f"API error: {exc}"
                )

    # -----------------------------------------------------
    # Pipeline workflow preview strip
    # -----------------------------------------------------

    st.html(
        """
        <div class="pipeline-workflow-strip">
            <div class="workflow-step-card">
                <div class="workflow-step-badge">Phase 01 &middot; SAST</div>
                <div class="workflow-step-name">Static Threat Modeling</div>
                <div class="workflow-step-desc">
                    AST decomposition, component extraction, and qualitative NIST AI 100-2 threat modeling across pipeline nodes.
                </div>
            </div>
            <div class="workflow-step-card">
                <div class="workflow-step-badge">Phase 02 &middot; DAST</div>
                <div class="workflow-step-name">Adversarial Testing Sandbox</div>
                <div class="workflow-step-desc">
                    Docker-isolated dynamic verification executing FGSM, PGD, HopSkipJump, and TextFooler attacks.
                </div>
            </div>
            <div class="workflow-step-card">
                <div class="workflow-step-badge">Phase 03 &middot; Reporting</div>
                <div class="workflow-step-name">Evidence-Informed Scoring</div>
                <div class="workflow-step-desc">
                    Empirical verification synthesis, theoretical baseline vs. final risk calculation, and executive reporting.
                </div>
            </div>
        </div>
        """
    )


# =========================================================
# RESULTS DASHBOARD
# =========================================================

else:

    result = st.session_state.audit_result

    report = result.get(
        "report",
        {},
    )

    findings = report.get(
        "findings",
        [],
    )

    overall = report.get(
        "overall_risk",
        {},
    )

    # Render the dashboard header and risk summary first.
    st.html(
        render_dashboard(
            result,
            section="summary",
        )
    )

    if "active_report_section" not in st.session_state:
        st.session_state.active_report_section = "Pipeline & findings"

    tab_names = [
        "Overview",
        "Pipeline & findings",
        "Governance mapping",
        "Full report",
    ]

    tab_cols = st.columns(
        [1.0, 1.55, 1.55, 1.0, 5.0],
        gap="small",
    )

    st.markdown(
        """
        <style>
        .aegis-tabs-anchor {
            display: none;
        }

        div[data-testid="stHorizontalBlock"]:has(.aegis-tabs-anchor) {
            gap: 0 !important;
            border-bottom: 1px solid #dedfd9;
        }

        div[data-testid="stHorizontalBlock"]:has(.aegis-tabs-anchor)
        .stButton > button {
            border: 0 !important;
            border-radius: 0 !important;
            background: transparent !important;
            box-shadow: none !important;
            color: #777b73 !important;
            min-height: 48px !important;
            font-weight: 500 !important;
            border-bottom: 2px solid transparent !important;
        }

        div[data-testid="stHorizontalBlock"]:has(.aegis-tabs-anchor)
        .stButton > button:hover {
            color: #30342c !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with tab_cols[4]:
        st.markdown(
            '<span class="aegis-tabs-anchor"></span>',
            unsafe_allow_html=True,
        )

    for tab_col, tab_name in zip(
        tab_cols[:4],
        tab_names,
    ):
        with tab_col:
            if st.button(
                tab_name,
                key=f"report_tab_{tab_name}",
            ):
                st.session_state.active_report_section = tab_name
                st.rerun()

    active_section = st.session_state.active_report_section
    active_index = tab_names.index(active_section) + 1

    st.markdown(
        f"""
        <style>
        div[data-testid="stHorizontalBlock"]:has(.aegis-tabs-anchor)
        > div:nth-child({active_index}) .stButton > button {{
            color: #30342c !important;
            border-bottom: 2px solid #4f8060 !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    if active_section == "Overview":

        overall_score = float(
            overall.get(
                "overall_risk_score",
                0.0,
            )
        )

        overall_severity = str(
            overall.get(
                "overall_severity",
                "Low",
            )
        )

        confirmed = int(
            overall.get(
                "confirmed_findings",
                0,
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

        summary = clean_executive_summary(
            report.get(
                "executive_summary",
                "",
            )
        )

        st.markdown("## Overview")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Final risk",
            f"{overall_score:.1f}/10",
        )

        c2.metric(
            "Severity",
            overall_severity,
        )

        c3.metric(
            "Confirmed findings",
            confirmed,
        )

        c4.metric(
            "Total findings",
            len(findings),
        )

        st.markdown(summary)

        st.caption(
            f"False positives: {false_positives} · "
            f"Hidden risks: {hidden_risks}"
        )

    elif active_section == "Pipeline & findings":

        # Only render the dashboard content here.
        # The header/summary was already rendered above the tabs.
        st.html(
            render_dashboard(
                result,
                section="content",
            )
        )

    elif active_section == "Governance mapping":

        st.markdown("## Governance mapping")

        st.markdown(
            "AegisML uses a **NIST-aligned risk assessment methodology**. "
            "Agent 1 provides theoretical/static threat context, "
            "Agent 2 performs empirical security testing, and "
            "Agent 3 correlates both sources to produce the final "
            "evidence-informed risk assessment."
        )

        st.caption(
            "This view summarizes assessment evidence and correlation "
            "results; it does not claim certification or introduce "
            "controls that were not assessed."
        )

        governance_rows = []

        for finding in findings:
            governance_rows.append(
                {
                    "ID": finding.get(
                        "vulnerability_id",
                        "Unknown",
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
                    "Final severity": finding.get(
                        "final_severity",
                        "Low",
                    ),
                }
            )

        if governance_rows:
            st.dataframe(
                governance_rows,
                use_container_width=True,
                hide_index=True,
            )

        st.markdown(
            "**Assessment flow:** "
            "Agent 1 → Agent 2 → Agent 3 → Final risk assessment"
        )

    else:

        st.markdown("## Full report")

        st.write(
            "Download the comprehensive PDF containing the overall risk "
            "assessment, executive summary, detailed findings, dynamic "
            "evidence, correlation results, and recommendations."
        )


    # -----------------------------------------------------
    # Bottom actions
    # -----------------------------------------------------

    st.divider()

    footer_info, spacer, rescan_col, download_col = (
        st.columns(
            [5, 1, 1.1, 1.7]
        )
    )

    with footer_info:

        st.caption(
            f"Scanned 3 files · "
            f"{len(findings)} findings · "
            "AegisML security assessment"
        )

    with rescan_col:

        if st.button(
            "Re-scan",
            use_container_width=True,
        ):

            st.session_state.audit_result = None

            st.rerun()

    with download_col:

        pdf_report = build_pdf_report(
            report
        )

        st.download_button(
            "Download full report",
            data=pdf_report,
            file_name=(
                "AegisML_Security_Audit_Report.pdf"
            ),
            mime="application/pdf",
            use_container_width=True,
        )