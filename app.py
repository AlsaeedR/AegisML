import io
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
import queue
import threading
import time

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

from ui.dashboard import (
    render_dashboard,
    render_interactive_pipeline_graph,
    render_attack_strategy_gate,
    render_report_signoff_gate,
    render_audit_ledger_panel,
)


API_URL = "http://127.0.0.1:8000"


# ---------------------------------------------------------
# Browser-persistent audit recovery
# ---------------------------------------------------------

def remember_audit_resume_state(
    audit_id: str,
    phase: str,
) -> None:
    """
    Persist only the audit identifier and workflow phase in the URL.

    The actual audit state remains in the backend persistent memory.
    No model, dataset, report, or security evidence is stored in the URL.
    """
    if not audit_id:
        return

    st.query_params["audit_id"] = audit_id
    st.query_params["audit_phase"] = phase


def clear_audit_resume_state() -> None:
    """Remove the browser-side pointer to a previous audit."""
    for key in ("audit_id", "audit_phase"):
        if key in st.query_params:
            del st.query_params[key]


def get_audit_resume_state() -> tuple[str | None, str | None]:
    """Read the persistent audit pointer from the current page URL."""
    audit_id = st.query_params.get("audit_id")
    phase = st.query_params.get("audit_phase")

    if isinstance(audit_id, list):
        audit_id = audit_id[0] if audit_id else None

    if isinstance(phase, list):
        phase = phase[0] if phase else None

    return audit_id, phase


def is_final_audit_payload(
    payload: Dict[str, Any],
) -> bool:
    """
    Only a completed correlated report may be rendered as final results.

    Partial/interrupted backend responses must never reach the dashboard.
    """
    if not isinstance(payload, dict):
        return False

    status = str(payload.get("status", "")).lower()
    if status not in {"awaiting_gate_2", "completed", "signed_off"}:
        return False

    report = payload.get("report")

    if not isinstance(report, dict) or not report:
        return False

    if not isinstance(report.get("findings"), list):
        return False

    if not isinstance(report.get("overall_risk"), dict):
        return False

    return True


def get_saved_audit_state(
    audit_id: str,
) -> requests.Response:
    """Read the backend checkpoint without executing any agent."""
    return requests.get(
        f"{API_URL}/audit/{audit_id}",
        timeout=10,
    )


def check_docker_available() -> bool:
    """
    Checks if Docker daemon is active and responsive.
    Queries the FastAPI backend endpoint first; falls back to local subprocess if offline.
    """
    try:
        resp = requests.get(f"{API_URL}/system/docker-status", timeout=2)
        if resp.status_code == 200:
            return bool(resp.json().get("docker_available", False))
    except Exception:
        pass

    try:
        from src.agents.testing_agent.sandbox_runner import is_docker_available
        return is_docker_available()
    except Exception:
        return False


def dismiss_docker_dialog():
    st.session_state.show_docker_dialog = False


@st.dialog("Docker Required for Security Audit", width="medium", on_dismiss=dismiss_docker_dialog)
def show_docker_unavailable_dialog():
    st.error(
        "AegisML cannot run a security audit because the Docker daemon is offline or unreachable."
    )
    st.markdown(
        """
        **Zero-Trust Isolation Requirement:**
        * **Container Isolation**: AegisML enforces strict container sandboxing to prevent untrusted 
          machine learning code, model deserialization (pickle / joblib), and dynamic attacks from running 
          on the host system.
        * **Host Protection**: Running unverified dynamic penetration tests outside of Docker risks arbitrary 
          code execution (RCE) and system compromise.

        **To proceed:**
        1. Open and start **Docker Desktop** (or start the Docker service).
        2. Wait until the Docker engine reports that it is running.
        3. Click **Re-check Docker** below to continue, or **Cancel** to abort.
        """
    )

    dialog_col1, dialog_col2 = st.columns(2, gap="medium")
    with dialog_col1:
        if st.button("Re-check Docker", type="primary", use_container_width=True, key="dialog_recheck_btn"):
            if check_docker_available():
                st.session_state.show_docker_dialog = False
                st.success("Docker daemon detected. Ready to proceed.")
                time.sleep(0.4)
                st.rerun()
            else:
                st.error("Docker daemon is still not reachable. Please verify Docker Desktop is running.")

    with dialog_col2:
        if st.button("Cancel", use_container_width=True, key="dialog_cancel_btn"):
            st.session_state.show_docker_dialog = False
            st.rerun()


def resume_saved_audit_once(
    audit_id: str,
    progress_placeholder: Any,
) -> requests.Response:
    """
    Resume one persisted audit exactly once.

    The backend execution lock prevents duplicate execution.
    Partial results are never treated as final.
    """
    state_response = requests.get(
        f"{API_URL}/audit/{audit_id}",
        timeout=10,
    )

    if state_response.status_code != 200:
        return state_response

    state_payload = state_response.json()

    if is_final_audit_payload(
        state_payload
    ):
        return state_response

    if state_payload.get(
        "execution_active",
        False,
    ):
        response = requests.Response()
        response.status_code = 409
        response._content = (
            b'{"detail":"This audit is already executing. '
            b'Wait for it to finish, then press Resume audit again."}'
        )
        response.headers["Content-Type"] = "application/json"
        return response

    return run_approved_audit_with_progress(
        audit_id,
        progress_placeholder,
    )


def restore_saved_report(
    audit_id: str,
    max_wait_seconds: int = 120,
) -> requests.Response:
    """
    Restore an already-completed audit result.

    This endpoint is read-only: it never re-runs Agent 1, Agent 2,
    Agent 3, or any dynamic security test. If the API is temporarily
    unavailable or restarting, retry until it becomes reachable.
    """
    deadline = time.time() + max_wait_seconds
    last_error = None

    while time.time() < deadline:
        try:
            response = requests.get(
                f"{API_URL}/audit/{audit_id}",
                timeout=5,
            )

            if response.status_code in {
                200,
                404,
            }:
                return response

            last_error = (
                f"HTTP {response.status_code}: "
                f"{response.text}"
            )

        except requests.exceptions.RequestException as exc:
            last_error = str(exc)

        time.sleep(2)

    raise requests.exceptions.Timeout(
        "The API did not become available in time. "
        f"Last detail: {last_error or 'service unavailable'}"
    )


def resume_previous_audit(
    audit_id: str,
    progress_placeholder: Any,
) -> requests.Response:
    """
    Resume an interrupted audit that has not produced a final report yet.

    The backend uses persistent checkpoints to skip completed steps.
    """
    return run_approved_audit_with_progress(
        audit_id,
        progress_placeholder,
    )


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

def render_agent_progress(
    progress_placeholder: Any,
    events: list[Dict[str, Any]],
) -> None:
    """Render the current multi-agent execution timeline in place."""
    stages = [
        ("Agent 1", "Static threat analysis", "Complete"),
        ("Agent 2", "Dynamic security testing", "Waiting"),
        ("Agent 2", "Forensic diagnosis", "Waiting"),
        ("Agent 3", "Evidence-informed reporting", "Waiting"),
    ]

    latest_by_step = {
        event.get("step"): event
        for event in events
        if event.get("step")
    }

    rows = []
    for agent, step, default_status in stages:
        event = latest_by_step.get(step, {})
        event_type = event.get("event")

        if event_type == "agent_step_started":
            status = "Running"
            marker = "running"
        elif event_type == "agent_step_finished":
            status = str(
                event.get(
                    "status",
                    "Complete",
                )
            ).replace(
                "_",
                " ",
            ).title()
            marker = "complete"
        elif default_status == "Complete":
            status = "Complete"
            marker = "complete"
        else:
            status = default_status
            marker = "waiting"

        message = event.get(
            "message",
            "",
        )

        rows.append(
            f"""
            <div class="agent-progress-row">
                <span class="agent-progress-marker {marker}"></span>
                <div class="agent-progress-copy">
                    <div class="agent-progress-step"><strong>{agent}</strong> · {step}</div>
                    <div class="agent-progress-message">{message or status}</div>
                </div>
                <span class="agent-progress-status {marker}">{status}</span>
            </div>
            """
        )

    progress_placeholder.html(
        """
        <div class="agent-progress-panel">
            <div class="agent-progress-heading">Audit execution</div>
            <div class="agent-progress-subheading">Live agent activity</div>
            {rows}
        </div>
        """.format(rows="".join(rows)),
    )


def run_approved_audit_with_progress(
    audit_id: str,
    progress_placeholder: Any,
    selected_tests: Optional[List[str]] = None,
) -> requests.Response:
    """Execute the approved audit while streaming agent telemetry to the UI."""
    event_queue: queue.Queue[Dict[str, Any]] = queue.Queue()
    response_holder: Dict[str, Any] = {}
    events: list[Dict[str, Any]] = []

    def read_telemetry() -> None:
        try:
            with requests.get(
                f"{API_URL}/audit/{audit_id}/telemetry/stream",
                stream=True,
                timeout=(10, 900),
            ) as stream_response:
                stream_response.raise_for_status()
                for raw_line in stream_response.iter_lines(decode_unicode=True):
                    if not raw_line or not raw_line.startswith("data: "):
                        continue
                    event_queue.put(json.loads(raw_line[6:]))
        except Exception as exc:
            event_queue.put({"event": "telemetry_error", "message": str(exc)})

    def execute_audit() -> None:
        try:
            payload_data: Dict[str, Any] = {"audit_id": audit_id}
            if selected_tests:
                payload_data["selected_tests"] = json.dumps(selected_tests)
            response_holder["response"] = requests.post(
                f"{API_URL}/audit/execute",
                data=payload_data,
                timeout=900,
            )
        except Exception as exc:
            response_holder["error"] = exc

    telemetry_thread = threading.Thread(target=read_telemetry, daemon=True)
    execute_thread = threading.Thread(target=execute_audit, daemon=True)
    telemetry_thread.start()
    execute_thread.start()

    render_agent_progress(progress_placeholder, events)
    while execute_thread.is_alive() or not event_queue.empty():
        try:
            while True:
                event = event_queue.get_nowait()
                if event.get("event") != "done":
                    events.append(event)
                render_agent_progress(progress_placeholder, events)
        except queue.Empty:
            pass
        time.sleep(0.15)

    execute_thread.join()
    while not event_queue.empty():
        event = event_queue.get_nowait()
        if event.get("event") != "done":
            events.append(event)
    render_agent_progress(progress_placeholder, events)

    if "error" in response_holder:
        raise response_holder["error"]
    return response_holder["response"]

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
                "risk_score",
                finding.get("final_risk_score", 0.0),
            )
        )

        final_severity = str(
            finding.get(
                "severity",
                finding.get("final_severity", "Low"),
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

        is_false_positive = "false positive" in correlation_status.lower() or (
            final_severity.lower() == "low" and str(finding.get("test_status", "")).lower() == "not_vulnerable"
        )
        is_confirmed = "confirmed" in correlation_status.lower() or final_severity.lower() in ["critical", "high"]

        if is_false_positive:
            desc_label = "Theoretical Concern (Static SAST)"
            recs_label = "Verification Outcome (Dynamic DAST)"
        elif is_confirmed:
            desc_label = "Root Cause & Description"
            recs_label = "Suggested Fix & Remediation"
        else:
            desc_label = "Static Observation"
            recs_label = "Hardening Guidance"

        if description:
            story.append(
                Paragraph(
                    f"<b>{desc_label}</b>",
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
                    f"<b>{recs_label}</b>",
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

    /* -----------------------------------------------------
       Streamlit top-right controls spacing
       ----------------------------------------------------- */

    /* Keep Stop / Deploy visible, but give the toolbar
       more breathing room so the controls do not look
       crowded together. */
    div[data-testid="stToolbar"] {
        gap: 1.25rem !important;
        padding-right: 0.75rem !important;
    }

    div[data-testid="stToolbar"] > div {
        gap: 1rem !important;
    }

    /* Add separation between Streamlit's native toolbar
       and AegisML's status badge below it. */
    .upload-hero {
        padding-top: 1.25rem;
    }

    .upload-status-badge {
        margin-top: 0.65rem;
        margin-right: 0.15rem;
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
if "audit_plan" not in st.session_state:
    st.session_state.audit_plan = None
if "audit_id" not in st.session_state:
    st.session_state.audit_id = None
if "report_signed_off" not in st.session_state:
    st.session_state.report_signed_off = False
if "show_docker_dialog" not in st.session_state:
    st.session_state.show_docker_dialog = False
if "audit_executing" not in st.session_state:
    st.session_state.audit_executing = False

if st.session_state.get("show_docker_dialog", False):
    show_docker_unavailable_dialog()

recoverable_audit_id, recoverable_audit_phase = get_audit_resume_state()

# Restore the audit identifier after a Streamlit/browser rerun.
# The security state itself is restored by FastAPI from persistent memory.
if (
    st.session_state.audit_id is None
    and recoverable_audit_id
):
    st.session_state.audit_id = recoverable_audit_id

# Automatic checkpoint restoration across browser refresh
if (
    st.session_state.audit_plan is None
    and st.session_state.audit_result is None
    and recoverable_audit_id
):
    try:
        saved_resp = requests.get(f"{API_URL}/audit/{recoverable_audit_id}", timeout=5)
        if saved_resp.status_code == 200:
            saved_data = saved_resp.json()
            if is_final_audit_payload(saved_data):
                st.session_state.audit_result = saved_data
                st.session_state.audit_id = recoverable_audit_id
                st.session_state.audit_result["checkpoint_loaded"] = True
            elif saved_data.get("status") == "awaiting_gate_1" or "attack_strategy_plan" in saved_data:
                st.session_state.audit_plan = saved_data
                st.session_state.audit_id = recoverable_audit_id
                st.session_state.audit_plan["checkpoint_loaded"] = True
    except Exception:
        pass


# =========================================================
# GATE 1 - HUMAN ATTACK STRATEGY APPROVAL
# =========================================================

if st.session_state.audit_result is None and st.session_state.audit_plan is not None:
    is_executing = st.session_state.get("audit_executing", False)

    if not is_executing:
        is_ckpt = st.session_state.audit_plan.get("checkpoint_loaded", False)
        gate_badge = "GATE 1 &middot; CHECKPOINT LOADED" if is_ckpt else "GATE 1 &middot; APPROVAL REQUIRED"

        st.html(
            f"""
            <div class="upload-hero">
                <div class="upload-hero-left">
                    <div class="upload-brand-row">
                        <div class="upload-logo">A</div>
                        <h1 class="upload-hero-title">Human-in-the-Loop Review</h1>
                    </div>
                    <p class="upload-hero-subtitle">
                        Agent 1 analysis is complete. Review Agent 2's proposed attack strategy before any dynamic tests execute.
                    </p>
                </div>
                <div class="upload-status-badge">{gate_badge}</div>
            </div>
            """
        )

        approved = render_attack_strategy_gate(st.session_state.audit_plan)

        if approved:
            if not check_docker_available():
                st.session_state.show_docker_dialog = True
                st.rerun()

            st.session_state.audit_executing = True
            remember_audit_resume_state(
                st.session_state.audit_id,
                "executing",
            )
            st.rerun()

        st.stop()

    # ---------------------------------------------------------
    # Execution phase: Human-in-the-Loop banner is bypassed completely
    # ---------------------------------------------------------
    st.html(
        """
        <div class="upload-hero">
            <div class="upload-hero-left">
                <div class="upload-brand-row">
                    <div class="upload-logo">A</div>
                    <h1 class="upload-hero-title">Executing Security Audit</h1>
                </div>
                <p class="upload-hero-subtitle">
                    Gate 1 approved. Running dynamic security tests and generating evidence-informed report...
                </p>
            </div>
            <div class="upload-status-badge">DYNAMIC TESTS ACTIVE</div>
        </div>
        """
    )

    progress_placeholder = st.empty()

    selected_tests = None
    if isinstance(st.session_state.audit_plan, dict):
        strat_plan = st.session_state.audit_plan.get("attack_strategy_plan") or {}
        selected_tests = strat_plan.get("selected_tests") or st.session_state.audit_plan.get("selected_tests")

    try:
        with st.spinner(
            "Running approved dynamic tests and evidence correlation..."
        ):
            response = run_approved_audit_with_progress(
                st.session_state.audit_id,
                progress_placeholder,
                selected_tests=selected_tests,
            )

        if response.status_code == 200:
            payload = response.json()

            if is_final_audit_payload(
                payload
            ):
                st.session_state.audit_result = payload
                st.session_state.audit_plan = None
                st.session_state.audit_executing = False
                st.session_state.report_signed_off = False

                remember_audit_resume_state(
                    st.session_state.audit_id,
                    "report",
                )

                st.rerun()

            else:
                st.warning(
                    "The audit stopped before the final report was ready. "
                    "No partial results were published. "
                    "Your checkpoint is saved."
                )
                st.session_state.audit_plan = None
                st.session_state.audit_executing = False
                st.rerun()

        elif response.status_code == 409:
            st.info(
                "This audit is already running. "
                "Wait for it to finish, then use Resume audit."
            )
            st.session_state.audit_plan = None
            st.session_state.audit_executing = False
            st.rerun()

        else:
            st.warning(
                "The audit was interrupted before completion. "
                "No partial results were published. "
                "Your checkpoint is saved."
            )
            st.session_state.audit_plan = None
            st.session_state.audit_executing = False
            st.rerun()

    except requests.exceptions.RequestException:
        st.warning(
            "The API connection was interrupted. "
            "No partial results were published. "
            "Your checkpoint is saved."
        )
        st.session_state.audit_plan = None
        st.session_state.audit_executing = False
        st.rerun()

    st.stop()


# =========================================================
# UPLOAD SCREEN
# =========================================================

if st.session_state.audit_result is None:

    # -----------------------------------------------------
    # Resume an interrupted approved audit
    # -----------------------------------------------------

    if (
        st.session_state.audit_plan is None
        and recoverable_audit_id
        and recoverable_audit_phase in {"executing", "report"}
    ):
        with st.container(border=True):
            st.markdown("### Previous audit detected")
            if recoverable_audit_phase == "report":
                st.write(
                    "AegisML found a completed saved audit. Restore the "
                    "persisted report without re-running any security test."
                )
            else:
                st.write(
                    "AegisML found a saved audit session. "
                    "Resume it safely from the latest completed checkpoint."
                )

            resume_info, resume_action = st.columns(
                [4, 1.35],
                gap="medium",
            )

            with resume_info:
                st.caption(
                    f"Audit ID: {recoverable_audit_id[:12]}… · "
                    f"Saved phase: {recoverable_audit_phase}"
                )

            with resume_action:
                action_label = (
                    "Restore report"
                    if recoverable_audit_phase == "report"
                    else "Resume audit"
                )

                if st.button(
                    action_label,
                    type="primary",
                    use_container_width=True,
                    key="resume_previous_audit",
                ):
                    if action_label == "Resume audit" and not check_docker_available():
                        st.session_state.show_docker_dialog = True
                        st.rerun()

                    try:
                        st.session_state.audit_id = recoverable_audit_id

                        if recoverable_audit_phase == "report":
                            response = restore_saved_report(
                                recoverable_audit_id
                            )
                        else:
                            progress_placeholder = st.empty()

                            with st.spinner(
                                "Resuming from the latest saved checkpoint..."
                            ):
                                response = resume_saved_audit_once(
                                    recoverable_audit_id,
                                    progress_placeholder,
                                )

                        if response.status_code == 200:
                            payload = response.json()

                            if is_final_audit_payload(
                                payload
                            ):
                                st.session_state.audit_result = payload
                                st.session_state.audit_plan = None
                                st.session_state.report_signed_off = False

                                remember_audit_resume_state(
                                    recoverable_audit_id,
                                    "report",
                                )

                                st.rerun()

                            else:
                                # A stale "report" URL can exist if shutdown
                                # happened before the report was fully persisted.
                                remember_audit_resume_state(
                                    recoverable_audit_id,
                                    "executing",
                                )
                                st.info(
                                    "The audit is saved but not complete yet. "
                                    "Press Resume audit to continue from the "
                                    "latest checkpoint."
                                )

                        elif response.status_code == 409:
                            st.info(
                                "This audit is already running. "
                                "Wait for it to finish, then press Resume audit again."
                            )

                        else:
                            try:
                                detail = response.json().get(
                                    "detail",
                                    "Could not restore the saved audit.",
                                )
                            except Exception:
                                detail = response.text

                            st.error(detail)

                    except requests.exceptions.RequestException:
                        st.warning(
                            "The API is unavailable right now. "
                            "Your checkpoint is still saved. "
                            "Start the API, then press Resume audit again."
                        )

            if st.button(
                "Discard saved audit",
                use_container_width=False,
                key="discard_saved_audit",
            ):
                clear_audit_resume_state()
                st.session_state.audit_id = None
                st.rerun()

        st.write("")

    elif (
        st.session_state.audit_plan is None
        and recoverable_audit_id
        and recoverable_audit_phase == "gate1"
    ):
        st.info(
            "A previous audit reached Gate 1 but was not approved yet. "
            "For safety, AegisML will not execute dynamic attacks automatically. "
            "Start a new scan unless the backend Gate 1 recovery view is enabled."
        )

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

        elif not check_docker_available():
            st.session_state.show_docker_dialog = True
            st.rerun()

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
                planning_progress = st.empty()
                render_agent_progress(
                    planning_progress,
                    [
                        {
                            "event": "agent_step_started",
                            "agent": "Agent 1",
                            "step": "Static threat analysis",
                            "message": "Parsing the pipeline and building the threat model.",
                        }
                    ],
                )

                with st.spinner(
                    "AegisML is analysing the ML pipeline..."
                ):

                    response = requests.post(
                        f"{API_URL}/audit/plan",
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

                    plan_payload = response.json()
                    st.session_state.audit_plan = plan_payload
                    st.session_state.audit_id = plan_payload.get("audit_id")
                    st.session_state.audit_result = None

                    remember_audit_resume_state(
                        st.session_state.audit_id,
                        "gate1",
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
        "Pipeline & findings",
        "Audit memory ledger",
        "Full report",
    ]

    # Reset old saved tab values from previous versions of the UI.
    if st.session_state.active_report_section not in tab_names:
        st.session_state.active_report_section = "Pipeline & findings"

    # Three equal-width tabs that fill the available page width.
    tab_cols = st.columns(
        [1, 1, 1],
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
            width: 100% !important;
        }

        div[data-testid="stHorizontalBlock"]:has(.aegis-tabs-anchor)
        > div {
            flex: 1 1 33.33% !important;
            width: 33.33% !important;
            max-width: 33.33% !important;
        }

        div[data-testid="stHorizontalBlock"]:has(.aegis-tabs-anchor)
        .stButton {
            width: 100% !important;
        }

        div[data-testid="stHorizontalBlock"]:has(.aegis-tabs-anchor)
        .stButton > button {
            width: 100% !important;
            border: 0 !important;
            border-radius: 0 !important;
            background: transparent !important;
            box-shadow: none !important;
            color: #777b73 !important;
            min-height: 54px !important;
            font-weight: 500 !important;
            font-size: 1rem !important;
            border-bottom: 2px solid transparent !important;
        }

        div[data-testid="stHorizontalBlock"]:has(.aegis-tabs-anchor)
        .stButton > button:hover {
            color: #30342c !important;
            background: rgba(79, 128, 96, 0.04) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    for index, (tab_col, tab_name) in enumerate(
        zip(
            tab_cols,
            tab_names,
        )
    ):
        with tab_col:
            if index == 0:
                st.markdown(
                    '<span class="aegis-tabs-anchor"></span>',
                    unsafe_allow_html=True,
                )

            if st.button(
                tab_name,
                key=f"report_tab_{tab_name}",
                use_container_width=True,
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
            border-bottom: 3px solid #4f8060 !important;
            font-weight: 650 !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    if active_section == "Pipeline & findings":

        if result.get("checkpoint_loaded"):
            st.info(
                "Audit Memory Active: Analysis results were restored from persistent step checkpoints. "
                "Review the 'Audit memory ledger' tab for granular node timings and SHA-256 verification."
            )

        st.markdown("## Interactive pipeline graph")
        render_interactive_pipeline_graph(result)
        st.divider()

        # Only render the dashboard content here.
        # The header/summary was already rendered above the tabs.
        st.html(
            render_dashboard(
                result,
                section="content",
            )
        )

        # Re-scan is a global audit action, so keep it available
        # from the Pipeline & findings tab as well.
        st.divider()

        pipeline_footer_info, pipeline_spacer, pipeline_rescan_col = (
            st.columns(
                [6, 1, 1.2]
            )
        )

        with pipeline_footer_info:
            st.caption(
                f"Scanned 3 files · "
                f"{len(findings)} findings · "
                "AegisML security assessment"
            )

        with pipeline_rescan_col:
            if st.button(
                "Re-scan",
                use_container_width=True,
                key="pipeline_rescan",
            ):
                st.session_state.audit_result = None
                st.session_state.audit_plan = None
                st.session_state.audit_id = None
                st.session_state.report_signed_off = False
                st.session_state.active_report_section = "Pipeline & findings"
                clear_audit_resume_state()
                st.rerun()

    elif active_section == "Audit memory ledger":

        render_audit_ledger_panel(result)
        st.divider()

        pipeline_footer_info, pipeline_spacer, pipeline_rescan_col = (
            st.columns(
                [6, 1, 1.2]
            )
        )

        with pipeline_footer_info:
            st.caption(
                f"Audit session {result.get('audit_id', '')[:12]}… · "
                "Step checkpoints verified in audit_memory.db"
            )

        with pipeline_rescan_col:
            if st.button(
                "Re-scan",
                use_container_width=True,
                key="ledger_rescan",
            ):
                st.session_state.audit_result = None
                st.session_state.audit_plan = None
                st.session_state.audit_id = None
                st.session_state.report_signed_off = False
                st.session_state.active_report_section = "Pipeline & findings"
                clear_audit_resume_state()
                st.rerun()

    else:

        st.markdown("## Full report")

        st.write(
            "Download the comprehensive PDF containing the overall risk "
            "assessment, executive summary, detailed findings, dynamic "
            "evidence, correlation results, and recommendations."
        )

        render_report_signoff_gate(result)

        # -------------------------------------------------
        # Full-report actions
        # -------------------------------------------------

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
                key="full_report_rescan",
            ):
                st.session_state.audit_result = None
                st.session_state.audit_plan = None
                st.session_state.audit_id = None
                st.session_state.report_signed_off = False
                st.session_state.active_report_section = "Pipeline & findings"
                clear_audit_resume_state()
                st.rerun()

        with download_col:
            pdf_report = build_pdf_report(
                report
            )

            st.download_button(
                (
                    "Download signed report"
                    if st.session_state.report_signed_off
                    else "Gate 2 approval required"
                ),
                data=pdf_report,
                file_name="AegisML_Security_Audit_Report.pdf",
                mime="application/pdf",
                use_container_width=True,
                disabled=not st.session_state.report_signed_off,
            )

