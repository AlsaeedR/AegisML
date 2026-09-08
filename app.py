import json
from pathlib import Path

import requests
import streamlit as st

from ui.dashboard import render_dashboard


API_URL = "http://127.0.0.1:8000"


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
        background: #fafaf7;

        border-radius: 10px;
    }

    .stButton > button {
        min-height: 45px;

        border-radius: 8px;

        font-weight: 600;
    }

    .stButton > button[kind="primary"] {
        border: none;

        background: #30342c;

        color: white;
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
        <div style="
            display: flex;
            align-items: center;
            gap: 15px;
            margin-bottom: 10px;
        ">

            <!-- Pipeline icon: ●—■—● -->
            <div style="
                display: flex;
                align-items: center;
                gap: 4px;
                height: 42px;
            ">

                <span style="
                    width: 9px;
                    height: 9px;
                    background: #30342c;
                    border-radius: 50%;
                    display: block;
                "></span>

                <span style="
                    width: 13px;
                    height: 2px;
                    background: #a8aaa3;
                    display: block;
                "></span>

                <span style="
                    width: 11px;
                    height: 11px;
                    background: #4f8060;
                    border-radius: 3px;
                    display: block;
                "></span>

                <span style="
                    width: 13px;
                    height: 2px;
                    background: #a8aaa3;
                    display: block;
                "></span>

                <span style="
                    width: 9px;
                    height: 9px;
                    background: #30342c;
                    border-radius: 50%;
                    display: block;
                "></span>

            </div>

            <h1 style="
                margin: 0;
                padding: 0;
                color: #30342c;
                font-size: 38px;
                font-weight: 700;
                line-height: 1.1;
            ">
                AegisML Pipeline Audit
            </h1>

        </div>

        <p style="
            margin: 0 0 28px 0;
            color: #6f726a;
            font-size: 17px;
        ">
            Upload your machine learning assets to run static threat analysis,
            dynamic security testing, and automated security reporting.
        </p>
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

        pipeline_file = st.file_uploader(
            "Pipeline source",
            type=["py"],
            help="Python ML pipeline source file.",
        )


    with col2:

        model_file = st.file_uploader(
            "Trained model",
            type=["pkl"],
            help="Pickle model generated by the ML pipeline.",
        )


    with col3:

        dataset_file = st.file_uploader(
            "Dataset",
            type=["csv"],
            help="CSV dataset used by the model.",
        )


    st.write("")


    # -----------------------------------------------------
    # Dataset settings
    # -----------------------------------------------------

    setting1, setting2 = st.columns(2)


    with setting1:

        text_column = st.text_input(
            "Text column",
            value="text",
        )


    with setting2:

        label_column = st.text_input(
            "Label column",
            value="label",
        )


    st.write("")


    # -----------------------------------------------------
    # Run audit
    # -----------------------------------------------------

    if st.button(
        "Run security audit",
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


# =========================================================
# RESULTS DASHBOARD
# =========================================================

else:

    result = st.session_state.audit_result


    # Render the custom HTML dashboard
    st.html(
        render_dashboard(
            result
        )
    )


    # -----------------------------------------------------
    # Bottom actions
    # -----------------------------------------------------

    report = result.get(
        "report",
        {},
    )

    findings = report.get(
        "findings",
        [],
    )


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

        report_json = json.dumps(
            report,
            indent=4,
            ensure_ascii=False,
        )

        st.download_button(
            "Download full report",
            data=report_json,
            file_name="aegisml_security_report.json",
            mime="application/json",
            use_container_width=True,
        )