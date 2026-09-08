import os
import shutil
import tempfile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from src.agents.pipeline_agent.pipeline_agent import run_pipeline_agent
from src.agents.testing_agent.testing_agent import run_testing_agent
from src.agents.reporting_agent.reporting_agent import run_reporting_agent


app = FastAPI(
    title="AegisML API",
    description="AI-powered ML pipeline security auditing API.",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "message": "AegisML API is running."
    }


def save_upload(
    uploaded_file: UploadFile,
    directory: str,
) -> str:
    """
    Save an uploaded file to a temporary directory.
    """

    file_path = os.path.join(
        directory,
        uploaded_file.filename,
    )

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(
            uploaded_file.file,
            buffer,
        )

    return file_path


@app.post("/audit")
def run_audit(
    pipeline_file: UploadFile = File(...),
    model_file: UploadFile = File(...),
    dataset_file: UploadFile = File(...),
    text_column: str = Form("text"),
    label_column: str = Form("label"),
):
    """
    Run the complete AegisML security audit.
    """

    # Validate uploaded file types
    if not pipeline_file.filename.endswith(".py"):
        raise HTTPException(
            status_code=400,
            detail="Pipeline must be a Python (.py) file.",
        )

    if not model_file.filename.endswith(".pkl"):
        raise HTTPException(
            status_code=400,
            detail="Model must be a pickle (.pkl) file.",
        )

    if not dataset_file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Dataset must be a CSV (.csv) file.",
        )

    try:
        # Create a temporary directory for uploaded files
        with tempfile.TemporaryDirectory() as temp_dir:

            pipeline_path = save_upload(
                pipeline_file,
                temp_dir,
            )

            model_path = save_upload(
                model_file,
                temp_dir,
            )

            dataset_path = save_upload(
                dataset_file,
                temp_dir,
            )

            # Read pipeline source code
            with open(
                pipeline_path,
                "r",
                encoding="utf-8",
            ) as file:
                python_code = file.read()

            # -------------------------------------------------
            # Agent 1 - Pipeline & Threat Model Agent
            # -------------------------------------------------

            agent_1_result = run_pipeline_agent(
                python_code
            )

            # -------------------------------------------------
            # Agent 2 - Vulnerability Testing Agent
            # -------------------------------------------------

            agent_2_result = run_testing_agent(
                agent_1_results=agent_1_result,
                model_path=model_path,
                dataset_path=dataset_path,
                pipeline_path=pipeline_path,
                text_column=text_column,
                label_column=label_column,
            )

            # -------------------------------------------------
            # Agent 3 - Reporting Agent
            # -------------------------------------------------

            reporting_result = run_reporting_agent(
                agent_1_results=agent_1_result,
                agent_2_results=agent_2_result,
            )

            # Return only the final report needed by Streamlit
            return {
                "status": "completed",
                "report": reporting_result.get(
                    "final_report"
                ),
            }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )