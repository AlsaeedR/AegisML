import os
import shutil
import tempfile
import traceback
import uuid
import json as _json
import threading
import asyncio
from typing import Optional, List, Dict, Any, Tuple

from dotenv import load_dotenv

# Ensure environment variables from .env are loaded before any agent or LangChain imports
load_dotenv()

# Synchronize LangSmith and LangChain tracing environment variables
if os.getenv("LANGSMITH_TRACING", "").lower() == "true" and not os.getenv("LANGCHAIN_TRACING_V2"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
elif os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true" and not os.getenv("LANGSMITH_TRACING"):
    os.environ["LANGSMITH_TRACING"] = "true"

if os.getenv("LANGSMITH_API_KEY") and not os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = os.environ["LANGSMITH_API_KEY"]
elif os.getenv("LANGCHAIN_API_KEY") and not os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_API_KEY"] = os.environ["LANGCHAIN_API_KEY"]

if os.getenv("LANGSMITH_PROJECT") and not os.getenv("LANGCHAIN_PROJECT"):
    os.environ["LANGCHAIN_PROJECT"] = os.environ["LANGSMITH_PROJECT"]
elif os.getenv("LANGCHAIN_PROJECT") and not os.getenv("LANGSMITH_PROJECT"):
    os.environ["LANGSMITH_PROJECT"] = os.environ["LANGCHAIN_PROJECT"]

if os.getenv("LANGSMITH_ENDPOINT") and not os.getenv("LANGCHAIN_ENDPOINT"):
    os.environ["LANGCHAIN_ENDPOINT"] = os.environ["LANGSMITH_ENDPOINT"]
elif os.getenv("LANGCHAIN_ENDPOINT") and not os.getenv("LANGSMITH_ENDPOINT"):
    os.environ["LANGSMITH_ENDPOINT"] = os.environ["LANGCHAIN_ENDPOINT"]

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    Request,
)
from fastapi.responses import StreamingResponse, JSONResponse

from src.agents.pipeline_agent.pipeline_agent import (
    run_pipeline_agent,
)

from src.agents.testing_agent.testing_agent import (
    run_testing_agent,
    node_prepare_metadata,
    node_reason_strategy,
    node_execute_sandbox,
    node_forensic_diagnosis,
    node_aggregate_results,
    create_stream,
    get_stream,
)

from src.agents.reporting_agent.reporting_agent import (
    run_reporting_agent,
)

from src.core.audit_memory import (
    load_session,
    save_session,
    initialize_memory,
    StaleArtifactError,
    register_audit_artifacts,
    verify_artifact_integrity,
    get_audit_ledger,
    get_all_subtests,
    find_audit_by_artifacts,
    get_audit_summary,
    invalidate_post_gate1_steps,
    sync_subtests_for_audit,
    get_subtests_by_artifacts,
    is_post_gate1_fully_cached,
)


app = FastAPI(
    title="AegisML API",
    description="AI-powered ML pipeline security auditing API.",
    version="1.0.0",
)


@app.on_event("startup")
def on_startup():
    initialize_memory()


@app.exception_handler(StaleArtifactError)
def stale_artifact_exception_handler(request: Request, exc: StaleArtifactError):
    return JSONResponse(
        status_code=409,
        content={
            "error": "stale_artifact_detected",
            "message": str(exc),
            "detail": "Target code, model, or dataset was modified since audit initialization. Resuming from this checkpoint would produce invalid findings.",
        },
    )


@app.get("/")
def root():
    return {
        "message": "AegisML API is running."
    }


@app.get("/system/docker-status")
def get_docker_status():
    from src.agents.testing_agent.sandbox_runner import is_docker_available
    return {
        "docker_available": is_docker_available(),
    }


def save_upload(
    uploaded_file: UploadFile,
    directory: str,
) -> str:
    file_path = os.path.join(
        directory,
        uploaded_file.filename,
    )

    with open(
        file_path,
        "wb",
    ) as buffer:
        shutil.copyfileobj(
            uploaded_file.file,
            buffer,
        )

    return file_path


AUDIT_SESSIONS = {}

# One in-process lock per audit prevents duplicate /audit/execute calls
# from running Agent 2 / Agent 3 concurrently for the same audit.
_EXECUTION_LOCKS = {}
_EXECUTION_LOCKS_GUARD = threading.Lock()


def _get_execution_lock(
    audit_id: str,
) -> threading.Lock:
    with _EXECUTION_LOCKS_GUARD:
        lock = _EXECUTION_LOCKS.get(
            audit_id
        )

        if lock is None:
            lock = threading.Lock()
            _EXECUTION_LOCKS[
                audit_id
            ] = lock

        return lock


def _is_execution_active(
    audit_id: str,
) -> bool:
    with _EXECUTION_LOCKS_GUARD:
        lock = _EXECUTION_LOCKS.get(
            audit_id
        )

    return bool(
        lock
        and lock.locked()
    )


def _validate_upload_types(
    pipeline_file: UploadFile,
    model_file: UploadFile,
    dataset_file: UploadFile,
) -> None:
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


def _deployment_context(
    agent_1_result,
):
    return (
        agent_1_result
        .get(
            "threat_model",
            {},
        )
        .get(
            "deployment_context",
            {},
        )
        or {}
    )


# =========================================================
# Persistent audit memory helpers
# =========================================================

def _save_audit_session(
    session,
):
    audit_id = session["audit_id"]

    AUDIT_SESSIONS[
        audit_id
    ] = session

    save_session(
        audit_id,
        session,
    )


def _get_audit_session(
    audit_id,
):
    session = AUDIT_SESSIONS.get(
        audit_id
    )

    if session is not None:
        return session

    session = load_session(
        audit_id
    )

    if session is not None:
        AUDIT_SESSIONS[
            audit_id
        ] = session

    return session


def _build_saved_audit_result(
    session,
):
    """
    Build the UI response from already-persisted audit state.

    This function never executes Agent 1, Agent 2, or Agent 3.
    """
    reporting_result = session.get(
        "reporting_result",
        {},
    )

    final_report = reporting_result.get(
        "final_report"
    )

    if final_report is None:
        return None

    agent_1_result = session.get(
        "agent_1_result",
        {},
    )

    audit_id = session.get("audit_id")
    audit_summary = get_audit_summary(audit_id) if audit_id else {}

    return {
        "status": session.get(
            "status",
            "awaiting_gate_2",
        ),
        "audit_id": audit_id,
        "report": final_report,
        "deployment_context": _deployment_context(
            agent_1_result
        ),
        "pipeline_graph": agent_1_result.get(
            "pipeline_graph",
            {},
        ),
        "pipeline_source": session.get(
            "pipeline_source",
            "",
        ),
        "attack_strategy_plan": session.get(
            "attack_strategy_plan",
            {},
        ),
        "checkpoint_loaded": session.get("checkpoint_loaded", True),
        "audit_memory": audit_summary,
    }


def _execute_agent_2_with_approved_plan(
    session,
):
    state = session.get(
        "agent_2_state"
    )

    if state:
        approved_tests = session.get("attack_strategy_plan", {}).get("selected_tests")
        if approved_tests:
            state["planned_tests"] = list(approved_tests)
            if "attack_strategy_plan" in state and isinstance(state["attack_strategy_plan"], dict):
                state["attack_strategy_plan"]["selected_tests"] = list(approved_tests)
    else:
        state = {
            "agent_1_results": (
                session[
                    "agent_1_result"
                ]
            ),
            "audit_id": (
                session.get(
                    "audit_id"
                )
            ),
            "model_path": (
                session[
                    "model_path"
                ]
            ),
            "dataset_path": (
                session[
                    "dataset_path"
                ]
            ),
            "pipeline_path": (
                session[
                    "pipeline_path"
                ]
            ),
            "text_column": (
                session[
                    "text_column"
                ]
            ),
            "label_column": (
                session[
                    "label_column"
                ]
            ),
            "vectorizer_path": None,
            "pipeline_source": session.get("pipeline_source", ""),
            "test_targets": None,
            "dataset_profile": (
                session.get(
                    "dataset_profile",
                    {},
                )
            ),
            "attack_strategy_plan": (
                session[
                    "attack_strategy_plan"
                ]
            ),
            "planned_tests": (
                session[
                    "attack_strategy_plan"
                ].get(
                    "selected_tests",
                    [],
                )
            ),
            "execution_plan_log": (
                list(
                    session.get(
                        "execution_plan_log",
                        [],
                    )
                )
                + [
                    (
                        "Gate 1 approved by human reviewer. "
                        "Proceeding with approved attack strategy."
                    )
                ]
            ),
            "status": (
                "gate_1_approved"
            ),
        }

    # -----------------------------------------------------
    # Agent 2 - Dynamic Sandbox (internal subtest caching & dispatch)
    # -----------------------------------------------------
    sandbox_update = (
        node_execute_sandbox(
            state
        )
        or {}
    )
    state.update(sandbox_update)

    # -----------------------------------------------------
    # Agent 2 - Forensic Diagnosis
    # -----------------------------------------------------
    forensics_update = (
        node_forensic_diagnosis(
            state
        )
        or {}
    )
    state.update(forensics_update)

    # -----------------------------------------------------
    # Agent 2 - Aggregate Results
    # -----------------------------------------------------
    aggregate_update = (
        node_aggregate_results(
            state
        )
        or {}
    )
    state.update(aggregate_update)

    completed_steps = set(session.get("completed_steps", []))
    completed_steps.update(["sandbox", "forensics", "aggregate"])
    session["completed_steps"] = list(completed_steps)
    session["agent_2_state"] = state
    _save_audit_session(session)

    return state


@app.post("/audit/plan")
def plan_audit(
    pipeline_file: UploadFile = File(...),
    model_file: UploadFile = File(...),
    dataset_file: UploadFile = File(...),
    text_column: str = Form("text"),
    label_column: str = Form("label"),
):
    from src.agents.testing_agent.sandbox_runner import is_docker_available
    if not is_docker_available():
        raise HTTPException(
            status_code=503,
            detail="Docker is unavailable. AegisML Zero-Trust policy requires an active Docker daemon to perform security audits.",
        )

    _validate_upload_types(
        pipeline_file,
        model_file,
        dataset_file,
    )

    audit_id = uuid.uuid4().hex

    runtime_root = os.path.join(
        ".aegisml_runtime",
        "audits",
    )

    os.makedirs(
        runtime_root,
        exist_ok=True,
    )

    temp_dir = tempfile.mkdtemp(
        prefix=f"{audit_id}_",
        dir=runtime_root,
    )

    try:
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

        with open(
            pipeline_path,
            "r",
            encoding="utf-8",
        ) as file:
            python_code = file.read()

        # Check if an identical audit already exists in audit_memory
        matching_audit_id = find_audit_by_artifacts(
            code=python_code,
            model_path=model_path,
            dataset_path=dataset_path,
        )

        checkpoint_loaded = False
        if matching_audit_id:
            audit_id = matching_audit_id
            checkpoint_loaded = True

        register_audit_artifacts(
            audit_id,
            code=python_code,
            model_path=model_path,
            dataset_path=dataset_path,
        )

        agent_1_result = (
            run_pipeline_agent(
                python_code,
                audit_id=audit_id,
            )
        )

        planning_state = {
            "audit_id": (
                audit_id
            ),
            "agent_1_results": (
                agent_1_result
            ),
            "dataset_path": (
                dataset_path
            ),
            "model_path": (
                model_path
            ),
            "pipeline_path": (
                pipeline_path
            ),
            "text_column": (
                text_column
            ),
            "label_column": (
                label_column
            ),
            "vectorizer_path": None,
            "test_targets": None,
            "execution_plan_log": [],
        }

        metadata_update = (
            node_prepare_metadata(
                planning_state,
            )
            or {}
        )

        planning_state.update(
            metadata_update
        )

        strategy_update = (
            node_reason_strategy(
                planning_state,
            )
            or {}
        )

        planning_state.update(
            strategy_update
        )

        create_stream(
            audit_id
        )

        session = {
            "audit_id": (
                audit_id
            ),
            "temp_dir": (
                temp_dir
            ),
            "pipeline_path": (
                pipeline_path
            ),
            "model_path": (
                model_path
            ),
            "dataset_path": (
                dataset_path
            ),
            "text_column": (
                text_column
            ),
            "label_column": (
                label_column
            ),
            "pipeline_source": (
                python_code
            ),
            "agent_1_result": (
                agent_1_result
            ),
            "dataset_profile": (
                planning_state.get(
                    "dataset_profile",
                    {},
                )
            ),
            "attack_strategy_plan": (
                planning_state.get(
                    "attack_strategy_plan",
                    {},
                )
            ),
            "execution_plan_log": (
                planning_state.get(
                    "execution_plan_log",
                    [],
                )
            ),
            "completed_steps": [
                "agent_1",
                "metadata",
                "strategy",
            ],
            "status": (
                "awaiting_gate_1"
            ),
            "checkpoint_loaded": checkpoint_loaded,
        }

        _save_audit_session(
            session
        )

        audit_summary = get_audit_summary(audit_id)

        return {
            "status": (
                "awaiting_gate_1"
            ),
            "audit_id": (
                audit_id
            ),
            "checkpoint_loaded": checkpoint_loaded,
            "audit_memory": audit_summary,
            "attack_strategy_plan": (
                planning_state.get(
                    "attack_strategy_plan",
                    {},
                )
            ),
            "pipeline_graph": (
                agent_1_result.get(
                    "pipeline_graph",
                    {},
                )
            ),
            "pipeline_source": (
                python_code
            ),
            "vulnerability_findings": (
                agent_1_result.get(
                    "vulnerability_findings",
                    {},
                )
            ),
            "deployment_context": (
                _deployment_context(
                    agent_1_result
                )
            ),
        }

    except HTTPException:
        shutil.rmtree(
            temp_dir,
            ignore_errors=True,
        )

        raise

    except Exception as exc:
        shutil.rmtree(
            temp_dir,
            ignore_errors=True,
        )

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


@app.get("/audit/{audit_id}")
def get_saved_audit(
    audit_id: str,
):
    """
    Restore a persisted audit without re-running any security test.

    If the final report already exists, the exact saved result is returned.
    If execution is still incomplete, the current checkpoint metadata is
    returned so the UI can decide whether to resume execution.
    """
    session = _get_audit_session(
        audit_id
    )

    if not session:
        raise HTTPException(
            status_code=404,
            detail="Saved audit session not found or expired.",
        )

    saved_result = _build_saved_audit_result(
        session
    )

    if saved_result is not None:
        saved_result["has_report"] = True
        saved_result["is_final"] = True
        saved_result["execution_active"] = _is_execution_active(
            audit_id
        )
        saved_result["checkpoint_loaded"] = True
        return saved_result

    audit_summary = get_audit_summary(audit_id)
    agent_1_result = session.get("agent_1_result", {})

    return {
        "status": session.get(
            "status",
            "interrupted",
        ),
        "audit_id": audit_id,
        "completed_steps": session.get(
            "completed_steps",
            [],
        ),
        "has_report": False,
        "is_final": False,
        "checkpoint_loaded": True,
        "audit_memory": audit_summary,
        "execution_active": _is_execution_active(
            audit_id
        ),
        "attack_strategy_plan": session.get(
            "attack_strategy_plan",
            {},
        ),
        "pipeline_graph": agent_1_result.get(
            "pipeline_graph",
            {},
        ),
        "pipeline_source": session.get(
            "pipeline_source",
            "",
        ),
        "vulnerability_findings": agent_1_result.get(
            "vulnerability_findings",
            {},
        ),
        "deployment_context": _deployment_context(
            agent_1_result
        ),
    }


@app.get("/audit/{audit_id}/ledger")
def get_audit_step_ledger(
    audit_id: str,
):
    """
    Returns the fine-grained chronological step-level checkpoint ledger
    and completed dynamic sub-tests for this audit.
    """
    ledger = get_audit_ledger(audit_id)
    subtests = get_all_subtests(audit_id)
    return {
        "audit_id": audit_id,
        "steps_count": len(ledger),
        "steps": ledger,
        "completed_subtests": list(subtests.keys()),
    }


@app.get(
    "/audit/{audit_id}/telemetry/stream"
)
def stream_audit_telemetry(
    audit_id: str,
):
    stream = get_stream(
        audit_id
    )

    if stream is None:
        session = _get_audit_session(
            audit_id
        )

        if session:
            stream = create_stream(
                audit_id
            )

    if stream is None:
        raise HTTPException(
            status_code=404,
            detail="No active telemetry stream for this audit_id.",
        )

    async def event_generator():
        try:
            while True:
                try:
                    event = await asyncio.wait_for(
                        asyncio.to_thread(
                            stream.get
                        ),
                        timeout=1.0,
                    )
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    continue

                yield (
                    f"data: "
                    f"{_json.dumps(event, default=str)}"
                    f"\n\n"
                )

                if event.get("event") == "done":
                    break

        except (
            asyncio.CancelledError,
            GeneratorExit,
        ):
            return

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.post("/audit/execute")
def execute_planned_audit(
    audit_id: str = Form(...),
    selected_tests: Optional[str] = Form(None),
):
    from src.agents.testing_agent.sandbox_runner import is_docker_available
    if not is_docker_available():
        raise HTTPException(
            status_code=503,
            detail="Docker is unavailable. AegisML Zero-Trust policy requires an active Docker daemon to perform security audits.",
        )

    session = (
        _get_audit_session(
            audit_id
        )
    )

    if not session:
        raise HTTPException(
            status_code=404,
            detail=(
                "Audit planning session "
                "not found or expired."
            ),
        )

    selected_tests_list = None
    if selected_tests:
        try:
            parsed = _json.loads(selected_tests)
            if isinstance(parsed, list) and parsed:
                selected_tests_list = parsed
                if "attack_strategy_plan" in session and isinstance(session["attack_strategy_plan"], dict):
                    session["attack_strategy_plan"]["selected_tests"] = parsed
                if "agent_2_state" in session and isinstance(session["agent_2_state"], dict):
                    session["agent_2_state"]["planned_tests"] = parsed
                    if "attack_strategy_plan" in session["agent_2_state"]:
                        session["agent_2_state"]["attack_strategy_plan"]["selected_tests"] = parsed
        except Exception:
            pass

    if not selected_tests_list:
        selected_tests_list = (
            session.get("attack_strategy_plan", {}).get("selected_tests")
            or ["V1_poisoning", "V2_preprocessing", "V3_validation", "V4_adversarial"]
        )

    sync_subtests_for_audit(
        audit_id,
        code=session.get("pipeline_source"),
        model_path=session.get("model_path"),
        dataset_path=session.get("dataset_path"),
    )

    if not is_post_gate1_fully_cached(audit_id, selected_tests_list):
        invalidate_post_gate1_steps(audit_id)
        session["completed_steps"] = [
            s for s in session.get("completed_steps", [])
            if s in ("agent_1", "metadata", "strategy")
        ]
        session.pop("reporting_result", None)
        session.pop("agent_2_result", None)

    _save_audit_session(session)

    execution_lock = _get_execution_lock(
        audit_id
    )

    if not execution_lock.acquire(
        blocking=False
    ):
        raise HTTPException(
            status_code=409,
            detail=(
                "This audit is already executing. "
                "Wait for the active execution to finish "
                "or reconnect to its saved state."
            ),
        )

    try:
        from src.agents.testing_agent.testing_agent import (
            publish,
        )

        # Re-create telemetry after an API restart.
        if get_stream(
            audit_id
        ) is None:
            create_stream(
                audit_id
            )

        agent_2_result = (
            _execute_agent_2_with_approved_plan(
                session
            )
        )

        session[
            "agent_2_result"
        ] = agent_2_result

        session[
            "status"
        ] = "agent_2_complete"

        _save_audit_session(
            session
        )

        publish(
            audit_id,
            {
                "event": "agent_step_finished",
                "agent": "Agent 2",
                "step": "Forensic diagnosis",
                "status": "completed",
                "message": "Dynamic evidence passed to Agent 3 for reporting.",
            },
        )

        # -------------------------------------------------
        # Agent 3 - Evidence-informed reporting
        # -------------------------------------------------
        publish(
            audit_id,
            {
                "event": (
                    "agent_step_started"
                ),
                "agent": (
                    "Agent 3"
                ),
                "step": (
                    "Evidence-informed reporting"
                ),
                "message": (
                    "Correlating static findings "
                    "and dynamic evidence."
                ),
            },
        )

        reporting_result = (
            run_reporting_agent(
                agent_1_results=(
                    session[
                        "agent_1_result"
                    ]
                ),
                agent_2_results=(
                    agent_2_result
                ),
                audit_id=audit_id,
            )
        )

        session[
            "reporting_result"
        ] = reporting_result

        session[
            "completed_steps"
        ] = list(
            set(
                session.get(
                    "completed_steps",
                    [],
                )
            )
            | {
                "agent_3"
            }
        )

        session[
            "status"
        ] = "awaiting_gate_2"

        _save_audit_session(
            session
        )

        publish(
            audit_id,
            {
                "event": (
                    "agent_step_finished"
                ),
                "agent": (
                    "Agent 3"
                ),
                "step": (
                    "Evidence-informed reporting"
                ),
                "status": (
                    "completed"
                ),
                "message": (
                    "Final security report is ready."
                ),
            },
        )

        saved_result = _build_saved_audit_result(
            session
        )

        if saved_result is None:
            raise HTTPException(
                status_code=500,
                detail="Audit completed but the final report was not persisted.",
            )

        return saved_result

    except HTTPException:
        raise

    except Exception as exc:
        session[
            "status"
        ] = "interrupted"

        _save_audit_session(
            session
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )

    finally:
        if execution_lock.locked():
            execution_lock.release()

        from src.agents.testing_agent.testing_agent import (
            close_stream,
        )

        close_stream(
            audit_id
        )


@app.post("/audit")
def run_audit(
    pipeline_file: UploadFile = File(...),
    model_file: UploadFile = File(...),
    dataset_file: UploadFile = File(...),
    text_column: str = Form("text"),
    label_column: str = Form("label"),
):
    _validate_upload_types(
        pipeline_file,
        model_file,
        dataset_file,
    )

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline_path = (
                save_upload(
                    pipeline_file,
                    temp_dir,
                )
            )

            model_path = (
                save_upload(
                    model_file,
                    temp_dir,
                )
            )

            dataset_path = (
                save_upload(
                    dataset_file,
                    temp_dir,
                )
            )

            with open(
                pipeline_path,
                "r",
                encoding="utf-8",
            ) as file:
                python_code = (
                    file.read()
                )

            register_audit_artifacts(
                audit_id,
                code=python_code,
                model_path=model_path,
                dataset_path=dataset_path,
            )

            agent_1_result = (
                run_pipeline_agent(
                    python_code,
                    audit_id=audit_id,
                )
            )

            agent_2_result = (
                run_testing_agent(
                    agent_1_results=(
                        agent_1_result
                    ),
                    model_path=(
                        model_path
                    ),
                    dataset_path=(
                        dataset_path
                    ),
                    pipeline_path=(
                        pipeline_path
                    ),
                    text_column=(
                        text_column
                    ),
                    label_column=(
                        label_column
                    ),
                    audit_id=audit_id,
                )
            )

            reporting_result = (
                run_reporting_agent(
                    agent_1_results=(
                        agent_1_result
                    ),
                    agent_2_results=(
                        agent_2_result
                    ),
                    audit_id=audit_id,
                )
            )

            deployment_context = (
                _deployment_context(
                    agent_1_result
                )
            )

            return {
                "status": (
                    "completed"
                ),
                "report": (
                    reporting_result.get(
                        "final_report"
                    )
                ),
                "deployment_context": (
                    deployment_context
                ),
                "pipeline_graph": (
                    agent_1_result.get(
                        "pipeline_graph",
                        {},
                    )
                ),
                "pipeline_source": (
                    python_code
                ),
                "attack_strategy_plan": (
                    agent_2_result.get(
                        "attack_strategy_plan"
                    )
                    or (
                        agent_2_result.get(
                            "structured_test_results"
                        )
                        or {}
                    ).get(
                        "attack_strategy_plan"
                    )
                    or {}
                ),
            }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )