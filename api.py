import os
import shutil
import tempfile
import uuid
import json as _json
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from src.agents.pipeline_agent.pipeline_agent import run_pipeline_agent
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
from src.agents.reporting_agent.reporting_agent import run_reporting_agent
app = FastAPI(title='AegisML API', description='AI-powered ML pipeline security auditing API.', version='1.0.0')

@app.get('/')
def root():
    return {'message': 'AegisML API is running.'}

def save_upload(uploaded_file: UploadFile, directory: str) -> str:
    file_path = os.path.join(directory, uploaded_file.filename)
    with open(file_path, 'wb') as buffer:
        shutil.copyfileobj(uploaded_file.file, buffer)
    return file_path
AUDIT_SESSIONS = {}

def _validate_upload_types(pipeline_file: UploadFile, model_file: UploadFile, dataset_file: UploadFile) -> None:
    if not pipeline_file.filename.endswith('.py'):
        raise HTTPException(status_code=400, detail='Pipeline must be a Python (.py) file.')
    if not model_file.filename.endswith('.pkl'):
        raise HTTPException(status_code=400, detail='Model must be a pickle (.pkl) file.')
    if not dataset_file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail='Dataset must be a CSV (.csv) file.')

def _deployment_context(agent_1_result):
    return agent_1_result.get('threat_model', {}).get('deployment_context', {}) or {}

def _execute_agent_2_with_approved_plan(session):
    state = {'agent_1_results': session['agent_1_result'], 'audit_id': session.get('audit_id'), 'model_path': session['model_path'], 'dataset_path': session['dataset_path'], 'pipeline_path': session['pipeline_path'], 'text_column': session['text_column'], 'label_column': session['label_column'], 'vectorizer_path': None, 'test_targets': None, 'dataset_profile': session.get('dataset_profile', {}), 'attack_strategy_plan': session['attack_strategy_plan'], 'planned_tests': session['attack_strategy_plan'].get('selected_tests', []), 'execution_plan_log': list(session.get('execution_plan_log', [])) + ['Gate 1 approved by human reviewer. Proceeding with approved attack strategy.'], 'status': 'gate_1_approved'}
    for node_function in (node_execute_sandbox, node_forensic_diagnosis, node_aggregate_results):
        update = node_function(state) or {}
        state.update(update)
    return state

@app.post('/audit/plan')
def plan_audit(pipeline_file: UploadFile=File(...), model_file: UploadFile=File(...), dataset_file: UploadFile=File(...), text_column: str=Form('text'), label_column: str=Form('label')):
    _validate_upload_types(pipeline_file, model_file, dataset_file)
    temp_dir = tempfile.mkdtemp(prefix='aegisml_hitl_')
    try:
        pipeline_path = save_upload(pipeline_file, temp_dir)
        model_path = save_upload(model_file, temp_dir)
        dataset_path = save_upload(dataset_file, temp_dir)
        with open(pipeline_path, 'r', encoding='utf-8') as file:
            python_code = file.read()
        agent_1_result = run_pipeline_agent(python_code)
        planning_state = {'agent_1_results': agent_1_result, 'dataset_path': dataset_path, 'model_path': model_path, 'pipeline_path': pipeline_path, 'text_column': text_column, 'label_column': label_column, 'vectorizer_path': None, 'test_targets': None, 'execution_plan_log': []}
        metadata_update = node_prepare_metadata(planning_state) or {}
        planning_state.update(metadata_update)
        strategy_update = node_reason_strategy(planning_state) or {}
        planning_state.update(strategy_update)
        audit_id = uuid.uuid4().hex
        create_stream(audit_id)
        AUDIT_SESSIONS[audit_id] = {'audit_id': audit_id, 'temp_dir': temp_dir, 'pipeline_path': pipeline_path, 'model_path': model_path, 'dataset_path': dataset_path, 'text_column': text_column, 'label_column': label_column, 'pipeline_source': python_code, 'agent_1_result': agent_1_result, 'dataset_profile': planning_state.get('dataset_profile', {}), 'attack_strategy_plan': planning_state.get('attack_strategy_plan', {}), 'execution_plan_log': planning_state.get('execution_plan_log', [])}
        return {'status': 'awaiting_gate_1', 'audit_id': audit_id, 'attack_strategy_plan': planning_state.get('attack_strategy_plan', {}), 'pipeline_graph': agent_1_result.get('pipeline_graph', {}), 'pipeline_source': python_code, 'vulnerability_findings': agent_1_result.get('vulnerability_findings', {}), 'deployment_context': _deployment_context(agent_1_result)}
    except HTTPException:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(exc))

@app.get('/audit/{audit_id}/telemetry/stream')
def stream_audit_telemetry(audit_id: str):
    stream = get_stream(audit_id)
    if stream is None:
        raise HTTPException(status_code=404, detail='No active telemetry stream for this audit_id. Either the audit_id is invalid, has already completed and been cleaned up, or /audit/plan was never called for it.')

    def event_generator():
        while True:
            event = stream.get()
            yield f'data: {_json.dumps(event, default=str)}\n\n'
            if event.get('event') == 'done':
                break
    return StreamingResponse(event_generator(), media_type='text/event-stream', headers={'Cache-Control': 'no-cache', 'Connection': 'keep-alive'})

@app.post('/audit/execute')
def execute_planned_audit(audit_id: str=Form(...)):
    session = AUDIT_SESSIONS.get(audit_id)
    if not session:
        raise HTTPException(status_code=404, detail='Audit planning session not found or expired.')
    try:
        from src.agents.testing_agent.testing_agent import publish
        agent_2_result = _execute_agent_2_with_approved_plan(session)
        publish(audit_id, {'event': 'agent_step_started', 'agent': 'Agent 3', 'step': 'Evidence-informed reporting', 'message': 'Correlating static findings and dynamic evidence.'})
        reporting_result = run_reporting_agent(agent_1_results=session['agent_1_result'], agent_2_results=agent_2_result)
        publish(audit_id, {'event': 'agent_step_finished', 'agent': 'Agent 3', 'step': 'Evidence-informed reporting', 'status': 'completed', 'message': 'Final security report is ready.'})
        return {'status': 'awaiting_gate_2', 'audit_id': audit_id, 'report': reporting_result.get('final_report'), 'deployment_context': _deployment_context(session['agent_1_result']), 'pipeline_graph': session['agent_1_result'].get('pipeline_graph', {}), 'pipeline_source': session['pipeline_source'], 'attack_strategy_plan': session['attack_strategy_plan']}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        from src.agents.testing_agent.testing_agent import close_stream
        close_stream(audit_id)
        shutil.rmtree(session.get('temp_dir', ''), ignore_errors=True)
        AUDIT_SESSIONS.pop(audit_id, None)

@app.post('/audit')
def run_audit(pipeline_file: UploadFile=File(...), model_file: UploadFile=File(...), dataset_file: UploadFile=File(...), text_column: str=Form('text'), label_column: str=Form('label')):
    _validate_upload_types(pipeline_file, model_file, dataset_file)
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline_path = save_upload(pipeline_file, temp_dir)
            model_path = save_upload(model_file, temp_dir)
            dataset_path = save_upload(dataset_file, temp_dir)
            with open(pipeline_path, 'r', encoding='utf-8') as file:
                python_code = file.read()
            agent_1_result = run_pipeline_agent(python_code)
            agent_2_result = run_testing_agent(agent_1_results=agent_1_result, model_path=model_path, dataset_path=dataset_path, pipeline_path=pipeline_path, text_column=text_column, label_column=label_column)
            reporting_result = run_reporting_agent(agent_1_results=agent_1_result, agent_2_results=agent_2_result)
            deployment_context = _deployment_context(agent_1_result)
            return {'status': 'completed', 'report': reporting_result.get('final_report'), 'deployment_context': deployment_context, 'pipeline_graph': agent_1_result.get('pipeline_graph', {}), 'pipeline_source': python_code, 'attack_strategy_plan': agent_2_result.get('attack_strategy_plan') or (agent_2_result.get('structured_test_results') or {}).get('attack_strategy_plan') or {}}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
