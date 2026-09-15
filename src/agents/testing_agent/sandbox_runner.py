import os
import sys
import json
import uuid
import shutil
import tempfile
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional
from .testing_agent import publish as publish_telemetry, close_stream

def is_docker_available() -> bool:
    try:
        res = subprocess.run(['docker', 'info'], capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=4)
        return res.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        return False

def dispatch_sandbox(model_path: str, dataset_path: str, pipeline_path: str, vectorizer_path: Optional[str], text_column: str, label_column: str, planned_tests: List[str], strategy_config: Optional[Dict[str, Any]]=None, agent_1_results: Optional[Dict[str, Any]]=None, timeout_seconds: int=180, audit_id: Optional[str]=None, max_oom_retries: Optional[int]=None) -> Dict[str, Any]:
    if not is_docker_available():
        print('\n[AegisML Security Gate] Docker sandbox is unavailable. Zero-Trust policy active: untrusted model/code will NOT be executed on host. Marking dynamic tests as skipped.')
        publish_telemetry(audit_id, {'event': 'sandbox_skipped', 'reason': 'docker_unavailable'})
        result = _build_fail_closed_skip_response(planned_tests=planned_tests, reason='Zero-Trust sandbox policy active: Docker daemon is unavailable. Untrusted model deserialization and pipeline execution were halted to protect the host environment from potential RCE or resource exhaustion.')
        close_stream(audit_id)
        return result

    result = _run_in_docker(model_path=model_path, dataset_path=dataset_path, pipeline_path=pipeline_path, vectorizer_path=vectorizer_path, text_column=text_column, label_column=label_column, planned_tests=planned_tests, strategy_config=strategy_config, agent_1_results=agent_1_results, timeout_seconds=timeout_seconds, audit_id=audit_id, max_oom_retries=max_oom_retries if max_oom_retries is not None else int(os.getenv('AEGISML_MAX_OOM_RETRIES', '1')))
    close_stream(audit_id)
    return result

def _poll_container_stats(container_name: str, audit_id: Optional[str], stop_event: threading.Event, memory_limit_bytes: int, interval_seconds: float=1.5) -> None:
    warned = False
    while not stop_event.is_set():
        try:
            res = subprocess.run(['docker', 'stats', container_name, '--no-stream', '--format', '{{json .}}'], capture_output=True, text=True, timeout=5)
            if res.returncode == 0 and res.stdout.strip():
                stats = json.loads(res.stdout.strip())
                publish_telemetry(audit_id, {'event': 'container_stats', 'container_id': container_name, 'cpu_percent': stats.get('CPUPerc'), 'mem_usage': stats.get('MemUsage'), 'mem_percent': stats.get('MemPerc'), 'net_io': stats.get('NetIO'), 'pids': stats.get('PIDs'), 'timestamp': time.time()})
                mem_pct_raw = (stats.get('MemPerc') or '0%').strip('%')
                try:
                    mem_pct = float(mem_pct_raw)
                except ValueError:
                    mem_pct = 0.0
                if not warned and mem_pct >= 85.0:
                    warned = True
                    publish_telemetry(audit_id, {'event': 'memory_warning', 'container_id': container_name, 'mem_percent': mem_pct, 'message': 'Container memory usage exceeded 85% of its limit. An OOM kill may occur; adaptive downscaling will trigger automatically on retry if the container is killed.'})
        except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
            pass
        stop_event.wait(interval_seconds)

def _was_oom_killed(container_name: str) -> bool:
    try:
        res = subprocess.run(['docker', 'inspect', '--format', '{{.State.OOMKilled}}', container_name], capture_output=True, text=True,encoding='utf-8', errors='replace',  timeout=5)
        return res.returncode == 0 and res.stdout.strip().lower() == 'true'
    except Exception:
        return False

def _remove_container(container_name: str) -> None:
    subprocess.run(['docker', 'rm', '-f', container_name], capture_output=True, timeout=10)

def _downscale_strategy_config(strategy_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    config = dict(strategy_config or {})
    adv_cfg = dict(config.get('adversarial_config') or {})
    current_sample_size = adv_cfg.get('sample_size', 50)
    adv_cfg['sample_size'] = max(10, int(current_sample_size // 2))
    config['adversarial_config'] = adv_cfg
    return config

def _run_in_docker(model_path: str, dataset_path: str, pipeline_path: str, vectorizer_path: Optional[str], text_column: str, label_column: str, planned_tests: List[str], strategy_config: Optional[Dict[str, Any]], agent_1_results: Optional[Dict[str, Any]], timeout_seconds: int, audit_id: Optional[str]=None, max_oom_retries: int=1) -> Dict[str, Any]:
    memory_limit = os.getenv('AEGISML_CONTAINER_MEMORY', '4g')
    memory_limit_bytes = _parse_memory_limit(memory_limit)
    attempt = 0
    current_strategy_config = strategy_config
    while True:
        attempt += 1
        container_name = f'aegisml-sandbox-{uuid.uuid4().hex[:8]}'
        image_name = os.getenv('AEGISML_DOCKER_IMAGE', 'aegisml-sandbox:latest')
        data_dir = os.path.abspath(os.path.dirname(dataset_path))
        temp_dir = tempfile.mkdtemp(prefix='aegisml_sandbox_')
        input_dir = os.path.join(temp_dir, 'input')
        output_dir = os.path.join(temp_dir, 'output')
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        clean_agent_1 = None
        if agent_1_results:
            clean_agent_1 = {k: v for k, v in agent_1_results.items() if k != 'networkx_graph'}
        strategy_payload = {'model_path': f'/workspace/data/{os.path.basename(model_path)}', 'dataset_path': f'/workspace/data/{os.path.basename(dataset_path)}', 'pipeline_path': f'/workspace/data/{os.path.basename(pipeline_path)}', 'vectorizer_path': f'/workspace/data/{os.path.basename(vectorizer_path)}' if vectorizer_path else None, 'text_column': text_column, 'label_column': label_column, 'planned_tests': planned_tests, 'agent_1_results': clean_agent_1, **(current_strategy_config or {})}
        input_json_path = os.path.join(input_dir, 'strategy.json')
        with open(input_json_path, 'w', encoding='utf-8') as f:
            json.dump(strategy_payload, f, indent=2, default=str)
        src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        docker_cmd = [
            'docker', 'run', '--name', container_name,
            '--network', 'none', '--memory', memory_limit,
            '--cpus', '2.0', '--pids-limit', '128',
            '-v', f'{data_dir}:/workspace/data:ro',
            '-v', f'{input_dir}:/workspace/input:ro',
            '-v', f'{output_dir}:/workspace/output:rw',
            '-v', f'{src_dir}:/app/src:ro',
            '--entrypoint', 'python',
            image_name,
            '-m', 'src.agents.testing_agent.sandbox.worker',
        ]
        publish_telemetry(audit_id, {'event': 'container_starting', 'container_id': container_name, 'attempt': attempt, 'planned_tests': planned_tests})
        stop_stats_event = threading.Event()
        stats_thread = threading.Thread(target=_poll_container_stats, args=(container_name, audit_id, stop_stats_event, memory_limit_bytes), daemon=True)
        try:
            proc = subprocess.Popen(docker_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,encoding='utf-8', errors='replace')
            stats_thread.start()
            try:
                stdout, stderr = proc.communicate(timeout=timeout_seconds)
                returncode = proc.returncode
            except subprocess.TimeoutExpired:
                subprocess.run(['docker', 'kill', container_name], capture_output=True, 
                    text=True, 
                    encoding='utf-8', 
                    errors='replace')
                stdout, stderr = proc.communicate()
                returncode = -1
            stop_stats_event.set()
            stats_thread.join(timeout=5)
            oom_killed = _was_oom_killed(container_name)
            if oom_killed and attempt <= max_oom_retries:
                publish_telemetry(audit_id, {'event': 'oom_detected_retrying', 'container_id': container_name, 'attempt': attempt, 'message': 'Container was killed by the OOM killer. Retrying with a downscaled attack configuration (reduced adversarial sample size).'})
                current_strategy_config = _downscale_strategy_config(current_strategy_config)
                _remove_container(container_name)
                shutil.rmtree(temp_dir, ignore_errors=True)
                continue
            output_json_path = os.path.join(output_dir, 'test_results.json')
            if returncode == 0 and os.path.exists(output_json_path):
                with open(output_json_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                results['sandbox_status'] = 'executed'
                results.setdefault('telemetry', {})['exit_code'] = returncode
                results['telemetry']['container_id'] = container_name
                results['telemetry']['attempts'] = attempt
                results['telemetry']['oom_recovered'] = attempt > 1
                publish_telemetry(audit_id, {'event': 'container_finished', 'container_id': container_name, 'exit_code': returncode, 'attempts': attempt})
                return results
            error_reason = f'Container was killed by the OOM killer and the retry budget ({max_oom_retries}) was exhausted.' if oom_killed else f'Container finished without producing output. Exit code: {returncode}. Stderr: {stderr}'
            publish_telemetry(audit_id, {'event': 'container_error', 'container_id': container_name, 'exit_code': returncode, 'oom_killed': oom_killed, 'stderr': stderr})
            return {'sandbox_status': 'error', 'error': error_reason, 'telemetry': {'exit_code': returncode, 'stderr': stderr, 'oom_killed': oom_killed, 'attempts': attempt}}
        except Exception as e:
            stop_stats_event.set()
            publish_telemetry(audit_id, {'event': 'container_error', 'container_id': container_name, 'error': str(e)})
            return {'sandbox_status': 'error', 'error': f'Sandbox execution failed: {str(e)}', 'telemetry': {'exception': str(e)}}
        finally:
            _remove_container(container_name)
            shutil.rmtree(temp_dir, ignore_errors=True)

def _parse_memory_limit(limit_str: str) -> int:
    units = {'k': 1024, 'm': 1024 ** 2, 'g': 1024 ** 3}
    limit_str = limit_str.strip().lower()
    if limit_str and limit_str[-1] in units:
        try:
            return int(float(limit_str[:-1]) * units[limit_str[-1]])
        except ValueError:
            pass
    try:
        return int(limit_str)
    except ValueError:
        return 4 * 1024 ** 3

def _build_fail_closed_skip_response(planned_tests: List[str], reason: str) -> Dict[str, Any]:

    def _make_skip(vid: str, name: str) -> Dict[str, Any]:
        return {'vulnerability_id': vid, 'vulnerability_name': name, 'status': 'unverified', 'severity': None, 'evidence': {'status': 'skipped', 'reason': reason}, 'summary': 'Dynamic test skipped: Docker sandbox isolation required by Zero-Trust policy.'}
    return {'sandbox_status': 'skipped_zero_trust', 'planned_tests': planned_tests, 'poisoning_evidence': _make_skip('V1', 'Data Poisoning'), 'preprocessing_evidence': _make_skip('V2', 'Preprocessing Attack Surface'), 'validation_evidence': _make_skip('V3', 'Data Validation Weaknesses'), 'adversarial_evidence': _make_skip('V4', 'Adversarial Robustness'), 'execution_log': ['Zero-Trust Policy Gate: Docker daemon offline.', 'Dynamic testing aborted on host to prevent untrusted code execution.', 'Passed unverified status to Agent 3 for static-only reporting.'], 'telemetry': {'execution_mode': 'fail_closed_skip', 'docker_available': False}}
