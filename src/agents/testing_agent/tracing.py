from __future__ import annotations
import functools
import os
import time
from typing import Any, Callable, Dict
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode
_TRACER_PROVIDER_INITIALIZED = False

def _summarize_state_for_span(state: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    if 'dataset_path' in state:
        summary['aegisml.dataset_path'] = str(state.get('dataset_path'))
    if 'model_path' in state:
        summary['aegisml.model_path'] = str(state.get('model_path'))
    if 'planned_tests' in state and state.get('planned_tests'):
        summary['aegisml.planned_tests'] = ','.join(state.get('planned_tests') or [])
    if 'sandbox_status' in state and state.get('sandbox_status'):
        summary['aegisml.sandbox_status'] = str(state.get('sandbox_status'))
    if 'status' in state and state.get('status'):
        summary['aegisml.status'] = str(state.get('status'))
    return summary

def _init_tracer_provider() -> None:
    global _TRACER_PROVIDER_INITIALIZED
    if _TRACER_PROVIDER_INITIALIZED:
        return
    resource = Resource.create({'service.name': 'aegisml-testing-agent'})
    provider = TracerProvider(resource=resource)
    exporter_kind = os.getenv('AEGISML_OTEL_EXPORTER', 'console').lower()
    if exporter_kind == 'otlp':
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            endpoint = os.getenv('AEGISML_OTEL_ENDPOINT', 'localhost:4317')
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
        except ImportError:
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    else:
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    _TRACER_PROVIDER_INITIALIZED = True

def _langsmith_available() -> bool:
    return bool(os.getenv('LANGCHAIN_TRACING_V2', '').lower() == 'true' and os.getenv('LANGCHAIN_API_KEY'))

def traced_node(node_name: str) -> Callable:
    _init_tracer_provider()
    tracer = trace.get_tracer('aegisml.testing_agent')

    def decorator(node_fn: Callable) -> Callable:

        @functools.wraps(node_fn)
        def otel_wrapped(state: Dict[str, Any]) -> Dict[str, Any]:
            start = time.time()
            with tracer.start_as_current_span(f'agent2.node.{node_name}') as span:
                span.set_attribute('aegisml.node_name', node_name)
                for key, value in _summarize_state_for_span(state).items():
                    span.set_attribute(key, value)
                try:
                    result = node_fn(state)
                    span.set_attribute('aegisml.duration_seconds', round(time.time() - start, 4))
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as exc:
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
                    span.record_exception(exc)
                    raise
        if not _langsmith_available():
            return otel_wrapped
        from langsmith import traceable

        @functools.wraps(node_fn)
        @traceable(name=f'agent2.{node_name}', run_type='chain')
        def fully_wrapped(state: Dict[str, Any]) -> Dict[str, Any]:
            return otel_wrapped(state)
        return fully_wrapped
    return decorator
