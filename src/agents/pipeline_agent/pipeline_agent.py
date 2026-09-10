from typing import Dict, Any, Optional
from .graph import build_pipeline_agent_graph
from .state import PipelineAgentState


def run_pipeline_agent(
    code: str,
    testing_agent_results: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Executes Agent 1 (Pipeline & Threat Modeling Agent).

    Takes the raw Python source code of a machine learning pipeline, constructs the
    structural execution graph using AST and NetworkX, derives a NIST AI 100-2e2025
    threat model, identifies the four MVP vulnerabilities through static threat analysis,
    and runs self-correction loops via Pydantic validation.

    Parameters:
        code: Python source code of the pipeline to analyze.
        testing_agent_results: Optional slot for downstream Agent 2 results.

    Returns:
        Dictionary containing the validated pipeline_graph, threat_model, and
        vulnerability_findings.
    """
    app = build_pipeline_agent_graph()

    initial_state: PipelineAgentState = {
        "code": code,
        "testing_agent_results": testing_agent_results or {},
        "retry_count": 0,
        "max_retries": 3,
        "status": "initialized",
    }

    return app.invoke(initial_state)
