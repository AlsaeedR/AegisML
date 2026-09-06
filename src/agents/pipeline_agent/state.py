from typing import TypedDict, Any, Dict

class PipelineAgentState(TypedDict, total=False):
    code: str
    testing_agent_results: Dict[str, Any]

    pipeline_graph: Dict[str, Any]
    networkx_graph: Any
    threat_model: Dict[str, Any]
    vulnerability_findings: Dict[str, Any]
