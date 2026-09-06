from langgraph.graph import StateGraph, END

from .state import PipelineAgentState
from .steps import (
    extract_pipeline_from_code,
    build_threat_model,
    identify_vulnerabilities,
)
from .code_parser import build_networkx_graph


def node_extract_pipeline(state: PipelineAgentState):
    code = state["code"]

    pipeline_graph = extract_pipeline_from_code(code)
    networkx_graph = build_networkx_graph(pipeline_graph)

    return {
        "pipeline_graph": pipeline_graph,
        "networkx_graph": networkx_graph,
    }


def node_build_threat_model(state: PipelineAgentState):
    code = state["code"]
    pipeline_graph = state["pipeline_graph"]
    testing_agent_results = state.get("testing_agent_results")

    threat_model = build_threat_model(code, pipeline_graph, testing_agent_results)

    return {
        "threat_model": threat_model,
    }


def node_identify_vulnerabilities(state: PipelineAgentState):
    code = state["code"]
    pipeline_graph = state["pipeline_graph"]
    threat_model = state["threat_model"]
    testing_agent_results = state.get("testing_agent_results")

    vuln = identify_vulnerabilities(
        code,
        pipeline_graph,
        threat_model,
        testing_agent_results,
    )

    return {
        "vulnerability_findings": vuln,
    }


def build_pipeline_agent_graph():
    graph = StateGraph(PipelineAgentState)

    graph.add_node("extract_pipeline", node_extract_pipeline)
    graph.add_node("build_threat_model", node_build_threat_model)
    graph.add_node("identify_vulnerabilities", node_identify_vulnerabilities)

    graph.set_entry_point("extract_pipeline")
    graph.add_edge("extract_pipeline", "build_threat_model")
    graph.add_edge("build_threat_model", "identify_vulnerabilities")
    graph.add_edge("identify_vulnerabilities", END)

    return graph.compile()
