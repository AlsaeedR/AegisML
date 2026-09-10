from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, END

from .state import PipelineAgentState
from .tools import (
    run_ast_extractor,
    run_networkx_builder,
    validate_threat_model_schema,
    validate_vulnerabilities_schema,
)
from .steps import (
    generate_threat_model_step,
    generate_vulnerabilities_step,
)


def node_extract_pipeline(state: PipelineAgentState) -> Dict[str, Any]:
    """
    Perception node: uses Python AST and NetworkX tools to construct
    the structural pipeline graph and analyze its topology.
    """
    code = state["code"]

    pipeline_graph = run_ast_extractor(code)
    networkx_graph, topology = run_networkx_builder(pipeline_graph)

    return {
        "pipeline_graph": pipeline_graph,
        "networkx_graph": networkx_graph,
        "graph_topology": topology,
        "retry_count": 0,
        "max_retries": 3,
        "validation_errors": None,
        "status": "pipeline_extracted",
    }


def node_reason_threat_model(state: PipelineAgentState) -> Dict[str, Any]:
    """
    Reasoning node: synthesizes code, AST findings, and graph topology
    to produce the NIST AI 100-2e2025 deployment context and threat model.
    """
    code = state["code"]
    pipeline_graph = state["pipeline_graph"]
    topology = state.get("graph_topology", {})
    validation_errors = state.get("validation_errors")

    threat_model_raw = generate_threat_model_step(
        code=code,
        pipeline_graph=pipeline_graph,
        graph_topology=topology,
        validation_errors=validation_errors,
    )

    return {
        "threat_model": threat_model_raw,
    }


def node_validate_threat_model(state: PipelineAgentState) -> Dict[str, Any]:
    """
    Validation node: runs Pydantic verification on the generated threat model.
    If errors are detected, registers diagnostics for self-correction.
    """
    raw_threat_model = state.get("threat_model", {})
    is_valid, errors, validated_model = validate_threat_model_schema(raw_threat_model)

    if is_valid and validated_model is not None:
        return {
            "threat_model": validated_model.model_dump(),
            "validation_errors": None,
            "retry_count": 0,
            "status": "threat_model_validated",
        }

    current_retries = state.get("retry_count", 0) + 1
    return {
        "validation_errors": errors,
        "retry_count": current_retries,
        "status": "threat_model_validation_failed",
    }


def route_after_threat_model_validation(
    state: PipelineAgentState,
) -> Literal["reason_threat_model", "reason_vulnerabilities"]:
    """
    Conditional router: implements the reflection loop if validation fails,
    or advances to vulnerability reasoning once valid.
    """
    has_errors = state.get("validation_errors") is not None
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)

    if has_errors and retry_count < max_retries:
        return "reason_threat_model"

    return "reason_vulnerabilities"


def node_reason_vulnerabilities(state: PipelineAgentState) -> Dict[str, Any]:
    """
    Reasoning node: analyzes the pipeline against the four MVP vulnerability
    classes and produces concrete remediation recommendations.
    """
    code = state["code"]
    pipeline_graph = state["pipeline_graph"]
    threat_model = state["threat_model"]
    validation_errors = state.get("validation_errors")

    vulnerabilities_raw = generate_vulnerabilities_step(
        code=code,
        pipeline_graph=pipeline_graph,
        threat_model=threat_model,
        validation_errors=validation_errors,
    )

    # Agent 1 focuses purely on qualitative threat modeling and vulnerability identification.
    # Mathematical risk scoring (both theoretical baseline and empirical) is centralized in Agent 3.
    return {
        "vulnerability_findings": vulnerabilities_raw,
    }


def node_validate_vulnerabilities(state: PipelineAgentState) -> Dict[str, Any]:
    """
    Validation node: evaluates the vulnerability report against Pydantic schema
    and confirms coverage of all four MVP classes.
    """
    raw_findings = state.get("vulnerability_findings", {})
    is_valid, errors, validated_report = validate_vulnerabilities_schema(raw_findings)

    if is_valid and validated_report is not None:
        return {
            "vulnerability_findings": validated_report.model_dump(),
            "validation_errors": None,
            "retry_count": 0,
            "status": "vulnerabilities_validated",
        }

    current_retries = state.get("retry_count", 0) + 1
    return {
        "validation_errors": errors,
        "retry_count": current_retries,
        "status": "vulnerability_validation_failed",
    }


def route_after_vulnerability_validation(
    state: PipelineAgentState,
) -> Literal["reason_vulnerabilities", "finalize_agent_results"]:
    """
    Conditional router: loops back to refine vulnerabilities if schema validation fails,
    or transitions to finalization once valid.
    """
    has_errors = state.get("validation_errors") is not None
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)

    if has_errors and retry_count < max_retries:
        return "reason_vulnerabilities"

    return "finalize_agent_results"


def node_finalize_agent_results(state: PipelineAgentState) -> Dict[str, Any]:
    """
    Final node: consolidates the validated pipeline graph, threat model, and
    vulnerability findings into the completed Agent 1 state.
    """
    return {
        "status": "completed",
        "validation_errors": None,
    }


def build_pipeline_agent_graph():
    """
    Assembles and compiles the StateGraph for Agent 1, wiring perception,
    reasoning, validation, and self-correction reflection edges.
    """
    workflow = StateGraph(PipelineAgentState)

    # Register workflow nodes
    workflow.add_node("extract_pipeline", node_extract_pipeline)
    workflow.add_node("reason_threat_model", node_reason_threat_model)
    workflow.add_node("validate_threat_model", node_validate_threat_model)
    workflow.add_node("reason_vulnerabilities", node_reason_vulnerabilities)
    workflow.add_node("validate_vulnerabilities", node_validate_vulnerabilities)
    workflow.add_node("finalize_agent_results", node_finalize_agent_results)

    # Establish entry point and linear connections
    workflow.set_entry_point("extract_pipeline")
    workflow.add_edge("extract_pipeline", "reason_threat_model")
    workflow.add_edge("reason_threat_model", "validate_threat_model")

    # Conditional reflection edge for threat modeling
    workflow.add_conditional_edges(
        "validate_threat_model",
        route_after_threat_model_validation,
        {
            "reason_threat_model": "reason_threat_model",
            "reason_vulnerabilities": "reason_vulnerabilities",
        },
    )

    workflow.add_edge("reason_vulnerabilities", "validate_vulnerabilities")

    # Conditional reflection edge for vulnerability identification
    workflow.add_conditional_edges(
        "validate_vulnerabilities",
        route_after_vulnerability_validation,
        {
            "reason_vulnerabilities": "reason_vulnerabilities",
            "finalize_agent_results": "finalize_agent_results",
        },
    )

    workflow.add_edge("finalize_agent_results", END)

    return workflow.compile()
