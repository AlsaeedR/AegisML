import json
import time
from typing import Any, Dict, List, Literal, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from src.core.llm import get_llm, is_llm_available
from src.core.audit_memory import (
    get_step_checkpoint,
    save_step_checkpoint,
    register_audit_artifacts,
    verify_artifact_integrity,
)
from .schemas import PipelineAgentState
from .tools import (
    run_ast_extractor,
    run_networkx_builder,
    validate_threat_model_schema,
    validate_threat_model_semantics,
    validate_vulnerabilities_schema,
    validate_vulnerabilities_semantics,
    make_graph_analysis_tools,
    generate_threat_model_step,
    generate_vulnerabilities_step,
)


# ⭐⭐ STATION 1: this is the function you talk about in your script
def node_extract_pipeline(state: PipelineAgentState) -> Dict[str, Any]:
    """Perception node: uses AST and NetworkX tools to construct the structural graph."""
    audit_id = state.get("audit_id")
    if audit_id:
        cached = get_step_checkpoint(audit_id, "extract_pipeline")
        if cached is not None:
            return cached

    t0 = time.time()
    code = state["code"]
    # ⭐⭐ "run_ast_extractor converts the raw code into the AST"
    pipeline_graph = run_ast_extractor(code)
    # ⭐⭐ "run_networkx_builder takes that AST and builds the graph structure"
    networkx_graph, topology = run_networkx_builder(pipeline_graph)

    result = {
        "pipeline_graph": pipeline_graph,
        "networkx_graph": networkx_graph,
        "graph_topology": topology,
        "retry_count": 0,
        "max_retries": 3,
        "validation_errors": None,
        "status": "pipeline_extracted",
    }
    if audit_id:
        save_step_checkpoint(audit_id, "Agent 1", "extract_pipeline", result, duration_seconds=round(time.time() - t0, 3))
    return result


def node_reason_threat_model(state: PipelineAgentState) -> Dict[str, Any]:
    """Reasoning node: ReAct exploration tool calling, then threat model formulation."""
    audit_id = state.get("audit_id")
    if audit_id and not state.get("validation_errors"):
        cached = get_step_checkpoint(audit_id, "reason_threat_model")
        if cached is not None:
            return cached

    t0 = time.time()
    code = state["code"]
    pipeline_graph = state["pipeline_graph"]
    networkx_graph = state.get("networkx_graph")
    topology = state.get("graph_topology", {})
    validation_errors = state.get("validation_errors")
    tool_context: str = ""

    if is_llm_available():
        graph_tools = make_graph_analysis_tools(
            pipeline_graph=pipeline_graph,
            code=code,
            networkx_graph=networkx_graph,
        )
        tool_map: Dict[str, Any] = {t.name: t for t in graph_tools}
        llm_with_tools = get_llm(temperature=0.0).bind_tools(graph_tools)

        error_note = (
            f"\nPRIOR VALIDATION ERRORS TO FIX:\n{validation_errors}\n"
            if validation_errors else ""
        )

        system_msg = SystemMessage(content=(
            "You are the AegisML Pipeline & Threat Modeling Agent.\n"
            "You have four active exploration tools to probe the pipeline before formulating the threat model:\n\n"
            "- query_trust_boundaries(): returns entry-point nodes and ingestion nodes where untrusted data enters.\n"
            "- query_components_by_type(component_type): filters nodes by functional role.\n"
            "- trace_node_lineage(node_id): traces upstream sources and downstream sinks in the dataflow graph.\n"
            "- inspect_component_source(component_id): extracts exact source code snippet and line numbers.\n\n"
            "Use these tools to ground your understanding of trust boundaries and protected assets. "
            "When finished probing, stop calling tools."
        ))

        human_msg = HumanMessage(content=(
            f"Graph topology summary:\n{json.dumps(topology, indent=2)}\n"
            f"{error_note}"
            "Call exploration tools to investigate trust boundaries and components as needed. "
            "When done, stop calling tools."
        ))

        messages: List[Any] = [system_msg, human_msg]

        # 🔴 the ReAct loop: ask LLM, maybe call a tool, feed result back, repeat
        for _ in range(2):
            response = llm_with_tools.invoke(messages)
            messages.append(response)

            if not response.tool_calls:
                break

            tool_findings: List[str] = []
            for tc in response.tool_calls:
                t_name = tc["name"]
                t_args = tc.get("args", {})
                t_id = tc["id"]

                if t_name in tool_map:
                    result = tool_map[t_name].invoke(t_args)
                    tool_findings.append(f"[{t_name}({t_args})]: {json.dumps(result, indent=2)}")
                else:
                    result = {"error": f"Unknown tool: {t_name}"}
                    tool_findings.append(f"[{t_name}]: rejected — unknown tool.")

                messages.append(ToolMessage(content=str(result), tool_call_id=t_id))

            if tool_findings:
                tool_context += "\n".join(tool_findings) + "\n"

    try:
        threat_model_raw = generate_threat_model_step(
            code=code,
            pipeline_graph=pipeline_graph,
            graph_topology=topology,
            validation_errors=validation_errors,
            tool_context=tool_context or None,
        )
    except Exception as exc:
        current_retries = state.get("retry_count", 0) + 1
        return {
            "threat_model": {},
            "validation_errors": f"Failed to parse LLM output into JSON: {exc}. Output must strictly be valid JSON.",
            "retry_count": current_retries,
            "status": "threat_model_validation_failed",
        }

    result = {"threat_model": threat_model_raw}
    if audit_id:
        save_step_checkpoint(audit_id, "Agent 1", "reason_threat_model", result, duration_seconds=round(time.time() - t0, 3))
    return result


def node_validate_threat_model(state: PipelineAgentState) -> Dict[str, Any]:
    """Validation node: syntactic and semantic verification with diagnostic registration."""
    if state.get("status") == "threat_model_validation_failed" and state.get("validation_errors"):
        return {
            "validation_errors": state.get("validation_errors"),
            "retry_count": state.get("retry_count", 0),
            "status": "threat_model_validation_failed",
        }

    raw_threat_model = state.get("threat_model", {})
    pipeline_graph = state.get("pipeline_graph", {})

    is_valid, errors, validated_model = validate_threat_model_schema(raw_threat_model)
    if not is_valid or validated_model is None:
        current_retries = state.get("retry_count", 0) + 1
        return {
            "validation_errors": errors,
            "retry_count": current_retries,
            "status": "threat_model_validation_failed",
        }

    # 🔴 grounds the LLM's claims against the real AST graph
    sem_valid, sem_errors = validate_threat_model_semantics(raw_threat_model, pipeline_graph)
    if not sem_valid:
        current_retries = state.get("retry_count", 0) + 1
        return {
            "validation_errors": sem_errors,
            "retry_count": current_retries,
            "status": "threat_model_validation_failed",
        }

    result = {
        "threat_model": validated_model.model_dump(),
        "validation_errors": None,
        "retry_count": 0,
        "status": "threat_model_validated",
    }
    audit_id = state.get("audit_id")
    if audit_id:
        save_step_checkpoint(audit_id, "Agent 1", "validate_threat_model", result)
    return result


def route_after_threat_model_validation(
    state: PipelineAgentState,
) -> Literal["reason_threat_model", "reason_vulnerabilities"]:
    """Conditional router: loops back on validation error or advances to vulnerabilities."""
    has_errors = state.get("validation_errors") is not None
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)

    # 🔴 retry logic: max 3 tries before giving up and moving on
    if has_errors and retry_count < max_retries:
        return "reason_threat_model"

    return "reason_vulnerabilities"


# ⭐⭐ STATION 4: this is the second function you talk about in your script
def node_reason_vulnerabilities(state: PipelineAgentState) -> Dict[str, Any]:
    """Reasoning node: actively investigates pipeline controls before formulating findings."""
    audit_id = state.get("audit_id")
    if audit_id and not state.get("validation_errors"):
        cached = get_step_checkpoint(audit_id, "reason_vulnerabilities")
        if cached is not None:
            return cached

    t0 = time.time()
    code = state["code"]
    pipeline_graph = state["pipeline_graph"]
    networkx_graph = state.get("networkx_graph")
    threat_model = state["threat_model"]
    validation_errors = state.get("validation_errors")
    tool_context: str = ""

    if is_llm_available():
        graph_tools = make_graph_analysis_tools(
            pipeline_graph=pipeline_graph,
            code=code,
            networkx_graph=networkx_graph,
        )
        tool_map: Dict[str, Any] = {t.name: t for t in graph_tools}
        llm_with_tools = get_llm(temperature=0.0).bind_tools(graph_tools)

        error_note = (
            f"\nPRIOR VALIDATION ERRORS TO FIX:\n{validation_errors}\n"
            if validation_errors else ""
        )

        # ⭐⭐ "It evaluates four core ML vulnerability classes: V1... V2... V3... V4..."
        system_msg = SystemMessage(content=(
            "You are the AegisML Pipeline Vulnerability Auditor.\n"
            "Analyze the pipeline against the four MVP vulnerability classes:\n"
            "1. V1 - Data Poisoning\n"
            "2. V2 - Preprocessing Attack Surface\n"
            "3. V3 - Data Validation Weaknesses\n"
            "4. V4 - Adversarial Robustness\n\n"
            "You have tools to actively verify whether code implements defensive controls:\n"
            "- inspect_component_source(component_id): inspects exact lines of code.\n"
            "- trace_node_lineage(node_id): traces whether unvalidated inputs reach downstream models.\n"
            "- query_trust_boundaries(): checks data ingress points.\n"
            "- query_components_by_type(component_type): locates specific modules.\n\n"
            "Use these tools to inspect specific components before determining their vulnerability status."
        ))

        human_msg = HumanMessage(content=(
            f"Threat Model Summary:\n{json.dumps(threat_model, indent=2)}\n"
            f"{error_note}"
            "Call inspection tools to verify defensive controls in the code. When finished, stop calling tools."
        ))

        messages: List[Any] = [system_msg, human_msg]

        for _ in range(2):
            response = llm_with_tools.invoke(messages)
            messages.append(response)

            if not response.tool_calls:
                break

            tool_findings: List[str] = []
            for tc in response.tool_calls:
                t_name = tc["name"]
                t_args = tc.get("args", {})
                t_id = tc["id"]

                if t_name in tool_map:
                    result = tool_map[t_name].invoke(t_args)
                    tool_findings.append(f"[{t_name}({t_args})]: {json.dumps(result, indent=2)}")
                else:
                    result = {"error": f"Unknown tool: {t_name}"}
                    tool_findings.append(f"[{t_name}]: rejected — unknown tool.")

                messages.append(ToolMessage(content=str(result), tool_call_id=t_id))

            if tool_findings:
                tool_context += "\n".join(tool_findings) + "\n"

    try:
        vulnerabilities_raw = generate_vulnerabilities_step(
            code=code,
            pipeline_graph=pipeline_graph,
            threat_model=threat_model,
            validation_errors=validation_errors,
            tool_context=tool_context or None,
        )
    except Exception as exc:
        current_retries = state.get("retry_count", 0) + 1
        return {
            "vulnerability_findings": {},
            "validation_errors": f"Failed to parse LLM output into JSON: {exc}. Output must strictly be valid JSON.",
            "retry_count": current_retries,
            "status": "vulnerability_validation_failed",
        }

    result = {"vulnerability_findings": vulnerabilities_raw}
    if audit_id:
        save_step_checkpoint(audit_id, "Agent 1", "reason_vulnerabilities", result, duration_seconds=round(time.time() - t0, 3))
    return result


def node_validate_vulnerabilities(state: PipelineAgentState) -> Dict[str, Any]:
    """Validation node: syntactic and semantic grounding verification for vulnerabilities."""
    if state.get("status") == "vulnerability_validation_failed" and state.get("validation_errors"):
        return {
            "validation_errors": state.get("validation_errors"),
            "retry_count": state.get("retry_count", 0),
            "status": "vulnerability_validation_failed",
        }

    raw_findings = state.get("vulnerability_findings", {})
    pipeline_graph = state.get("pipeline_graph", {})

    is_valid, errors, validated_report = validate_vulnerabilities_schema(raw_findings)
    if not is_valid or validated_report is None:
        current_retries = state.get("retry_count", 0) + 1
        return {
            "validation_errors": errors,
            "retry_count": current_retries,
            "status": "vulnerability_validation_failed",
        }

    sem_valid, sem_errors = validate_vulnerabilities_semantics(validated_report, pipeline_graph)
    if not sem_valid:
        current_retries = state.get("retry_count", 0) + 1
        return {
            "validation_errors": sem_errors,
            "retry_count": current_retries,
            "status": "vulnerability_validation_failed",
        }

    # 🔴 the smart auto-override: no inference call found → V4 forced to "not_applicable"
    has_inference_surface = any(
        str(node.get("type", "")).lower() == "inference"
        or str(node.get("component_type", "")).lower() == "inference"
        for node in pipeline_graph.get("nodes", [])
    )
    if not has_inference_surface:
        for finding in validated_report.vulnerabilities:
            if finding.vulnerability_id == "V4":
                finding.status = "not_applicable"
                finding.mitigating_controls = []

    result = {
        "vulnerability_findings": validated_report.model_dump(),
        "validation_errors": None,
        "retry_count": 0,
        "status": "vulnerabilities_validated",
    }
    audit_id = state.get("audit_id")
    if audit_id:
        save_step_checkpoint(audit_id, "Agent 1", "validate_vulnerabilities", result)
    return result


def route_after_vulnerability_validation(
    state: PipelineAgentState,
) -> Literal["reason_vulnerabilities", "finalize_agent_results"]:
    """Conditional router: loops back to refine findings or finalizes."""
    has_errors = state.get("validation_errors") is not None
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)

    if has_errors and retry_count < max_retries:
        return "reason_vulnerabilities"

    return "finalize_agent_results"


def node_finalize_agent_results(state: PipelineAgentState) -> Dict[str, Any]:
    """Final node: completes the state transition."""
    result = {
        "status": "completed",
        "validation_errors": None,
    }
    audit_id = state.get("audit_id")
    if audit_id:
        save_step_checkpoint(audit_id, "Agent 1", "finalize_agent_results", result)
    return result


# 🔴 wires all the stations above together into one graph, in the right order
def build_pipeline_agent_graph():
    """Assembles and compiles the StateGraph for Agent 1."""
    workflow = StateGraph(PipelineAgentState)

    workflow.add_node("extract_pipeline", node_extract_pipeline)
    workflow.add_node("reason_threat_model", node_reason_threat_model)
    workflow.add_node("validate_threat_model", node_validate_threat_model)
    workflow.add_node("reason_vulnerabilities", node_reason_vulnerabilities)
    workflow.add_node("validate_vulnerabilities", node_validate_vulnerabilities)
    workflow.add_node("finalize_agent_results", node_finalize_agent_results)

    workflow.set_entry_point("extract_pipeline")
    workflow.add_edge("extract_pipeline", "reason_threat_model")
    workflow.add_edge("reason_threat_model", "validate_threat_model")

    workflow.add_conditional_edges(
        "validate_threat_model",
        route_after_threat_model_validation,
        {
            "reason_threat_model": "reason_threat_model",
            "reason_vulnerabilities": "reason_vulnerabilities",
        },
    )

    workflow.add_edge("reason_vulnerabilities", "validate_vulnerabilities")

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


def run_pipeline_agent(
    code: str,
    testing_agent_results: Optional[Dict[str, Any]] = None,
    audit_id: Optional[str] = None,
    force_resume: bool = False,
) -> Dict[str, Any]:
    """Executes Agent 1 (Pipeline & Threat Modeling Agent)."""
    if audit_id:
        verify_artifact_integrity(audit_id, code=code, force=force_resume)

    app = build_pipeline_agent_graph()

    initial_state: PipelineAgentState = {
        "code": code,
        "testing_agent_results": testing_agent_results or {},
        "retry_count": 0,
        "max_retries": 3,
        "status": "initialized",
        "audit_id": audit_id,
    }

    return app.invoke(initial_state)