import ast
from typing import Dict, Any, List, Tuple, Optional
import networkx as nx
from pydantic import ValidationError
from langchain_core.tools import tool

from .code_parser import parse_python_pipeline
from .networkx_utils import build_networkx_graph, summarize_graph_topology
from .schemas import ThreatModel, VulnerabilitiesReport


def run_ast_extractor(code: str) -> Dict[str, Any]:
    """
    Tool wrapping Python AST extraction to discover components, routines,
    and sequence edges from Python ML pipeline code.
    """
    return parse_python_pipeline(code)


def run_networkx_builder(pipeline_graph: Dict[str, Any]) -> Tuple[nx.DiGraph, Dict[str, Any]]:
    """
    Tool wrapping NetworkX graph construction and topological structure analysis.
    """
    graph = build_networkx_graph(pipeline_graph)
    topology = summarize_graph_topology(graph)
    return graph, topology


def validate_threat_model_schema(raw_data: Any) -> Tuple[bool, Optional[str], Optional[ThreatModel]]:
    """
    Tool wrapping Pydantic validation for the extracted threat model.
    Returns a success flag, an error diagnostic string if invalid, and the validated model.
    """
    try:
        validated = ThreatModel.model_validate(raw_data)
        return True, None, validated
    except ValidationError as e:
        error_lines = []
        for err in e.errors():
            loc = " -> ".join(str(p) for p in err.get("loc", []))
            msg = err.get("msg", "Invalid value")
            error_lines.append(f"Field '{loc}': {msg}")
        return False, "\n".join(error_lines), None


def validate_threat_model_semantics(
    raw_data: Any,
    pipeline_graph: Dict[str, Any],
) -> Tuple[bool, Optional[str]]:
    """
    Semantic validation for the threat model.
    Verifies that pipeline components and trust boundaries correspond to
    grounded nodes discovered in the extracted pipeline AST.
    """
    nodes: List[Dict[str, Any]] = pipeline_graph.get("nodes", [])
    if not nodes:
        return True, None

    valid_node_ids = {n.get("id", "").lower() for n in nodes if n.get("id")}
    valid_node_descriptions = {n.get("description", "").lower() for n in nodes if n.get("description")}

    # Extract components from raw_data
    pipeline_components = []
    if isinstance(raw_data, dict):
        pipeline_components = raw_data.get("pipeline_components", [])
        if not pipeline_components and "deployment_context" in raw_data:
            pipeline_components = raw_data["deployment_context"].get("pipeline_components", [])

    ungrounded_components = []
    for comp in pipeline_components:
        comp_name = comp if isinstance(comp, str) else comp.get("name", "") or comp.get("id", "")
        comp_lower = comp_name.lower().strip()
        if not comp_lower:
            continue

        # Check if matched by node id, description, or substring
        matched = (
            comp_lower in valid_node_ids
            or any(comp_lower in nid for nid in valid_node_ids)
            or any(comp_lower in ndesc for ndesc in valid_node_descriptions)
            or any(nid in comp_lower for nid in valid_node_ids)
        )
        if not matched:
            ungrounded_components.append(comp_name)

    if ungrounded_components:
        available_ids = [n.get("id") for n in nodes if n.get("id")]
        return False, (
            f"Semantic Error: The following components in the threat model are not grounded in "
            f"the extracted AST pipeline nodes: {ungrounded_components}. "
            f"Ground your findings strictly in the discovered AST nodes: {available_ids}."
        )

    return True, None


def validate_vulnerabilities_schema(raw_data: Any) -> Tuple[bool, Optional[str], Optional[VulnerabilitiesReport]]:
    """
    Tool wrapping Pydantic validation for the four MVP threat evaluation findings.
    Ensures all four threat classes are audited with valid discriminative statuses.
    """
    try:
        validated = VulnerabilitiesReport.model_validate(raw_data)

        # Confirm all 4 MVP categories are represented
        required_categories = {
            "Data Poisoning",
            "Preprocessing Attack Surface",
            "Data Validation Weaknesses",
            "Adversarial Robustness",
        }
        present_categories = {v.category for v in validated.vulnerabilities}
        missing = required_categories - present_categories
        if missing:
            return False, f"Missing required MVP vulnerability categories: {', '.join(missing)}", None

        return True, None, validated
    except ValidationError as e:
        error_lines = []
        for err in e.errors():
            loc = " -> ".join(str(p) for p in err.get("loc", []))
            msg = err.get("msg", "Invalid value")
            error_lines.append(f"Field '{loc}': {msg}")
        return False, "\n".join(error_lines), None


def validate_vulnerabilities_semantics(
    validated_report: VulnerabilitiesReport,
    pipeline_graph: Dict[str, Any],
) -> Tuple[bool, Optional[str]]:
    """
    Semantic validation for vulnerability findings.
    Verifies that affected_components match discovered AST nodes, and confirms
    that 'mitigated' findings supply explicit mitigating controls.
    """
    nodes: List[Dict[str, Any]] = pipeline_graph.get("nodes", [])
    if not nodes:
        return True, None

    valid_node_ids = {n.get("id", "").lower() for n in nodes if n.get("id")}
    valid_node_descriptions = {n.get("description", "").lower() for n in nodes if n.get("description")}

    semantic_errors: List[str] = []

    for v in validated_report.vulnerabilities:
        # Check affected components
        for comp in v.affected_components:
            comp_lower = comp.lower().strip()
            if not comp_lower:
                continue
            matched = (
                comp_lower in valid_node_ids
                or any(comp_lower in nid for nid in valid_node_ids)
                or any(comp_lower in ndesc for ndesc in valid_node_descriptions)
                or any(nid in comp_lower for nid in valid_node_ids)
            )
            if not matched:
                available_ids = [n.get("id") for n in nodes if n.get("id")]
                semantic_errors.append(
                    f"Finding '{v.category}' references affected_component '{comp}' which is not in "
                    f"the extracted AST graph. Discovered nodes: {available_ids}."
                )

        # Check mitigated status justifications
        if v.status == "mitigated":
            has_controls = False
            if isinstance(v.mitigating_controls, list):
                has_controls = any(isinstance(c, str) and len(c.strip()) >= 5 for c in v.mitigating_controls)
            elif isinstance(v.mitigating_controls, str):
                has_controls = len(v.mitigating_controls.strip()) >= 5
            if not has_controls:
                semantic_errors.append(
                    f"Finding '{v.category}' is marked 'mitigated' but lacks specific mitigating_controls documentation."
                )

    if semantic_errors:
        return False, "\n".join(semantic_errors)

    return True, None


def make_graph_analysis_tools(
    pipeline_graph: Dict[str, Any],
    code: str = "",
    networkx_graph: Optional[nx.DiGraph] = None,
) -> List[Any]:
    """
    Factory that produces an expanded suite of LangChain @tool functions pre-bound
    to the current pipeline_graph, networkx_graph, and source code.

    Tools provided:
    - query_trust_boundaries(): locates entry points and external ingress nodes.
    - query_components_by_type(component_type): filters pipeline nodes by functional role.
    - trace_node_lineage(node_id): traces upstream sources and downstream sinks in NetworkX.
    - inspect_component_source(component_id): extracts exact source code snippet and lines.
    """
    nodes: List[Dict[str, Any]] = pipeline_graph.get("nodes", [])
    edges: List[Dict[str, Any]] = pipeline_graph.get("edges", [])

    # Ensure we have a NetworkX graph instance
    if networkx_graph is None:
        graph = build_networkx_graph(pipeline_graph)
    else:
        graph = networkx_graph

    @tool
    def query_trust_boundaries() -> Dict[str, Any]:
        """
        Identify pipeline components that are trust boundary candidates.
        Returns nodes with no incoming edges (entry points) and nodes whose
        name suggests external data ingestion (load, read, input, fetch, ingest).
        Call this to locate where untrusted data first enters the NLP pipeline.
        """
        has_incoming = {
            e.get("target") or e.get("to")
            for e in edges
            if e.get("target") or e.get("to")
        }

        ingestion_keywords = ["load", "read", "input", "fetch", "ingest", "source", "import"]

        boundary_nodes = [
            n for n in nodes
            if n.get("id", "") not in has_incoming
            or any(kw in n.get("id", "").lower() for kw in ingestion_keywords)
        ]
        return {
            "trust_boundary_candidates": boundary_nodes,
            "total": len(boundary_nodes),
            "note": "These are entry points or ingestion nodes where untrusted data crosses into the pipeline.",
        }

    @tool
    def query_components_by_type(component_type: str) -> Dict[str, Any]:
        """
        Filter pipeline nodes by a keyword matched against their id or type fields.
        Use this to precisely locate specific functional steps before writing the threat model.

        Example keywords for NLP pipelines:
        - 'vectoriz' or 'tfidf'  -> find the feature extraction step
        - 'classif' or 'model'   -> find the classifier
        - 'clean' or 'preprocess' -> find text cleaning steps
        - 'train' or 'fit'       -> find training steps
        - 'predict' or 'infer'   -> find inference steps

        Args:
            component_type: keyword to search for in component names/types (case-insensitive).
        """
        kw = component_type.lower()
        matched = [
            n for n in nodes
            if kw in n.get("id", "").lower() or kw in n.get("type", "").lower()
        ]
        return {
            "matched_components": matched,
            "total": len(matched),
            "keyword_used": component_type,
        }

    @tool
    def trace_node_lineage(node_id: str) -> Dict[str, Any]:
        """
        Trace upstream sources (predecessors) and downstream sinks (successors)
        for a specific component node ID in the pipeline graph.
        Use this to verify dataflow paths, trust boundaries, and whether untrusted data reaches the model.

        Args:
            node_id: The node identifier to inspect (e.g. 'data_ingestion_1', 'preprocessing_2').
        """
        if node_id not in graph:
            # Attempt case-insensitive match
            matched_id = next((n for n in graph.nodes if n.lower() == node_id.lower()), None)
            if not matched_id:
                return {
                    "error": f"Node '{node_id}' not found in pipeline graph.",
                    "available_nodes": list(graph.nodes),
                }
            node_id = matched_id

        preds = list(graph.predecessors(node_id))
        succs = list(graph.successors(node_id))
        ancestors = list(nx.ancestors(graph, node_id)) if graph.has_node(node_id) else []
        descendants = list(nx.descendants(graph, node_id)) if graph.has_node(node_id) else []

        node_attrs = dict(graph.nodes[node_id])
        return {
            "node_id": node_id,
            "node_attributes": node_attrs,
            "immediate_predecessors": preds,
            "immediate_successors": succs,
            "all_upstream_ancestors": ancestors,
            "all_downstream_descendants": descendants,
        }

    @tool
    def inspect_component_source(component_id: str) -> Dict[str, Any]:
        """
        Inspect the exact Python source code slice corresponding to a pipeline component.
        Use this to verify whether sanitization, length clamping, or input validation exists in the code.

        Args:
            component_id: The node identifier (e.g. 'preprocessing_2', 'data_ingestion_1') or function keyword.
        """
        if not code:
            return {"error": "Source code not loaded in tool context."}

        target_node = None
        for n in nodes:
            if n.get("id", "").lower() == component_id.lower():
                target_node = n
                break

        line_no = target_node.get("line_number") if target_node else None
        code_lines = code.splitlines()

        if line_no and 1 <= line_no <= len(code_lines):
            # Extract a context window around the line
            start = max(0, line_no - 4)
            end = min(len(code_lines), line_no + 8)
            snippet = "\n".join(f"{i+1}: {code_lines[i]}" for i in range(start, end))
            return {
                "component_id": component_id,
                "line_number": line_no,
                "source_snippet": snippet,
                "node_description": target_node.get("description") if target_node else "",
            }

        # Fallback: search for keyword in code
        kw = component_id.lower()
        matched_lines = [
            f"{i+1}: {line}"
            for i, line in enumerate(code_lines)
            if kw in line.lower()
        ]
        if matched_lines:
            return {
                "component_id": component_id,
                "matched_lines": matched_lines[:10],
                "total_matches": len(matched_lines),
            }

        return {
            "error": f"Could not locate source code for component '{component_id}'.",
            "available_nodes": [n.get("id") for n in nodes if n.get("id")],
        }

    return [
        query_trust_boundaries,
        query_components_by_type,
        trace_node_lineage,
        inspect_component_source,
    ]
