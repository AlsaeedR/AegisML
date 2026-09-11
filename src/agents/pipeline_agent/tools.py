from typing import Dict, Any, Tuple, Optional
import networkx as nx
from pydantic import ValidationError

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

