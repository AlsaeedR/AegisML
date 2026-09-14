import ast
from typing import Any, Dict, List, Optional, Tuple
import networkx as nx

NODE_TYPE_COLORS = {
    "data_ingestion": "#B45309",
    "preprocessing": "#2563EB",
    "preprocessing_routine": "#3B82F6",
    "model_training": "#7C3AED",
    "training_routine": "#8B5CF6",
    "inference": "#059669",
    "validation": "#DC2626",
    "persistence": "#4B5563",
    "error": "#991B1B",
    "unclassified": "#6B7280",
}


class PipelineExtractor(ast.NodeVisitor):
    """
    AST visitor that scans Python source code to identify core machine learning
    pipeline components and build a structured representation of the execution graph.
    """

    def __init__(self, source_code: str):
        self.nodes: List[Dict[str, Any]] = []
        self.edges: List[Dict[str, str]] = []
        self.last_node_id: Optional[str] = None
        self.source_code = source_code
        self.source_lines = source_code.splitlines()

    def _get_source_snippet(self, line_number: Optional[int], radius: int = 2) -> str:
        if not line_number or line_number < 1 or line_number > len(self.source_lines):
            return ""
        start = max(1, line_number - radius)
        end = min(len(self.source_lines), line_number + radius)
        return "\n".join(
            f"{line_no:>4} | {self.source_lines[line_no - 1]}"
            for line_no in range(start, end + 1)
        )

    def _add_node(
        self,
        step_type: str,
        description: str,
        line_number: Optional[int] = None,
        ast_node_type: Optional[str] = None,
        name: Optional[str] = None,
    ) -> str:
        node_id = f"{step_type}_{len(self.nodes) + 1}"
        node_data = {
            "id": node_id,
            "name": name or description,
            "type": step_type,
            "component_type": step_type,
            "description": description,
            "line_number": line_number,
            "ast_node_type": ast_node_type or "Unknown",
            "source_snippet": self._get_source_snippet(line_number),
        }
        self.nodes.append(node_data)
        if self.last_node_id is not None:
            self.edges.append({
                "from": self.last_node_id,
                "to": node_id,
                "source": self.last_node_id,
                "target": node_id,
            })
        self.last_node_id = node_id
        return node_id

    def visit_Call(self, node: ast.Call):
        call_name = ""
        if isinstance(node.func, ast.Attribute):
            call_name = node.func.attr.lower()
        elif isinstance(node.func, ast.Name):
            call_name = node.func.id.lower()

        line = getattr(node, "lineno", None)
        ast_type = type(node).__name__

        ingestion_identifiers = {
            "read_csv", "read_json", "read_parquet", "read_excel", "read_table",
            "load_dataset", "open", "input", "file_uploader", "from_csv",
        }
        if call_name in ingestion_identifiers:
            self._add_node(
                step_type="data_ingestion",
                description=f"Loads data via '{call_name}'",
                line_number=line,
                ast_node_type=ast_type,
                name=call_name,
            )

        preprocessing_keywords = {
            "preprocess", "clean", "tokenize", "stem", "normalize",
            "transform", "vectorize",
        }
        if any(keyword in call_name for keyword in preprocessing_keywords):
            if call_name not in ingestion_identifiers:
                self._add_node(
                    step_type="preprocessing",
                    description=f"Executes data transformation via '{call_name}'",
                    line_number=line,
                    ast_node_type=ast_type,
                    name=call_name,
                )

        training_identifiers = {"fit", "fit_transform", "train"}
        if call_name in training_identifiers:
            self._add_node(
                step_type="model_training",
                description=f"Fits model or feature extractor via '{call_name}'",
                line_number=line,
                ast_node_type=ast_type,
                name=call_name,
            )

        inference_identifiers = {"predict", "predict_proba", "infer", "evaluate", "score"}
        if call_name in inference_identifiers:
            self._add_node(
                step_type="inference",
                description=f"Executes model inference or evaluation via '{call_name}'",
                line_number=line,
                ast_node_type=ast_type,
                name=call_name,
            )

        serialization_identifiers = {"dump", "save", "save_weights", "to_pickle", "to_parquet"}
        if call_name in serialization_identifiers:
            self._add_node(
                step_type="persistence",
                description=f"Serializes artifact via '{call_name}'",
                line_number=line,
                ast_node_type=ast_type,
                name=call_name,
            )

        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        name_lower = node.name.lower()
        line = getattr(node, "lineno", None)
        ast_type = type(node).__name__

        if any(keyword in name_lower for keyword in ["preprocess", "clean", "sanitize", "transform"]):
            self._add_node(
                step_type="preprocessing_routine",
                description=f"Defines data processing routine '{node.name}'",
                line_number=line,
                ast_node_type=ast_type,
                name=node.name,
            )
        elif any(keyword in name_lower for keyword in ["train", "fit", "evaluate"]):
            self._add_node(
                step_type="training_routine",
                description=f"Defines model pipeline routine '{node.name}'",
                line_number=line,
                ast_node_type=ast_type,
                name=node.name,
            )
        self.generic_visit(node)

    def visit_If(self, node: ast.If):
        line = getattr(node, "lineno", None)
        ast_type = type(node).__name__

        if isinstance(node.test, ast.Call):
            if isinstance(node.test.func, ast.Name) and node.test.func.id == "isinstance":
                self._add_node(
                    step_type="validation",
                    description="Type verification using isinstance",
                    line_number=line,
                    ast_node_type=ast_type,
                    name="isinstance validation",
                )
            elif isinstance(node.test.func, ast.Attribute) and node.test.func.attr.lower() in {"isna", "isnull", "empty"}:
                self._add_node(
                    step_type="validation",
                    description="Null or empty input verification",
                    line_number=line,
                    ast_node_type=ast_type,
                    name="null validation",
                )
        elif isinstance(node.test, ast.Compare):
            self._add_node(
                step_type="validation",
                description="Conditional range or equality check",
                line_number=line,
                ast_node_type=ast_type,
                name="conditional validation",
            )
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert):
        line = getattr(node, "lineno", None)
        self._add_node(
            step_type="validation",
            description="Validation assertion via assert statement",
            line_number=line,
            ast_node_type=type(node).__name__,
            name="assert validation",
        )
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try):
        line = getattr(node, "lineno", None)
        self._add_node(
            step_type="validation",
            description="Defensive exception handling via try/except block",
            line_number=line,
            ast_node_type=type(node).__name__,
            name="exception handling",
        )
        self.generic_visit(node)


def parse_python_pipeline(code: str) -> Dict[str, Any]:
    """
    Parse Python pipeline source code and return structured nodes and edges.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return {
            "nodes": [
                {
                    "id": "parse_error_1",
                    "name": "Syntax error",
                    "type": "error",
                    "component_type": "error",
                    "description": f"Syntax error during code parsing: {exc}",
                    "line_number": getattr(exc, "lineno", None),
                    "ast_node_type": "SyntaxError",
                    "source_snippet": "",
                }
            ],
            "edges": [],
        }

    extractor = PipelineExtractor(source_code=code)
    extractor.visit(tree)
    nodes = extractor.nodes
    edges = extractor.edges

    if not nodes:
        nodes = [
            {
                "id": "unclassified_1",
                "name": "Unclassified pipeline",
                "type": "unclassified",
                "component_type": "unclassified",
                "description": "No explicit ML pipeline stages matched static heuristics",
                "line_number": None,
                "ast_node_type": "Module",
                "source_snippet": "",
            }
        ]

    return {"nodes": nodes, "edges": edges}


def parse_pipeline_code(code: str) -> Dict[str, Any]:
    """Alias for parse_python_pipeline."""
    return parse_python_pipeline(code)


def build_networkx_graph(pipeline_graph: Dict[str, Any]) -> nx.DiGraph:
    """Constructs a NetworkX directed graph from pipeline nodes and edges."""
    graph = nx.DiGraph()
    for node in pipeline_graph.get("nodes", []):
        node_id = node.get("id")
        if node_id:
            graph.add_node(
                node_id,
                type=node.get("type"),
                description=node.get("description"),
                line_number=node.get("line_number"),
            )
    for edge in pipeline_graph.get("edges", []):
        source = edge.get("from")
        target = edge.get("to")
        if source and target:
            graph.add_edge(source, target)
    return graph


def compute_layered_layout(
    graph: nx.DiGraph,
    x_gap: int = 260,
    y_gap: int = 120,
) -> Dict[str, Dict[str, float]]:
    """Generates a left-to-right layered topological layout for visualization."""
    if graph.number_of_nodes() == 0:
        return {}

    try:
        layers = list(nx.topological_generations(graph))
    except Exception:
        ordered_nodes = list(graph.nodes())
        layers = [ordered_nodes] if ordered_nodes else []

    layout: Dict[str, Dict[str, float]] = {}
    seen = set()

    for level, layer_nodes in enumerate(layers):
        ordered_nodes = sorted(layer_nodes)
        count = len(ordered_nodes)
        if count == 0:
            continue
        total_height = (count - 1) * y_gap
        for index, node_id in enumerate(ordered_nodes):
            x = level * x_gap
            y = (index * y_gap) - (total_height / 2)
            layout[node_id] = {"x": float(x), "y": float(y), "level": level}
            seen.add(node_id)

    missing_nodes = [node for node in graph.nodes() if node not in seen]
    if missing_nodes:
        extra_level = len(layers)
        count = len(missing_nodes)
        total_height = (count - 1) * y_gap
        for index, node_id in enumerate(sorted(missing_nodes)):
            x = extra_level * x_gap
            y = (index * y_gap) - (total_height / 2)
            layout[node_id] = {"x": float(x), "y": float(y), "level": extra_level}

    return layout


def enrich_pipeline_graph_for_ui(pipeline_graph: Dict[str, Any]) -> Dict[str, Any]:
    """Enriches pipeline nodes with layout coordinates and UI color metadata."""
    graph = build_networkx_graph(pipeline_graph)
    layout = compute_layered_layout(graph)
    enriched_nodes: List[Dict[str, Any]] = []

    for node in pipeline_graph.get("nodes", []):
        node_id = node.get("id")
        coords = layout.get(node_id, {"x": 0.0, "y": 0.0, "level": 0})
        node_type = node.get("type", "unclassified")
        enriched_nodes.append({
            **node,
            "x": coords["x"],
            "y": coords["y"],
            "level": coords["level"],
            "color": NODE_TYPE_COLORS.get(node_type, NODE_TYPE_COLORS["unclassified"]),
        })

    return {
        "nodes": enriched_nodes,
        "edges": pipeline_graph.get("edges", []),
        "topology": summarize_graph_topology(graph),
    }


def summarize_graph_topology(graph: nx.DiGraph) -> Dict[str, Any]:
    """Extracts topological properties (entry/exit nodes, degrees, counts)."""
    if len(graph.nodes) == 0:
        return {
            "total_nodes": 0,
            "total_edges": 0,
            "entry_nodes": [],
            "exit_nodes": [],
            "node_types": {},
        }

    entry_nodes = [n for n in graph.nodes if graph.in_degree(n) == 0]
    exit_nodes = [n for n in graph.nodes if graph.out_degree(n) == 0]
    node_types: Dict[str, int] = {}
    for _, data in graph.nodes(data=True):
        ntype = data.get("type", "unknown")
        node_types[ntype] = node_types.get(ntype, 0) + 1

    return {
        "total_nodes": graph.number_of_nodes(),
        "total_edges": graph.number_of_edges(),
        "entry_nodes": entry_nodes,
        "exit_nodes": exit_nodes,
        "node_types": node_types,
    }

