import ast
from typing import Any, Dict, List, Optional
import networkx as nx

PIPELINE_STAGES = {
    "data_ingestion": {"stage_index": 0, "stage_name": "Data Ingestion", "icon": "input"},
    "validation": {"stage_index": 1, "stage_name": "Validation & Guards", "icon": "shield"},
    "preprocessing": {"stage_index": 2, "stage_name": "Feature Engineering", "icon": "filter"},
    "preprocessing_routine": {"stage_index": 2, "stage_name": "Feature Engineering", "icon": "filter"},
    "model_training": {"stage_index": 3, "stage_name": "Model Architecture", "icon": "cpu"},
    "training_routine": {"stage_index": 3, "stage_name": "Model Architecture", "icon": "cpu"},
    "inference": {"stage_index": 4, "stage_name": "Inference & Decision", "icon": "check"},
    "persistence": {"stage_index": 4, "stage_name": "Inference & Persistence", "icon": "save"},
    "error": {"stage_index": 1, "stage_name": "Validation & Guards", "icon": "alert"},
    "unclassified": {"stage_index": 2, "stage_name": "Pipeline Logic", "icon": "code"},
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

    def _extract_string_arg(self, node: ast.Call, arg_idx: int = 0) -> str:
        """Extract a string literal argument (e.g. filename) from a call node."""
        if len(node.args) > arg_idx and isinstance(node.args[arg_idx], ast.Constant):
            val = str(node.args[arg_idx].value)
            return val.split("/")[-1].split("\\")[-1]
        return ""

    def _add_node(
        self,
        step_type: str,
        description: str,
        line_number: Optional[int] = None,
        ast_node_type: Optional[str] = None,
        name: Optional[str] = None,
    ) -> str:
        node_id = f"{step_type}_{len(self.nodes) + 1}"
        stage_info = PIPELINE_STAGES.get(step_type, PIPELINE_STAGES["unclassified"])
        node_data = {
            "id": node_id,
            "name": name or description,
            "type": step_type,
            "component_type": step_type,
            "stage_index": stage_info["stage_index"],
            "stage_name": stage_info["stage_name"],
            "description": description,
            "line_number": line_number,
            "ast_node_type": ast_node_type or "Unknown",
            "source_snippet": self._get_source_snippet(line_number),
        }
        self.nodes.append(node_data)
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
            "load_dataset", "file_uploader", "from_csv",
        }
        if call_name in ingestion_identifiers:
            filename = self._extract_string_arg(node, 0)
            disp_name = f"pd.{call_name}('{filename}')" if filename else f"pd.{call_name}()"
            self._add_node(
                step_type="data_ingestion",
                description=f"Loads data via '{call_name}'" + (f" ({filename})" if filename else ""),
                line_number=line,
                ast_node_type=ast_type,
                name=disp_name,
            )

        elif call_name in {"fit_transform", "transform", "vectorize", "stem", "tokenize", "normalize"}:
            self._add_node(
                step_type="preprocessing",
                description=f"Executes data transformation via '{call_name}'",
                line_number=line,
                ast_node_type=ast_type,
                name=f"{call_name}()",
            )

        elif call_name in {"fit", "train"}:
            self._add_node(
                step_type="model_training",
                description=f"Fits model or feature extractor via '{call_name}'",
                line_number=line,
                ast_node_type=ast_type,
                name="model.fit()",
            )

        elif call_name in {"predict", "predict_proba", "infer"}:
            self._add_node(
                step_type="inference",
                description=f"Executes model inference via '{call_name}'",
                line_number=line,
                ast_node_type=ast_type,
                name="model.predict()",
            )

        elif call_name in {"dump", "save", "save_weights", "to_pickle", "to_parquet"}:
            target_name = self._extract_string_arg(node, 1) or self._extract_string_arg(node, 0)
            disp_name = f"joblib.dump('{target_name}')" if target_name else "joblib.dump()"
            self._add_node(
                step_type="persistence",
                description=f"Serializes artifact via '{call_name}'" + (f" to '{target_name}'" if target_name else ""),
                line_number=line,
                ast_node_type=ast_type,
                name=disp_name,
            )

        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        name_lower = node.name.lower()
        line = getattr(node, "lineno", None)
        ast_type = type(node).__name__

        if any(keyword in name_lower for keyword in ["preprocess", "clean", "sanitize", "transform", "tokenize"]):
            self._add_node(
                step_type="preprocessing_routine",
                description=f"Defines data processing routine '{node.name}'",
                line_number=line,
                ast_node_type=ast_type,
                name=f"{node.name}()",
            )
        elif any(keyword in name_lower for keyword in ["train", "fit", "evaluate"]):
            self._add_node(
                step_type="training_routine",
                description=f"Defines model pipeline routine '{node.name}'",
                line_number=line,
                ast_node_type=ast_type,
                name=f"{node.name}()",
            )
        self.generic_visit(node)

    def visit_If(self, node: ast.If):
        line = getattr(node, "lineno", None)
        ast_type = type(node).__name__

        test_node = node.test
        if isinstance(test_node, ast.UnaryOp) and isinstance(test_node.op, ast.Not):
            test_node = test_node.operand

        if isinstance(test_node, ast.Call):
            if isinstance(test_node.func, ast.Name) and test_node.func.id == "isinstance":
                self._add_node(
                    step_type="validation",
                    description="Type verification using isinstance",
                    line_number=line,
                    ast_node_type=ast_type,
                    name="isinstance() guard",
                )
            elif isinstance(test_node.func, ast.Attribute) and test_node.func.attr.lower() in {"isna", "isnull", "empty", "isnan"}:
                self._add_node(
                    step_type="validation",
                    description="Null or empty input verification",
                    line_number=line,
                    ast_node_type=ast_type,
                    name="null/empty guard",
                )
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert):
        line = getattr(node, "lineno", None)
        self._add_node(
            step_type="validation",
            description="Validation assertion via assert statement",
            line_number=line,
            ast_node_type=type(node).__name__,
            name="assert statement",
        )
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try):
        line = getattr(node, "lineno", None)
        body_dump = ast.dump(node).lower()
        # Skip package downloads or environment checks like nltk.download
        if "download" in body_dump and "nltk" in body_dump:
            self.generic_visit(node)
            return

        self._add_node(
            step_type="validation",
            description="Defensive exception handling via try/except block",
            line_number=line,
            ast_node_type=type(node).__name__,
            name="try/except guard",
        )
        self.generic_visit(node)


def construct_hierarchical_edges(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Construct a true multi-branch convergent/divergent DAG connecting the 5 pipeline lifecycle stages:
    Stage 0 (Ingestion) -> Stage 1 (Validation Guards) -> Stage 2 (Feature Engineering) ->
    Stage 3 (Model Architecture) -> Stage 4 (Inference & Persistence).
    """
    by_stage: Dict[int, List[Dict[str, Any]]] = {0: [], 1: [], 2: [], 3: [], 4: []}
    for n in nodes:
        stage_idx = int(n.get("stage_index", 2))
        by_stage.setdefault(stage_idx, []).append(n)

    edges: List[Dict[str, Any]] = []
    seen = set()

    def add_edge(src_id: str, dst_id: str, label: str = ""):
        if src_id and dst_id and src_id != dst_id and (src_id, dst_id) not in seen:
            seen.add((src_id, dst_id))
            edges.append({
                "from": src_id,
                "to": dst_id,
                "source": src_id,
                "target": dst_id,
                "label": label,
            })

    ing_nodes = by_stage.get(0, [])
    val_nodes = by_stage.get(1, [])
    prep_nodes = by_stage.get(2, [])
    model_nodes = by_stage.get(3, [])
    out_nodes = by_stage.get(4, [])

    # 1. Stage 0 Ingestion feeds Stage 1 Validation guards or Stage 2 Preprocessing
    if ing_nodes:
        if val_nodes:
            for ing in ing_nodes:
                for val in val_nodes:
                    add_edge(ing["id"], val["id"], "guards")
            if prep_nodes:
                for val in val_nodes:
                    add_edge(val["id"], prep_nodes[0]["id"], "validates")
        elif prep_nodes:
            for ing in ing_nodes:
                add_edge(ing["id"], prep_nodes[0]["id"], "raw data")

    # 2. Sequence through preprocessing transformations within Stage 2
    if len(prep_nodes) > 1:
        for i in range(len(prep_nodes) - 1):
            add_edge(prep_nodes[i]["id"], prep_nodes[i + 1]["id"], "transforms")

    # 3. Last Preprocessing node feeds Stage 3 Model Training and any preprocessor persistence
    if prep_nodes:
        last_prep = prep_nodes[-1]
        for m in model_nodes:
            add_edge(last_prep["id"], m["id"], "features")
        for out in out_nodes:
            name_lower = out.get("name", "").lower()
            if any(k in name_lower for k in ["vectorizer", "scaler", "encoder", "tfidf", "vocab"]):
                add_edge(last_prep["id"], out["id"], "artifact")

    # 4. Stage 3 Model Training feeds Stage 4 Inference and Model Persistence
    if model_nodes:
        last_model = model_nodes[-1]
        for out in out_nodes:
            name_lower = out.get("name", "").lower()
            if not any(k in name_lower for k in ["vectorizer", "scaler", "encoder", "tfidf", "vocab"]):
                add_edge(last_model["id"], out["id"], "weights")

    # 5. Inference feeds downstream model persistence or evaluation
    infer_nodes = [o for o in out_nodes if o.get("type") == "inference"]
    persist_model_nodes = [
        o for o in out_nodes
        if o.get("type") == "persistence"
        and not any(k in o.get("name", "").lower() for k in ["vectorizer", "scaler", "encoder", "tfidf", "vocab"])
    ]
    for inf in infer_nodes:
        for p in persist_model_nodes:
            add_edge(inf["id"], p["id"], "evaluated")

    return edges


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
    edges = construct_hierarchical_edges(nodes)

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
    """Enriches pipeline nodes with stage classification and layout coordinates."""
    graph = build_networkx_graph(pipeline_graph)
    layout = compute_layered_layout(graph)
    enriched_nodes: List[Dict[str, Any]] = []

    for node in pipeline_graph.get("nodes", []):
        node_id = node.get("id")
        coords = layout.get(node_id, {"x": 0.0, "y": 0.0, "level": 0})
        node_type = node.get("type", "unclassified")
        stage_info = PIPELINE_STAGES.get(node_type, PIPELINE_STAGES["unclassified"])
        enriched_nodes.append({
            **node,
            "x": coords["x"],
            "y": coords["y"],
            "level": coords["level"],
            "stage_index": stage_info["stage_index"],
            "stage_name": stage_info["stage_name"],
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

