import ast
from typing import Dict, Any, List, Optional


class PipelineExtractor(ast.NodeVisitor):
    """
    AST visitor that scans Python source code to identify core machine learning
    pipeline components and build a structured representation of the execution graph.
    """

    def __init__(self):
        self.nodes: List[Dict[str, Any]] = []
        self.edges: List[Dict[str, str]] = []
        self.last_node_id: Optional[str] = None

    def _add_node(self, step_type: str, description: str, line_number: Optional[int] = None) -> str:
        node_id = f"{step_type}_{len(self.nodes) + 1}"
        node_data = {
            "id": node_id,
            "type": step_type,
            "description": description,
            "line_number": line_number,
        }
        self.nodes.append(node_data)

        # Connect sequential steps to establish pipeline flow
        if self.last_node_id is not None:
            self.edges.append({
                "from": self.last_node_id,
                "to": node_id,
            })

        self.last_node_id = node_id
        return node_id

    def visit_Call(self, node: ast.Call):
        # Extract call identifier whether attribute or simple function name
        call_name = ""
        if isinstance(node.func, ast.Attribute):
            call_name = node.func.attr.lower()
        elif isinstance(node.func, ast.Name):
            call_name = node.func.id.lower()

        line = getattr(node, "lineno", None)

        # 1. Data Ingestion Checks
        ingestion_identifiers = {
            "read_csv", "read_json", "read_parquet", "read_excel", "read_table",
            "load_dataset", "open", "input", "file_uploader", "from_csv"
        }
        if call_name in ingestion_identifiers:
            self._add_node("data_ingestion", f"Loads data via '{call_name}'", line)

        # 2. Preprocessing & Feature Engineering Checks
        preprocessing_keywords = {"preprocess", "clean", "tokenize", "stem", "normalize", "transform", "vectorize"}
        if any(keyword in call_name for keyword in preprocessing_keywords):
            if call_name not in ingestion_identifiers:
                self._add_node("preprocessing", f"Executes data transformation via '{call_name}'", line)

        # 3. Model Training & Fitting
        training_identifiers = {"fit", "fit_transform", "train"}
        if call_name in training_identifiers:
            self._add_node("model_training", f"Fits model or feature extractor via '{call_name}'", line)

        # 4. Model Inference & Evaluation
        inference_identifiers = {"predict", "predict_proba", "infer", "evaluate", "score"}
        if call_name in inference_identifiers:
            self._add_node("inference", f"Executes model inference or evaluation via '{call_name}'", line)

        # 5. Serialization and Persistence
        serialization_identifiers = {"dump", "save", "save_weights", "to_pickle", "to_parquet"}
        if call_name in serialization_identifiers:
            self._add_node("persistence", f"Serializes artifact via '{call_name}'", line)

        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        name_lower = node.name.lower()
        line = getattr(node, "lineno", None)

        # Identify custom preprocessing and evaluation routines
        if any(k in name_lower for k in ["preprocess", "clean", "sanitize", "transform"]):
            self._add_node("preprocessing_routine", f"Defines data processing routine '{node.name}'", line)
        elif any(k in name_lower for k in ["train", "fit", "evaluate"]):
            self._add_node("training_routine", f"Defines model pipeline routine '{node.name}'", line)

        self.generic_visit(node)

    def visit_If(self, node: ast.If):
        line = getattr(node, "lineno", None)

        # Detect defensive programming or input validation checks
        if isinstance(node.test, ast.Call):
            if isinstance(node.test.func, ast.Name) and node.test.func.id == "isinstance":
                self._add_node("validation", "Type verification using isinstance", line)
            elif isinstance(node.test.func, ast.Attribute) and node.test.func.attr.lower() in ["isna", "isnull", "empty"]:
                self._add_node("validation", "Null or empty input verification", line)

        if isinstance(node.test, ast.Compare):
            self._add_node("validation", "Conditional range or equality check", line)

        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert):
        line = getattr(node, "lineno", None)
        self._add_node("validation", "Validation assertion via assert statement", line)
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try):
        line = getattr(node, "lineno", None)
        self._add_node("validation", "Defensive exception handling via try/except block", line)
        self.generic_visit(node)


def parse_python_pipeline(code: str) -> Dict[str, Any]:
    """
    Parses Python pipeline code and returns structured nodes and edges.
    Provides a fallback if no specific steps could be statically matched.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {
            "nodes": [
                {
                    "id": "parse_error_1",
                    "type": "error",
                    "description": f"Syntax error during code parsing: {e}",
                    "line_number": getattr(e, "lineno", None)
                }
            ],
            "edges": []
        }

    extractor = PipelineExtractor()
    extractor.visit(tree)

    nodes = extractor.nodes
    edges = extractor.edges

    if not nodes:
        nodes = [
            {
                "id": "unclassified_1",
                "type": "unclassified",
                "description": "No explicit ML pipeline stages matched static heuristics",
                "line_number": None
            }
        ]

    return {
        "nodes": nodes,
        "edges": edges,
    }
