import ast
import networkx as nx
from typing import Dict, Any, List


class PipelineExtractor(ast.NodeVisitor):
    """
    Extracts ML pipeline structure from Python code using AST.
    Focuses ONLY on MVP vulnerability categories:
    - data_ingestion
    - preprocessing
    - validation
    - inference
    """

    def __init__(self):
        self.nodes = []
        self.edges = []
        self.last_node = None

    def add_step(self, step_type: str, description: str):
        node_id = f"{step_type}_{len(self.nodes)+1}"
        self.nodes.append({
            "id": node_id,
            "type": step_type,
            "description": description,
        })

        if self.last_node:
            self.edges.append({"from": self.last_node, "to": node_id})

        self.last_node = node_id

    # -------------------------
    # DATA INGESTION DETECTION
    # -------------------------
    def visit_Call(self, node):
        # Detect dataset loading
        if isinstance(node.func, ast.Attribute):
            attr = node.func.attr.lower()

            if attr in ["read_csv", "read_json", "load_dataset"]:
                self.add_step("data_ingestion", f"Loads data via {attr}")

        if isinstance(node.func, ast.Name):
            name = node.func.id.lower()

            if name in ["input", "open"]:
                self.add_step("data_ingestion", f"Reads external input via {name}")

        # Detect preprocessing
        if isinstance(node.func, ast.Name):
            if "preprocess" in node.func.id.lower():
                self.add_step("preprocessing", f"Calls {node.func.id}")

        # Detect inference
        if isinstance(node.func, ast.Name):
            if "predict" in node.func.id.lower() or "infer" in node.func.id.lower():
                self.add_step("inference", f"Calls {node.func.id}")

        self.generic_visit(node)

    # -------------------------
    # VALIDATION DETECTION
    # -------------------------
    def visit_If(self, node):
        """
        Detect validation logic inside if-statements:
        - type checks
        - boundary checks
        - shape checks
        - sanitization
        """
        # Type checks
        if isinstance(node.test, ast.Call):
            if isinstance(node.test.func, ast.Name):
                if node.test.func.id == "isinstance":
                    self.add_step("validation", "Type validation via isinstance")

        # Boundary checks
        if isinstance(node.test, ast.Compare):
            self.add_step("validation", "Boundary validation via comparison")

        self.generic_visit(node)

    def visit_Assert(self, node):
        self.add_step("validation", "Validation via assert")
        self.generic_visit(node)

    def visit_Raise(self, node):
        self.add_step("validation", "Validation via raise")
        self.generic_visit(node)

    def visit_Try(self, node):
        self.add_step("validation", "Validation via try/except")
        self.generic_visit(node)


def parse_python_pipeline(code: str) -> Dict[str, Any]:
    """
    Parse Python ML pipeline code and extract pipeline_graph JSON.
    """
    tree = ast.parse(code)
    extractor = PipelineExtractor()
    extractor.visit(tree)

    # Always return valid structure
    return {
        "nodes": extractor.nodes or [
            {
                "id": "empty_1",
                "type": "empty",
                "description": "No pipeline steps detected"
            }
        ],
        "edges": extractor.edges or []
    }



def build_networkx_graph(pipeline_graph: Dict[str, Any]) -> nx.DiGraph:
    G = nx.DiGraph()

    for node in pipeline_graph["nodes"]:
        G.add_node(node["id"], type=node["type"], description=node["description"])

    for edge in pipeline_graph["edges"]:
        G.add_edge(edge["from"], edge["to"])

    return G
