import ast
from typing import Dict, Any, List, Optional


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

    def _get_source_snippet(
        self,
        line_number: Optional[int],
        radius: int = 2,
    ) -> str:
        """
        Return a short source-code snippet around the AST node line.
        """

        if not line_number:
            return ""

        if (
            line_number < 1
            or line_number > len(self.source_lines)
        ):
            return ""

        start = max(
            1,
            line_number - radius,
        )

        end = min(
            len(self.source_lines),
            line_number + radius,
        )

        return "\n".join(
            f"{line_no:>4} | {self.source_lines[line_no - 1]}"
            for line_no in range(
                start,
                end + 1,
            )
        )

    def _add_node(
        self,
        step_type: str,
        description: str,
        line_number: Optional[int] = None,
        ast_node_type: Optional[str] = None,
        name: Optional[str] = None,
    ) -> str:
        """
        Add a pipeline node enriched with AST and source metadata.
        """

        node_id = (
            f"{step_type}_{len(self.nodes) + 1}"
        )

        node_data = {
            "id": node_id,
            "name": (
                name
                or description
            ),
            "type": step_type,
            "component_type": step_type,
            "description": description,
            "line_number": line_number,
            "ast_node_type": (
                ast_node_type
                or "Unknown"
            ),
            "source_snippet": (
                self._get_source_snippet(
                    line_number
                )
            ),
        }

        self.nodes.append(
            node_data
        )

        # Connect sequential steps to establish pipeline flow
        if self.last_node_id is not None:
            self.edges.append(
                {
                    "from": self.last_node_id,
                    "to": node_id,
                    "source": self.last_node_id,
                    "target": node_id,
                }
            )

        self.last_node_id = node_id

        return node_id

    def visit_Call(
        self,
        node: ast.Call,
    ):
        """
        Extract function and method calls relevant to the ML pipeline.
        """

        call_name = ""

        if isinstance(
            node.func,
            ast.Attribute,
        ):
            call_name = (
                node.func.attr.lower()
            )

        elif isinstance(
            node.func,
            ast.Name,
        ):
            call_name = (
                node.func.id.lower()
            )

        line = getattr(
            node,
            "lineno",
            None,
        )

        ast_type = type(
            node
        ).__name__

        # -------------------------------------------------
        # 1. Data Ingestion
        # -------------------------------------------------

        ingestion_identifiers = {
            "read_csv",
            "read_json",
            "read_parquet",
            "read_excel",
            "read_table",
            "load_dataset",
            "open",
            "input",
            "file_uploader",
            "from_csv",
        }

        if call_name in ingestion_identifiers:
            self._add_node(
                step_type="data_ingestion",
                description=(
                    f"Loads data via '{call_name}'"
                ),
                line_number=line,
                ast_node_type=ast_type,
                name=call_name,
            )

        # -------------------------------------------------
        # 2. Preprocessing & Feature Engineering
        # -------------------------------------------------

        preprocessing_keywords = {
            "preprocess",
            "clean",
            "tokenize",
            "stem",
            "normalize",
            "transform",
            "vectorize",
        }

        if any(
            keyword in call_name
            for keyword in preprocessing_keywords
        ):
            if call_name not in ingestion_identifiers:
                self._add_node(
                    step_type="preprocessing",
                    description=(
                        "Executes data transformation "
                        f"via '{call_name}'"
                    ),
                    line_number=line,
                    ast_node_type=ast_type,
                    name=call_name,
                )

        # -------------------------------------------------
        # 3. Model Training & Fitting
        # -------------------------------------------------

        training_identifiers = {
            "fit",
            "fit_transform",
            "train",
        }

        if call_name in training_identifiers:
            self._add_node(
                step_type="model_training",
                description=(
                    "Fits model or feature extractor "
                    f"via '{call_name}'"
                ),
                line_number=line,
                ast_node_type=ast_type,
                name=call_name,
            )

        # -------------------------------------------------
        # 4. Model Inference & Evaluation
        # -------------------------------------------------

        inference_identifiers = {
            "predict",
            "predict_proba",
            "infer",
            "evaluate",
            "score",
        }

        if call_name in inference_identifiers:
            self._add_node(
                step_type="inference",
                description=(
                    "Executes model inference or evaluation "
                    f"via '{call_name}'"
                ),
                line_number=line,
                ast_node_type=ast_type,
                name=call_name,
            )

        # -------------------------------------------------
        # 5. Serialization & Persistence
        # -------------------------------------------------

        serialization_identifiers = {
            "dump",
            "save",
            "save_weights",
            "to_pickle",
            "to_parquet",
        }

        if call_name in serialization_identifiers:
            self._add_node(
                step_type="persistence",
                description=(
                    f"Serializes artifact via '{call_name}'"
                ),
                line_number=line,
                ast_node_type=ast_type,
                name=call_name,
            )

        self.generic_visit(
            node
        )

    def visit_FunctionDef(
        self,
        node: ast.FunctionDef,
    ):
        """
        Detect user-defined preprocessing, training, and evaluation routines.
        """

        name_lower = (
            node.name.lower()
        )

        line = getattr(
            node,
            "lineno",
            None,
        )

        ast_type = type(
            node
        ).__name__

        if any(
            keyword in name_lower
            for keyword in [
                "preprocess",
                "clean",
                "sanitize",
                "transform",
            ]
        ):
            self._add_node(
                step_type="preprocessing_routine",
                description=(
                    "Defines data processing routine "
                    f"'{node.name}'"
                ),
                line_number=line,
                ast_node_type=ast_type,
                name=node.name,
            )

        elif any(
            keyword in name_lower
            for keyword in [
                "train",
                "fit",
                "evaluate",
            ]
        ):
            self._add_node(
                step_type="training_routine",
                description=(
                    "Defines model pipeline routine "
                    f"'{node.name}'"
                ),
                line_number=line,
                ast_node_type=ast_type,
                name=node.name,
            )

        self.generic_visit(
            node
        )

    def visit_If(
        self,
        node: ast.If,
    ):
        """
        Detect validation and defensive checks.
        """

        line = getattr(
            node,
            "lineno",
            None,
        )

        ast_type = type(
            node
        ).__name__

        # -------------------------------------------------
        # isinstance(...)
        # -------------------------------------------------

        if isinstance(
            node.test,
            ast.Call,
        ):
            if (
                isinstance(
                    node.test.func,
                    ast.Name,
                )
                and node.test.func.id == "isinstance"
            ):
                self._add_node(
                    step_type="validation",
                    description=(
                        "Type verification using isinstance"
                    ),
                    line_number=line,
                    ast_node_type=ast_type,
                    name="isinstance validation",
                )

            elif (
                isinstance(
                    node.test.func,
                    ast.Attribute,
                )
                and node.test.func.attr.lower()
                in {
                    "isna",
                    "isnull",
                    "empty",
                }
            ):
                self._add_node(
                    step_type="validation",
                    description=(
                        "Null or empty input verification"
                    ),
                    line_number=line,
                    ast_node_type=ast_type,
                    name="null validation",
                )

        # -------------------------------------------------
        # Comparisons
        # -------------------------------------------------

        if isinstance(
            node.test,
            ast.Compare,
        ):
            self._add_node(
                step_type="validation",
                description=(
                    "Conditional range or equality check"
                ),
                line_number=line,
                ast_node_type=ast_type,
                name="conditional validation",
            )

        self.generic_visit(
            node
        )

    def visit_Assert(
        self,
        node: ast.Assert,
    ):
        """
        Detect assertion-based validation.
        """

        line = getattr(
            node,
            "lineno",
            None,
        )

        self._add_node(
            step_type="validation",
            description=(
                "Validation assertion via assert statement"
            ),
            line_number=line,
            ast_node_type=type(node).__name__,
            name="assert validation",
        )

        self.generic_visit(
            node
        )

    def visit_Try(
        self,
        node: ast.Try,
    ):
        """
        Detect exception-handling controls.
        """

        line = getattr(
            node,
            "lineno",
            None,
        )

        self._add_node(
            step_type="validation",
            description=(
                "Defensive exception handling via "
                "try/except block"
            ),
            line_number=line,
            ast_node_type=type(node).__name__,
            name="exception handling",
        )

        self.generic_visit(
            node
        )


def parse_python_pipeline(
    code: str,
) -> Dict[str, Any]:
    """
    Parse Python pipeline source code and return structured
    nodes and edges enriched with source metadata.

    The returned graph is used by Agent 1 and by the
    interactive Streamlit pipeline inspection view.
    """

    try:
        tree = ast.parse(
            code
        )

    except SyntaxError as exc:
        return {
            "nodes": [
                {
                    "id": "parse_error_1",
                    "name": "Syntax error",
                    "type": "error",
                    "component_type": "error",
                    "description": (
                        "Syntax error during code parsing: "
                        f"{exc}"
                    ),
                    "line_number": getattr(
                        exc,
                        "lineno",
                        None,
                    ),
                    "ast_node_type": "SyntaxError",
                    "source_snippet": "",
                }
            ],
            "edges": [],
        }

    extractor = PipelineExtractor(
        source_code=code
    )

    extractor.visit(
        tree
    )

    nodes = extractor.nodes
    edges = extractor.edges

    # -----------------------------------------------------
    # Fallback
    # -----------------------------------------------------

    if not nodes:
        nodes = [
            {
                "id": "unclassified_1",
                "name": "Unclassified pipeline",
                "type": "unclassified",
                "component_type": "unclassified",
                "description": (
                    "No explicit ML pipeline stages "
                    "matched static heuristics"
                ),
                "line_number": None,
                "ast_node_type": "Module",
                "source_snippet": "",
            }
        ]

    return {
        "nodes": nodes,
        "edges": edges,
    }