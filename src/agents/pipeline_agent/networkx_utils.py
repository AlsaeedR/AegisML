import networkx as nx
from typing import Dict, Any, List


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


def build_networkx_graph(pipeline_graph: Dict[str, Any]) -> nx.DiGraph:
    """
    Constructs a NetworkX directed graph from extracted pipeline nodes and edges.
    """
    graph = nx.DiGraph()

    nodes = pipeline_graph.get("nodes", [])
    edges = pipeline_graph.get("edges", [])

    for node in nodes:
        node_id = node.get("id")
        if node_id:
            graph.add_node(
                node_id,
                type=node.get("type"),
                description=node.get("description"),
                line_number=node.get("line_number"),
            )

    for edge in edges:
        source = edge.get("from")
        target = edge.get("to")
        if source and target:
            graph.add_edge(source, target)

    return graph


def _fallback_layers(graph: nx.DiGraph) -> List[List[str]]:
    """
    Fallback layout if the graph is not a clean DAG.
    Keeps nodes grouped in a single level rather than collapsing visually.
    """
    ordered_nodes = list(graph.nodes())
    return [ordered_nodes] if ordered_nodes else []


def compute_layered_layout(
    graph: nx.DiGraph,
    x_gap: int = 260,
    y_gap: int = 120,
) -> Dict[str, Dict[str, float]]:
    """
    Create a clean left-to-right layered layout for a DAG.

    Each topological generation becomes a column.
    Nodes inside the same generation are vertically spaced.
    """
    if graph.number_of_nodes() == 0:
        return {}

    try:
        layers = list(nx.topological_generations(graph))
    except Exception:
        layers = _fallback_layers(graph)

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

            layout[node_id] = {
                "x": float(x),
                "y": float(y),
                "level": level,
            }
            seen.add(node_id)

    # Catch any node not included for safety
    missing_nodes = [node for node in graph.nodes() if node not in seen]
    if missing_nodes:
        extra_level = len(layers)
        count = len(missing_nodes)
        total_height = (count - 1) * y_gap

        for index, node_id in enumerate(sorted(missing_nodes)):
            x = extra_level * x_gap
            y = (index * y_gap) - (total_height / 2)
            layout[node_id] = {
                "x": float(x),
                "y": float(y),
                "level": extra_level,
            }

    return layout


def enrich_pipeline_graph_for_ui(
    pipeline_graph: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Enrich pipeline nodes with layout coordinates and UI metadata
    so the dashboard can render a readable graph.
    """
    graph = build_networkx_graph(pipeline_graph)
    layout = compute_layered_layout(graph)

    enriched_nodes: List[Dict[str, Any]] = []

    for node in pipeline_graph.get("nodes", []):
        node_id = node.get("id")
        coords = layout.get(
            node_id,
            {"x": 0.0, "y": 0.0, "level": 0},
        )

        node_type = node.get("type", "unclassified")

        enriched_nodes.append(
            {
                **node,
                "x": coords["x"],
                "y": coords["y"],
                "level": coords["level"],
                "color": NODE_TYPE_COLORS.get(
                    node_type,
                    NODE_TYPE_COLORS["unclassified"],
                ),
            }
        )

    return {
        "nodes": enriched_nodes,
        "edges": pipeline_graph.get("edges", []),
        "topology": summarize_graph_topology(graph),
    }


def summarize_graph_topology(graph: nx.DiGraph) -> Dict[str, Any]:
    """
    Extracts topological properties from the pipeline graph.
    """
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