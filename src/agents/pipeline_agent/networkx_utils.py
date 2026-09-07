import networkx as nx
from typing import Dict, Any, List


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


def summarize_graph_topology(graph: nx.DiGraph) -> Dict[str, Any]:
    """
    Extracts topological properties from the pipeline graph to help identify
    entry points, trust boundaries, and processing sinks.
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
