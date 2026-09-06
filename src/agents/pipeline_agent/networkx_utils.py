import networkx as nx
from typing import Dict, Any, List


def build_networkx_graph(pipeline_graph: Dict[str, Any]) -> nx.DiGraph:
    G = nx.DiGraph()

    nodes = pipeline_graph.get("nodes", [])
    edges = pipeline_graph.get("edges", [])

    for node in nodes:
        G.add_node(
            node["id"],
            type=node.get("type"),
            description=node.get("description")
        )

    for edge in edges:
        if "from" in edge and "to" in edge:
            G.add_edge(edge["from"], edge["to"])

    return G




def debug_print_graph(G: nx.DiGraph):
    print("Nodes:", list(G.nodes(data=True)))
    print("Edges:", list(G.edges()))
