"""Render the Tool Inertia Graph - one figure for your report/slides."""
from __future__ import annotations


def plot_tool_graph(graph, path: str = "tool_inertia_graph.png") -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    g = nx.DiGraph()
    for u, v, d in graph.g.edges(data=True):
        if (d.get("kind") == "sequence" and d.get("weight", 0) > 0
                and graph.g.nodes[u].get("kind") == "tool"
                and graph.g.nodes[v].get("kind") == "tool"):
            g.add_edge(u, v, weight=d["weight"])
    if g.number_of_edges() == 0:
        raise ValueError("No sequence edges yet - run the demo or train first.")

    pos = nx.spring_layout(g, seed=42)
    widths = [2 * g[u][v]["weight"] ** 0.5 for u, v in g.edges()]
    fig, ax = plt.subplots(figsize=(8, 5))
    nx.draw_networkx_nodes(g, pos, ax=ax, node_size=2400, node_color="#4c9be8")
    nx.draw_networkx_labels(g, pos, ax=ax, font_size=7, font_color="white")
    nx.draw_networkx_edges(g, pos, ax=ax, width=widths, alpha=0.75,
                           arrowstyle="-|>", connectionstyle="arc3,rad=0.12")
    nx.draw_networkx_edge_labels(g, pos, edge_labels={(u, v): f"{g[u][v]['weight']:.0f}"
                                                      for u, v in g.edges()},
                                 ax=ax, font_size=7)
    ax.set_title("Tool Inertia Graph (weight = count + success - failure)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    return path