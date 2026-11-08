"""Render the Tool Inertia Graph - publication-quality figure.

Clean version: layered SOP layout, sqrt-normalised bounded edge widths,
sequential colormap instead of black, short wrapped labels, self-loops
omitted from the drawing but reported in the caption.
"""
from __future__ import annotations

SHORT_LABELS = {
    "__start__": "START",
    "keyword_toxicity_filter": "Keyword\nFilter",
    "detoxify_classifier": "Detoxify",
    "moderation_api": "ToxicBERT\n(mod. api)",
    "llm_toxicity_judge": "LLM Judge\n(Qwen)",
    "explain_toxicity_result": "Explain\n(Qwen)",
}

_LAYER_X = {
    "__start__": 0.0,
    "keyword_toxicity_filter": 1.0,
    "detoxify_classifier": 1.0,
    "moderation_api": 1.0,
    "llm_toxicity_judge": 1.0,
    "explain_toxicity_result": 2.0,
}


def plot_tool_graph(graph, path: str = "tool_inertia_graph.png",
                    min_weight: float = 1.0, show_self_loops: bool = False) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    g = nx.DiGraph()
    self_loops = {}
    for u, v, d in graph.g.edges(data=True):
        if d.get("kind") != "sequence" or d.get("weight", 0) < min_weight:
            continue
        if graph.g.nodes[u].get("kind") != "tool" or graph.g.nodes[v].get("kind") != "tool":
            continue
        if u == v:
            if show_self_loops:
                self_loops[u] = d["weight"]
            continue
        g.add_edge(u, v, weight=d["weight"])
    if g.number_of_edges() == 0:
        raise ValueError("No sequence edges yet - run the demo or train first.")

    # ---- layered layout: START | detectors | explain ----
    layers: dict[float, list[str]] = {}
    for n in g.nodes:
        layers.setdefault(_LAYER_X.get(n, 1.0), []).append(n)
    pos = {}
    for x, members in layers.items():
        members.sort(key=lambda n: SHORT_LABELS.get(n, n))
        span = 0.55
        for i, n in enumerate(members):
            pos[n] = (x, (len(members) - 1) * span / 2 - i * span)

    # ---- visual encodings (bounded, colour-coded) ----
    weights = [g[u][v]["weight"] for u, v in g.edges]
    wmax = max(weights)
    norm = [g[u][v]["weight"] / wmax for u, v in g.edges]
    widths = [0.9 + 4.6 * (n ** 0.5) for n in norm]
    cmap = plt.get_cmap("YlGnBu")
    colors = [cmap(0.25 + 0.65 * n) for n in norm]

    throughput = {
        n: sum(d["weight"] for _, _, d in g.in_edges(n, data=True))
           + sum(d["weight"] for _, _, d in g.out_edges(n, data=True))
        for n in g
    }
    tmax = max(throughput.values())
    sizes = [3000 + 2500 * (throughput[n] / tmax) for n in g.nodes]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    nx.draw_networkx_nodes(g, pos, ax=ax, node_size=sizes, node_color="#5aa9e6",
                           edgecolors="white", linewidths=2.0, alpha=0.95)
    nx.draw_networkx_labels(g, pos, ax=ax,
                            labels={n: SHORT_LABELS.get(n, n) for n in g},
                            font_size=8.5, font_color="white", font_weight="bold")
    nx.draw_networkx_edges(g, pos, ax=ax, edge_color=colors, width=widths,
                           alpha=0.9, arrows=True, arrowsize=16,
                           arrowstyle="-|>", connectionstyle="arc3,rad=0.10",
                           node_size=sizes)
    lab = {(u, v): f"{g[u][v]['weight']:.0f}"
           for u, v in g.edges if g[u][v]["weight"] >= 0.02 * wmax}
    nx.draw_networkx_edge_labels(
        g, pos, edge_labels=lab, ax=ax, font_size=8, font_color="#222222",
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85),
        rotate=False,
    )
    ax.set_title("Tool Inertia Graph (edge width & colour ∝ learned weight)")
    caption = "weight = count + success − failure; self-loops omitted"
    if self_loops:
        caption += " (" + ", ".join(f"{k}: {v:.0f}" for k, v in self_loops.items()) + ")"
    ax.text(0.01, -0.02, caption, transform=ax.transAxes, fontsize=8, color="#555555")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path