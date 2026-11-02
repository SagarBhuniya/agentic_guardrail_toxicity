"""Tool Inertia Graph (TIG) - AutoTool Sec. 4.3/4.4, implemented on networkx.

Node kinds:  tool | param_in | param_out
Edge kinds:  sequence (tool->tool) | param_flow (param_out->param_in)
Sequence edge weight = count + success - failure   (efficacy-aware, Sec. 4.4)
"""
from __future__ import annotations

import json
from typing import Any, Iterable

import networkx as nx

from .tools import START, TOOL_REGISTRY


class ToolInertiaGraph:
    def __init__(self) -> None:
        self.g = nx.DiGraph()

    # ---------------- node helpers ----------------
    def _ensure_tool(self, tool: str) -> None:
        if not self.g.has_node(tool):
            self.g.add_node(tool, kind="tool", uses=0)

    def _param_node(self, tool: str, param: str, direction: str) -> str:
        node = f"{tool}.{direction}.{param}"
        if not self.g.has_node(node):
            self.g.add_node(node, kind=f"param_{direction}", tool=tool, param=param)
        return node

    # ---------------- learning (Sec. 4.4) ----------------
    def record_execution(self, source: str, target: str, success: bool | None = None,
                         param_flows: Iterable[tuple[str, str, str, str]] | None = None) -> None:
        """source -> target happened; optionally record parameter data flows."""
        self._ensure_tool(source)
        self._ensure_tool(target)
        self.g.nodes[target]["uses"] = self.g.nodes[target].get("uses", 0) + 1

        if not self.g.has_edge(source, target):
            self.g.add_edge(source, target, kind="sequence",
                            count=0.0, success=0.0, failure=0.0, weight=0.0)
        edge = self.g[source][target]
        edge["count"] += 1
        if success is True:
            edge["success"] += 1
        if success is False:
            edge["failure"] += 1
        edge["weight"] = max(0.0, edge["count"] + edge["success"] - edge["failure"])

        for src_tool, src_param, dst_tool, dst_param in param_flows or []:
            u = self._param_node(src_tool, src_param, "out")
            v = self._param_node(dst_tool, dst_param, "in")
            if self.g.has_edge(u, v):
                self.g[u][v]["count"] += 1
            else:
                self.g.add_edge(u, v, kind="param_flow", count=1.0)

    # ---------------- querying (Sec. 4.5) ----------------
    def sequence_edges(self, source: str, available: list[str] | None = None):
        out = []
        for _, tgt, data in self.g.out_edges(source, data=True):
            if data.get("kind") != "sequence" or data.get("weight", 0.0) <= 0:
                continue
            if available is not None and tgt not in available:
                continue
            out.append((tgt, data))
        return out

    def candidates(self, source: str, available: list[str]) -> list[str]:
        historical = [t for t, _ in self.sequence_edges(source, available)]
        return historical or list(available)

    def frequency_score(self, source: str, target: str, available: list[str]) -> float:
        """Score_freq: normalised edge weight; global prior at the start node."""
        edges = self.sequence_edges(source, available)
        total = sum(d["weight"] for _, d in edges)
        if total > 0:
            for tgt, data in edges:
                if tgt == target:
                    return data["weight"] / total
            return 0.0
        if source == START:
            uses = [self.g.nodes[t].get("uses", 0) for t in available if self.g.has_node(t)]
            total_uses = sum(uses)
            if total_uses == 0:
                return 1.0 / len(available) if available else 0.0
            return self.g.nodes[target].get("uses", 0) / total_uses if self.g.has_node(target) else 0.0
        return 0.0

    def param_sources(self, tool: str, param: str) -> list[tuple[str, str, float]]:
        """Ranked (src_tool, src_param, count) candidates for backtracking."""
        v = f"{tool}.in.{param}"
        if not self.g.has_node(v):
            return []
        srcs = []
        for u, _, data in self.g.in_edges(v, data=True):
            if data.get("kind") != "param_flow":
                continue
            node = self.g.nodes[u]
            srcs.append((node.get("tool", ""), node.get("param", ""), data.get("count", 1.0)))
        srcs.sort(key=lambda x: x[2], reverse=True)
        return srcs

    # ---------------- persistence / bootstrap ----------------
    def is_empty(self) -> bool:
        return not any(d.get("kind") == "sequence" for _, _, d in self.g.edges(data=True))

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [{"id": n, **d} for n, d in self.g.nodes(data=True)],
            "edges": [{"source": u, "target": v, **d} for u, v, d in self.g.edges(data=True)],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolInertiaGraph":
        graph = cls()
        for node in data.get("nodes", []):
            payload = dict(node)
            graph.g.add_node(payload.pop("id"), **payload)
        for edge in data.get("edges", []):
            payload = dict(edge)
            graph.g.add_edge(payload.pop("source"), payload.pop("target"), **payload)
        return graph

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "ToolInertiaGraph":
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return cls.from_dict(json.load(fh))
        except (OSError, json.JSONDecodeError):
            return cls()


def bootstrap_default_graph(graph: ToolInertiaGraph) -> None:
    """Small prior so cold starts behave sensibly (paper allows bootstrapping)."""
    if not graph.is_empty():
        return
    sequences = [
        (START, "keyword_toxicity_filter", 5),
        (START, "moderation_api", 3),
        ("keyword_toxicity_filter", "moderation_api", 3),
        ("keyword_toxicity_filter", "detoxify_classifier", 2),
        ("moderation_api", "llm_toxicity_judge", 2),
        ("moderation_api", "explain_toxicity_result", 2),
        ("llm_toxicity_judge", "explain_toxicity_result", 2),
        ("detoxify_classifier", "explain_toxicity_result", 1),
    ]
    for source, target, count in sequences:
        if target not in TOOL_REGISTRY:
            continue
        for _ in range(count):
            graph.record_execution(source, target, success=True)