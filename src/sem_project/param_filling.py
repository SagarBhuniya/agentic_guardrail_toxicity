"""Hierarchical, non-LLM parameter filling (AutoTool Sec. 4.5, Fig. 2c).

Priority order:
  1. dependency backtracking  - follow param_flow edges to a previous tool output
  2. environment/state match  - response_text, inferred categories, last analysis
  3. heuristic default        - the tool spec's default value
If a REQUIRED parameter cannot be filled, filling aborts (returns None)
and the selector must fall back - exactly like the paper.
"""
from __future__ import annotations

from typing import Any

from .embeddings import tokenize
from .inertia_graph import ToolInertiaGraph
from .tools import TOOL_REGISTRY

_EMPTY = (None, "", [], {})

CATEGORY_KEYWORDS = {
    "harassment": {"harassment", "harass", "bully", "insult", "stupid", "useless", "trash", "idiot"},
    "hate": {"hate", "hateful", "racist", "bigot", "slur"},
    "self-harm": {"suicide", "self", "harm"},
    "sexual": {"sexual", "nsfw"},
    "violence": {"kill", "attack", "weapon", "violent"},
}


def infer_categories(text: str) -> list[str]:
    tokens = set(tokenize(text or ""))
    cats = [cat for cat, kws in CATEGORY_KEYWORDS.items() if tokens & kws]
    return cats or ["toxicity"]


def fill_parameters(tool: str, state: dict[str, Any], graph: ToolInertiaGraph
                    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    spec = TOOL_REGISTRY.get(tool)
    if spec is None:
        return None, []

    outputs = state.get("tool_outputs") or {}
    context = f"{state.get('task_context', '')} {state.get('response_text', '')}"
    params: dict[str, Any] = {}
    trace: list[dict[str, Any]] = []

    for name, pspec in spec.parameters.items():
        value: Any = None
        via: str | None = None
        source: str | None = None

        # 1) dependency backtracking over the TIG
        for src_tool, src_param, _count in graph.param_sources(tool, name):
            candidate = (outputs.get(src_tool) or {}).get(src_param)
            if candidate not in _EMPTY:
                value, via, source = candidate, "backtrack", f"{src_tool}.{src_param}"
                break

        # 2) environment / state matching
        if value in _EMPTY:
            if pspec.kind == "text":
                value = state.get("response_text") or (state.get("last_tool_output") or {}).get("text")
                via = "state"
            elif pspec.kind == "categories":
                value = infer_categories(context)
                via = "state"
            elif pspec.kind == "analysis":
                value = (state.get("last_tool_output") or {}).get("analysis")
                via = "state"

        # 3) heuristic default
        if value in _EMPTY:
            value = pspec.default
            via = "default"

        if value in _EMPTY:
            if pspec.required:
                return None, trace          # abort: inertial call not allowed
            continue

        params[name] = value
        trace.append({"param": name, "via": via, "source": source})

    return params, trace