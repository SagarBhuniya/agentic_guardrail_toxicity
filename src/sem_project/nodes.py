"""LangGraph nodes: plain callables, state-in / partial-state-out."""
from __future__ import annotations

from typing import Any

from .config import SelectorConfig
from .executors import execute_tool
from .selector import ToxicityToolSelector
from .state import ToxicityState
from .tools import START

_CACHE: dict[str, ToxicityToolSelector] = {}


def get_selector(graph_path: str | None = None,
                 config: SelectorConfig | None = None) -> ToxicityToolSelector:
    config = config or SelectorConfig()
    if graph_path:
        config = config.model_copy(update={"graph_path": graph_path})
    key = config.graph_path or ":memory:"
    if key not in _CACHE:
        _CACHE[key] = ToxicityToolSelector.load(config)
    return _CACHE[key]


def selection_node(state: ToxicityState) -> dict[str, Any]:
    """Inertia Sensing + Parameter Filling. No LLM, no API calls."""
    selector = get_selector(state.get("graph_path"))
    result = selector.select(dict(state))
    selector.save()
    return result.to_state_update()


def execute_node(state: ToxicityState) -> dict[str, Any]:
    """Runs the selected tool and stores its output for parameter backtracking."""
    tool = state.get("selected_tool") or ""
    result = execute_tool(tool, state.get("tool_params") or {})
    outputs = {**(state.get("tool_outputs") or {}), tool: result.output}
    return {
        "tool_outputs": outputs,
        "last_tool_output": result.output,
        "last_tool_success": result.success,
        "events": [{"type": "execution", "tool": tool,
                    "success": result.success, "error": result.error}],
    }


def feedback_node(state: ToxicityState) -> dict[str, Any]:
    """Online TIG update: reinforce/penalise the used sequence edge."""
    selector = get_selector(state.get("graph_path"))
    history = state.get("tool_history") or []
    if not history:
        return {}
    target = history[-1]
    source = history[-2] if len(history) >= 2 else START

    flows: list[tuple[str, str, str, str]] = []
    for entry in state.get("param_trace") or []:
        if entry.get("via") == "backtrack" and entry.get("source"):
            src_tool, _, src_param = entry["source"].partition(".")
            flows.append((src_tool, src_param, target, entry["param"]))

    selector.graph.record_execution(source, target,
                                    success=state.get("last_tool_success"),
                                    param_flows=flows)
    selector.save()
    return {"events": [{"type": "feedback", "edge": f"{source}->{target}",
                        "success": state.get("last_tool_success")}]}


def route_after_selection(state: ToxicityState) -> str:
    """Optional safety valve: uncertain selections go to review."""
    return "human_or_llm_review" if state.get("needs_review") else "execute"