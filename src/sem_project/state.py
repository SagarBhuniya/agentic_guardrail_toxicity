"""LangGraph-compatible workflow state."""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class ToxicityState(TypedDict, total=False):
    # ---- inputs (filled by your friend's graph) ----
    response_text: str                            # the LLM response to check
    task_context: str                             # e.g. "check harassment before reply"
    available_tools: list[str]
    graph_path: str | None                        # ":memory:" for tests/demos

    # ---- append-only logs (LangGraph reducers) ----
    tool_history: Annotated[list[str], operator.add]
    events: Annotated[list[dict[str, Any]], operator.add]

    # ---- execution feedback ----
    tool_outputs: dict[str, dict[str, Any]]       # tool name -> last output
    last_tool_output: dict[str, Any]
    last_tool_success: bool | None

    # ---- selection outputs ----
    selected_tool: str | None
    tool_params: dict[str, Any]
    param_trace: list[dict[str, Any]]             # how each parameter was filled
    selection_route: str                          # "inertia" | "context" | "default_fallback"
    selection_score: float
    selection_reason: str
    used_inertia: bool
    needs_review: bool
    candidates: list[dict[str, Any]]

    # ---- inertia policy counters ----
    selection_count: int
    inertia_count: int
    last_selection_used_inertia: bool

    # ---- final result ----
    toxicity_report: dict[str, Any]