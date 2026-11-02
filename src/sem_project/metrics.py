"""Paper-style efficiency metrics (AutoTool Sec. 5.1/5.2).

Every inertial selection bypasses one LLM inference, so
llm_calls_avoided == number of inertia calls (the paper's LLMC reduction).
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Iterable


def summarize_events(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    events = list(events)
    selections = [e for e in events if e.get("type") == "selection"]
    executions = [e for e in events if e.get("type") == "execution"]
    routes = Counter(s.get("route") for s in selections)
    inertia = routes.get("inertia", 0)
    total = len(selections)
    return {
        "selections": total,
        "inertia_calls": inertia,
        "inertia_ratio": round(inertia / total, 3) if total else 0.0,
        "llm_calls_avoided": inertia,
        "context_fallbacks": routes.get("context", 0),
        "default_fallbacks": routes.get("default_fallback", 0),
        "executions": len(executions),
        "failed_executions": sum(1 for e in executions if not e.get("success")),
    }