"""Inertia Sensing: CIPS scoring + inertia policy (AutoTool Sec. 4.5 & 5.2)."""
from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, Field

from .config import SelectorConfig
from .embeddings import build_scorer
from .inertia_graph import ToolInertiaGraph, bootstrap_default_graph
from .param_filling import fill_parameters
from .tools import DEFAULT_TOOL, START, TOOL_REGISTRY


class SelectionResult(BaseModel):
    tool: str
    params: dict[str, Any] = Field(default_factory=dict)
    param_trace: list[dict[str, Any]] = Field(default_factory=list)
    score: float = 0.0
    route: str = "context"                    # inertia | context | default_fallback
    used_inertia: bool = False
    reason: str = ""
    needs_review: bool = False
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    selection_count: int = 0
    inertia_count: int = 0

    def to_state_update(self) -> dict[str, Any]:
        """Partial state update for LangGraph."""
        return {
            "selected_tool": self.tool,
            "tool_params": self.params,
            "param_trace": self.param_trace,
            "selection_route": self.route,
            "selection_score": self.score,
            "selection_reason": self.reason,
            "used_inertia": self.used_inertia,
            "needs_review": self.needs_review,
            "candidates": self.candidates,
            "selection_count": self.selection_count,
            "inertia_count": self.inertia_count,
            "last_selection_used_inertia": self.used_inertia,
            "tool_history": [self.tool],
            "events": [{"type": "selection", "tool": self.tool,
                        "route": self.route, "score": round(self.score, 4)}],
        }


class ToxicityToolSelector:
    def __init__(self, config: SelectorConfig | None = None,
                 graph: ToolInertiaGraph | None = None, scorer=None) -> None:
        self.config = config or SelectorConfig()
        self.graph = graph or ToolInertiaGraph()
        if self.config.bootstrap and self.graph.is_empty():
            bootstrap_default_graph(self.graph)
        self.scorer = scorer or build_scorer(self.config)

    @classmethod
    def load(cls, config: SelectorConfig | None = None) -> "ToxicityToolSelector":
        config = config or SelectorConfig()
        graph = ToolInertiaGraph()
        if config.graph_path and config.graph_path != ":memory:" and os.path.exists(config.graph_path):
            graph = ToolInertiaGraph.load(config.graph_path)
        return cls(config=config, graph=graph)

    def save(self) -> None:
        if self.config.graph_path and self.config.graph_path != ":memory:":
            self.graph.save(self.config.graph_path)

    def _context(self, state: dict[str, Any]) -> str:
        parts = [str(state.get("task_context", "")), str(state.get("response_text", ""))]
        return " | ".join(p for p in parts if p)[: self.config.max_context_chars]

    def select(self, state: dict[str, Any]) -> SelectionResult:
        cfg = self.config
        available = [t for t in (state.get("available_tools") or list(TOOL_REGISTRY))
                     if t in TOOL_REGISTRY] or [DEFAULT_TOOL]
        history = list(state.get("tool_history") or [])
        source = history[-1] if history else START
        context = self._context(state)

        # candidates: historically-followed tools first, then everything available
        candidates = list(dict.fromkeys(self.graph.candidates(source, available) + available))

        scored = []
        for tool in candidates:
            freq = self.graph.frequency_score(source, tool, available)
            spec = TOOL_REGISTRY[tool]
            ctx = self.scorer.similarity(context, f"{spec.description} {' '.join(spec.tags)}")
            cips = (1.0 - cfg.alpha) * freq + cfg.alpha * ctx          # Eq. 1
            scored.append({"tool": tool, "freq": round(freq, 4),
                           "ctx": round(ctx, 4), "cips": round(cips, 4)})
        scored.sort(key=lambda row: row["cips"], reverse=True)

        # first candidate whose parameters can be FULLY filled wins (paper rule)
        chosen, params, trace = None, None, []
        for row in scored:
            params, trace = fill_parameters(row["tool"], state, self.graph)
            if params is not None:
                chosen = row
                break
        fillable = chosen is not None
        if not fillable:
            chosen = {"tool": DEFAULT_TOOL if DEFAULT_TOOL in available else available[0],
                      "freq": 0.0, "ctx": 0.0, "cips": 0.0}
            params = {"text": state.get("response_text", "")}
            trace = [{"param": "text", "via": "fallback", "source": None}]

        # ---- inertia policy (Sec. 5.2): 30% cap + no consecutive inertia ----
        selection_count = int(state.get("selection_count", 0)) + 1
        inertia_count = int(state.get("inertia_count", 0))
        policy_ok = (
            not (cfg.forbid_consecutive_inertia and state.get("last_selection_used_inertia"))
            and inertia_count / max(1, selection_count) < cfg.max_inertia_ratio
        )
        used_inertia = (
            fillable and policy_ok and source != START
            and chosen["freq"] > 0 and chosen["cips"] >= cfg.inertia_threshold
        )

        if used_inertia:
            inertia_count += 1
            route = "inertia"
            reason = (f"Inertial call: '{chosen['tool']}' after '{source}' "
                      f"(freq={chosen['freq']:.2f}, ctx={chosen['ctx']:.2f}, "
                      f"CIPS={chosen['cips']:.2f} >= theta={cfg.inertia_threshold}).")
        else:
            route = "context" if fillable else "default_fallback"
            reason = f"Graph/context selection of '{chosen['tool']}' (CIPS={chosen['cips']:.2f})."
            if not fillable:
                reason = "No candidate had fillable parameters; safe default used."
            elif not policy_ok:
                reason += " Inertia blocked by policy (30% cap / no consecutive calls)."
            elif source == START:
                reason += " First step of episode: no inertia evidence yet."
            elif chosen["cips"] < cfg.inertia_threshold:
                reason += f" CIPS below theta_inertial={cfg.inertia_threshold}."

        needs_review = (not fillable) or chosen["cips"] < cfg.inertia_threshold \
            or not str(params.get("text", "")).strip()

        return SelectionResult(tool=chosen["tool"], params=params, param_trace=trace,
                               score=chosen["cips"], route=route, used_inertia=used_inertia,
                               reason=reason, needs_review=needs_review, candidates=scored,
                               selection_count=selection_count, inertia_count=inertia_count)