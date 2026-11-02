"""Runtime configuration: the paper's hyper-parameters (AutoTool Sec. 5.2)."""
from __future__ import annotations

import os
from typing import Any, Callable

from pydantic import BaseModel, Field


def _from_env(name: str, default: Any, cast: Callable[[str], Any]) -> Any:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return cast(raw)
    except (TypeError, ValueError):
        return default


class SelectorConfig(BaseModel):
    # CIPS = (1 - alpha) * Score_freq + alpha * Score_ctx      (paper Eq. 1)
    alpha: float = Field(default_factory=lambda: _from_env("SEM_ALPHA", 0.5, float))
    # theta_inertial: minimum CIPS to trust an inertial (LLM-free) call.
    inertia_threshold: float = Field(default_factory=lambda: _from_env("SEM_THETA_INERTIAL", 0.15, float))
    # Paper constraint: inertia calls <= 30% of all operations.
    max_inertia_ratio: float = Field(default_factory=lambda: _from_env("SEM_MAX_INERTIA_RATIO", 0.30, float))
    # Paper constraint: never two inertial calls in a row.
    forbid_consecutive_inertia: bool = True
    max_context_chars: int = 1000
    # "auto" = sentence-transformers if installed, else lexical fallback.
    embedding_backend: str = Field(default_factory=lambda: os.getenv("SEM_EMBED_BACKEND", "auto"))
    embedding_model: str = Field(default_factory=lambda: os.getenv("SEM_EMBED_MODEL", "all-MiniLM-L6-v2"))
    # TIG persistence. ":memory:" / None = RAM only.
    graph_path: str | None = Field(default_factory=lambda: os.getenv("SEM_TOOL_GRAPH", ".sem_tool_graph.json"))
    bootstrap: bool = True