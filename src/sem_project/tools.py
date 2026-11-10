"""Tool catalog: the Tool Nodes (V) of the Tool Inertia Graph."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

START = "__start__"
DEFAULT_TOOL = "keyword_toxicity_filter"


class ParamSpec(BaseModel):
    required: bool = True
    default: Any = None
    kind: str = "text"          # text | categories | number | analysis | policy


class ToolSpec(BaseModel):
    name: str
    description: str
    tags: list[str] = Field(default_factory=list)
    parameters: dict[str, ParamSpec] = Field(default_factory=dict)
    notes: str = ""


_SPECS = [
    ToolSpec(
        name="keyword_toxicity_filter",
        description="Fast local keyword and pattern filter for obvious toxic, hateful or harassing language.",
        tags=["fast", "cheap", "local", "screening", "toxicity", "hate", "harassment"],
        parameters={
            "text": ParamSpec(kind="text"),
            "categories": ParamSpec(required=False, kind="categories",
                                    default=["toxicity", "harassment", "hate"]),
        },
        notes="Low-cost first pass; misses sarcasm.",
    ),
    ToolSpec(
        name="detoxify_classifier",
        description="Pretrained transformer classifier for toxicity, severe toxicity, insult, threat and identity hate.",
        tags=["model", "classifier", "toxicity", "insult", "threat", "identity-hate"],
        parameters={"text": ParamSpec(kind="text")},
        notes="Optional heavy dependency (detoxify).",
    ),
    ToolSpec(
        name="moderation_api",
        description="Production moderation service classifying harassment, hate, self-harm, sexual content and violence.",
        tags=["moderation", "api", "production", "harassment", "hate", "self-harm", "sexual", "violence"],
        parameters={
            "text": ParamSpec(kind="text"),
            "categories": ParamSpec(required=False, kind="categories",
                                    default=["harassment", "hate", "self-harm", "sexual", "violence"]),
            "threshold": ParamSpec(required=False, kind="number", default=0.5),
        },
        notes="Swap the mock executor for a real API client.",
    ),
    ToolSpec(
        name="llm_toxicity_judge",
        description="LLM reviewer for nuanced toxicity, sarcasm, context-dependent abuse and policy explanations.",
        tags=["llm", "nuanced", "sarcasm", "context", "policy", "review"],
        parameters={
            "text": ParamSpec(kind="text"),
            "policy": ParamSpec(required=False, kind="policy", default="community_safety_policy"),
            "temperature": ParamSpec(required=False, kind="number", default=0.0),
        },
        notes="Use when context matters or tools disagree.",
    ),
    ToolSpec(
        name="explain_toxicity_result",
        description="Explains a previous toxicity detection result in user-friendly language.",
        tags=["explain", "explanation", "reason", "user-facing", "report"],
        parameters={
            "text": ParamSpec(kind="text"),
            "analysis": ParamSpec(required=False, kind="analysis", default=""),
        },
        notes="Typically follows a detection tool (parameter backtracking demo).",
    ),
    ToolSpec(
        name="jailbreak_classifier",
        description="Transformer classifier specifically trained to detect prompt injections, jailbreaks, and adversarial attacks.",
        tags=["jailbreak", "injection", "adversarial", "attack", "bypass", "imagine", "pretend"],
        parameters={"text": ParamSpec(kind="text", required=True)},
    ),
]

TOOL_REGISTRY: dict[str, ToolSpec] = {spec.name: spec for spec in _SPECS}