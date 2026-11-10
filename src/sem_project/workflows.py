"""
Full toxicity-moderation workflows for LangGraph.
Optimized for 8GB RAM using Qwen2.5-1.5B-Instruct (Local Hugging Face Model).
"""
from __future__ import annotations

import json
import ast
import os
import re
import time
from typing import Any, Dict, Literal

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from .embeddings import LexicalScorer
from .inertia_graph import ToolInertiaGraph
from .nodes import execute_node, feedback_node, selection_node
from .param_filling import fill_parameters
from .state import ToxicityState
from .tools import TOOL_REGISTRY
from .local_llm import get_local_llm

load_dotenv()


# -----------------------------------------------------------------------------
# 2. Helpers & Simulated Fallback
# -----------------------------------------------------------------------------
DETECTORS = {"keyword_toxicity_filter", "detoxify_classifier", "moderation_api", "llm_toxicity_judge", "prompt_guard_fallback"}

# Update the prompt (Make it Rule 1!):
MAX_STEPS = 4

def _enrich_tags() -> None:
    kw = TOOL_REGISTRY["keyword_toxicity_filter"].tags
    if "scan" not in kw:
        kw += ["scan", "screen", "obvious", "insult", "idiot", "trash"]
    mo = TOOL_REGISTRY["moderation_api"].tags
    if "escalate" not in mo:
        mo += ["escalate", "violation", "flagged", "log", "threat"]
    lj = TOOL_REGISTRY["llm_toxicity_judge"].tags
    if "sarcasm" not in lj:
        lj += ["subtle", "sarcasm", "sarcastic", "nuanced", "review"]

_enrich_tags()

_SCORER = LexicalScorer()
_EMPTY_GRAPH = ToolInertiaGraph()

import numpy as np
_RNG = np.random.default_rng(7)

def _simulated_llm(state: ToxicityState) -> dict[str, Any]:
    """Fallback if local model fails to load."""
    text = state.get("response_text", "")
    blob = f"{text} | {state.get('task_context', '')}"
    best, best_score = None, -1.0
    for name, spec in TOOL_REGISTRY.items():
        score = _SCORER.similarity(blob, f"{spec.description} {' '.join(spec.tags)}")
        if score > best_score: best, best_score = name, score
    if _RNG.random() < 0.05: best = str(_RNG.choice([t for t in TOOL_REGISTRY if t != best]))
    params, trace = fill_parameters(best, dict(state), _EMPTY_GRAPH)
    return {
        "selected_tool": best, "tool_params": params or {"text": text}, "param_trace": trace,
        "used_inertia": False, "selection_route": "llm_fallback",
        "selection_reason": f"Simulated LLM chose '{best}'.", "needs_review": False,
        "tool_history": [best], "events": [{"type": "llm_select", "tool": best, "tokens": 400, "latency": 1.5}],
    }

def _extract_json_from_text(text: str) -> Dict[str, Any] | None:
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*```$', '', text, flags=re.MULTILINE)
    start, end = text.find('{'), text.rfind('}')
    if start == -1 or end <= start:
        return None
    blob = text[start:end + 1]
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        pass
    try:                      # small models emit single quotes / trailing commas
        parsed = ast.literal_eval(blob)
        return parsed if isinstance(parsed, dict) else None
    except (ValueError, SyntaxError):
        return None

# -----------------------------------------------------------------------------
# 3. LangGraph Nodes
# -----------------------------------------------------------------------------
def llm_select_node(state: ToxicityState) -> dict[str, Any]:
    """Real ReAct LLM decision using LOCAL Qwen2.5-1.5B-Instruct."""
    pipe = get_local_llm()
    if pipe is None:
        return _simulated_llm(state)

    text = state.get("response_text", "")
    context = state.get("task_context", "")
    
    tools_desc = "\n".join([
        f"- {name}: {spec.description}. Required Params: {list(spec.parameters.keys())}" 
        for name, spec in TOOL_REGISTRY.items()
    ])
    
    system_prompt = f"""You are an expert toxicity moderation agent. 
    Your task is to select the most appropriate tool to process the user's comment.
    
    Available Tools:
    {tools_desc}
    
    STRICT RULES:
    1. Normal/Clean text -> keyword_toxicity_filter
    2. General toxicity/profanity probability -> detoxify_classifier
    3. Severe/Threats/Hate -> moderation_api
    4. Sarcasm/subtle abuse -> llm_toxicity_judge
    5. Explaining a result -> explain_toxicity_result

    
    You MUST reply with ONLY a valid JSON object. No markdown, no explanations.
    Format: {{"reasoning": "brief reason", "tool_name": "exact_tool_name", "tool_params": {{"text": "..."}}}}"""

    user_prompt = f"Comment: {text}\nCurrent Task: {context}"

    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Run local inference (max_new_tokens kept low for speed)
        outputs = pipe(messages, max_new_tokens=128, do_sample=False, temperature=0.1)
        response_text = outputs[0]["generated_text"][-1]["content"]
        
        # Parse JSON
        parsed = _extract_json_from_text(response_text)
        if not parsed or "tool_name" not in parsed:
            raise ValueError("LLM returned invalid JSON format.")
            
        tool_name = parsed["tool_name"]
        if tool_name not in TOOL_REGISTRY:
            tool_name = "keyword_toxicity_filter" # Safety fallback
            
        params = parsed.get("tool_params", {})
        if "text" not in params: params["text"] = text
            
        return {
            "selected_tool": tool_name,
            "tool_params": params,
            "param_trace": [{"param": k, "via": "llm_generated"} for k in params],
            "used_inertia": False,
            "selection_route": "llm_fallback",
            "selection_reason": parsed.get("reasoning", "Local LLM decision"),
            "needs_review": False,
            "tool_history": [tool_name],
            "events": [{"type": "llm_select", "tool": tool_name, "tokens": 400, "latency": 2.0}],
        }
    except Exception as e:
        print(f"[LOCAL LLM ERROR] {e}. Falling back to keyword filter.")
        return {
            "selected_tool": "keyword_toxicity_filter", "tool_params": {"text": text},
            "tool_history": ["keyword_toxicity_filter"],
            "events": [{"type": "llm_select", "tool": "keyword_toxicity_filter", "tokens": 0, "error": str(e)}]
        }


def subgoal_node(state: ToxicityState) -> dict[str, Any]:
    tool = state.get("selected_tool")
    out = state.get("last_tool_output") or {}
    steps = len(state.get("tool_history") or [])
    failed = state.get("last_tool_success") is False
    tool_history = state.get("tool_history") or []
    
    score = float(out.get("score", 0) or 0)
    cats = out.get("categories") or ["none"]
    
    # 1. Check if the current tool declared the text "Clean" (0.0 score)
    is_clean = (tool in DETECTORS and not failed and score == 0 and cats == ["none"])
    
    # 2. FALLBACK LOGIC: If it's clean, but we haven't asked Prompt Guard yet, force a check.
    # We check tool_history to prevent an infinite loop.
    needs_fallback = (is_clean and tool != "prompt_guard_fallback" and "prompt_guard_fallback" not in tool_history)
    
    if needs_fallback:
        return {
            "task_context": "Fallback Safety Check",
            "selected_tool": "prompt_guard_fallback",
            "tool_params": {"text": state.get("response_text", "")},
            "used_inertia": False,
            "selection_route": "fallback_policy",
            "selection_reason": "Clean Exit Fallback: Forcing Prompt Guard safety check",
            "events": [{"type": "subgoal", "action": "continue", "reason": "fallback"}],
            "tool_history": ["prompt_guard_fallback"],
            "param_trace": [{"param": "text", "via": "fallback_policy", "source": None}],
        }

    # 3. Original Clean Exit Logic (Only triggers if Prompt Guard ALSO says 0.0)
    if failed:   # fail closed: no auto-verdict, flag for human review
        return {"task_context": "DONE", "needs_review": True,
                "events": [{"type": "subgoal", "action": "end", "reason": "executor_failure"}]}
                
    if tool == "explain_toxicity_result" or is_clean or steps >= MAX_STEPS:
        return {"task_context": "DONE", "needs_review": False,
                "events": [{"type": "subgoal", "action": "end"}]}
                
    return {"task_context": "Escalate the flagged finding and produce a user-facing explanation",
            "events": [{"type": "subgoal", "action": "continue"}]}

def _after_subgoal(state: ToxicityState) -> str:
    if state.get("task_context") == "DONE":
        return "end"
    if state.get("selection_route") == "fallback_policy":
        return "fallback"          # go straight to execute, no re-selection
    return "continue"
def _after_selection(state: ToxicityState) -> str: return "execute" if state.get("used_inertia") else "llm_select"

def build_autotool_workflow():
    g = StateGraph(ToxicityState)
    g.add_node("auto_select", selection_node)
    g.add_node("llm_select", llm_select_node)
    g.add_node("execute", execute_node)
    g.add_node("feedback", feedback_node)
    g.add_node("subgoal", subgoal_node)
    g.add_edge(START, "auto_select")
    g.add_conditional_edges("auto_select", _after_selection, {"execute": "execute", "llm_select": "llm_select"})
    g.add_edge("llm_select", "execute")
    g.add_edge("execute", "feedback")
    g.add_edge("feedback", "subgoal")
    g.add_conditional_edges("subgoal", _after_subgoal,
    {"continue": "auto_select", "end": END, "fallback": "execute"})
    return g.compile()

def build_react_workflow():
    g = StateGraph(ToxicityState)
    g.add_node("llm_select", llm_select_node)
    g.add_node("execute", execute_node)
    g.add_node("subgoal", subgoal_node)
    g.add_edge(START, "llm_select")
    g.add_edge("llm_select", "execute")
    g.add_edge("execute", "subgoal")
    g.add_conditional_edges("subgoal", _after_subgoal,
    {"continue": "llm_select", "end": END, "fallback": "execute"})
    return g.compile()