"""Real open-source tool implementations. No mocks."""
from __future__ import annotations
import re
import json
from dataclasses import dataclass, field
from typing import Any

@dataclass
class ToolResult:
    success: bool
    output: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

# ==============================================================================
# 1. REAL KEYWORD FILTER (better-profanity)
# ==============================================================================
def run_keyword_toxicity_filter(params: dict[str, Any]) -> ToolResult:
    from better_profanity import profanity
    text = params.get("text", "")
    has_profanity = profanity.contains_profanity(text)
    
    # Contract: Clean text MUST be score 0.0 and categories ["none"]
    score = 1.0 if has_profanity else 0.0
    categories = ["profanity"] if has_profanity else ["none"]
    
    return ToolResult(True, {
        "score": score,
        "categories": categories,
        "analysis": f"Real profanity filter: detected={has_profanity}",
        "real_tool": True,
    })

# ==============================================================================
# 2. REAL MODERATION API (HuggingFace ToxicBERT)
# ==============================================================================
_MODERATION_PIPE = None
def _get_moderation_pipe():
    global _MODERATION_PIPE
    if _MODERATION_PIPE is None:
        from transformers import pipeline
        _MODERATION_PIPE = pipeline("text-classification", model="unitary/toxic-bert")
    return _MODERATION_PIPE

def run_moderation_api(params: dict[str, Any]) -> ToolResult:
    try:
        pipe = _get_moderation_pipe()
        text = params.get("text", "")
        result = pipe(text)[0]
        
        # ToxicBERT returns label "toxic" or "non-toxic" with a score
        is_toxic = result["label"].lower() == "toxic"
        score = result["score"] if is_toxic else 0.0
        categories = ["toxicity"] if is_toxic else ["none"]
        
        return ToolResult(True, {
            "score": round(score, 3),
            "categories": categories,
            "analysis": f"ToxicBERT: label={result['label']}, score={score:.3f}",
            "real_tool": True,
        })
    except Exception as e:
        return ToolResult(False, error=str(e))

# ==============================================================================
# 3. REAL LLM JUDGE (Local Qwen)
# ==============================================================================
def run_llm_toxicity_judge(params: dict[str, Any]) -> ToolResult:
    from .local_llm import get_local_llm
    pipe = get_local_llm()
    if pipe is None:
        return ToolResult(False, error="Local LLM unavailable")

    text = params.get("text", "")
    prompt = f"""Analyze this comment for subtle toxicity, sarcasm, or passive aggression. 
    Reply ONLY with JSON: {{"is_toxic": true/false, "reason": "short reason"}}
    Comment: {text}"""
    
    try:
        out = pipe(prompt, max_new_tokens=50, do_sample=False, temperature=0.1)[0]["generated_text"]
        # Extract JSON from LLM output
        match = re.search(r'\{.*\}', out, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
            is_toxic = parsed.get("is_toxic", False)
            score = 0.8 if is_toxic else 0.0
            categories = ["nuanced_toxicity"] if is_toxic else ["none"]
            return ToolResult(True, {
                "score": score,
                "categories": categories,
                "analysis": f"LLM Judge: {parsed.get('reason', '')}",
                "real_tool": True,
            })
        return ToolResult(False, error="LLM JSON parse failed")
    except Exception as e:
        return ToolResult(False, error=str(e))

# ==============================================================================
# 4. REAL EXPLAINER (Local Qwen)
# ==============================================================================
def run_explain_toxicity_result(params: dict[str, Any]) -> ToolResult:
    from .local_llm import get_local_llm
    pipe = get_local_llm()
    if pipe is None:
        return ToolResult(False, error="Local LLM unavailable")

    text = params.get("text", "")
    analysis = params.get("analysis", "unknown reason")
    
    prompt = f"""Write a polite, 1-sentence explanation to the user why their comment was flagged.
    Reason: {analysis}
    Comment: {text}
    Explanation:"""
    
    try:
        out = pipe(prompt, max_new_tokens=50, do_sample=False, temperature=0.3)[0]["generated_text"]
        # Clean up the output to just get the explanation
        explanation = out.split("Explanation:")[-1].strip() if "Explanation:" in out else out.strip()
        return ToolResult(True, {
            "explanation": explanation,
            "analysis": analysis,
            "real_tool": True,
        })
    except Exception as e:
        return ToolResult(False, error=str(e))

# ==============================================================================
# 5. REAL DETOXIFY (Already implemented, keep as is)
# ==============================================================================
_DETOXIFY_MODEL = None
def _get_detoxify():
    global _DETOXIFY_MODEL
    if _DETOXIFY_MODEL is None:
        from detoxify import Detoxify
        _DETOXIFY_MODEL = Detoxify("unbiased")
    return _DETOXIFY_MODEL

def run_detoxify_classifier(params: dict[str, Any]) -> ToolResult:
    try:
        model = _get_detoxify()
        preds = {k: round(float(v), 3) for k, v in model.predict(params.get("text", "")).items()}
        flagged = [k for k, v in preds.items() if v >= 0.5]
        score = max(preds[k] for k in flagged) if flagged else 0.0
        categories = flagged if flagged else ["none"]
        return ToolResult(True, {
            "score": score, "categories": categories, "probs": preds,
            "analysis": f"Detoxify flagged={categories}", "real_tool": True,
        })
    except Exception as e:
        return ToolResult(False, error=str(e))

# ==============================================================================
# REAL JAILBREAK CLASSIFIER (ProtectAI DeBERTa-v3)
# ==============================================================================
_JAILBREAK_PIPE = None
def _get_jailbreak_pipe():
    global _JAILBREAK_PIPE
    if _JAILBREAK_PIPE is None:
        from transformers import pipeline
        # This automatically downloads the model on first run (~300MB)
        _JAILBREAK_PIPE = pipeline("text-classification", model="protectai/deberta-v3-base-prompt-injection-v2")
    return _JAILBREAK_PIPE

def run_jailbreak_classifier(params: dict[str, Any]) -> ToolResult:
    try:
        pipe = _get_jailbreak_pipe()
        text = params.get("text", "")
        result = pipe(text)[0]
        
        is_injection = result["label"] == "INJECTION"
        score = result["score"] if is_injection else 0.0
        categories = ["jailbreak"] if is_injection else ["none"]
        
        return ToolResult(True, {
            "score": round(score, 3),
            "categories": categories,
            "analysis": f"DeBERTa Classifier: {result['label']} ({score:.3f})",
            "real_tool": True,
        })
    except Exception as e:
        return ToolResult(False, error=str(e))

# ==============================================================================
# 7. REAL PROMPT GUARD FALLBACK (Llama Prompt Guard 2 - 86M classifier)
# ==============================================================================
_PROMPT_GUARD_PIPE = None

def _get_prompt_guard():
    global _PROMPT_GUARD_PIPE
    if _PROMPT_GUARD_PIPE is None:
        from transformers import pipeline
        print("[INFO] Loading Llama Prompt Guard 2 (86M) for fallback...")
        _PROMPT_GUARD_PIPE = pipeline(
            "text-classification",                      # classifier, NOT text-generation
            model="meta-llama/Llama-Prompt-Guard-2-86M",
        )
        print("[SUCCESS] Prompt Guard loaded.")
    return _PROMPT_GUARD_PIPE


def run_prompt_guard_fallback(params: dict[str, Any]) -> ToolResult:
    try:
        pipe = _get_prompt_guard()
        text = params.get("text", "")
        result = pipe(text)[0]                          # top-1 label + confidence

        label = result["label"]
        conf = result["score"]

        # Your diagnostic proved: LABEL_0 = benign, LABEL_1 = jailbreak/injection
        is_unsafe = label in ["LABEL_1", "jailbreak", "prompt_injection", "unsafe"]

        final_score = max(conf, 0.9) if is_unsafe else 0.0
        categories = ["jailbreak"] if is_unsafe else ["none"]

        return ToolResult(True, {
            "score": round(final_score, 3),
            "categories": categories,
            "analysis": f"Prompt Guard 2 (86M): {label} ({conf:.3f})",
            "real_tool": True,
        })
    except Exception as e:
        return ToolResult(False, error=str(e))
    
EXECUTORS = {
    "keyword_toxicity_filter": run_keyword_toxicity_filter,
    "detoxify_classifier": run_detoxify_classifier,
    "moderation_api": run_moderation_api,
    "llm_toxicity_judge": run_llm_toxicity_judge,
    "explain_toxicity_result": run_explain_toxicity_result,
    "prompt_guard_fallback": run_prompt_guard_fallback,
}

def execute_tool(tool: str, params: dict[str, Any]) -> ToolResult:
    runner = EXECUTORS.get(tool)
    if runner is None:
        return ToolResult(False, error=f"No executor for '{tool}'")
    try:
        return runner(params or {})
    except Exception as exc:
        return ToolResult(False, error=f"{type(exc).__name__}: {exc}")