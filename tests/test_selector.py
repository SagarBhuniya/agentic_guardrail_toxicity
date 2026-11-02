from sem_project import SelectorConfig, ToxicityToolSelector
from sem_project.param_filling import fill_parameters
from sem_project.tools import START


def make_selector():
    return ToxicityToolSelector(
        config=SelectorConfig(graph_path=None, embedding_backend="lexical"))


def test_select_fills_text_and_metadata():
    sel = make_selector()
    res = sel.select({"response_text": "you are stupid", "task_context": "detect toxicity"})
    assert res.params["text"] == "you are stupid"
    assert res.route in {"inertia", "context", "default_fallback"}
    assert res.candidates and res.reason


def test_parameter_backtracking_uses_previous_tool_output():
    sel = make_selector()
    sel.graph.record_execution(
        "moderation_api", "explain_toxicity_result", success=True,
        param_flows=[("moderation_api", "analysis", "explain_toxicity_result", "analysis")])
    state = {"response_text": "x",
             "tool_outputs": {"moderation_api": {"analysis": "harassment detected"}}}
    params, trace = fill_parameters("explain_toxicity_result", state, sel.graph)
    assert params["analysis"] == "harassment detected"
    assert any(t["param"] == "analysis" and t["via"] == "backtrack" for t in trace)


def test_inertia_blocked_by_30_percent_cap():
    sel = make_selector()
    sel.graph.record_execution(START, "keyword_toxicity_filter", success=True)
    sel.graph.record_execution("keyword_toxicity_filter", "moderation_api", success=True)
    state = {"response_text": "you are stupid", "task_context": "detect toxicity",
             "tool_history": ["keyword_toxicity_filter"],
             "selection_count": 10, "inertia_count": 4}   # 4/11 >= 0.30 -> blocked
    res = sel.select(state)
    assert res.used_inertia is False


def test_inertia_blocked_when_consecutive():
    sel = make_selector()
    sel.graph.record_execution(START, "keyword_toxicity_filter", success=True)
    sel.graph.record_execution("keyword_toxicity_filter", "moderation_api", success=True)
    state = {"response_text": "you are stupid", "task_context": "detect toxicity",
             "tool_history": ["keyword_toxicity_filter"],
             "last_selection_used_inertia": True}
    res = sel.select(state)
    assert res.used_inertia is False