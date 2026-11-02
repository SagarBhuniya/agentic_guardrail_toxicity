"""Complete LangGraph integration: select -> (review?) -> execute -> feedback."""
from langgraph.graph import END, START, StateGraph

from sem_project import (ToxicityState, execute_node, feedback_node,
                         route_after_selection, selection_node)


def human_or_llm_review(state: ToxicityState):
    """Stub: plug your LLM selector / human approval here."""
    return {"events": [{"type": "review", "tool": state.get("selected_tool")}]}


builder = StateGraph(ToxicityState)
builder.add_node("select", selection_node)
builder.add_node("review", human_or_llm_review)
builder.add_node("execute", execute_node)
builder.add_node("feedback", feedback_node)

builder.add_edge(START, "select")
builder.add_conditional_edges("select", route_after_selection,
                              {"execute": "execute", "human_or_llm_review": "review"})
builder.add_edge("review", "execute")
builder.add_edge("execute", "feedback")
builder.add_edge("feedback", END)

app = builder.compile()

if __name__ == "__main__":
    result = app.invoke({
        "response_text": "You are stupid and your answer is trash.",
        "task_context": "Check harassment before sending reply",
        "graph_path": ":memory:",
    })
    print("selected :", result["selected_tool"])
    print("params   :", result["tool_params"])
    print("trace    :", result["param_trace"])
    print("output   :", result["last_tool_output"])