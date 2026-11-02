from langgraph.graph import END, START, StateGraph

from sem_project import (ToxicityState, execute_node, feedback_node,
                         selection_node, summarize_events)


def test_mini_workflow_end_to_end():
    builder = StateGraph(ToxicityState)
    builder.add_node("select", selection_node)
    builder.add_node("execute", execute_node)
    builder.add_node("feedback", feedback_node)
    builder.add_edge(START, "select")
    builder.add_edge("select", "execute")
    builder.add_edge("execute", "feedback")
    builder.add_edge("feedback", END)
    workflow = builder.compile()

    final = workflow.invoke({
        "response_text": "You are stupid and your answer is trash.",
        "task_context": "Check harassment before sending reply",
        "graph_path": ":memory:",
    })

    assert final["selected_tool"]
    assert final["tool_params"]["text"]
    assert final["last_tool_success"] in (True, False)
    summary = summarize_events(final["events"])
    assert summary["selections"] == 1 and summary["executions"] == 1