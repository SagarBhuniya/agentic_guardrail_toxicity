"""CLI: demo / select / show-graph / plot / train / reset."""
from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .config import SelectorConfig
from .metrics import summarize_events
from .nodes import execute_node, feedback_node, get_selector, selection_node
from .state import ToxicityState
from .tools import START

app = typer.Typer(help="AutoTool-style toxicity tool selection (final-year project).")
console = Console()

SAMPLES = [
    {"response_text": "You are stupid and your answer is trash.",
     "task_context": "Check harassment before sending reply"},
    {"response_text": "Here is a step-by-step solution to your math question.",
     "task_context": "Check harassment before sending reply"},
    {"response_text": "Oh great, another genius answer. Nobody could have done worse.",
     "task_context": "Detect sarcasm and subtle toxicity"},
]


def _run_episode(samples: list[dict], episode: int) -> list[dict]:
    from langgraph.graph import END, START as GSTART, StateGraph   # lazy import

    builder = StateGraph(ToxicityState)
    builder.add_node("select", selection_node)
    builder.add_node("execute", execute_node)
    builder.add_node("feedback", feedback_node)
    builder.add_edge(GSTART, "select")
    builder.add_edge("select", "execute")
    builder.add_edge("execute", "feedback")
    builder.add_edge("feedback", END)
    workflow = builder.compile()

    events: list[dict] = []
    table = Table(title=f"Episode {episode} (select -> execute -> feedback)")
    for col in ("#", "selected_tool", "route", "CIPS", "exec_ok"):
        table.add_column(col)
    for i, sample in enumerate(samples, 1):
        final = workflow.invoke(dict(sample))
        ev = final.get("events", [])
        events.extend(ev)
        sel = next(e for e in ev if e["type"] == "selection")
        exe = next((e for e in ev if e["type"] == "execution"), {})
        table.add_row(str(i), sel["tool"], sel["route"], f"{sel['score']:.2f}",
                      str(exe.get("success")))
    console.print(table)
    return events


@app.command()
def demo(episodes: int = 2) -> None:
    """Run episodes; episode 2+ shows inertia kicking in as the TIG warms up."""
    all_events: list[dict] = []
    for ep in range(1, episodes + 1):
        all_events += _run_episode(SAMPLES, ep)
    table = Table(title="Efficiency summary (paper-style)")
    table.add_column("metric")
    table.add_column("value")
    for key, value in summarize_events(all_events).items():
        table.add_row(key, str(value))
    console.print(table)


@app.command("select")
def select_cmd(text: str, context: str = "detect toxicity") -> None:
    """One-shot selection without execution."""
    result = get_selector().select({"response_text": text, "task_context": context})
    console.print_json(result.model_dump_json())


@app.command("show-graph")
def show_graph() -> None:
    table = Table(title="Sequence edges of the Tool Inertia Graph")
    for col in ("source", "target", "count", "success", "failure", "weight"):
        table.add_column(col)
    for u, v, d in get_selector().graph.g.edges(data=True):
        if d.get("kind") == "sequence":
            table.add_row(u, v, f"{d['count']:.0f}", f"{d['success']:.0f}",
                          f"{d['failure']:.0f}", f"{d['weight']:.1f}")
    console.print(table)


@app.command("plot")
def plot_cmd(out: Path = Path("tool_inertia_graph.png")) -> None:
    from .viz import plot_tool_graph
    console.print(f"Saved {plot_tool_graph(get_selector().graph, str(out))}")


@app.command("train")
def train_cmd(file: Path) -> None:
    """Bootstrap the TIG from historical trajectories (jsonl, see examples/)."""
    selector = get_selector()
    n = 0
    for line in file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        prev = START
        for tool in rec.get("sequence", []):
            selector.graph.record_execution(prev, tool, success=rec.get("success"),
                                              param_flows=rec.get("param_flows"))
            prev = tool
        n += 1
    selector.save()
    console.print(f"Trained on {n} trajectories; edges={selector.graph.g.number_of_edges()}")


@app.command("reset")
def reset_cmd() -> None:
    cfg = SelectorConfig()
    if cfg.graph_path and cfg.graph_path != ":memory:" and Path(cfg.graph_path).exists():
        Path(cfg.graph_path).unlink()
        console.print(f"Deleted {cfg.graph_path}")
    else:
        console.print("Nothing to reset.")


if __name__ == "__main__":
    app()