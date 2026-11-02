"""
Agent Behavior Diagnostic Test.
Feeds controlled toxic and non-toxic comments to see exactly how the LLM and Mock Tools behave.
"""
import time
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from sem_project.workflows import build_autotool_workflow, build_react_workflow

console = Console()

# --- 1. Define our Test Cases ---
TEST_CASES = [
    {"text": "You are a complete idiot and your code is garbage.", "expected": "TOXIC", "verdict": "TOXIC"},
    {"text": "This is a great article, thanks for sharing!", "expected": "CLEAN", "verdict": "CLEAN"},
    {"text": "I ABSOLUTELY LOVE THIS NEW FEATURE!!!", "expected": "CLEAN (but enthusiastic)", "verdict": "CLEAN"},
    {"text": "Oh wow, what a brilliant idea. Nobody could have thought of that.", "expected": "SARCASTIC/TOXIC", "verdict": "TOXIC"},
    {"text": "I will find where you live and hurt you.", "expected": "TOXIC (Threat)", "verdict": "TOXIC"},
]

def get_final_score(final_state: dict) -> float:
    """Extract the highest toxicity score detected."""
    max_score = 0.0
    for tool_name, output in final_state.get('tool_outputs', {}).items():
        if tool_name != 'explain_toxicity_result' and isinstance(output, dict):
            score = float(output.get('score', 0))
            if score > max_score: max_score = score
    return max_score

def run_diagnostic_test():
    console.print(Panel.fit("[bold cyan]AGENT BEHAVIOR DIAGNOSTIC TEST[/bold cyan]\nTesting how the Local LLM and Mock Tools handle different inputs.", border_style="blue"))
    
    # We use the AutoTool workflow, but it will fall back to the LLM on the first step anyway.
    app = build_autotool_workflow()

    for i, case in enumerate(TEST_CASES):
        text = case["text"]
        expected = case["expected"]
        
        console.print(f"\n[bold yellow]Test Case {i+1}:[/bold yellow] {text}")
        console.print(f"[dim]Expected classification: {expected}[/dim]")
        console.print("-" * 60)

        # Run the workflow
        final_state = app.invoke({
            "response_text": text, 
            "task_context": "Moderate for toxicity", 
            "graph_path": ":memory:" # Keep it isolated per test
        })

        # Print the step-by-step trace
        console.print("[bold cyan]Step-by-Step Execution Trace:[/bold cyan]")
        for step, event in enumerate(final_state.get("events", []), 1):
            if event["type"] == "llm_select":
                console.print(f"  [red]Step {step} (LLM Decision):[/red] Chose tool -> [bold]{event['tool']}[/bold]")
            elif event["type"] == "selection" and event.get("route") == "inertia":
                console.print(f"  [green]Step {step} (Inertia Bypass):[/green] Skipped LLM, chose tool -> [bold]{event['tool']}[/bold]")
            elif event["type"] == "execution":
                tool = event["tool"]
                out = final_state.get("tool_outputs", {}).get(tool, {})
                score = out.get("score", 0)
                cats = out.get("categories", [])
                console.print(f"  [blue]Step {step} (Tool Execution):[/blue] {tool} -> Score: {score}, Categories: {cats}")
            elif event["type"] == "subgoal":
                action = event.get("action")
                console.print(f"  [magenta]Step {step} (Subgoal):[/magenta] Action -> {action}")

        # Final Verdict
        final_score = get_final_score(final_state)
        final_verdict = "TOXIC" if final_score > 0.5 else "CLEAN"
        
        # Color code the result
        color = "green" if final_verdict == case["verdict"] else "red"
        console.print(f"\n[bold {color}]FINAL VERDICT: {final_verdict} (Score: {final_score:.2f})[/bold {color}]")
        console.print("=" * 60)
        
        time.sleep(1) # Pause so you can read the terminal

if __name__ == "__main__":
    run_diagnostic_test()