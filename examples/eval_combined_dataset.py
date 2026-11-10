"""
Extended Evaluation: Evaluates the agent against a custom combined dataset.
Limited to 60 examples for faster testing.
"""
import csv
import time
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.table import Table

# Make src/ importable for Colab
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_SRC = os.path.join(_PROJECT_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from sem_project.workflows import build_autotool_workflow, build_react_workflow

console = Console()
MAX_ROWS = 60  # Limit to 60 examples

# ==============================================================================
# 1. LOAD CUSTOM DATASET
# ==============================================================================
def load_custom_dataset(file_path: str):
    """Loads a local CSV with 'text' and 'expected_verdict' columns, limited to MAX_ROWS."""
    rows = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if len(rows) >= MAX_ROWS:
                    break
                
                # Safe parse: handles missing column AND empty cells (None)
                verdict = (row.get("expected_verdict") or "CLEAN").strip().upper()
                is_toxic = verdict == "TOXIC"
                category = (row.get("category") or "general").strip()
                
                rows.append({
                    "text": row["text"],
                    "is_toxic": is_toxic,
                    "category": category
                })
        console.print(f"[green]Successfully loaded {len(rows)} rows from {file_path}[/green]")
        return rows
    except FileNotFoundError:
        console.print(f"[bold red]Error: File '{file_path}' not found.[/bold red]")
        console.print("Please ensure your friend saved the CSV in the project root.")
        raise SystemExit
    except KeyError as e:
        console.print(f"[bold red]Error: Missing required column {e} in CSV.[/bold red]")
        raise SystemExit

# ==============================================================================
# 2. RUN BENCHMARK
# ==============================================================================
def run_combined_benchmark(dataset_path: str = "combined_dataset.csv"):
    rows = load_custom_dataset(dataset_path)
    console.print(f"[bold cyan]Starting live local LLM A/B test on {len(rows)} custom prompts...[/bold cyan]")

    # Build workflows once
    autotool_app = build_autotool_workflow()
    react_app = build_react_workflow()

    stats = {
        "react": {"calls": 0, "tokens": 0, "latency": 0.0}, 
        "autotool": {"calls": 0, "tokens": 0, "latency": 0.0, "inertia": 0}
    }
    
    parity_count = 0
    vs_label_count = 0

    for i, row in enumerate(rows):
        text = row["text"]
        is_toxic = row["is_toxic"]
        category = row["category"]
        
        console.print(f"\n[yellow][{i+1}/{len(rows)}] [{category.upper()}] {text[:50]}...[/yellow]")
        
        # --- 1. Run Standard ReAct ---
        start_time = time.time()
        react_final = react_app.invoke({"response_text": text, "task_context": "Moderate for toxicity"})
        react_latency = time.time() - start_time
        
        for ev in react_final.get("events", []):
            if ev.get("type") == "llm_select":
                stats["react"]["calls"] += 1
                stats["react"]["tokens"] += ev.get("tokens", 400)
        stats["react"]["latency"] += react_latency
        
        # --- 2. Run AutoTool ---
        start_time = time.time()
        # Use :memory: so the graph learns across the dataset but doesn't persist to disk
        at_final = autotool_app.invoke({
            "response_text": text, 
            "task_context": "Moderate for toxicity", 
            "graph_path": ":memory:"
        })
        at_latency = time.time() - start_time
        
        inertia_used = 0
        for ev in at_final.get("events", []):
            if ev.get("type") == "llm_select":
                stats["autotool"]["calls"] += 1
                stats["autotool"]["tokens"] += ev.get("tokens", 400)
            elif ev.get("type") == "selection" and ev.get("route") == "inertia":
                inertia_used += 1
        stats["autotool"]["inertia"] += inertia_used
        stats["autotool"]["latency"] += at_latency  # Counted once per comment

        # --- 3. Calculate Parity and Accuracy ---
        def get_max_score(final_state):
            max_s = 0.0
            for tool_name, output in final_state.get('tool_outputs', {}).items():
                if tool_name != 'explain_toxicity_result' and isinstance(output, dict):
                    max_s = max(max_s, float(output.get('score', 0)))
            return max_s

        pred_r = 1 if get_max_score(react_final) > 0.5 else 0
        pred_a = 1 if get_max_score(at_final) > 0.5 else 0
        
        if pred_r == pred_a:
            parity_count += 1
        if pred_a == (1 if is_toxic else 0):
            vs_label_count += 1

    # ==============================================================================
    # 3. PRINT RESULTS
    # ==============================================================================
    console.print("\n")
    table = Table(title=f"🏆 EXTENDED BENCHMARK: Combined Dataset ({len(rows)} Prompts)")
    table.add_column("Metric", style="bold")
    table.add_column("Standard ReAct", justify="right")
    table.add_column("ReAct + AutoTool", justify="right", style="green")
    
    r, a = stats["react"], stats["autotool"]
    call_reduction = (1 - a["calls"] / r["calls"]) * 100 if r["calls"] > 0 else 0
    latency_savings = r["latency"] - a["latency"]

    table.add_row("Total Tool Steps", str(r["calls"]), str(a["calls"] + a["inertia"]))
    table.add_row("Real Local LLM Calls", str(r["calls"]), f"{a['calls']} [dim]({a['inertia']} bypassed)[/dim]")
    table.add_row("LLM Call Reduction", "-", f"[bold red]-{call_reduction:.1f}%[/bold red]")
    table.add_row("Real CPU Latency", f"{r['latency']:.1f}s", f"{a['latency']:.1f}s [dim](-{latency_savings:.1f}s)[/dim]")
    table.add_row("Detection Parity (ReAct vs AutoTool)", "-", f"{parity_count}/{len(rows)}")
    table.add_row("AutoTool vs Ground Truth Label", "-", f"{vs_label_count}/{len(rows)}")

    console.print(table)

    # --- Save Chart ---
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(2)
    w = 0.35
    ax.bar(x - w/2, [r["calls"], r["tokens"]], w, label='Standard ReAct', color='salmon')
    ax.bar(x + w/2, [a["calls"], a["tokens"]], w, label='ReAct + AutoTool', color='skyblue')
    ax.set_xticks(x)
    ax.set_xticklabels(['Real LLM Calls', 'Real Tokens Consumed'])
    ax.set_title(f"Combined Dataset Efficiency: {call_reduction:.1f}% Fewer LLM Calls")
    ax.legend()
    fig.tight_layout()
    fig.savefig("eval_combined_dataset_efficiency.png", dpi=300)
    console.print("\n[green]Saved: eval_combined_dataset_efficiency.png[/green]")

if __name__ == "__main__":
    # Ensure the CSV is in the root directory
    run_combined_benchmark(dataset_path="combined_dataset.csv")