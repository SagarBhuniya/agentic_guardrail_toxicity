"""
Final Evaluation: Generates the 3 core charts for the Final Year Defense.
FIXED: Now simulates full multi-step workflows so inertia can actually trigger.
"""
import itertools
import matplotlib
import sys
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.table import Table

from sem_project.config import SelectorConfig
from sem_project.selector import ToxicityToolSelector
from sem_project.tools import START, TOOL_REGISTRY
from sem_project.viz import plot_tool_graph
from sem_project.embeddings import LexicalScorer

console = Console()
rng_auto = np.random.default_rng(42)
rng_base = np.random.default_rng(1234)

# --- 1. Ground Truth Mapping (Jigsaw -> Tools) ---
def get_truth_pipeline(row: dict) -> list[str]:
    t = float(row.get('toxicity', 0.0) or 0.0)
    sev = float(row.get('severe_toxicity', 0.0) or 0.0)
    ident = float(row.get('identity_attack', 0.0) or 0.0)
    threat = float(row.get('threat', 0.0) or 0.0)
    
    if t < 0.1: return []  # Clean
    if sev > 0.5 or ident > 0.5 or threat > 0.5: return ["moderation_api", "explain_toxicity_result"]
    if 0.3 <= t <= 0.7: return ["llm_toxicity_judge", "explain_toxicity_result"]
    if t > 0.7: return ["keyword_toxicity_filter", "explain_toxicity_result"]
    return []

def load_data(limit=300):
    try:
        from datasets import load_dataset
        ds = load_dataset("google/civil_comments", split="train", streaming=True)
        return list(itertools.islice(ds, limit))
    except Exception as e:
        console.print(f"[yellow]Dataset unavailable ({e}). Using fallback.[/yellow]")
        return [{"text": "You are an idiot.", "toxicity": 0.9}, {"text": "I will hurt you.", "toxicity": 0.9, "threat": 0.9}] * 50

# --- 2. Simulated LLM ---
def simulated_llm_choice(truth_tool: str, rng) -> str:
    if rng.random() < 0.95:
        return truth_tool
    return str(rng.choice([t for t in TOOL_REGISTRY if t != truth_tool]))

# --- 3. The Evaluation Loop (FIXED: Full multi-step simulation) ---
def run_evaluation(episodes=30):
    console.print(f"[bold cyan]Running Large-Scale Evaluation over {episodes} Episodes...[/bold cyan]")
    
    dataset = load_data(300)

    mode = sys.argv[1] if len(sys.argv) > 1 else "capped"
    capped = (mode == "capped")
    
    cfg = SelectorConfig(
        graph_path=None,
        embedding_backend="lexical",
        bootstrap=False,
        max_inertia_ratio=0.30 if capped else 1.0,
        forbid_consecutive_inertia=capped,
    )
    autotool = ToxicityToolSelector(config=cfg)
    
    at_pr_per_ep, base_pr_per_ep = [], []
    at_calls, base_calls = 0, 0
    at_tokens, base_tokens = 0, 0
    total_steps = 0

    carry = {"selection_count": 0, "inertia_count": 0, "last_selection_used_inertia": False}

    for ep in range(1, episodes + 1):
        ep_at_ok, ep_base_ok, ep_steps = 0, 0, 0
        
        for row in dataset:
            truth_pipeline = get_truth_pipeline(row)
            if not truth_pipeline:
                continue
            
            # Simulate the FULL pipeline (not just step 1)
            tool_history = []
            for step_idx, truth_tool in enumerate(truth_pipeline):
                ep_steps += 1
                total_steps += 1
                
                # Build state with ACTUAL history from previous steps
                state = {
                    "response_text": row["text"],
                    "task_context": "Moderate comment",
                    "tool_history": tool_history.copy(),  # Pass the real history!
                    **carry,                              # Pass the cap counters!
                }
                
                # --- ReAct + AutoTool ---
                res = autotool.select(state)
                carry = {   # remember for the NEXT step (the cap is global, like the paper)
                    "selection_count": res.selection_count,
                    "inertia_count": res.inertia_count,
                    "last_selection_used_inertia": res.used_inertia,
                }
                
                if res.used_inertia:
                    chosen = res.tool
                    # Inertial call: no LLM cost, and we assume it's correct if the graph learned well
                    # (In reality, inertial calls are ~98% accurate based on your earlier results)
                    is_correct = (chosen == truth_tool)
                else:
                    # LLM fallback: simulated 95% competence
                    chosen = simulated_llm_choice(truth_tool, rng_auto)
                    is_correct = (chosen == truth_tool)
                    at_calls += 1
                    at_tokens += 400
                
                if is_correct:
                    ep_at_ok += 1
                
                # Update Graph with the ACTUAL transition (not just START -> tool)
                source = tool_history[-1] if tool_history else START
                autotool.graph.record_execution(source, chosen, success=is_correct)
                tool_history.append(chosen)
                
                # --- ReAct Baseline (always uses LLM) ---
                base_chosen = simulated_llm_choice(truth_tool, rng_base)
                base_calls += 1
                base_tokens += 400
                if base_chosen == truth_tool:
                    ep_base_ok += 1

        at_pr_per_ep.append(ep_at_ok / ep_steps)
        base_pr_per_ep.append(ep_base_ok / ep_steps)

    # --- 4. Generate Chart 1: Efficiency ---
    console.print("[green]Generating Chart 1: Efficiency...[/green]")
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(2)
    w = 0.35
    ax.bar(x - w/2, [base_calls, base_tokens], w, label='Standard ReAct', color='salmon')
    ax.bar(x + w/2, [at_calls, at_tokens], w, label='ReAct + AutoTool', color='skyblue')
    ax.set_xticks(x)
    ax.set_xticklabels(['LLM API Calls', 'Estimated Tokens'])
    reduction = (1 - at_calls / base_calls) * 100 if base_calls > 0 else 0
    ax.set_title(f"Efficiency [{mode}]: {reduction:.1f}% Reduction in LLM Calls\n(Jigsaw Civil Comments Dataset)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"chart_1_efficiency_{mode}.png", dpi=300)

    # --- 5. Generate Chart 2: Tool Selection Accuracy ---
    console.print("[green]Generating Chart 2: Tool Selection Accuracy...[/green]")
    fig, ax = plt.subplots(figsize=(10, 6))
    eps = np.arange(1, episodes + 1)
    
    ax.plot(eps, base_pr_per_ep, "o--", color="red", alpha=0.6, label="ReAct Baseline (per episode)")
    ax.plot(eps, at_pr_per_ep, "s-", color="blue", alpha=0.6, label="AutoTool (per episode)")
    
    at_cum = np.cumsum(at_pr_per_ep) / eps
    base_cum = np.cumsum(base_pr_per_ep) / eps
    ax.plot(eps, base_cum, "-", color="red", lw=3, label="ReAct (Cumulative Mean)")
    ax.plot(eps, at_cum, "-", color="blue", lw=3, label="AutoTool (Cumulative Mean)")
    
    ax.set_title("Tool Selection Accuracy (Progress Rate) Over Time")
    ax.set_xlabel("Episode (Workflow Executions)")
    ax.set_ylabel("Selection Accuracy (%)")
    ax.set_ylim(0.7, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(f"chart_2_accuracy_{mode}.png", dpi=300)

    # --- 6. Generate Chart 3: Tool Inertia Graph ---
    console.print("[green]Generating Chart 3: Tool Inertia Graph...[/green]")
    plot_tool_graph(autotool.graph, path=f"chart_3_tig_{mode}.png")

    # --- 7. Print Summary Table ---
    table = Table(title=f"Final Evaluation Summary [{mode}]")
    table.add_column("Metric", style="bold")
    table.add_column("ReAct", justify="right")
    table.add_column("AutoTool", justify="right", style="green")
    table.add_row("Total Steps", str(total_steps), str(total_steps))
    table.add_row("LLM Calls", str(base_calls), str(at_calls))
    table.add_row("Tokens", f"{base_tokens:,}", f"{at_tokens:,}")
    table.add_row("LLM Reduction", "-", f"{reduction:.1f}%")
    table.add_row("Final Accuracy", f"{base_pr_per_ep[-1]:.3f}", f"{at_pr_per_ep[-1]:.3f}")
    console.print(table)
    console.print("[bold green]SUCCESS! Saved chart_1_efficiency.png, chart_2_accuracy.png, chart_3_tig.png[/bold green]")

if __name__ == "__main__":
    run_evaluation(episodes=30)