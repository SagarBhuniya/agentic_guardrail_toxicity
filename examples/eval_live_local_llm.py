"""
Live Local LLM Benchmark: Real Qwen 1.5B vs AutoTool on Jigsaw Toxicity data.
Focuses purely on real-world hardware efficiency (LLM calls, tokens, and latency saved).
"""
import itertools
import time
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.table import Table

from sem_project.workflows import build_autotool_workflow, build_react_workflow

console = Console()

def get_final_score(final_state: dict) -> float:
    max_score = 0.0
    for tool_name, output in final_state.get('tool_outputs', {}).items():
        if tool_name != 'explain_toxicity_result' and isinstance(output, dict):
            max_score = max(max_score, float(output.get('score', 0)))
    return max_score

def load_toxic_comments(limit=15):
    from datasets import load_dataset
    ds = load_dataset("google/civil_comments", split="train", streaming=True)
    toxic_rows, clean_rows = [], []
    # Load a balanced mix to show the router handles both clean and toxic text
    for row in itertools.islice(ds, 2000):
        t = float(row.get("toxicity") or 0)
        if t > 0.5 and len(toxic_rows) < limit: toxic_rows.append(row)
        elif t <= 0.1 and len(clean_rows) < limit: clean_rows.append(row)
        if len(toxic_rows) >= limit and len(clean_rows) >= limit: break
    return toxic_rows + clean_rows

def run_real_benchmark(num_comments=15):
    console.print(f"[bold cyan]Loading {num_comments} toxic & {num_comments} clean comments...[/bold cyan]")
    rows = load_toxic_comments(num_comments)
    console.print(f"[green]Loaded {len(rows)} comments. Starting live local LLM A/B test...[/green]")

    autotool_app = build_autotool_workflow()
    react_app = build_react_workflow()

    stats = {"react": {"calls": 0, "tokens": 0, "latency": 0}, 
             "autotool": {"calls": 0, "tokens": 0, "latency": 0, "inertia": 0}}

    parity = 0      # comments where both arms agree on TOXIC/CLEAN
    vs_label = 0    # comments where AutoTool's verdict matches the Jigsaw label

    for i, row in enumerate(rows):
        text = row["text"]
        is_toxic = float(row.get("toxicity") or 0) > 0.5
        
        console.print(f"\n[yellow][{i+1}/{len(rows)}] Comment:[/yellow] {text[:60]}... ({'TOXIC' if is_toxic else 'CLEAN'})")
        
        # --- 1. Run Standard ReAct (Real Local LLM every step) ---
        console.print("  -> Running [red]ReAct (Local Qwen 1.5B)[/red]...")
        start_time = time.time()
        react_final = react_app.invoke({"response_text": text, "task_context": "Moderate for toxicity"})
        react_latency = time.time() - start_time
        
        for ev in react_final.get("events", []):
            if ev.get("type") == "llm_select":
                stats["react"]["calls"] += 1
                stats["react"]["tokens"] += ev.get("tokens", 400)
        stats["react"]["latency"] += react_latency      # ONCE per comment
        
        time.sleep(0.2) # Small pause to keep terminal readable

        # --- 2. Run AutoTool (Router + Real Local LLM fallback) ---
        console.print("  -> Running [blue]AutoTool + Local Qwen 1.5B[/blue]...")
        start_time = time.time()
        at_final = autotool_app.invoke({"response_text": text, "task_context": "Moderate for toxicity", "graph_path": ":memory:"})
        at_latency = time.time() - start_time
        
        inertia_used = 0
        for ev in at_final.get("events", []):
            if ev.get("type") == "llm_select":
                stats["autotool"]["calls"] += 1
                stats["autotool"]["tokens"] += ev.get("tokens", 400)
            elif ev.get("type") == "selection" and ev.get("route") == "inertia":
                inertia_used += 1
        stats["autotool"]["inertia"] += inertia_used
        stats["autotool"]["latency"] += at_latency      # ONCE per comment, like the ReAct arm

        # Detection parity: did both arms reach the same TOXIC/CLEAN verdict?
        pred_r = 1 if get_final_score(react_final) > 0.5 else 0
        pred_a = 1 if get_final_score(at_final) > 0.5 else 0
        parity += int(pred_r == pred_a)
        vs_label += int(pred_a == (1 if is_toxic else 0))


    # --- Print Final Real-World Results ---
    console.print("\n")
    table = Table(title=f"🏆 LIVE LOCAL LLM BENCHMARK: Real Hardware Efficiency ({len(rows)} Comments)")
    table.add_column("Metric", style="bold")
    table.add_column("Standard ReAct", justify="right")
    table.add_column("ReAct + AutoTool", justify="right", style="green")
    
    r, a = stats["react"], stats["autotool"]
    call_reduction = (1 - a["calls"] / r["calls"]) * 100 if r["calls"] > 0 else 0
    token_savings = r["tokens"] - a["tokens"]
    latency_savings = r["latency"] - a["latency"]

    table.add_row("Total Tool Steps", str(r["calls"]), str(a["calls"] + a["inertia"]))
    table.add_row("Real Local LLM Calls", str(r["calls"]), f"{a['calls']} [dim]({a['inertia']} bypassed)[/dim]")
    table.add_row("LLM Call Reduction", "-", f"[bold red]-{call_reduction:.1f}%[/bold red]")
    table.add_row("Real Tokens Saved", "-", f"[bold]-{token_savings:,}[/bold]")
    table.add_row("Real CPU Latency", f"{r['latency']:.1f}s", f"{a['latency']:.1f}s [dim](-{latency_savings:.1f}s)[/dim]")
    table.add_row("Detection parity (ReAct vs AutoTool)", "-", f"{parity}/{len(rows)}")
    table.add_row("AutoTool vs Jigsaw label", "-", f"{vs_label}/{len(rows)}")

    console.print(table)

    # --- Save a simple efficiency chart for the slides ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].bar(["ReAct", "ReAct+AutoTool"], [r["calls"], a["calls"]], color=["salmon", "skyblue"])
    axes[0].set_title("Real LLM calls")
    axes[1].bar(["ReAct", "ReAct+AutoTool"], [r["tokens"], a["tokens"]], color=["salmon", "skyblue"])
    axes[1].set_title("Estimated tokens")
    fig.suptitle(f"Live Local LLM Efficiency: {call_reduction:.1f}% fewer LLM calls (Qwen2.5-1.5B)")
    fig.tight_layout()
    fig.savefig("eval_9_live_local_llm_efficiency.png", dpi=300)
    console.print("\n[green]Saved: eval_9_live_local_llm_efficiency.png[/green]")

if __name__ == "__main__":
    run_real_benchmark(num_comments=15) # 15 toxic + 15 clean = 30 comments