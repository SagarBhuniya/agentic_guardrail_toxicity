"""Real-detector probe: sanity cases + agreement with Jigsaw ground truth."""
import itertools
from rich.console import Console
from rich.table import Table
from sem_project.executors import execute_tool

console = Console()

CASES = [
    ("You are a complete idiot and your code is garbage.", "TOXIC"),
    ("This is a great article, thanks for sharing!", "CLEAN"),
    ("I ABSOLUTELY LOVE THIS NEW FEATURE!!!", "CLEAN"),
    ("I will find where you live and hurt you.", "TOXIC"),
]

def main(n=400):
    table = Table(title="Detoxify probe (real pretrained model)")
    table.add_column("Text"); table.add_column("Expected")
    table.add_column("Score"); table.add_column("Flagged categories")
    for text, exp in CASES:
        r = execute_tool("detoxify_classifier", {"text": text})
        out = r.output if r.success else {"score": "-", "categories": [r.error]}
        table.add_row(text[:45], exp, str(out.get("score")), str(out.get("categories")))
    console.print(table)

    try:
        from datasets import load_dataset
        ds = load_dataset("google/civil_comments", split="train", streaming=True)
        tp = fp = tn = fn = 0
        for row in itertools.islice(ds, n):
            label = 1 if float(row.get("toxicity") or 0) > 0.5 else 0
            r = execute_tool("detoxify_classifier", {"text": row["text"]})
            pred = 1 if (r.success and r.output.get("score", 0) > 0) else 0
            tp += pred and label; fp += pred and not label
            fn += (not pred) and label; tn += (not pred) and (not label)
        total = tp + fp + tn + fn
        console.print(
            f"\n[bold green]Real detection vs Jigsaw labels ({total} comments):[/bold green] "
            f"accuracy={(tp + tn) / total:.3f}, precision={tp / max(1, tp + fp):.3f}, "
            f"recall={tp / max(1, tp + fn):.3f}"
        )
    except Exception as e:
        console.print(f"[yellow]Jigsaw probe skipped: {e}[/yellow]")

if __name__ == "__main__":
    main()