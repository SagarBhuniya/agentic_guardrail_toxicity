"""Router overhead (paper Table 5 analogue): ms per inertia-sensing decision."""
import time
from sem_project.config import SelectorConfig
from sem_project.selector import ToxicityToolSelector

def main(n=1000):
    sel = ToxicityToolSelector(
        config=SelectorConfig(graph_path=None,
                              embedding_backend=os_backend()))
    state = {"response_text": "you are stupid and your answer is trash",
             "task_context": "Moderate comment",
             "tool_history": ["keyword_toxicity_filter"]}
    sel.select(state)                      # warm-up
    t0 = time.perf_counter()
    for _ in range(n):
        sel.select(state)
    ms = (time.perf_counter() - t0) / n * 1000
    print(f"AutoTool decision overhead: {ms:.3f} ms/step ({n} steps)")
    print("Reference: one local Qwen2.5-1.5B inference ~ 2000-3000 ms on CPU")

def os_backend():
    import os
    return os.getenv("SEM_EMBED_BACKEND", "lexical")

if __name__ == "__main__":
    main()