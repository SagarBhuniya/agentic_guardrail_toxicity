"""Shared local LLM loader to prevent circular imports."""
import os
from dotenv import load_dotenv

load_dotenv()

LOCAL_MODEL_ID = os.getenv("SEM_LOCAL_MODEL", "Qwen/Qwen2.5-7B-Instruct")
_LOCAL_PIPE = None

def get_local_llm():
    """Lazy-loads the local LLM only when called, saving RAM."""
    global _LOCAL_PIPE
    if _LOCAL_PIPE is None:
        # Guard for pytest/CI environments
        if os.getenv("SEM_DISABLE_LOCAL_LLM") == "1":
            return None
            
        try:
            import torch
            from transformers import pipeline
            print(f"[INFO] Loading local LLM: {LOCAL_MODEL_ID}...")
            _LOCAL_PIPE = pipeline(
                "text-generation", 
                model=LOCAL_MODEL_ID,
                device_map="auto" if torch.cuda.is_available() else "cpu"
            )
            print("[SUCCESS] Local LLM loaded.")
        except Exception as e:
            print(f"[WARNING] Local LLM failed to load: {e}")
            return None
    return _LOCAL_PIPE