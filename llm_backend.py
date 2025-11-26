from __future__ import annotations

import os
from typing import List, Dict, Optional

from huggingface_hub import InferenceClient

# ============================================================
# Configuration (Qwen 2.5 Instruct via HF Inference API)
# ============================================================

HF_MODEL_ID = os.getenv("HF_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")

# We will NOT read HF_TOKEN at import time anymore.
# Instead we read it lazily inside _get_client().
_CLIENT: Optional[InferenceClient] = None


def _get_client() -> InferenceClient:
    """
    Lazily create and cache a global InferenceClient.
    Fails with a clear error if HF_TOKEN is missing.
    """
    global _CLIENT

    if _CLIENT is not None:
        return _CLIENT

    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "[llm_backend] HF_TOKEN environment variable is not set.\n"
            "Set it in a cell *before calling call_llm()*, e.g.:\n"
            "  import os; os.environ['HF_TOKEN'] = 'hf_...your-token-here...'\n"
        )

    _CLIENT = InferenceClient(model=HF_MODEL_ID, token=token)
    print(f"[llm_backend] Using HF InferenceClient chat_completion with model={HF_MODEL_ID}")
    return _CLIENT


# ============================================================
# Main API: call_llm
# ============================================================

def call_llm(messages: List[Dict[str, str]]) -> str:
    """
    Main LLM hook used by ontobpr_llm.extract_schema_with_llm.

    messages: list of {"role": "system"|"user"|"assistant", "content": "..."}
    Returns: plain text string (the assistant's content).
    """
    client = _get_client()

    try:
        response = client.chat_completion(
            messages=messages,
            max_tokens=1024,
            temperature=0.0,  # deterministic for JSON extraction
        )
    except Exception as e:
        # This is *fatal* – we do not silently continue with dummy output.
        raise RuntimeError(f"[llm_backend] HF chat_completion failed: {e}") from e

    # response.choices[0].message.content in current HF Hub client
    try:
        choice = response.choices[0]
        msg = choice.message
        if isinstance(msg, dict):
            content = msg.get("content", "")
        else:
            content = getattr(msg, "content", "")
    except Exception:
        content = str(response)

    return content.strip()
