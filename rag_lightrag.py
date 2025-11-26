# rag_lightrag.py
from __future__ import annotations
from pathlib import Path
from typing import List
import json
import shutil
import inspect

try:
    from lightrag import LightRAG, QueryParam
    _HAS_LIGHTRAG = True
except Exception as e:
    _HAS_LIGHTRAG = False
    _IMPORT_ERROR = e

def _load_chunk_texts(rag_dir: Path) -> List[str]:
    chunks_path = rag_dir / "chunks.jsonl"
    if not chunks_path.exists():
        raise FileNotFoundError(f"[rag_lightrag] chunks.jsonl not found at {chunks_path}")
    texts: List[str] = []
    with chunks_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                txt = obj.get("text") or ""
                txt = str(txt).strip()
                if txt:
                    texts.append(txt)
            except Exception:
                continue
    return texts

def build_lightrag_index(case_id: str, rag_dir: Path) -> Path:
    """
    Build a LightRAG index from chunks.jsonl.
    Returns the store_dir path.
    """
    if not _HAS_LIGHTRAG:
        raise RuntimeError(f"[rag_lightrag] LightRAG not available: {_IMPORT_ERROR}")

    texts = _load_chunk_texts(rag_dir)
    if not texts:
        raise RuntimeError("[rag_lightrag] No chunk texts found; cannot build LightRAG index.")

    store_dir = rag_dir / "lightrag_store"
    if store_dir.exists():
        shutil.rmtree(store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)

    rag = LightRAG(working_dir=str(store_dir))

    # Be robust to different insert() signatures
    sig = inspect.signature(rag.insert)
    params = sig.parameters

    print(f"[rag_lightrag] Building LightRAG index for {case_id} with {len(texts)} chunks…")
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            if "docs" in params:
                rag.insert(docs=batch)
            else:
                # Old signature: first positional arg is docs
                rag.insert(batch)
        except TypeError as e:
            # If even this fails, surface the problem – user can adapt to their lightrag version
            raise RuntimeError(f"[rag_lightrag] insert() failed: {e}")

    print(f"[rag_lightrag] LightRAG index build finished at {store_dir}")
    return store_dir

def retrieve_context_for_case(
    case_id: str,
    rag_dir: Path,
    query: str,
    top_k: int = 10,
    mode: str = "hybrid",
) -> str:
    """
    Retrieve context from LightRAG index for the given query.
    Returns a plain text block (string).
    """
    if not _HAS_LIGHTRAG:
        raise RuntimeError(f"[rag_lightrag] LightRAG not available: {_IMPORT_ERROR}")

    store_dir = rag_dir / "lightrag_store"
    if not store_dir.exists():
        raise FileNotFoundError(f"[rag_lightrag] Store dir does not exist: {store_dir}")

    rag = LightRAG(working_dir=str(store_dir))

    try:
        result = rag.query(query, param=QueryParam(mode=mode, top_k=top_k))
    except TypeError:
        # Older versions may not support param kw
        result = rag.query(query)

    # Try to extract contexts if result is a dict-like structure
    if isinstance(result, dict):
        ctxs = result.get("contexts") or result.get("chunks") or []
        if isinstance(ctxs, list):
            texts = []
            for c in ctxs:
                if isinstance(c, dict):
                    t = c.get("text") or c.get("content") or ""
                    t = str(t).strip()
                    if t:
                        texts.append(t)
            if texts:
                return "\n\n".join(texts)
        # fallback: stringify dict
        return json.dumps(result, ensure_ascii=False, indent=2)
    else:
        return str(result)
