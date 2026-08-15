"""
embedding.py

Embedding pipeline for RAG document processing.
Takes chunks (output of chunking.py) and converts them into vector
embeddings, ready for storage in a vector database.

Supports two backends:
  1. OpenAI embeddings API   (requires OPENAI_API_KEY env var)
  2. sentence-transformers   (local, free, no API key needed)

All vectors are L2-normalized so cosine similarity search works correctly.
"""

import os
import time
import numpy as np
from typing import List, Dict, Optional


# ---------------------------------------------------------------------------
# Helper: L2 normalize a vector (unit length) — required for cosine similarity
# ---------------------------------------------------------------------------
def normalize_vector(vector: List[float]) -> List[float]:
    arr = np.array(vector, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm == 0:
        return arr.tolist()
    return (arr / norm).tolist()


# ---------------------------------------------------------------------------
# Backend 1: OpenAI embeddings
# ---------------------------------------------------------------------------
def embed_with_openai(texts: List[str], model: str = "text-embedding-3-small",
                       batch_size: int = 100, max_retries: int = 3) -> List[List[float]]:
    """
    Embed a list of texts using OpenAI's embedding API, in batches.
    Requires: pip install openai   and   OPENAI_API_KEY env var set.
    """
    from openai import OpenAI
    client = OpenAI()

    all_embeddings: List[List[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        for attempt in range(max_retries):
            try:
                response = client.embeddings.create(model=model, input=batch)
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                wait = 2 ** attempt
                print(f"OpenAI embedding error: {e}. Retrying in {wait}s...")
                time.sleep(wait)

    return all_embeddings


# ---------------------------------------------------------------------------
# Backend 2: sentence-transformers (local, free)
# ---------------------------------------------------------------------------
_ST_MODEL_CACHE = {}

def embed_with_sentence_transformers(texts: List[str],
                                      model_name: str = "all-MiniLM-L6-v2",
                                      batch_size: int = 32) -> List[List[float]]:
    """
    Embed a list of texts locally using sentence-transformers.
    Requires: pip install sentence-transformers
    """
    from sentence_transformers import SentenceTransformer

    if model_name not in _ST_MODEL_CACHE:
        _ST_MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    model = _ST_MODEL_CACHE[model_name]

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return embeddings.tolist()


# ---------------------------------------------------------------------------
# Orchestrator: embed a list of chunk dicts (from chunking.py)
# ---------------------------------------------------------------------------
def embed_chunks(chunks: List[Dict], backend: str = "sentence_transformers",
                  model: Optional[str] = None, normalize: bool = True) -> List[Dict]:
    """
    Takes chunk dicts (with a "text" field, as produced by chunking.py)
    and adds an "embedding" field to each, plus the model used.

    backend: "openai" or "sentence_transformers"
    model: override the default model for the chosen backend
    normalize: L2-normalize each vector (recommended for cosine similarity)
    """
    texts = [c["text"] for c in chunks]

    if backend == "openai":
        model = model or "text-embedding-3-small"
        vectors = embed_with_openai(texts, model=model)
    elif backend == "sentence_transformers":
        model = model or "all-MiniLM-L6-v2"
        vectors = embed_with_sentence_transformers(texts, model_name=model)
    else:
        raise ValueError(f"Unknown backend '{backend}'. Choose 'openai' or 'sentence_transformers'")

    for chunk, vector in zip(chunks, vectors):
        if normalize:
            vector = normalize_vector(vector)
        chunk["embedding"] = vector
        chunk["embedding_model"] = model
        chunk["embedding_dim"] = len(vector)

    return chunks


# ---------------------------------------------------------------------------
# Utility: cosine similarity (for quick sanity checks / testing retrieval)
# ---------------------------------------------------------------------------
def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


if __name__ == "__main__":

    sample_chunks = [
        {"chunk_id": "1", "index": 0, "text": "RAG combines retrieval with generation."},
        {"chunk_id": "2", "index": 1, "text": "Chunk size affects retrieval quality significantly."},
        {"chunk_id": "3", "index": 2, "text": "Bananas are a good source of potassium."},
    ]

    # Uses local sentence-transformers by default (no API key needed)
    embedded = embed_chunks(sample_chunks, backend="sentence_transformers")

    for c in embedded:
        print(f"[{c['index']}] dim={c['embedding_dim']} | {c['text']}")

    # quick relevance sanity check
    sim = cosine_similarity(embedded[0]["embedding"], embedded[1]["embedding"])
    print(f"\nSimilarity (RAG-related chunks): {sim:.3f}")

    sim2 = cosine_similarity(embedded[0]["embedding"], embedded[2]["embedding"])
    print(f"Similarity (unrelated chunk): {sim2:.3f}")