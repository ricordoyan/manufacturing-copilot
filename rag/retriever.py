"""
Vector similarity search over a FAISS index with source-diversity guarantees.

Given a user query, this module embeds it with the NVIDIA NIM embedding
endpoint (using ``input_type="query"`` for asymmetric retrieval), then
finds the closest document chunks in the FAISS index while ensuring the
result set references multiple source documents.
"""

from collections import defaultdict
from typing import Optional

import faiss
import numpy as np

from rag.ingest import get_embeddings_batch


def retrieve_relevant_docs(
    query: str,
    index: faiss.IndexFlatL2,
    chunks: list[dict],
    top_k: int = 8,
    min_unique_sources: int = 3,
) -> list[dict]:
    """Retrieve relevant document chunks with source diversity.

    Ensures results include chunks from at least *min_unique_sources*
    different documents so the LLM can cross-reference multiple SOPs,
    QA reports, and maintenance logs.

    Algorithm
    ---------
    1. Over-fetch ``top_k * 3`` candidates from FAISS.
    2. Group candidates by source document.
    3. Guarantee at least one chunk from each unique source (up to
       *min_unique_sources*).
    4. Fill remaining slots with highest-scoring chunks regardless of
       source.
    5. Return the top *top_k* results sorted by ascending L2 distance.

    Returns
    -------
    list[dict]
        Each dict contains ``content``, ``source``, ``score`` (L2
        distance — lower is better), and ``chunk_index`` (position in
        the original chunk list).
    """
    if index is None or not chunks:
        return []

    # ── 1. Embed query ──────────────────────────────────────────────────
    query_vec = get_embeddings_batch([query], input_type="query")

    # ── 2. Over-fetch from FAISS ────────────────────────────────────────
    fetch_k = min(top_k * 3, index.ntotal)
    distances, indices = index.search(query_vec, fetch_k)

    # Build candidate list
    candidates: list[dict] = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue
        chunk = chunks[idx]
        candidates.append(
            {
                "content": chunk["content"],
                "source": chunk["source"],
                "score": round(float(dist), 4),
                "chunk_index": int(idx),
            }
        )

    if not candidates:
        return []

    # ── 3. Group by source ──────────────────────────────────────────────
    by_source: dict[str, list[dict]] = defaultdict(list)
    for c in candidates:
        by_source[c["source"]].append(c)

    # ── 4. Diversity selection ──────────────────────────────────────────
    selected: list[dict] = []
    used_indices: set[int] = set()

    # First pass: pick the best chunk from each unique source
    source_bests = sorted(
        [(src, grp[0]) for src, grp in by_source.items()],
        key=lambda x: x[1]["score"],
    )

    for src, best in source_bests:
        if len(selected) >= min_unique_sources:
            break
        selected.append(best)
        used_indices.add(best["chunk_index"])

    # Second pass: fill remaining slots with top-scoring chunks
    for c in candidates:
        if len(selected) >= top_k:
            break
        if c["chunk_index"] not in used_indices:
            selected.append(c)
            used_indices.add(c["chunk_index"])

    # ── 5. Sort final results by score ──────────────────────────────────
    selected.sort(key=lambda x: x["score"])
    return selected[:top_k]
