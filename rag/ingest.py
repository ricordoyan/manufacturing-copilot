"""
Document ingestion, chunking, embedding, and FAISS index creation.

This module handles the offline pipeline:
  1. Load manufacturing documents (Markdown, TXT, PDF) from disk.
  2. Split them into retrieval-friendly chunks.
  3. Embed each chunk via the NVIDIA NIM embedding endpoint.
  4. Build a FAISS IndexFlatL2 and persist it alongside chunk metadata.
"""

import os
import pickle
from typing import Optional

import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

from config import (
    DOCS_DIR,
    FAISS_INDEX_PATH,
    NVIDIA_API_KEY,
    NVIDIA_BASE_URL,
    NVIDIA_EMBED_MODEL,
)

# ── NVIDIA NIM client (OpenAI-compatible) ───────────────────────────────────
_client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY)


# ── 1. Load documents ──────────────────────────────────────────────────────

def load_documents(docs_dir: str = DOCS_DIR) -> list[dict]:
    """Load all .md, .txt, and .pdf files from *docs_dir*.

    Returns a list of ``{"content": str, "source": str}`` dicts.
    PDF support requires ``pypdf``; if it is not installed, PDFs are skipped
    with a warning rather than crashing.
    """
    documents: list[dict] = []

    if not os.path.isdir(docs_dir):
        print(f"⚠️  Documents directory '{docs_dir}' not found.")
        return documents

    for fname in sorted(os.listdir(docs_dir)):
        fpath = os.path.join(docs_dir, fname)
        if not os.path.isfile(fpath):
            continue

        ext = os.path.splitext(fname)[1].lower()

        if ext in {".md", ".txt"}:
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
            if text.strip():
                documents.append({"content": text, "source": fname})

        elif ext == ".pdf":
            try:
                from pypdf import PdfReader  # type: ignore

                reader = PdfReader(fpath)
                pages = [p.extract_text() or "" for p in reader.pages]
                text = "\n".join(pages).strip()
                if text:
                    documents.append({"content": text, "source": fname})
            except ImportError:
                print(f"⚠️  Skipping '{fname}' — install `pypdf` for PDF support.")
            except Exception as exc:
                print(f"⚠️  Failed to read '{fname}': {exc}")

    return documents


# ── 2. Chunk documents ─────────────────────────────────────────────────────

def chunk_documents(
    documents: list[dict],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> list[dict]:
    """Split documents into retrieval-sized chunks.

    ``chunk_size=800`` characters keeps key paragraphs (root-cause
    sections, SOP procedures) intact within a single chunk.
    ``chunk_overlap=150`` provides generous overlap so that boundary
    information is not lost.

    Each chunk is prefixed with ``[Source: <filename>]`` so the LLM and
    the embedding model always know which document a chunk belongs to.

    Returns a list of ``{"content": str, "source": str}`` dicts.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[dict] = []
    for doc in documents:
        parts = splitter.split_text(doc["content"])
        source = doc["source"]
        for part in parts:
            # Prepend source context for better retrieval & LLM grounding
            prefixed = f"[Source: {source}]\n{part}"
            chunks.append({"content": prefixed, "source": source})

    return chunks


# ── 3. Embed texts via NVIDIA NIM ──────────────────────────────────────────

def get_embeddings_batch(
    texts: list[str],
    batch_size: int = 10,
    input_type: str = "passage",
) -> np.ndarray:
    """Embed a list of texts using the NVIDIA NIM embedding endpoint.

    The NVIDIA embedding API has per-request limits, so we process in
    batches of *batch_size*.  ``input_type`` should be ``"passage"`` for
    document chunks and ``"query"`` for user queries (asymmetric retrieval).

    Returns an (N, D) float32 numpy array.
    """
    all_embeddings: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        try:
            response = _client.embeddings.create(
                input=batch,
                model=NVIDIA_EMBED_MODEL,
                encoding_format="float",
                extra_body={"input_type": input_type},
            )
            for item in response.data:
                all_embeddings.append(item.embedding)
        except Exception as exc:
            print(f"❌  Embedding API error (batch {start}–{start + len(batch)}): {exc}")
            # Insert zero vectors so indices stay aligned
            dim = len(all_embeddings[0]) if all_embeddings else 1024
            for _ in batch:
                all_embeddings.append([0.0] * dim)

    return np.array(all_embeddings, dtype=np.float32)


# ── 4. Build & save FAISS index ────────────────────────────────────────────

def build_faiss_index(chunks: list[dict]) -> faiss.IndexFlatL2:
    """Embed all chunks, build a FAISS L2 index, and save to disk.

    Saves two files:
      - ``{FAISS_INDEX_PATH}.index``  — the FAISS binary index
      - ``{FAISS_INDEX_PATH}.pkl``    — pickled chunk metadata list
    """
    texts = [c["content"] for c in chunks]
    embeddings = get_embeddings_batch(texts, input_type="passage")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Persist
    os.makedirs(os.path.dirname(FAISS_INDEX_PATH) or ".", exist_ok=True)
    faiss.write_index(index, f"{FAISS_INDEX_PATH}.index")
    with open(f"{FAISS_INDEX_PATH}.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print(f"✅  FAISS index built — {index.ntotal} vectors, dimension {dim}")
    return index


# ── 5. Load saved index ────────────────────────────────────────────────────

def load_faiss_index() -> tuple[Optional[faiss.IndexFlatL2], Optional[list[dict]]]:
    """Load a previously saved FAISS index and its chunk metadata.

    Returns ``(index, chunks)`` or ``(None, None)`` if the files are missing.
    """
    index_path = f"{FAISS_INDEX_PATH}.index"
    meta_path = f"{FAISS_INDEX_PATH}.pkl"

    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        return None, None

    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        chunks = pickle.load(f)

    return index, chunks
