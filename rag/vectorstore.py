"""
rag/vectorstore.py - FAISS index management.

Responsibility:
  - Build a FAISS index from a list of text chunks + their embeddings.
  - Save the index (and the associated docstore) to disk.
  - Load a previously saved index from disk.
  - Perform nearest-neighbour search given a query embedding.
"""

import logging
import pickle
from pathlib import Path

import faiss
import numpy as np

from config import FAISS_INDEX_FILE, FAISS_DOCSTORE_FILE, TOP_K_RESULTS

logger = logging.getLogger(__name__)

# ─── Module-level cache so we don't reload the index on every request ─────────
_faiss_index: faiss.IndexFlatL2 | None = None
_docstore: list[str] | None = None          # Parallel list: index i → chunk text


def build_and_save_index(chunks: list[str], embeddings: list[list[float]]) -> None:
    """
    Create a FAISS flat L2 index from chunks + embeddings and persist to disk.

    Args:
        chunks:     List of raw text chunks (used as the document store).
        embeddings: Corresponding embedding vectors (must be same length as chunks).
    """
    global _faiss_index, _docstore

    vectors = np.array(embeddings, dtype="float32")
    dimension = vectors.shape[1]   # e.g. 384 for all-MiniLM-L6-v2

    # IndexFlatL2: exact nearest-neighbour search using L2 (Euclidean) distance.
    # Good choice for moderate corpus sizes (hundreds of thousands of vectors).
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)

    logger.info(f"Built FAISS index with {index.ntotal} vectors (dim={dimension}).")

    # Save index binary
    faiss.write_index(index, str(FAISS_INDEX_FILE))
    # Save docstore (plain list of chunk strings) alongside the index
    with open(FAISS_DOCSTORE_FILE, "wb") as f:
        pickle.dump(chunks, f)

    logger.info(f"Index saved to {FAISS_INDEX_FILE}")

    # Update in-memory cache
    _faiss_index = index
    _docstore = chunks


def load_index() -> tuple[faiss.IndexFlatL2, list[str]]:
    """
    Load the FAISS index and docstore from disk.
    Uses a module-level cache so disk I/O happens only once per process.

    Returns:
        (faiss_index, docstore) tuple.

    Raises:
        FileNotFoundError: If no index has been saved yet.
    """
    global _faiss_index, _docstore

    # Return cached version if already loaded
    if _faiss_index is not None and _docstore is not None:
        return _faiss_index, _docstore

    if not FAISS_INDEX_FILE.exists() or not FAISS_DOCSTORE_FILE.exists():
        raise FileNotFoundError(
            "No FAISS index found. Please upload a PDF first."
        )

    logger.info("Loading FAISS index from disk…")
    _faiss_index = faiss.read_index(str(FAISS_INDEX_FILE))

    with open(FAISS_DOCSTORE_FILE, "rb") as f:
        _docstore = pickle.load(f)

    logger.info(f"Index loaded: {_faiss_index.ntotal} vectors, {len(_docstore)} chunks.")
    return _faiss_index, _docstore


def search(query_embedding: list[float], k: int = TOP_K_RESULTS) -> list[str]:
    """
    Return the top-k most relevant text chunks for a query embedding.

    Args:
        query_embedding: Embedded query vector.
        k:               Number of results to return.

    Returns:
        List of matching text chunks (closest first).
    """
    index, docstore = load_index()

    query_vector = np.array([query_embedding], dtype="float32")
    # D = distances, I = indices of nearest neighbours
    distances, indices = index.search(query_vector, k)

    results = []
    for idx in indices[0]:
        if idx != -1:           # -1 means no result was found for that slot
            results.append(docstore[idx])

    return results


def invalidate_cache() -> None:
    """
    Clear the in-memory index cache.
    Call this after building a new index so the next query loads the fresh data.
    """
    global _faiss_index, _docstore
    _faiss_index = None
    _docstore = None
